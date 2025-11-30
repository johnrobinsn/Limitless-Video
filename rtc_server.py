# Infinite Forcing Demo with webRTC (c) 2025 John Robinson
# WebRTC server for real-time video generation using Infinite Forcing model

import os
import random
import time
import argparse
import urllib.request
import numpy as np
import torch
from omegaconf import OmegaConf
import queue
from threading import Event
import uuid

from pipeline import CausalInferencePipeline
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from demo_utils.utils import generate_timestamp
from demo_utils.memory import gpu, get_cuda_free_memory_gb

import traceback
import sys

import asyncio
from aiohttp import web
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCSessionDescription, RTCIceCandidate
import json
import av
import logging

# squelch logs
_log_aiohttp = logging.getLogger("aiohttp")
_log_aiohttp.setLevel(logging.WARNING)
_log_aiortc = logging.getLogger("aiortc")
_log_aiortc.setLevel(logging.WARNING)
_log_aioice = logging.getLogger("aioice")
_log_aioice.setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Infinite Forcing")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=5001)
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument("--config_path", type=str, default='configs/self_forcing_dmd_vsink1.yaml')
args = parser.parse_args()

# Constants
WIDTH, HEIGHT = 832, 480  # landscape
# WIDTH, HEIGHT = 480, 832  # portrait
WLATENT, HLATENT = WIDTH // 8, HEIGHT // 8
MAX_FPS, MIN_FPS, TGT_FPS = 60, 3, 16
TGT_T = 1000 / TGT_FPS

log.info(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')

# Store the main event loop globally for cross-thread access
main_event_loop = None
loop_time = None

def send_status(context, message):
    """Send status message if data channel is available"""
    dc = context.get('data_channel')
    if not dc or dc.readyState != "open":
        return
    
    def do_send():
        dc.send(json.dumps({'type': 'status', 'message': message}))
        log.debug(f"Sent status: {message}")
    
    main_event_loop.call_soon_threadsafe(do_send)

def initialize_vae_decoder():

    from demo_utils.taehv import TAEHV

    class DotDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    class TAEHVDiffusersWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dtype = torch.float16
            self.taehv = TAEHV(checkpoint_path="checkpoints/taew2_1.pth").to(self.dtype)
            self.config = DotDict(scaling_factor=1.0)  

        def decode(self, latents, return_dict=None):        
            return self.taehv.decode_video(latents,show_progress_bar=False).mul_(2).sub_(1)

    current_vae_decoder = TAEHVDiffusersWrapper()

    return current_vae_decoder

# Global models and loading state
text_encoder = None
transformer = None
vae_decoder = None
models_loaded = False

def load_models():
    """Load all models once at startup"""
    global text_encoder, transformer, vae_decoder, models_loaded
    
    if models_loaded:
        return
    
    log.info("Loading models (one-time initialization)...")

    log.info("Loading text encoder...")
    text_encoder = WanTextEncoder()  
    
    log.info("Loading causal diffusion model...")
    transformer = WanDiffusionWrapper(is_causal=True, local_attn_size=6, sink_size=1, timestep_shift=5)
    state_dict = torch.load("checkpoints/ema_model.pt", map_location="cpu")
    state_dict = state_dict['generator_ema']
    state_dict = {k.replace('._fsdp_wrapped_module', ''):v for k, v in state_dict.items()}
    transformer.load_state_dict(state_dict)

    log.info("Loading VAE decoder...")
    vae_decoder = initialize_vae_decoder()

    # Set to eval mode and freeze
    text_encoder.eval()
    transformer.eval()
    vae_decoder.eval()

    text_encoder.to(dtype=torch.bfloat16)
    transformer.to(dtype=torch.float16)
    vae_decoder.to(dtype=torch.float16)

    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae_decoder.requires_grad_(False)

    # Move to GPU
    text_encoder.to(gpu)
    transformer.to(gpu)
    vae_decoder.to(gpu)
    
    models_loaded = True
    log.info("Models loaded successfully!")

@torch.no_grad()
def generate_video_stream(prompt, seed, session_id):
    global text_encoder, transformer, vae_decoder, models_loaded
    context = sessions[session_id]['context']
    
    # Send status: waiting for models
    send_status(context, 'Loading models, please wait...')
    
    # Wait for models
    while not models_loaded:
        time.sleep(0.5)
        if not context.get('generation_active'):
            return
    
    # Send status: models loaded
    send_status(context, 'Models loaded! Starting generation...')
    
    log.info(f"Session {session_id}: Models ready, starting generation...")
    
    context = sessions[session_id]['context']
    stop_event = context['stop_event']
    frame_send_queue = context['frame_send_queue']

    # Create per-session pipeline using shared models
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    
    pipeline = CausalInferencePipeline(
        config,
        device=gpu,
        generator=transformer,
        text_encoder=text_encoder,
        vae=vae_decoder
    )
    
    context['pipeline'] = pipeline

    try:
        context['generation_active'] = True
        stop_event.clear()
        
        # Send status: starting generation
        send_status(context, 'Encoding prompt and initializing...')

        # Encode prompt
        context['conditional_dict'] = {
            k: v.to(dtype=torch.float16) 
            for k, v in text_encoder(text_prompts=[prompt]).items()
        }

        rnd = torch.Generator(gpu).manual_seed(seed)

        pipeline._initialize_kv_cache(batch_size=1, dtype=torch.float16, device=gpu)
        pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.float16, device=gpu)

        # Generation parameters
        current_num_frames = 3
        current_start_frame = 0
        vae_cache = None

        frameTimes = []
        frameIntervals = []

        idx = 0
        generation_id = context.get('generation_id', 0)
        conditional_dict = context['conditional_dict']
        while True:
            if generation_id != context.get('generation_id'):
                log.debug(f"Generation ID changed for session {session_id}: {generation_id} -> {context.get('generation_id')}")
                idx = 0
                current_start_frame = 0
                vae_cache = None
                pipeline._initialize_kv_cache(batch_size=1, dtype=torch.float16, device=gpu)
                pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.float16, device=gpu)          
                conditional_dict = context['conditional_dict']
                generation_id = context.get('generation_id')
                # Drain the queue
                while not frame_send_queue.empty():
                    try:
                        frame_send_queue.get_nowait()
                        frame_send_queue.task_done()
                    except queue.Empty:
                        break


            # Check stop condition at the start of each iteration
            if not context['generation_active'] or stop_event.is_set():
                log.debug(f"Generation stopped for session {session_id} at block {idx}")
                break

            block_start_time = time.time()
            
            noisy_input = torch.randn([1, current_num_frames, 16, HLATENT, WLATENT], device=gpu, dtype=torch.float16, generator=rnd)

            # Denoising loop
            for index, current_timestep in enumerate(pipeline.denoising_step_list):
                # Check stop condition more frequently during denoising
                if not context['generation_active'] or stop_event.is_set():
                    log.debug(f"Generation stopped during denoising for session {session_id}")
                    break

                timestep = torch.ones([1, current_num_frames], device=noisy_input.device,
                                      dtype=torch.int64) * current_timestep

                _, denoised_pred = transformer(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=pipeline.kv_cache1,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start_frame * pipeline.frame_seq_length
                )
                if index < len(pipeline.denoising_step_list) - 1:
                    next_timestep = pipeline.denoising_step_list[index + 1]
                    noisy_input = pipeline.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones([1 * current_num_frames], device=noisy_input.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])                

            if not context['generation_active'] or stop_event.is_set():
                break

            decode_start = time.time()

            if vae_cache is None:
                vae_cache = denoised_pred
            else:
                denoised_pred = torch.cat([vae_cache, denoised_pred], dim=1)
                vae_cache = denoised_pred[:, -3:, :, :, :]
            pixels = vae_decoder.decode(denoised_pred.half())  
            pixels = pixels[:, -12:, :, :, :]           

            decode_time = time.time() - decode_start
            log.debug(f"Block {idx+1} VAE decoding completed in {decode_time:.2f}s")

            block_frames = pixels.shape[1]

            pixels = pixels.cpu()

            block_frames = pixels.shape[1]

            for frame_idx in range(block_frames):
                frame_send_queue.put((pixels[0, frame_idx], idx))
                frameTimes.append(loop_time())

                if len(frameTimes) > 1:
                    frameIntervals.append(loop_time() - frameTimes[-2])

                    if len(frameTimes) > 50:
                        frameTimes = frameTimes[-50:]
                        frameIntervals = frameIntervals[-49:]
                
            if len(frameIntervals) > 0:
                avgInterval = sum(frameIntervals) / len(frameIntervals)
                if frame_send_queue.qsize() > 20:
                    avgInterval *= 0.8
                elif frame_send_queue.qsize() > 40:
                    avgInterval *= 0.5
                context['cur_t'] = context['cur_t'] * 0.9 + avgInterval * 0.1

            block_time = time.time() - block_start_time
            status_msg = f"Block {idx+1} completed in {block_time:.2f}s ({block_frames} frames) fps: {1/context['cur_t']:.2f}"
            print(status_msg, ' ' * 50, end="\r")
   
            # Send status to client
            send_status(context, 
                f"Block {idx+1} completed in {block_time:.2f}s "
                f"({block_frames} frames) fps: {1/context['cur_t']:.2f}")

            current_start_frame += current_num_frames
            idx += 1

    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__, file=sys.stderr)

        # Also capture as string (useful for logs)
        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        print("Current call stack:")
        print(tb_str)
        
        # Send error status to client
        send_status(context, f'Error during generation: {str(e)}')

    finally:
        context['generation_active'] = False
        stop_event.set()
        
        # Send completion status to client
        frame_send_queue.put(None)
        send_status(context, 'Generation stopped.')

def handle_prompt_update(data, session_id):
    """Handle prompt update - start generation or update existing"""
    context = sessions[session_id]['context']

    prompt = data.get('prompt', '')
    if not context['generation_active']:
        seed = data.get('seed', random.randint(0, 2**32))
        context['generation_active'] = True
        asyncio.create_task(asyncio.to_thread(generate_video_stream, prompt, seed, session_id))
    else:
        # Update prompt mid-generation
        conditional_dict = {
            k: v.to(dtype=torch.float16)
            for k, v in text_encoder(text_prompts=[prompt]).items()
        }
        context['conditional_dict'] = conditional_dict
        context['generation_id'] += 1  # Increment generation ID to signal new prompt

def stop_generation(session_id):
    """Stop generation for a specific session"""
    if session_id not in sessions:
        return
    
    context = sessions[session_id]['context']
    context['generation_active'] = False
    context['stop_event'].set()
    
    # Clear the queue to unblock any waiting operations
    frame_send_queue = context['frame_send_queue']
    try:
        # Drain the queue
        while not frame_send_queue.empty():
            try:
                frame_send_queue.get_nowait()
                frame_send_queue.task_done()
            except queue.Empty:
                break
        # Signal end
        frame_send_queue.put(None)
    except Exception as e:
        print(f"Error stopping generation for session {session_id}: {e}")

#### WebRTC Server Setup ####

class CustomVideoStreamTrack(VideoStreamTrack):
    """Video track that serves frames from queue with adaptive timing"""
    def __init__(self, session_id):
        super().__init__()
        self.session_id = session_id
        self.last_frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 128
        self.last_frame_time = 0

    async def recv(self):
        context = sessions[self.session_id]['context']
        cur_t = context['cur_t']
        frame_send_queue = context['frame_send_queue']
        
        now = loop_time()
        elapsed = now - self.last_frame_time

        # Get new frame if enough time has passed
        if elapsed > cur_t:
            try:
                result = frame_send_queue.get(block=False)
                if result is None:  # End signal
                    frame = self.last_frame
                else:
                    frame, _ = result
                    frame = (frame + 1.0) * 127.5
                    frame = frame.to(torch.uint8).cpu().numpy()
                    frame = np.transpose(frame, (1, 2, 0))
                    self.last_frame = frame
                    self.last_frame_time = now
            except queue.Empty:
                frame = self.last_frame
        else:
            frame = self.last_frame

        video_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        video_frame.pts, video_frame.time_base = await self.next_timestamp()
        return video_frame

sessions = {}  # Global dict: session_id -> {'pc': pc, 'context': {...}}

async def index(request):
    return web.FileResponse('static/index.html')

async def offer_handler(request):
    """Handle WebRTC offer and create new session"""
    data = await request.json()
    session_id = str(uuid.uuid4())
    log.debug(f'Received offer for session {session_id}')

    pc = RTCPeerConnection()

    # Initialize session
    sessions[session_id] = {
        'pc': pc,
        'context': {
            'conditional_dict': None,
            'cur_t': 1.0/12.0,
            'generation_active': False,
            'stop_event': Event(),
            'frame_send_queue': queue.Queue(),
            'data_channel': None,
            'pipeline': None,
            'generation_id': 0
        }
    }

    # Add video track
    pc.addTrack(CustomVideoStreamTrack(session_id))

    # Setup data channel
    dc = pc.createDataChannel("server")
    sessions[session_id]['context']['data_channel'] = dc

    @dc.on("message")
    def on_message(message):
        cmd_data = json.loads(message)
        if cmd_data.get('cmd') == 'updatePrompt':
            handle_prompt_update(cmd_data, session_id)

    @pc.on("connectionstatechange")
    async def on_state_change():
        if pc.connectionState == "closed":
            stop_generation(session_id)
            await asyncio.sleep(0.2)
            sessions.pop(session_id, None)
            log.info(f"Session {session_id} cleaned up")

    # Create answer
    await pc.setRemoteDescription(RTCSessionDescription(data['sdp'], 'offer'))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        'sdp': pc.localDescription.sdp,
        'session_id': session_id,
        'models_loaded': models_loaded
    })

async def on_startup(app):
    """Load models in background on startup"""
    global main_event_loop, loop_time
    main_event_loop = asyncio.get_event_loop()
    loop_time = main_event_loop.time

    log.info("Server starting, loading models in background...")
    asyncio.create_task(asyncio.to_thread(load_models))

async def ice_candidate_handler(request):
    data = await request.json()
    session_id = data.get('session_id', 'default')
    candidate_dict = data.get('candidate')

    session = sessions.get(session_id)
    if not session:
        return web.Response(status=404, text="PeerConnection not found")
    pc = session['pc']

    # Handle end-of-candidates or empty payload
    if not candidate_dict or not candidate_dict.get('candidate'):
        await pc.addIceCandidate(None)
        return web.Response(text="ICE candidate: end-of-candidates")

    log.debug(f'candidate_dict: {candidate_dict}')
    candidate_str = candidate_dict['candidate']
    # Split the candidate string into its components
    parts = candidate_str.split()

    # Ensure the string is not empty and has enough parts
    if parts and parts[0] == 'candidate:':
        # Remove the 'candidate:' prefix for easier processing
        parts[0] = parts[0][10:]

    # Extract the required information manually
    foundation = parts[0]
    component = int(parts[1])
    protocol = parts[2]
    priority = int(parts[3])
    ip = parts[4]
    port = int(parts[5])
    typ = parts[7] # 'typ' followed by 'host', 'srflx', etc.

    candidate = RTCIceCandidate(
        foundation=foundation,
        component=component,
        protocol=protocol,
        priority=priority,
        ip=ip,
        port=port,
        type=typ,
        sdpMid=candidate_dict.get('sdpMid'),
        sdpMLineIndex=candidate_dict.get('sdpMLineIndex')
    )
    await pc.addIceCandidate(candidate)
    return web.Response(text="ICE candidate added")

async def on_shutdown(app):
    """Cleanup on shutdown"""
    log.info("Shutting down...")
     
    # Stop all sessions
    for session_id in list(sessions.keys()):
        stop_generation(session_id)
     
    # Give threads a moment to exit gracefully
    await asyncio.sleep(0.5)
     
    # Close connections
    for session_data in sessions.values():
        await session_data['pc'].close()
     
    # Clear sessions
    sessions.clear()
    
    log.info("Shutdown complete")

app = web.Application()

# Add routes
app.router.add_get('/', index)
app.router.add_post('/offer', offer_handler)
app.router.add_post('/ice-candidate', ice_candidate_handler)
app.router.add_static('/static/', 'static/')

# Register startup and cleanup handlers
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

if __name__ == '__main__':
    log.info("Starting server immediately (models will load in background)...")
    try:
        web.run_app(app, port=args.port, host=args.host, handle_signals=True)
    except KeyboardInterrupt:
        log.info("Received keyboard interrupt, shutting down...")

