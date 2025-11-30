const videoElement = document.getElementById('video');
const updatePromptButton = document.getElementById('updatePrompt');
const promptTextarea = document.querySelector('textarea');
const statusElement = document.getElementById('status');
const placeholderElement = document.getElementById('placeholder');

const pc = new RTCPeerConnection();

const clientChannel = pc.createDataChannel("client");
clientChannel.onopen = () => {
    console.log("Client channel open");
};
clientChannel.onmessage = (e) => {
    console.log("Message from server (client channel):", e.data);
    try {
        const data = JSON.parse(e.data);
        if (data.type === 'status' && data.message) {
            statusElement.textContent = data.message;
        }
    } catch (err) {
        // Not JSON, ignore
    }
};

let serverChannel = null;

// Handle server-initiated channel
pc.ondatachannel = (event) => {
    const ch = event.channel;
    console.log("Server channel arrived:", ch.label);
    ch.onmessage = (e) => {
        console.log("Server channel message:", e.data);
        try {
            const data = JSON.parse(e.data);
            if (data.type === 'status' && data.message) {
                statusElement.textContent = data.message;
            }
        } catch (err) {
            // Not JSON, ignore
        }
    };
    ch.onopen = () => {
        console.log("Server channel open");
    }
    serverChannel = ch; 
};

pc.ontrack = (event) => {
    console.log('Received track:', event.track.kind);
    videoElement.srcObject = event.streams[0];
    videoElement.play();
};

// Handle ICE candidates (send to server)
pc.onicecandidate = (event) => {
    if (event.candidate) {
        fetch('/ice-candidate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ candidate: event.candidate, session_id: 'default' })
        });
    }
};

// Add transceiver for receiving video from server
pc.addTransceiver('video', { direction: 'recvonly' });

async function start() {    

    console.log('Starting connection...');
    // Create offer from client
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    // Send offer to server and get answer
    const response = await fetch('/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sdp: offer.sdp, session_id: 'default' })
    });    
    const answerData = await response.json();
    
    // Set server's answer as remote description
    await pc.setRemoteDescription({ type: 'answer', sdp: answerData.sdp });
}

function updatePlaceholder() {
    if (promptTextarea.value.trim() === '') {
        placeholderElement.classList.add('show');
    } else {
        placeholderElement.classList.remove('show');
    }
}

window.onload = async () => {
    await start();
    promptTextarea.focus();
    promptTextarea.select();
    updatePlaceholder();
};

function sendEcho() {
    if (serverChannel && serverChannel.readyState === 'open') {
        let j = JSON.stringify({cmd: 'updatePrompt', prompt: 'A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.'});
        serverChannel.send(j);
    }
}

function updatePrompt() {
    if (serverChannel && serverChannel.readyState === 'open') {
        let prompt = promptTextarea.value.trim();
        if (prompt != '') {
            let j = JSON.stringify({cmd: 'updatePrompt', prompt: prompt});
            promptTextarea.value = "";
            serverChannel.send(j);
        }
    }
    promptTextarea.focus();
    updatePlaceholder();
}

updatePromptButton.onclick = updatePrompt;

promptTextarea.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        event.preventDefault();
        updatePrompt();
    }
});

promptTextarea.addEventListener('input', updatePlaceholder);
