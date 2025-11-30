# Limitless Video Demo

Watch text prompts come to life as **streaming video** — this WebRTC application demonstrates true real-time generative diffusion at 12-15 fps, generating infinite-length video on consumer hardware.

![Demo Video](https://github.com/user-attachments/assets/08251fd8-8144-440f-b85d-68bd67a7b761)

Building on the great Wan 2.1 Video Generation model that has undergone a post-training regime based on [Self Forcing](https://arxiv.org/abs/2506.08009)/[Infinite Forcing](https://github.com/SOTAMak1r/Infinite-Forcing) to transform it from a model that can generate very short fixed length video clips (5 seconds at a time) into a causal autoregressive model that can generate infinite length video streams in "real-time" 12-15 fps using a single consumer-grade gpu (NVIDIA 5090).

If you'd like more background you can check out my blog article, [Limitless Video](https://www.storminthecastle.com/posts/limitless/).

The full working source for this application is this repo.

## Project Structure

```
limitless-demo
├── rtc_server.py          # Main server application
├── static
│   ├── index.html     # HTML structure for the webpage
│   └── client.js      # Client-side JavaScript for WebRTC
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone https://github.com/johnrobinsn/Limitless-Video
   cd Limitless-Video
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment. You can create one using:
   ```
   conda create -n limitless python=3.13
   conda activate limitless
   ```
   Then install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download model checkpoints into a directory called `checkpoints`

   ``` bash
   wget -P checkpoints https://huggingface.co/johnrobinsn/Infinite-Forcing/resolve/main/ema_model.pt
   wget -P checkpoints https://huggingface.co/johnrobinsn/Infinite-Forcing/resolve/main/taew2_1.pth
   ```

4. **Run the server:**
   Start the server by running:

   ``` bash
   python rtc_server.py
   ```

5. **Access the application:**
   Open your web browser and navigate to the displayed URL to start generating video given a prompt.  The models will take a couple of minutes to load when the application first starts.

## Usage

Once the server is running, the webpage will display an area for the generated video to be displayed and an area for you to enter a text prompt.  An initial sample prompt will be provided but you can change that to whatever you want and then click the "Update Prompt" button to start generating video.  You can change the prompt at any time just click the "Update Prompt" button again to switch to that prompt.


