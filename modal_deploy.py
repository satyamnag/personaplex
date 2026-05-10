# modal_deploy.py
# Deploy the PersonaPlex demo server on modal.com
# Uses an A100 GPU (or L40S for budget) with persistent HF cache and warm containers.
#
# Before deploying:
# 1) Create a volume for caching:  modal volume create hf-cache
# 2) Then run:  modal deploy modal_deploy.py

from modal import App, Image, Volume, asgi_app

# ---------- Configuration ----------
MODEL_REPO = "nvidia/personaplex-7b-v1"
GPU_TYPE = "A100-40GB"  # or "L40S"
KEEP_WARM = 1            # Always keep one instance alive to avoid cold‑starts
CONTAINER_IDLE_TIMEOUT = 600   # Keep container alive 10 min after last request

# Persistent volume for Hugging Face cache and voice files
hf_cache_volume = Volume.from_name("hf-cache", create_if_missing=True)

# ---------- Container Image ----------
image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("libopus-dev", "build-essential", "pkg-config")
    .pip_install(
        "aiohttp>=3.10.5,<3.11",
        "huggingface_hub>=0.24,<0.25",
        "sentencepiece==0.2",
        "sphn>=0.1.4,<0.2",
        "torch>=2.2.0,<2.5",
        "safetensors>=0.4.0,<0.5",
        "einops==0.7",
        "numpy>=1.26,<2.2",
        "sounddevice==0.5",        # unused but required by some import
        "aiohttp-asgi",            # to wrap aiohttp app as ASGI
    )
    # Copy the entire local PersonaPlex repository into the image
    .add_local_dir(".", "/app", copy=True)
    .workdir("/app")
)

app = App("personaplex-bio-teacher", image=image)

@app.function(
    volumes={"/cache": hf_cache_volume},
    gpu=GPU_TYPE,
    container_idle_timeout=CONTAINER_IDLE_TIMEOUT,
    keep_warm=KEEP_WARM,
    timeout=1200,
)
@asgi_app()
def serve():
    """ASGI entry point – loads models once and returns the aiohttp WebSocket app."""
    import asyncio
    import logging
    import secrets
    import os, sys, tarfile
    from pathlib import Path

    import torch
    import sentencepiece
    from aiohttp import web
    from aiohttp_asgi import ASGIApplication

    # Ensure Hugging Face cache is in the persistent volume
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"
    Path("/cache/huggingface").mkdir(parents=True, exist_ok=True)

    # Import PersonaPlex internals (now accessible because we copied the whole repo)
    from moshi.models import loaders, MimiModel, LMGen
    from moshi.utils.logging import setup_logger

    logger = setup_logger(__name__)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # ---------- Download & load models (cached by volume) ----------
    logger.info("Loading Mimi...")
    mimi_weight = loaders.hf_hub_download(MODEL_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)
    logger.info("Mimi loaded")

    logger.info("Loading text tokenizer...")
    tokenizer_path = loaders.hf_hub_download(MODEL_REPO, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
    logger.info("Tokenizer loaded")

    logger.info("Loading Moshi LM...")
    moshi_weight = loaders.hf_hub_download(MODEL_REPO, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=False)
    lm.eval()
    logger.info("Moshi loaded")

    # ---------- Extract voice prompts if not already present ----------
    voice_dir = Path("/cache/voices")
    if not voice_dir.exists():
        logger.info("Downloading and extracting voice prompts...")
        voices_tgz = loaders.hf_hub_download(MODEL_REPO, "voices.tgz")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path="/cache")
        # The archive should create a 'voices' subdirectory
        logger.info("Voice prompts extracted")

    # ---------- Create the server state (replaces ServerState from server.py) ----------
    frame_size = int(mimi.sample_rate / mimi.frame_rate)

    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        save_voice_prompt_embeddings=False,
        use_sampling=True,
        temp=0.8,   # audio temperature
        temp_text=0.7,
        top_k=250,
        top_k_text=25,
    )

    # Warmup (required for CUDA graph capture)
    logger.info("Warming up models...")
    for _ in range(4):
        chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
        codes = mimi.encode(chunk)
        _ = other_mimi.encode(chunk)
        for c in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, c: c + 1])
            if tokens is not None:
                _ = mimi.decode(tokens[:, 1:9])
                _ = other_mimi.decode(tokens[:, 1:9])
    if device == "cuda":
        torch.cuda.synchronize()
    logger.info("Warmup complete")

    # We reuse the same models and lm_gen for all connections (protected by a lock)
    state_lock = asyncio.Lock()
    # Keep streaming state open forever; we reset per connection
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    # ---------- WebSocket server (adapted from server.py) ----------
    async def chat_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        peer = request.remote
        logger.info(f"New connection from {peer}")

        # Extract query parameters (same as in server.py)
        text_prompt = request.query.get("text_prompt", "You are a biology teacher.")
        voice_prompt_file = request.query.get("voice_prompt", "NATF0.pt")
        seed = int(request.query.get("seed", "-1"))
        audio_temp = float(request.query.get("audio_temperature", "0.8"))
        audio_topk = int(request.query.get("audio_topk", "250"))
        text_temp = float(request.query.get("text_temperature", "0.7"))
        text_topk = int(request.query.get("text_topk", "25"))

        voice_prompt_path = voice_dir / voice_prompt_file
        if not voice_prompt_path.exists():
            logger.error(f"Voice prompt not found: {voice_prompt_path}")
            await ws.close()
            return ws

        # Set per-connection sampling parameters
        lm_gen.temp = audio_temp
        lm_gen.top_k = audio_topk
        lm_gen.temp_text = text_temp
        lm_gen.top_k_text = text_topk

        # Load prompts
        lm_gen.load_voice_prompt(str(voice_prompt_path))
        lm_gen.text_prompt_tokens = text_tokenizer.encode(
            "<system> " + text_prompt + " <system>"
        )

        if seed != -1:
            torch.manual_seed(seed)
            import random
            random.seed(seed)

        close = False
        opus_writer = sphn.OpusStreamWriter(mimi.sample_rate)
        opus_reader = sphn.OpusStreamReader(mimi.sample_rate)

        async def recv_loop():
            nonlocal close
            try:
                async for msg in ws:
                    if msg.type == web.WSMsgType.BINARY:
                        message = msg.data
                        if len(message) == 0:
                            continue
                        kind = message[0]
                        if kind == 1:  # audio
                            opus_reader.append_bytes(message[1:])
                    elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                        break
            finally:
                close = True
                logger.info("Receive loop ended")

        async def opus_loop():
            nonlocal close
            pcm_buffer = np.empty(0, dtype=np.float32)
            while not close:
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                pcm_buffer = np.concatenate([pcm_buffer, pcm]) if len(pcm_buffer) else pcm
                while pcm_buffer.shape[-1] >= frame_size:
                    chunk = pcm_buffer[:frame_size]
                    pcm_buffer = pcm_buffer[frame_size:]
                    chunk = torch.from_numpy(chunk).to(device=device)[None, None]
                    codes = mimi.encode(chunk)
                    for c in range(codes.shape[-1]):
                        tokens = lm_gen.step(codes[:, :, c: c + 1])
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == lm_gen.lm_model.dep_q + 1
                        main_pcm = mimi.decode(tokens[:, 1:9])
                        opus_writer.append_pcm(main_pcm[0, 0].cpu().numpy())
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = text_tokenizer.id_to_piece(text_token)
                            _text = _text.replace("▁", " ")
                            await ws.send_bytes(b"\x02" + _text.encode("utf-8"))
            # End of loop
            # If there is remaining pcm, flush? (optional)

        async def send_loop():
            nonlocal close
            while not close:
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        # Lock to serialize inference (only one conversation at a time)
        async with state_lock:
            # Reset streaming state for this session
            mimi.reset_streaming()
            other_mimi.reset_streaming()
            lm_gen.reset_streaming()

            # Run the prompt phase (voice + text)
            async def is_alive():
                return not close and not ws.closed
            await lm_gen.step_system_prompts_async(mimi, is_alive=is_alive)
            mimi.reset_streaming()

            # Send handshake
            await ws.send_bytes(b"\x00")
            logger.info("Handshake sent, entering main conversation loop")

            tasks = [
                asyncio.create_task(recv_loop()),
                asyncio.create_task(opus_loop()),
                asyncio.create_task(send_loop()),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await ws.close()
            logger.info("Session closed")
        return ws

    # Build aiohttp app
    app_web = web.Application()
    app_web.router.add_get("/api/chat", chat_handler)

    # Convert to ASGI application
    return ASGIApplication(app_web)
