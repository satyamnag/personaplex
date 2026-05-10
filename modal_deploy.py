# modal_deploy.py
import modal
from modal import App, Image, asgi_app

# Create Modal app
app = App("personaplex-bio-teacher")

# Define container image
image = (
    Image.debian_slim()
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
    )
    .pip_install("accelerate==0.33.0")  # for CPU offload if needed
    .run_commands(
        "mkdir -p /app/moshi",
        "mkdir -p /app/cache",
    )
)

@app.function(
    image=image,
    gpu="A100-40GB",   # or "L40S" for lower cost; A100 for best perf
    container_idle_timeout=300,
    timeout=600,
    keep_warm=1,       # always keep one container alive for zero cold-start
)
@asgi_app()
def serve():
    import asyncio
    import sys
    import torch
    import sentencepiece
    from moshi.models import loaders, MimiModel, LMGen
    from moshi.utils.connection import create_ssl_context
    from moshi.utils.logging import setup_logger
    from aiohttp import web
    import secrets

    # Load models (cached by Modal)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = setup_logger(__name__)
    logger.info("Loading Mimi...")
    mimi = loaders.get_mimi(device=device)
    other_mimi = loaders.get_mimi(device=device)
    logger.info("Loading tokenizer...")
    text_tokenizer = sentencepiece.SentencePieceProcessor(loaders.TEXT_TOKENIZER_NAME)
    logger.info("Loading Moshi LM...")
    lm = loaders.get_moshi_lm(device=device, cpu_offload=False)  # GPU only for speed
    lm.eval()

    logger.info("Creating LMGen...")
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        use_sampling=True,
        temp=0.8,
        temp_text=0.7,
        top_k=250,
        top_k_text=25,
    )

    # Warmup
    for _ in range(4):
        chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
        codes = mimi.encode(chunk)
        _ = other_mimi.encode(chunk)
        for c in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, c:c + 1])
            if tokens is not None:
                _ = mimi.decode(tokens[:, 1:9])
                _ = other_mimi.decode(tokens[:, 1:9])
    if device == "cuda":
        torch.cuda.synchronize()
    logger.info("Model loaded and warmed up!")

    # ---- Build aiohttp WebSocket server ----
    # We replicate the core of server.py but without the opus stuff;
    # we'll use raw PCM streaming (16-bit signed ints) as in the client.
    app_web = web.Application()

    async def chat_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        peer = request.remote
        logger.info(f"Connection from {peer}")

        # Extract query params
        text_prompt = request.query.get("text_prompt", "You are a biology teacher.")
        voice_prompt_path = request.query.get("voice_prompt", "NATF0.pt")
        seed = int(request.query.get("seed", "-1"))

        if seed != -1:
            torch.manual_seed(seed)
            random.seed(seed)

        # Load voice prompt (adjust path to your voice directory if needed)
        voice_prompt = f"/app/moshi/voices/{voice_prompt_path}"
        lm_gen.load_voice_prompt(voice_prompt)
        lm_gen.text_prompt_tokens = text_tokenizer.encode(
            "<system> " + text_prompt + " <system>"
        )

        close = False
        lock = asyncio.Lock()

        # PCM buffers for streaming
        pcm_buffer = bytearray()

        async def recv_loop():
            nonlocal close
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY:
                    # data is raw PCM 16-bit mono at 24000 Hz
                    pcm_buffer.extend(msg.data)
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"ws error: {ws.exception()}")
                    break
                else:
                    break
            close = True

        async def inference_loop():
            nonlocal close
            while not close:
                # Process whenever we have enough samples for one frame
                while len(pcm_buffer) >= 2 * frame_size:
                    frame_bytes = pcm_buffer[:2 * frame_size]
                    del pcm_buffer[:2 * frame_size]
                    # Convert to tensor
                    samples = torch.frombuffer(bytearray(frame_bytes), dtype=torch.int16).float() / 32768.0
                    samples = samples.to(device).unsqueeze(0).unsqueeze(0)  # [1,1,T]
                    codes = mimi.encode(samples)
                    for c in range(codes.shape[-1]):
                        tokens = lm_gen.step(codes[:, :, c:c + 1])
                        if tokens is None:
                            continue
                        main_pcm = mimi.decode(tokens[:, 1:9])
                        main_pcm = main_pcm[0, 0].cpu().numpy()
                        # Convert to int16 and send back
                        audio_out = (main_pcm * 32767).astype("int16").tobytes()
                        await ws.send_bytes(b"\x01" + audio_out)
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            text = text_tokenizer.id_to_piece(text_token)
                            text = text.replace("▁", " ").encode("utf-8")
                            await ws.send_bytes(b"\x02" + text)
                await asyncio.sleep(0.001)
            logger.info("Inference loop ended.")

        async with lock:
            # System prompt steps (voice + text)
            mimi.reset_streaming()
            other_mimi.reset_streaming()
            lm_gen.reset_streaming()
            await lm_gen.step_system_prompts_async(mimi)
            mimi.reset_streaming()

            # Handshake
            await ws.send_bytes(b"\x00")
            logger.info("Handshake sent, starting loops.")
            recv_task = asyncio.create_task(recv_loop())
            inf_task = asyncio.create_task(inference_loop())
            done, pending = await asyncio.wait(
                [recv_task, inf_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await ws.close()
            logger.info("Session closed.")
        return ws

    app_web.router.add_get("/api/chat", chat_handler)
    return app_web
