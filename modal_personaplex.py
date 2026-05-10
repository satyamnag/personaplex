import modal

app = modal.App("biology-teacher-personaplex")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
        "libsndfile1",
        "libopus-dev"
    )
    .pip_install(
        "torch",
        "torchaudio",
        "torchvision",
        "accelerate",
        "transformers",
        "websockets",
        "fastapi",
        "uvicorn",
        "numpy",
        "soundfile"
    )
    .run_commands(
        "git clone https://github.com/NVIDIA/personaplex.git /root/personaplex",
        "cd /root/personaplex && pip install -e ."
    )
)

volume = modal.Volume.from_name(
    "personaplex-cache",
    create_if_missing=True
)

hf_secret = modal.Secret.from_name("huggingface-secret")

@app.function(
    gpu="H100:1",
    timeout=86400,
    scaledown_window=900,
    image=image,
    volumes={"/cache": volume},
    secrets=[hf_secret],
    memory=65536,
    cpu=16
)
@modal.web_server(port=8000)
def run_server():

    import os
    import subprocess

    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"

    os.chdir("/root/personaplex")

    subprocess.Popen([
        "python",
        "server.py",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--voice",
        "NATF2",
        "--persona-file",
        "/root/personaplex/personas/biology_teacher.txt"
    ]).wait()
