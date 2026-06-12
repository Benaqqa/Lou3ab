#THIS DOESNT WORK




FROM python:3.11-slim

# Install system dependencies for Pygame (SDL2 + X11) + audio/font fixes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libx11-dev \
    libxext-dev \
    xvfb \
    fontconfig \
    libasound2-dev \
    alsa-utils \
    pulseaudio \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Disable ALSA/audio to avoid sound errors (Pygame will fall back to no-sound mode)
ENV SDL_AUDIODRIVER=dummy
ENV XDG_RUNTIME_DIR=/tmp/runtime-root

CMD ["python", "experiments/play.py", "900011"]