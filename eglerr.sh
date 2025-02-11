#!/bin/bash

# Install required packages
apt-get update && apt-get install -y \
    xvfb \
    mesa-utils \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3-dev \
    patchelf

# Set up environment variables
export PYOPENGL_PLATFORM=osmesa
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330
export LIBGL_ALWAYS_SOFTWARE=1

# Optional: Set up virtual framebuffer if needed
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & 
export DISPLAY=:99

# Python environment setup
pip install PyOpenGL PyOpenGL-accelerate
