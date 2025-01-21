FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
ENV PATH="/root/miniconda3/bin:$PATH"
ARG PATH="/root/miniconda3/bin:$PATH"
RUN apt-get update && apt-get install -y libopenblas-dev
RUN apt-get update && apt-get install -y curl wget gcc build-essential
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN apt install libx11-dev -y
RUN apt-get install libgles2-mesa-dev -y
# install conda
# Install Miniconda on x86 or ARM platforms
RUN DEBIAN_FRONTEND=noninteractive apt-get install python3-opengl x11-apps zip libglib2.0-0 libsm6 libxrender1 libxext6 -y
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

ENV PULSE_SERVER 127.0.0.1:4713

# Default environment variables (password is "mypasswd")
ENV TZ UTC
ENV REFRESH 60
ENV PASSWD mypasswd
ENV NOVNC_ENABLE false
ENV WEBRTC_ENCODER nvh264enc
ENV WEBRTC_ENABLE_RESIZE false
ENV ENABLE_AUDIO false
ENV ENABLE_BASIC_AUTH true

# Temporary fix for NVIDIA container repository
RUN apt-get clean && \
    apt-key adv --fetch-keys "https://developer.download.nvidia.com/compute/cuda/repos/$(cat /etc/os-release | grep '^ID=' | awk -F'=' '{print $2}')$(cat /etc/os-release | grep '^VERSION_ID=' | awk -F'=' '{print $2}' | sed 's/[^0-9]*//g')/x86_64/3bf863cc.pub" && \
    rm -rf /var/lib/apt/lists/*

# Install locales to prevent errors
RUN apt-get clean && \
    apt-get update && apt-get install --no-install-recommends -y locales && \
    rm -rf /var/lib/apt/lists/* && \
    locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install Xvfb, Xfce Desktop, and others
RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install --no-install-recommends -y \
        software-properties-common \
        apt-transport-https \
        apt-utils \
        build-essential \
        ca-certificates \
        cups-filters \
        cups-common \
        cups-pdf \
        curl \
        file \
        wget \
        bzip2 \
        gzip \
        p7zip-full \
        xz-utils \
        zip \
        unzip \
        zstd \
        gcc \
        git \
        jq \
        make \
        python \
        python-numpy \
        python3 \
        python3-cups \
        python3-numpy \
        mlocate \
        nano \
        vim \
        htop \
        xarchiver \
        brltty \
        brltty-x11 \
        desktop-file-utils \
        gucharmap \
        mpd \
        onboard \
        orage \
        parole \
        policykit-desktop-privileges \
        libpulse0 \
        pavucontrol \
        ristretto \
        supervisor \
        thunar \
        thunar-volman \
        thunar-archive-plugin \
        thunar-media-tags-plugin \
        net-tools \
        libgtk-3-bin \
        vainfo \
        vdpauinfo \
        mesa-utils \
        mesa-utils-extra \
        dbus-x11 \
        libdbus-c++-1-0v5 \
        dmz-cursor-theme \
        numlockx \
        xcursor-themes \
        xvfb \
        xfburn &&\
    apt-get install -y libreoffice && \
    # Support libva and VA-API through NVIDIA VDPAU
    curl -fsSL -o /tmp/vdpau-va-driver.deb "https://launchpad.net/~saiarcot895/+archive/ubuntu/chromium-dev/+files/vdpau-va-driver_0.7.4-6ubuntu2~ppa1~18.04.1_amd64.deb" && apt-get install --no-install-recommends -y /tmp/vdpau-va-driver.deb && rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/*

# Install Vulkan (for offscreen rendering only)
RUN if [ "${UBUNTU_RELEASE}" = "18.04" ]; then apt-get update && apt-get install --no-install-recommends -y libvulkan1 vulkan-utils; else apt-get update && apt-get install --no-install-recommends -y libvulkan1 vulkan-tools; fi && \
    rm -rf /var/lib/apt/lists/* && \
    VULKAN_API_VERSION=$(dpkg -s libvulkan1 | grep -oP 'Version: [0-9|\.]+' | grep -oP '[0-9]+(\.[0-9]+)(\.[0-9]+)') && \
    mkdir -p /etc/vulkan/icd.d/ && \
    echo "{\n\
    \"file_format_version\" : \"1.0.0\",\n\
    \"ICD\": {\n\
        \"library_path\": \"libGLX_nvidia.so.0\",\n\
        \"api_version\" : \"${VULKAN_API_VERSION}\"\n\
    }\n\
}" > /etc/vulkan/icd.d/nvidia_icd.json

# Install VirtualGL
ARG VIRTUALGL_VERSION_MIN=3.0.2
RUN VIRTUALGL_VERSION=$(curl -fsSL "https://api.github.com/repos/VirtualGL/virtualgl/releases/67016359" | jq -r '.tag_name' | sed 's/[^0-9\.\-]*//g') && \
    if [ "$(echo "${VIRTUALGL_VERSION_MIN}" "${VIRTUALGL_VERSION}" | tr " " "\n" | sort -V | head -n 1)" = "${VIRTUALGL_VERSION_MIN}" ]; then \
    curl -fsSL -O https://sourceforge.net/projects/virtualgl/files/virtualgl_${VIRTUALGL_VERSION}_amd64.deb && \
    curl -fsSL -O https://sourceforge.net/projects/virtualgl/files/virtualgl32_${VIRTUALGL_VERSION}_amd64.deb; \
    else VIRTUALGL_VERSION=${VIRTUALGL_VERSION_MIN} && \
    curl -fsSL -O https://sourceforge.net/projects/virtualgl/files/virtualgl_${VIRTUALGL_VERSION}_amd64.deb && \
    curl -fsSL -O https://sourceforge.net/projects/virtualgl/files/virtualgl32_${VIRTUALGL_VERSION}_amd64.deb; fi &&\
    apt-get update && apt-get install -y --no-install-recommends ./virtualgl_${VIRTUALGL_VERSION}_amd64.deb ./virtualgl32_${VIRTUALGL_VERSION}_amd64.deb && \
    rm virtualgl_${VIRTUALGL_VERSION}_amd64.deb virtualgl32_${VIRTUALGL_VERSION}_amd64.deb && \
    rm -rf /var/lib/apt/lists/* && \
    chmod u+s /usr/lib/libvglfaker.so && \
    chmod u+s /usr/lib/libdlfaker.so && \
    chmod u+s /usr/lib32/libvglfaker.so && \
    chmod u+s /usr/lib32/libdlfaker.so && \
    chmod u+s /usr/lib/i386-linux-gnu/libvglfaker.so && \
    chmod u+s /usr/lib/i386-linux-gnu/libdlfaker.so

# Install Python application, and web application
RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        python3 \
        python3-pip \
        python3-dev \
        python3-gi \
        python3-setuptools \
        python3-wheel \
        tzdata \
        sudo \
        udev \
        xclip \
        x11-utils \
        xdotool \
        wmctrl \
        jq \
        gdebi-core \
        x11-xserver-utils \
        xserver-xorg-core \
        libopus0 \
        libgdk-pixbuf2.0-0 \
        libsrtp2-1 \
        libxdamage1 \
        libxml2-dev \
        libcairo-gobject2 \
        libpulse0 \
        libpangocairo-1.0-0 \
        libgirepository1.0-dev \
        libjpeg-dev \
        zlib1g-dev \
	netcat \
	iproute2 \ 
	iputils-ping \ 
	openssh-client \
        x264 && \
    rm -rf /var/lib/apt/lists/*

ENV DISPLAY :0
ENV VGL_REFRESHRATE 60
ENV VGL_ISACTIVE 1
ENV VGL_DISPLAY egl
ENV VGL_WM 1

WORKDIR /app
COPY ./ /app
RUN chmod 755 /app/virtualgl/virtualgl.sh

# create env with python 3.8
RUN conda init bash \
    && . ~/.bashrc \
    && conda create --name calvin python=3.8.5 \
    && conda activate calvin \
    && pip install setuptools==58 numpy==1.23.5 \
    && pip install pyyaml \
    && git clone --recurse-submodules https://github.com/mees/calvin.git \
    && export CALVIN_ROOT=$(pwd)/calvin \
    && cd calvin \
    && sh install.sh \
    && cd .. \
    && conda env update --name calvin --file environment.yaml \ 
    && pip install diffusers["torch"] \
    && pip install dgl==1.1.3+cu116 -f https://data.dgl.ai/wheels/cu116/dgl-1.1.3%2Bcu116-cp38-cp38-manylinux1_x86_64.whl \
    && pip install packaging \
    && pip install ninja \
    && pip install flash-attn==2.5.9.post1 --no-build-isolation \
    && pip install -e .

RUN rm /root/miniconda3/envs/calvin/lib/python3.8/site-packages/pyrender/platforms/egl.py
COPY virtualgl/egl.py /root/miniconda3/envs/calvin/lib/python3.8/site-packages/pyrender/platforms/


ENTRYPOINT ["virtualgl/virtualgl.sh"]
CMD ["/bin/bash"]
