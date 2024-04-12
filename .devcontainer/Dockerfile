FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Create a non-root user
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user with the specified UID and GID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 gcc g++

# Switch to the new user
USER $USERNAME

RUN sudo apt install -y wget && \
    mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh
RUN ~/miniconda3/bin/conda init bash
RUN conda create -y -n openodd python=3.11 pytorch pytorch-cuda=12.1 torchvision scikit-learn json5 matplotlib scipy tqdm pandas 'cython>=0.29.30' 'faiss-gpu>=1.7.2' 'gdown>=4.7.1'  -c pytorch -c nvidia -c conda-forge
RUN bash -c "source activate openodd && pip install --no-cache-dir timm 'libmr>=0.1.9' 'diffdist>=0.1' 'opencv-python>=4.4.0.46' 'imgaug>=0.4.0' 'pyyaml>=5.4.1'" 

# Set the default work directory
WORKDIR /workspace
