FROM python:3.10
WORKDIR /home/Symmetric-Meta-Reinforcement-Learning

# Copy packages
COPY smrl smrl
COPY specific specific
COPY symmetric_networks symmetric_networks
COPY experiments experiments
COPY configs configs
COPY submodules submodules
COPY ./*.py .
COPY README.md .

# Set up environment
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get install unzip && \
    apt-get install -y nano
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e ./submodules/meta-environments && \
    pip install -e ./submodules/MRL-analysis-tools && \
    pip install -e ./submodules/rlkit && \
    pip install -e ./submodules/symmetrizer && \
    pip install -e .

# MuJoCo
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
RUN mkdir -p /home/.mujoco
RUN tar -xf mujoco210-linux-x86_64.tar.gz -C /home/.mujoco
ENV LD_LIBRARY_PATH=/home/.mujoco/mujoco210/bin
ENV MUJOCO_PY_MUJOCO_PATH=/home/.mujoco/mujoco210
RUN apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
RUN pip install mujoco-py


CMD ["python", "check_setup.py"]