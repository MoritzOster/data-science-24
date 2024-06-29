# Use the PyTorch image with CUDA 11.1 and cuDNN 8
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install the required Python packages in a single RUN command
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    threadpoolctl \
    pandas \
    numpy \
    pickle5 \
    scipy \
    DEAP \
    update-checker \
    tqdm \
    stopit \
    joblib \
    scikit-learn \
    xgboost \
    torch \
<<<<<<< HEAD
    pyarrow \
=======
>>>>>>> 7eea67d6a81066d28bf7a5630d745c6acf2ebb90
    PyWavelets && \
    pip install --no-cache-dir tpot