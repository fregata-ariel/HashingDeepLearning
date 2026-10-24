# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
# CUDA 12.6 retains SM75 support. Do not use a floating CUDA/torch tag.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04@sha256:badf6c452e8b1efea49d0bb956bef78adcf60e7f87ac77333208205f00ac9ade AS components

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/gpu-venv \
    PATH=/opt/gpu-venv/bin:/usr/local/cuda/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda \
    CC=gcc-12 CXX=g++-12 CUDAHOSTCXX=g++-12 \
    MAX_JOBS=2 OMP_NUM_THREADS=2 TORCH_CUDA_ARCH_LIST=7.5 \
    PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git python3 python3-dev python3-venv \
    gcc-12 g++-12 cmake ninja-build time xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv "$VIRTUAL_ENV"
COPY docker/gpu/requirements.txt /opt/gpu-requirements.txt
# The official CUDA wheel index supplies Torch; resolved versions are retained.
RUN python -m pip install --no-cache-dir pip==25.3 \
    && python -m pip install --no-cache-dir --index-url https://pypi.org/simple \
       --extra-index-url https://download.pytorch.org/whl/cu126 -r /opt/gpu-requirements.txt \
    && python -m pip freeze --all > /opt/python-packages.txt \
    && dpkg-query -W > /opt/os-packages.txt
RUN git clone https://github.com/idiap/fast-transformers.git /opt/fast-transformers \
    && git -C /opt/fast-transformers checkout --detach f22c13716fc748bb21a7b226ada7f7b5f87f867f \
    && test "$(git -C /opt/fast-transformers rev-parse HEAD:fast_transformers/local_product/local_product_cuda.cu)" = 4d10eee240a53ae0bda4499e92b5d64115578310 \
    && test "$(git -C /opt/fast-transformers rev-parse HEAD:fast_transformers/causal_product/causal_product_cuda.cu)" = af60163dc631be5e008aa7d6f0af935bdb5f719b
COPY tools/bootstrap_nvidia_sanitizer.sh /opt/bootstrap_nvidia_sanitizer.sh
RUN bash /opt/bootstrap_nvidia_sanitizer.sh /opt/compute-sanitizer
WORKDIR /workspace

FROM components AS sycl
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang-18 libclang-18-dev llvm-18-dev lld-18 libomp-18-dev \
    libboost-context-dev libboost-fiber-dev libboost-filesystem-dev \
    libboost-test-dev libboost-system-dev libnuma-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git /opt/AdaptiveCpp \
    && git -C /opt/AdaptiveCpp checkout --detach 31db2e83903c1db9f50b03b620eb0a09224cd0eb \
    && git -C /opt/AdaptiveCpp archive HEAD > /tmp/acpp.tar \
    && echo '1f0816f8c6272334ee4c5ba8315828178c28a2bf35d82c82c07f3e3118be661a  /tmp/acpp.tar' | sha256sum --check \
    && rm /tmp/acpp.tar \
    && cmake -S /opt/AdaptiveCpp -B /tmp/acpp-build \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/acpp \
       -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 \
       -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm -DClang_DIR=/usr/lib/llvm-18/lib/cmake/clang \
       -DACPP_COMPILER_FEATURE_PROFILE=full -DWITH_CUDA_BACKEND=ON \
       -DWITH_ROCM_BACKEND=OFF -DWITH_OPENCL_BACKEND=OFF -DWITH_LEVEL_ZERO_BACKEND=OFF -DBUILD_TESTING=OFF \
    && cmake --build /tmp/acpp-build --parallel 2 \
    && cmake --install /tmp/acpp-build \
    && rm -rf /tmp/acpp-build /opt/AdaptiveCpp/.git \
    && dpkg-query -W > /opt/os-packages.txt
ENV ACPP_TARGETS=generic ACPP_VISIBILITY_MASK=cuda \
    CMAKE_PREFIX_PATH=/opt/acpp LD_LIBRARY_PATH=/opt/acpp/lib:/usr/local/cuda/lib64 \
    ACPP_INSTALL_DIR=/opt/acpp PATH=/opt/acpp/bin:${PATH}
