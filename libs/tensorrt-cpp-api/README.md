# TensorRT C++ API helper

This directory contains a small C++ wrapper that makes it easier to load
ONNX models into TensorRT and run inference from a standalone binary. The
original upstream repository includes many samples and build options; this
fork keeps only what we need for the XvRT video processor.

## What you get

- Minimal CMake project that builds a static library and a simple CLI tool
- Helpers to parse ONNX models, build a TensorRT engine, and execute batches
- Utilities for timing, logging, and command-line flags

## Requirements

- CUDA Toolkit (tested with 11.x)
- TensorRT SDK installed and available to CMake (`TensorRT_DIR`)
- C++17 compiler

Optional: `python3` if you want to regenerate bindings or run scripts shipped
with TensorRT.

## Building

```bash
mkdir -p build
cmake -S . -B build \
  -DTensorRT_DIR=/opt/tensorrt \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
cmake --build build
```

After the build finishes you will find the static library and sample binaries
under `build/`.

## Running the sample CLI

The CLI expects an ONNX model path and an output directory for the generated
TensorRT engine:

```bash
./build/bin/trt_benchmark \
  --onnx models/yolov8n.onnx \
  --precision fp16 \
  --warmup 10 \
  --runs 100
```

Adjust the flags to match your hardware and desired precision. The tool prints
basic statistics (average latency, throughput, etc.).

## Integrating in video_processor

The video_processor build links against the static library produced here. Make
sure the library is built before compiling the NIF or port driver, and that the
TensorRT runtime shared libraries are available at runtime (`LD_LIBRARY_PATH`).

If you make local changes to the TensorRT wrapper, re-run the build commands and
re-link the video processor.

## Updating from upstream

This subtree is based on https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP.
When updating, review upstream changes and keep only the pieces needed for
video_processor to limit build time and external dependencies.
