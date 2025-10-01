## YOLOv8 TensorRT C++ SDK

YOLOv8 inference using TensorRT (C++), supporting Object Detection, Segmentation, and Pose Estimation. The repo is organized as an SDK for easy integration and testing.

### Layout
- `include/yolov8/yolov8.hpp`: Public SDK header (primary application API)
- `src/`: Library implementation (no direct include needed)
- `examples/`: CLI samples (image, video/webcam, CSI camera, benchmark)
- `tests/`: Lightweight tests to verify SDK headers
- `libs/tensorrt-cpp-api`: Internal TensorRT helper library

### Requirements
- Ubuntu 20.04/22.04 (Windows not supported)
- CUDA >= 12.0, cuDNN >= 8
- OpenCV >= 4.7 (uses `cv::dnn::NMSBoxesBatched`), preferably built with CUDA
- TensorRT >= 10.0
- CMake >= 3.22, g++ (C++17)

Note: set `TensorRT_DIR` in `CMakeLists.txt` to your TensorRT installation path.

### Clone
- `git clone --recursive <repo>` (submodules required)

### Convert PyTorch model to ONNX
- Install Ultralytics: `pip3 install ultralytics`
- Script: `python3 scripts/pytorch2onnx.py --pt_path /path/to/model.pt`
- Disable `end2end` if using a different exporter (SDK performs decode + NMS in C++)

### Build SDK, Examples, Tests
1) Edit `TensorRT_DIR` in `CMakeLists.txt`.
2) Build:
   - `cmake -S . -B build -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON`
   - `cmake --build build -j`

CMake options:
- `ENABLE_BENCHMARKS=ON`: Print preprocess/inference/postprocess timings
- `BUILD_EXAMPLES=ON|OFF`: Build CLI samples
- `BUILD_TESTS=ON|OFF`: Build tests

### Run examples
Notes:
- First run may take minutes to build the TRT engine from ONNX (cached afterwards)
- Works with Ultralytics pretrained models (detect/seg/pose)

Binaries are in `build/`:
- Image: `./detect_object_image --onnx /path/to/model.onnx --input images/example.jpg`
- Webcam/Video: `./detect_object_video --onnx /path/to/model.onnx --input 0` or a video file path
- CSI Jetson: `./detect_object_csi_jetson --onnx /path/to/model.onnx --input 0` (GStreamer pipeline in example)
- Benchmark: `./benchmark --onnx /path/to/model.onnx --input images/640_640.jpg`

You can also use a prebuilt TensorRT engine:
- `--engine /path/to/model.engine` (instead of `--onnx`)

Common flags:
- `--precision fp32|fp16|int8` (default: fp16)
- `--calib-dir /path/to/calib_images` (required for `int8`)
- `--prob-threshold`, `--nms-threshold`, `--top-k`
- `--seg-channels`, `--seg-h`, `--seg-w`, `--seg-threshold`
- `--class-names <space separated class names>`

Example:
- `./detect_object_image --onnx models/yolov8n.onnx --input images/bus.jpg --precision fp16 --prob-threshold 0.25 --nms-threshold 0.65`

### Run tests
- Configure with: `-DBUILD_TESTS=ON`
- After build: `ctest --test-dir build`
  - Current tests verify headers and defaults (no GPU/engine required)

### INT8 Inference (optional)
- Requires a representative calibration set (recommended >= 1000 images)
- Flags: `--precision int8 --calib-dir /path/to/calib`
- If you hit OOM during calibration, reduce `Options.calibrationBatchSize` (see `libs/tensorrt-cpp-api`)

### Debug tips
- If engine building fails, open `libs/tensorrt-cpp-api/src/engine.cpp`, set log severity to `kVERBOSE`, rebuild, and rerun to get detailed logs.

### Integrate the SDK into your app
- Include: `#include <yolov8/yolov8.hpp>` (legacy compatible: `#include "yolov8.h"`)
- Link target: `YoloV8_TRT` (alias: `yolov8::sdk`)

Minimal example:

```cpp
#include <yolov8/yolov8.hpp>

int main() {
    YoloV8Config cfg; // defaults
    YoloV8 yolo("models/yolov8n.onnx", "", cfg);
    cv::Mat img = cv::imread("images/bus.jpg");
    auto objects = yolo.detectObjects(img);
    yolo.drawObjectLabels(img, objects);
    cv::imwrite("out.jpg", img);
}
```

---

If desired, we can add install/export rules to package the SDK (headers + library) for system-wide use.
