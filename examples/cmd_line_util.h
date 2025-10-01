#include "yolov8.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

// NOTE: This is copied from src/cmd_line_util.h to examples for SDK layout.
//       Keeping same content for compatibility.

inline void printUsage() {
    std::cout << "YOLOv8 TensorRT SDK examples\n"
              << "Usage: --onnx <path> | --engine <path> --input <path_or_device> [options]\n\n"
              << "Options:\n"
              << "--precision <fp32|fp16|int8>        Inference precision (default: fp16)\n"
              << "--calib-dir <path>                  INT8 calibration images dir\n"
              << "--prob-threshold <float>            Detection threshold (default: 0.25)\n"
              << "--nms-threshold <float>             NMS threshold (default: 0.65)\n"
              << "--top-k <int>                        Max detections per image (default: 100)\n"
              << "--seg-channels <int>                 Segmentation channels (default: 32)\n"
              << "--seg-h <int>                        Segmentation mask height (default: 160)\n"
              << "--seg-w <int>                        Segmentation mask width (default: 160)\n"
              << "--seg-threshold <float>              Segmentation threshold (default: 0.5)\n"
              << "--class-names <string list>          Override class names (space separated)\n"
              << std::endl;
}

inline bool parseArgumentsBase(int argc, char *argv[], YoloV8Config &config, std::string &onnxModelPath, std::string &trtModelPath,
                               std::string &input) {
    if (argc < 3) {
        printUsage();
        return false;
    }
    std::vector<std::string> values;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0) {
            std::string flag = arg.substr(2);
            values.clear();
            // collect following non-flag values
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                values.emplace_back(argv[++i]);
            }
            if (flag == "onnx") {
                onnxModelPath = values.empty() ? std::string() : values[0];
            } else if (flag == "engine") {
                trtModelPath = values.empty() ? std::string() : values[0];
            } else if (flag == "input") {
                input = values.empty() ? std::string() : values[0];
            } else if (flag == "precision") {
                if (!values.empty()) {
                    if (values[0] == "fp32") config.precision = Precision::FP32;
                    else if (values[0] == "fp16") config.precision = Precision::FP16;
                    else if (values[0] == "int8") config.precision = Precision::INT8;
                }
            } else if (flag == "calib-dir") {
                config.calibrationDataDirectory = values.empty() ? std::string() : values[0];
            } else if (flag == "prob-threshold") {
                if (!values.empty()) config.probabilityThreshold = std::stof(values[0]);
            } else if (flag == "nms-threshold") {
                if (!values.empty()) config.nmsThreshold = std::stof(values[0]);
            } else if (flag == "top-k") {
                if (!values.empty()) config.topK = std::stoi(values[0]);
            } else if (flag == "seg-channels") {
                if (!values.empty()) config.segChannels = std::stoi(values[0]);
            } else if (flag == "seg-h") {
                if (!values.empty()) config.segH = std::stoi(values[0]);
            } else if (flag == "seg-w") {
                if (!values.empty()) config.segW = std::stoi(values[0]);
            } else if (flag == "seg-threshold") {
                if (!values.empty()) config.segmentationThreshold = std::stof(values[0]);
            } else if (flag == "class-names") {
                config.classNames = values;
            }
        }
    }
    if (onnxModelPath.empty() && trtModelPath.empty()) {
        std::cerr << "Error: provide either --onnx or --engine" << std::endl;
        return false;
    }
    if (input.empty()) {
        std::cerr << "Error: provide --input image/video/cam index" << std::endl;
        return false;
    }
    return true;
}

inline bool parseArguments(int argc, char *argv[], YoloV8Config &config, std::string &onnxModelPath, std::string &trtModelPath,
                           std::string &inputImage) {
    return parseArgumentsBase(argc, argv, config, onnxModelPath, trtModelPath, inputImage);
}

inline bool parseArgumentsVideo(int argc, char *argv[], YoloV8Config &config, std::string &onnxModelPath, std::string &trtModelPath,
                                std::string &inputVideo) {
    return parseArgumentsBase(argc, argv, config, onnxModelPath, trtModelPath, inputVideo);
}

