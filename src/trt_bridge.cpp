#include "yolo_trt_bridge.h"

#include <algorithm>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <yolov8.hpp>

namespace {

struct BridgeContext {
  std::unique_ptr<YoloV8> detector;
};

thread_local std::string g_last_error;

void set_last_error(std::string message) {
  g_last_error = std::move(message);
}

YoloV8Config build_config(const yolo_trt_options *options, yolo_trt_status *status) {
  YoloV8Config config;
  if (options == nullptr) {
    return config;
  }

  switch (options->precision) {
  case YOLO_TRT_PRECISION_FP32:
    config.precision = Precision::FP32;
    break;
  case YOLO_TRT_PRECISION_FP16:
    config.precision = Precision::FP16;
    break;
  case YOLO_TRT_PRECISION_INT8:
    config.precision = Precision::INT8;
    if (options->calibration_dir == nullptr || options->calibration_dir[0] == '\0') {
      if (status) {
        *status = YOLO_TRT_STATUS_INVALID_ARGUMENT;
      }
      set_last_error("INT8 precision requires calibration_dir");
    } else {
      config.calibrationDataDirectory = options->calibration_dir;
    }
    break;
  case YOLO_TRT_PRECISION_DEFAULT:
  default:
    config.precision = Precision::FP16;
    break;
  }

  if (options->prob_threshold > 0.f)
    config.probabilityThreshold = options->prob_threshold;
  if (options->nms_threshold > 0.f)
    config.nmsThreshold = options->nms_threshold;
  if (options->top_k > 0)
    config.topK = options->top_k;
  if (options->seg_channels > 0)
    config.segChannels = options->seg_channels;
  if (options->seg_h > 0)
    config.segH = options->seg_h;
  if (options->seg_w > 0)
    config.segW = options->seg_w;
  if (options->seg_threshold >= 0.f)
    config.segmentationThreshold = options->seg_threshold;
  if (options->num_kps > 0)
    config.numKPS = options->num_kps;
  if (options->kps_threshold >= 0.f)
    config.kpsThreshold = options->kps_threshold;

  if (options->class_names.items != nullptr && options->class_names.count > 0) {
    config.classNames.clear();
    config.classNames.reserve(options->class_names.count);
    for (size_t i = 0; i < options->class_names.count; ++i) {
      const char *entry = options->class_names.items[i];
      if (entry != nullptr) {
        config.classNames.emplace_back(entry);
      }
    }
  }

  if (status && *status == YOLO_TRT_STATUS_INVALID_ARGUMENT) {
    return config;
  }

  if (status)
    *status = YOLO_TRT_STATUS_OK;
  return config;
}

cv::Mat make_bgr_view(const yolo_trt_frame &frame, cv::Mat &scratch, yolo_trt_status &status) {
  status = YOLO_TRT_STATUS_OK;
  if (frame.data == nullptr || frame.width <= 0 || frame.height <= 0) {
    status = YOLO_TRT_STATUS_INVALID_ARGUMENT;
    set_last_error("invalid frame");
    return cv::Mat();
  }

  const int stride = frame.stride > 0 ? frame.stride : frame.width * 3;
  if (stride < frame.width * 3) {
    status = YOLO_TRT_STATUS_INVALID_ARGUMENT;
    set_last_error("stride smaller than width * channels");
    return cv::Mat();
  }

  cv::Mat input(frame.height, frame.width, CV_8UC3,
                const_cast<uint8_t *>(frame.data), stride);

  if (frame.format == YOLO_TRT_FORMAT_RGB) {
    cv::cvtColor(input, scratch, cv::COLOR_RGB2BGR);
    return scratch;
  }

  return input;
}

} // namespace

extern "C" {

yolo_trt_handle yolo_trt_create(const yolo_trt_options *options,
                                yolo_trt_status *status) {
  if (status)
    *status = YOLO_TRT_STATUS_OK;

  if (options == nullptr || options->engine_path == nullptr || options->engine_path[0] == '\0') {
    set_last_error("engine_path is required");
    if (status)
      *status = YOLO_TRT_STATUS_INVALID_ARGUMENT;
    return nullptr;
  }

  std::unique_ptr<BridgeContext> ctx(new (std::nothrow) BridgeContext());
  if (!ctx) {
    set_last_error("out of memory");
    if (status)
      *status = YOLO_TRT_STATUS_OUT_OF_MEMORY;
    return nullptr;
  }

  yolo_trt_status cfg_status = YOLO_TRT_STATUS_OK;
  YoloV8Config config = build_config(options, &cfg_status);
  if (cfg_status != YOLO_TRT_STATUS_OK) {
    if (status)
      *status = cfg_status;
    return nullptr;
  }

  std::string onnx_path;
  if (options->onnx_path != nullptr)
    onnx_path = options->onnx_path;

  try {
    ctx->detector = std::make_unique<YoloV8>(onnx_path, options->engine_path, config);
  } catch (const std::exception &ex) {
    set_last_error(ex.what());
    if (status)
      *status = YOLO_TRT_STATUS_BACKEND_ERROR;
    return nullptr;
  }

  return ctx.release();
}

void yolo_trt_destroy(yolo_trt_handle handle) {
  auto *ctx = reinterpret_cast<BridgeContext *>(handle);
  delete ctx;
}

yolo_trt_status yolo_trt_detect(yolo_trt_handle handle,
                                const yolo_trt_frame *frame,
                                yolo_trt_detection_callback callback,
                                void *user_data,
                                int *out_count) {
  if (out_count)
    *out_count = 0;

  if (handle == nullptr) {
    set_last_error("handle is null");
    return YOLO_TRT_STATUS_NOT_INITIALIZED;
  }

  if (frame == nullptr) {
    set_last_error("frame is null");
    return YOLO_TRT_STATUS_INVALID_ARGUMENT;
  }

  auto *ctx = reinterpret_cast<BridgeContext *>(handle);
  if (!ctx->detector) {
    set_last_error("detector not initialized");
    return YOLO_TRT_STATUS_NOT_INITIALIZED;
  }

  cv::Mat scratch;
  yolo_trt_status frame_status;
  cv::Mat bgr = make_bgr_view(*frame, scratch, frame_status);
  if (frame_status != YOLO_TRT_STATUS_OK) {
    return frame_status;
  }

  std::vector<Object> detections;
  try {
    detections = ctx->detector->detectObjects(bgr);
  } catch (const std::exception &ex) {
    set_last_error(ex.what());
    return YOLO_TRT_STATUS_BACKEND_ERROR;
  }

  if (out_count)
    *out_count = static_cast<int>(detections.size());

  if (callback == nullptr) {
    return YOLO_TRT_STATUS_OK;
  }

  for (const auto &det : detections) {
    yolo_trt_detection dto{};
    dto.label_id = det.label;
    dto.confidence = det.probability;
    dto.bbox.x = det.rect.x;
    dto.bbox.y = det.rect.y;
    dto.bbox.width = det.rect.width;
    dto.bbox.height = det.rect.height;
    dto.keypoints = det.kps.empty() ? nullptr : det.kps.data();
    dto.keypoint_count = det.kps.size();

    if (!det.boxMask.empty()) {
      dto.mask_data = det.boxMask.ptr<uint8_t>();
      dto.mask_rows = det.boxMask.rows;
      dto.mask_cols = det.boxMask.cols;
      dto.mask_stride = det.boxMask.step;
    }

    int cb_status = callback(&dto, user_data);
    if (cb_status != 0) {
      break;
    }
  }

  return YOLO_TRT_STATUS_OK;
}

const char *yolo_trt_get_last_error(void) {
  return g_last_error.c_str();
}

} // extern "C"
