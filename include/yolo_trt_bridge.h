#ifndef YOLO_TRT_BRIDGE_H
#define YOLO_TRT_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/** @brief Opaque handle to a YOLO TensorRT detector instance. */
typedef void *yolo_trt_handle;

/** @brief Status codes returned by the C bridge API. */
typedef enum {
  YOLO_TRT_STATUS_OK = 0,
  YOLO_TRT_STATUS_INVALID_ARGUMENT = 1,
  YOLO_TRT_STATUS_OUT_OF_MEMORY = 2,
  YOLO_TRT_STATUS_BACKEND_ERROR = 3,
  YOLO_TRT_STATUS_NOT_INITIALIZED = 4
} yolo_trt_status;

/** @brief Precision options supported by the detector. */
typedef enum {
  YOLO_TRT_PRECISION_DEFAULT = 0,
  YOLO_TRT_PRECISION_FP32 = 1,
  YOLO_TRT_PRECISION_FP16 = 2,
  YOLO_TRT_PRECISION_INT8 = 3
} yolo_trt_precision;

/** @brief Supported image colour format encodings. */
typedef enum {
  YOLO_TRT_FORMAT_BGR = 0,
  YOLO_TRT_FORMAT_RGB = 1
} yolo_trt_format;

/** @brief Helper structure representing a list of strings. */
typedef struct {
  const char **items;
  size_t count;
} yolo_trt_string_list;

/**
 * @brief Runtime options used when creating a detector.
 *
 * Missing or invalid values fall back to their default equivalents.
 */
typedef struct {
  const char *engine_path;       /* Required */
  const char *onnx_path;         /* Optional - use when engine absent */
  yolo_trt_precision precision;  /* Default: FP16 */
  float prob_threshold;          /* <= 0.f -> default */
  float nms_threshold;           /* <= 0.f -> default */
  int top_k;                     /* <= 0 -> default */
  const char *calibration_dir;   /* Required for INT8 */
  int seg_channels;              /* <= 0 -> default */
  int seg_h;                     /* <= 0 -> default */
  int seg_w;                     /* <= 0 -> default */
  float seg_threshold;           /* < 0.f -> default */
  int num_kps;                   /* <= 0 -> default */
  float kps_threshold;           /* < 0.f -> default */
  yolo_trt_string_list class_names; /* Optional override */
} yolo_trt_options;

/**
 * @brief Image descriptor accepted by yolo_trt_detect.
 *
 * The image data is not copied and must remain valid during inference.
 */
typedef struct {
  const uint8_t *data;
  int width;
  int height;
  int stride;            /* Bytes per row. <= 0 -> inferred */
  yolo_trt_format format;
} yolo_trt_frame;

/** @brief Floating-point rectangle describing a detection bounding box. */
typedef struct {
  float x;
  float y;
  float width;
  float height;
} yolo_trt_rect;

/**
 * @brief Inference result object delivered through the callback.
 *
 * Optional fields (keypoints and mask data) are only set when the model
 * produces the corresponding outputs.
 */
typedef struct {
  int label_id;
  float confidence;
  yolo_trt_rect bbox;
  const float *keypoints;       /* Optional. Lifetime: inside callback */
  size_t keypoint_count;
  const uint8_t *mask_data;     /* Optional */
  int mask_rows;
  int mask_cols;
  size_t mask_stride;
} yolo_trt_detection;

/** @brief Callback invoked for each detection produced by yolo_trt_detect. */
typedef int (*yolo_trt_detection_callback)(const yolo_trt_detection *det,
                                           void *user_data);

/**
 * @brief Create a YOLO TensorRT detector instance.
 *
 * @param options Creation options; must not be null.
 * @param status Optional status output parameter.
 * @return A newly created detector handle or NULL on failure.
 */
yolo_trt_handle yolo_trt_create(const yolo_trt_options *options,
                                yolo_trt_status *status);

/**
 * @brief Destroy a detector instance and release its resources.
 *
 * @param handle Detector handle returned by yolo_trt_create.
 */
void yolo_trt_destroy(yolo_trt_handle handle);

/**
 * @brief Run inference on an image frame.
 *
 * @param handle Detector handle.
 * @param frame Input frame descriptor.
 * @param callback Callback invoked per detection; may be NULL.
 * @param user_data User context passed to the callback.
 * @param out_count Optional number of detections produced.
 * @return Status code describing the result.
 */
yolo_trt_status yolo_trt_detect(yolo_trt_handle handle,
                                const yolo_trt_frame *frame,
                                yolo_trt_detection_callback callback,
                                void *user_data,
                                int *out_count);

/**
 * @brief Retrieve the last human-readable error message.
 *
 * @return Pointer to a static string describing the most recent error.
 */
const char *yolo_trt_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* YOLO_TRT_BRIDGE_H */
