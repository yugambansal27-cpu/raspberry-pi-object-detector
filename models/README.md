# Pi Detector - Model Directory

This directory should contain your TensorFlow Lite models for object detection.

## Recommended Model

For Raspberry Pi, we recommend using **MobileNet SSD v2** with TensorFlow Lite:

### Download Instructions

1. Download the model from TensorFlow:
   ```bash
   wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
   unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
   mv detect.tflite models/mobilenet_ssd_v2.tflite
   ```

2. Or download directly from the official TensorFlow model zoo:
   - Visit: https://www.tensorflow.org/lite/models/object_detection/overview
   - Download a pre-trained model suitable for Raspberry Pi

## Model Requirements

- Format: TensorFlow Lite (.tflite)
- Input: RGB image (typically 300x300 or 640x480)
- Output: Bounding boxes, class IDs, and confidence scores
- Optimized for edge devices (quantized models preferred)

## Supported Models

- MobileNet SSD v1/v2
- EfficientDet Lite
- YOLO Lite models

## Custom Models

If you want to use a custom model:

1. Ensure it outputs detection boxes, classes, and scores
2. Update the model path in `config/settings.json`
3. Modify the label mapping in `src/pi_detector/detector.py` if needed

## Note

The application will work without a model (using a dummy detector for testing), but for real detection, you must download and place a proper TFLite model in this directory.
