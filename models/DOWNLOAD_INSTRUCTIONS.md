# Model Download Instructions

## For Raspberry Pi (Recommended)

Download a TensorFlow Lite model for object detection:

```bash
# Download MobileNet SSD v1 (quantized)
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip

# Extract
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip

# Move to models directory
mv detect.tflite models/mobilenet_ssd_v2.tflite
```

## Alternative: Download from TensorFlow Hub

Visit: https://www.tensorflow.org/lite/models/object_detection/overview

Choose a lightweight model suitable for Raspberry Pi (MobileNet or EfficientDet Lite).

## Note

This placeholder file will be replaced by your actual model file.
