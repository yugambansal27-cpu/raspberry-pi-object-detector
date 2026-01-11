# Raspberry Pi AI Camera Detection System

A Python application for real-time detection of humans and animals using a Raspberry Pi AI camera with audio output through a speaker.

## Features

- Real-time object detection using AI/ML models
- Raspberry Pi camera integration
- Human and animal classification
- Audio output system for announcing detected objects
- Configurable detection thresholds
- Logging and monitoring

## Requirements

- Raspberry Pi (3B+, 4, or 5 recommended)
- Raspberry Pi AI Camera (or compatible camera module)
- Speaker or audio output device
- Python 3.8+

## Installation

1. Clone this repository or copy the files to your Raspberry Pi

2. Install system dependencies (on Raspberry Pi):
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv portaudio19-dev
sudo apt-get install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── src/
│   └── pi_detector/
│       ├── __init__.py
│       ├── main.py              # Main application entry point
│       ├── camera.py            # Camera handling
│       ├── detector.py          # Object detection logic
│       ├── audio.py             # Audio output system
│       └── config.py            # Configuration management
├── models/                      # AI models directory
├── config/
│   └── settings.json           # Configuration file
├── examples/
│   └── basic_detection.py      # Example usage
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## Configuration

Edit `config/settings.json` to customize detection settings:

```json
{
  "camera": {
    "resolution": [640, 480],
    "framerate": 30
  },
  "detection": {
    "confidence_threshold": 0.5,
    "model_path": "models/mobilenet_ssd_v2.tflite"
  },
  "audio": {
    "enabled": true,
    "volume": 80
  }
}
```

## Usage

Run the main detection application:

```bash
python src/pi_detector/main.py
```

Or use the example script:

```bash
python examples/basic_detection.py
```

## Supported Objects

The default model can detect:
- Humans (person class)
- Common animals (dog, cat, bird, horse, cow, etc.)

## Hardware Setup

1. Connect your Raspberry Pi camera to the CSI port
2. Connect a speaker to the 3.5mm audio jack or via USB
3. Ensure the camera is enabled in `raspi-config`

## Troubleshooting

- **Camera not detected**: Run `vcgencmd get_camera` to verify camera status
- **Audio issues**: Test with `speaker-test -t wav -c 2`
- **Performance**: Consider using lighter models or reducing resolution

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
