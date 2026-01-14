import argparse
import sys
import time
import threading
from functools import lru_cache
from collections import defaultdict

import cv2
import numpy as np
import lgpio as GPIO

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)
# Try different audio methods
try:
    import pyttsx3
    AUDIO_METHOD = "pyttsx3"
except ImportError:
    try:
        from gtts import gTTS
        import os
        AUDIO_METHOD = "gtts"
    except ImportError:
        import subprocess
        AUDIO_METHOD = "espeak"

# Set pins
TRIG = 23
ECHO = 24

# Open the GPIO chip and set the GPIO direction
h = GPIO.gpiochip_open(0)
GPIO.gpio_claim_output(h, TRIG)
GPIO.gpio_claim_input(h, ECHO)

last_detections = []
last_announced = {}  # Track last announcement time for each object
announcement_cooldown = 3.0  # Seconds between announcements for same object
audio_queue = []
audio_lock = threading.Lock()

# Ultrasonic sensor state
sensor_active = False
sensor_lock = threading.Lock()
camera_active = False

# Define human and animal categories (COCO dataset labels)
HUMANS = {'person', 'human'}
ANIMALS = {
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'mouse', 'rabbit', 'fox', 'raccoon', 'deer',
    'skunk', 'squirrel', 'lion', 'tiger', 'monkey', 'gorilla', 'wolf'
}

def get_distance():
    # Set TRIG LOW
    GPIO.gpio_write(h, TRIG, 0)
    time.sleep(2)

    # Send 10us pulse to TRIG
    GPIO.gpio_write(h, TRIG, 1)
    time.sleep(0.00001)
    GPIO.gpio_write(h, TRIG, 0)

    # Start recording the time when the wave is sent
    while GPIO.gpio_read(h, ECHO) == 0:
        pulse_start = time.time()

    # Record time of arrival
    while GPIO.gpio_read(h, ECHO) == 1:
        pulse_end = time.time()

    # Calculate the difference in times
    pulse_duration = pulse_end - pulse_start

    # Multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = pulse_duration * 17150
    distance = round(distance, 2)

    return distance


class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def speak_text(text):
    """Speak the given text using available audio method."""
    if args.no_audio:
        return
    
    if AUDIO_METHOD == "pyttsx3":
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Audio error: {e}")
    
    elif AUDIO_METHOD == "gtts":
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save("/tmp/detection.mp3")
            os.system("mpg123 -q /tmp/detection.mp3 2>/dev/null")
        except Exception as e:
            print(f"Audio error: {e}")
    
    else:  # espeak fallback
        try:
            subprocess.run(['espeak', text], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Audio error: {e}")


def audio_worker():
    """Background thread to handle audio announcements."""
    global audio_queue
    while True:
        with audio_lock:
            if audio_queue:
                text = audio_queue.pop(0)
            else:
                text = None
        
        if text:
            speak_text(text)
        else:
            time.sleep(0.1)


def is_human_or_animal(label):
    """Check if the detected object is a human or animal."""
    label_lower = label.lower()
    return label_lower in HUMANS or label_lower in ANIMALS


def announce_detections(detections):
    """Announce detected objects (humans and animals only) with cooldown to avoid spam."""
    global last_announced
    
    # Only announce if sensor is active
    with sensor_lock:
        if not camera_active:
            return
    
    if not detections or args.no_audio:
        return
    
    current_time = time.time()
    labels = get_labels()
    detected_objects = defaultdict(int)
    
    # Count each object type (only humans and animals)
    for detection in detections:
        label = labels[int(detection.category)]
        if is_human_or_animal(label):
            detected_objects[label] += 1
    
    # Build announcement
    announcements = []
    for label, count in detected_objects.items():
        # Check cooldown
        if label not in last_announced or (current_time - last_announced[label]) > announcement_cooldown:
            last_announced[label] = current_time
            if count == 1:
                announcements.append(f"{label} detected")
            else:
                announcements.append(f"{count} {label}s detected")
    
    # Queue announcement
    if announcements:
        announcement_text = ". ".join(announcements)
        with audio_lock:
            # Limit queue size
            if len(audio_queue) < 3:
                audio_queue.append(announcement_text)
        # Also print to console
        print(f"🚨 [ALERT] {announcement_text}")


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_detections
    
    # Only process if camera is active
    with sensor_lock:
        if not camera_active and args.ultrasonic_enable:
            return []
    
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = \
            postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                          max_out_dets=max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    
    # Announce detections (only humans and animals)
    announce_detections(last_detections)
    
    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def draw_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    detections = last_results
    
    labels = get_labels()
    with MappedArray(request, stream) as m:
        # Show sensor status on screen
        if args.ultrasonic_enable:
            with sensor_lock:
                status_text = "ACTIVE" if camera_active else "STANDBY"
                status_color = (0, 255, 0) if camera_active else (128, 128, 128)
            cv2.putText(m.array, f"Sensor: {status_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        if detections is None or not detections:
            return
            
        for detection in detections:
            x, y, w, h = detection.box
            label = labels[int(detection.category)]
            label_text = f"{label} ({detection.conf:.2f})"
            
            # Highlight humans and animals with different color
            is_target = is_human_or_animal(label)
            box_color = (0, 0, 255) if is_target else (0, 255, 0)  # Red for humans/animals, green for others
            text_color = (0, 0, 255) if is_target else (0, 0, 255)  # Red text for all

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + 5
            text_y = y + 15

            # Create a copy of the array to draw the background with opacity
            overlay = m.array.copy()

            # Draw the background rectangle on the overlay
            cv2.rectangle(overlay,
                          (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255),  # Background color (white)
                          cv2.FILLED)

            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

            # Draw text on top of the background
            cv2.putText(m.array, label_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            # Draw detection box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), box_color, thickness=2)

        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)  # red
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                        help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="Run post process of type")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    # Audio options
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable audio announcements")
    parser.add_argument("--audio-cooldown", type=float, default=3.0,
                        help="Seconds between announcements for same object (default: 3.0)")
    # Ultrasonic sensor options
    parser.add_argument("--ultrasonic-enable", action="store_true",
                        help="Enable ultrasonic sensor trigger")
    parser.add_argument("--trig-pin", type=int, default=23,
                        help="GPIO pin for ultrasonic TRIG (default: 23)")
    parser.add_argument("--echo-pin", type=int, default=24,
                        help="GPIO pin for ultrasonic ECHO (default: 24)")
    parser.add_argument("--distance-threshold", type=float, default=1.0,
                        help="Distance threshold in meters (default: 1.0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    announcement_cooldown = args.audio_cooldown

    print(f"Audio method: {AUDIO_METHOD}")
    if AUDIO_METHOD == "espeak":
        print("Note: Install pyttsx3 or gtts for better audio quality")
    print("\n🎯 Detection Mode: HUMANS AND ANIMALS ONLY")
    print(f"Humans: {', '.join(sorted(HUMANS))}")
    print(f"Animals: {', '.join(sorted(ANIMALS))}")
    
    print("\n")
    
    # Start audio worker thread
    if not args.no_audio:
        audio_thread = threading.Thread(target=audio_worker, daemon=True)
        audio_thread.start()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    last_results = None
    picam2.pre_callback = draw_detections
    
    try:
        while True:
            dist = get_distance()
            print("Measured Distance = {:.2f} cm".format(dist))
            if dist < 100:
                last_results = parse_detections(picam2.capture_metadata())
                time.sleep(3)

    except KeyboardInterrupt:
        print("\n\nStopping...")
        GPIO.gpiochip_close(h)
        picam2.stop()
