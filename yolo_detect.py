import os
import sys
import argparse
import glob
import time
import threading
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# Audio library detection with improved error handling
AUDIO_METHOD = "none"
pyttsx3_engine = None

try:
    import pyttsx3
    # Try to initialize pyttsx3 early to catch errors
    try:
        test_engine = pyttsx3.init()
        # Try to set a voice to test if it works
        voices = test_engine.getProperty('voices')
        if voices:
            test_engine.setProperty('voice', voices[0].id)
        test_engine.stop()
        AUDIO_METHOD = "pyttsx3"
        print("✓ pyttsx3 initialized successfully")
    except Exception as e:
        print(f"pyttsx3 available but failed to initialize: {e}")
        raise ImportError
except ImportError:
    try:
        from gtts import gTTS
        AUDIO_METHOD = "gtts"
        print("✓ Using gTTS for audio")
    except ImportError:
        try:
            import subprocess
            # Test if espeak is available
            result = subprocess.run(['which', 'espeak'], capture_output=True)
            if result.returncode == 0:
                AUDIO_METHOD = "espeak"
                print("✓ Using espeak for audio")
            else:
                print("⚠️ No audio method available. Install: sudo apt-get install espeak")
        except:
            print("⚠️ No audio method available")

# Audio state management
audio_queue = []
audio_lock = threading.Lock()
last_announced = {}
announcement_cooldown = 3.0

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), or index of Picamera ("picamera0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')
parser.add_argument('--no-audio', help='Disable audio announcements',
                    action='store_true')
parser.add_argument('--audio-cooldown', type=float, default=3.0,
                    help='Seconds between announcements for same object (default: 3.0)')
parser.add_argument('--announce-all', help='Announce all detected objects individually (default: only total count)',
                    action='store_true')
parser.add_argument('--audio-method', choices=['pyttsx3', 'gtts', 'espeak', 'auto'], default='auto',
                    help='Force specific audio method (default: auto)')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record
announcement_cooldown = args.audio_cooldown

# Override audio method if specified
if args.audio_method != 'auto':
    AUDIO_METHOD = args.audio_method
    print(f"Using forced audio method: {AUDIO_METHOD}")


def speak_announcement(text):
    """Convert text to speech using available method."""
    if args.no_audio or AUDIO_METHOD == "none":
        return
    
    if AUDIO_METHOD == "pyttsx3":
        try:
            # Create a fresh engine for each announcement to avoid state issues
            engine = pyttsx3.init()
            
            # Configure engine with safe settings for Raspberry Pi
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            
            # Try to use the first available voice
            voices = engine.getProperty('voices')
            if voices:
                # Use first voice (usually more reliable on Pi)
                engine.setProperty('voice', voices[0].id)
            
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine  # Clean up
            
        except Exception as err:
            print(f"pyttsx3 error: {err}")
            print("Falling back to espeak...")
            # Fallback to espeak
            try:
                subprocess.run(['espeak', text], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
            except:
                pass
    
    elif AUDIO_METHOD == "gtts":
        try:
            audio_obj = gTTS(text=text, lang='en', slow=False)
            audio_obj.save("/tmp/detection.mp3")
            # Try multiple players
            for player in ['mpg123', 'mpg321', 'ffplay']:
                result = subprocess.run(['which', player], capture_output=True)
                if result.returncode == 0:
                    subprocess.run([player, '-q', '/tmp/detection.mp3'], 
                                 stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL)
                    break
        except Exception as err:
            print(f"gTTS error: {err}")
    
    elif AUDIO_METHOD == "espeak":
        try:
            subprocess.run(['espeak', text], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
        except Exception as err:
            print(f"espeak error: {err}")


def process_audio_queue():
    """Background thread for processing audio announcements."""
    global audio_queue
    while True:
        with audio_lock:
            if audio_queue:
                announcement = audio_queue.pop(0)
            else:
                announcement = None
        
        if announcement:
            speak_announcement(announcement)
        else:
            time.sleep(0.1)


def create_announcement(detected_objects):
    """Generate and queue audio announcements for detections."""
    global last_announced
    
    if args.no_audio or not detected_objects or AUDIO_METHOD == "none":
        return
    
    current_time = time.time()
    messages = []
    
    if args.announce_all:
        # Announce each type of object
        for obj_label, obj_count in detected_objects.items():
            # Apply cooldown per object type
            if obj_label not in last_announced or (current_time - last_announced[obj_label]) > announcement_cooldown:
                last_announced[obj_label] = current_time
                if obj_count == 1:
                    messages.append(f"{obj_label} detected")
                else:
                    messages.append(f"{obj_count} {obj_label}s detected")
    else:
        # Announce total object count
        total = sum(detected_objects.values())
        if 'total' not in last_announced or (current_time - last_announced['total']) > announcement_cooldown:
            last_announced['total'] = current_time
            if total == 1:
                messages.append("1 object detected")
            else:
                messages.append(f"{total} objects detected")
    
    # Add to queue
    if messages:
        full_message = ". ".join(messages)
        with audio_lock:
            # Limit queue to prevent backlog
            if len(audio_queue) < 3:
                audio_queue.append(full_message)
        print(f"[AUDIO] {full_message}")


# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Display audio configuration
if not args.no_audio:
    print(f"\n🔊 Audio Configuration:")
    print(f"   Method: {AUDIO_METHOD}")
    if AUDIO_METHOD == "none":
        print("   ⚠️ No audio available. To enable:")
        print("      sudo apt-get install espeak")
        print("      pip install pyttsx3")
    elif AUDIO_METHOD == "espeak":
        print("   Note: For better quality, install: pip install pyttsx3")
    elif AUDIO_METHOD == "gtts":
        print("   Note: Requires internet connection")
        print("   Install player: sudo apt-get install mpg123")
    print(f"   Cooldown: {announcement_cooldown}s")
    print(f"   Mode: {'All objects' if args.announce_all else 'Total count only'}\n")
    
    # Start audio processing thread
    if AUDIO_METHOD != "none":
        audio_thread = threading.Thread(target=process_audio_queue, daemon=True)
        audio_thread.start()
else:
    print("🔇 Audio disabled\n")

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':

    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Set camera or video resolution if specified by user
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Begin inference loop
while True:

    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    
    elif source_type == 'usb':
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    elif source_type == 'picamera':
        frame = cap.capture_array()
        if (frame is None):
            print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Resize frame to desired display resolution
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    # Run inference on frame
    results = model(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Track object counts
    object_count = 0
    detected_objects = defaultdict(int)

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):

        # Get bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > min_thresh:

            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Count objects
            object_count = object_count + 1
            detected_objects[classname] += 1

    # Generate audio announcement for detected objects
    create_announcement(detected_objects)

    # Calculate and draw framerate
    if source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    
    # Display detection results
    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow('YOLO detection results',frame)
    if record: recorder.write(frame)

    # Handle key presses
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):
        cv2.imwrite('capture.png',frame)
    
    # Calculate FPS
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Update FPS buffer
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS
    avg_frame_rate = np.mean(frame_rate_buffer)


# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()
