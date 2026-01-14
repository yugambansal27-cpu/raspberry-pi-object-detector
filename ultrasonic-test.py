#!/usr/bin/env python3
"""
Ultrasonic sensor test with improved error handling
"""
import RPi.GPIO as GPIO
import time

# GPIO pins
TRIG_PIN = 23
ECHO_PIN = 24

def setup_sensor():
    """Setup GPIO pins for ultrasonic sensor."""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.output(TRIG_PIN, False)
    print(f"Sensor setup complete. TRIG: GPIO{TRIG_PIN}, ECHO: GPIO{ECHO_PIN}")
    time.sleep(0.5)  # Allow sensor to settle

def measure_distance():
    """Measure distance with timeout protection."""
    try:
        # Ensure trigger is low
        GPIO.output(TRIG_PIN, False)
        time.sleep(0.01)
        
        # Send 10us pulse
        GPIO.output(TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, False)
        
        # Wait for echo start with timeout
        timeout = time.time() + 0.1  # 100ms timeout
        while GPIO.input(ECHO_PIN) == 0:
            pulse_start = time.time()
            if time.time() > timeout:
                print("⚠️  Timeout waiting for echo start (ECHO stuck LOW)")
                return None
        
        # Wait for echo end with timeout
        timeout = time.time() + 0.1
        while GPIO.input(ECHO_PIN) == 1:
            pulse_end = time.time()
            if time.time() > timeout:
                print("⚠️  Timeout waiting for echo end (ECHO stuck HIGH)")
                # Force reset
                GPIO.setup(ECHO_PIN, GPIO.OUT)
                GPIO.output(ECHO_PIN, False)
                time.sleep(0.01)
                GPIO.setup(ECHO_PIN, GPIO.IN)
                return None
        
        # Calculate distance
        pulse_duration = pulse_end - pulse_start
        distance_cm = (pulse_duration * 34300) / 2
        
        # Filter invalid readings
        if distance_cm < 2 or distance_cm > 400:
            return None
        
        return distance_cm
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print("=" * 50)
    print("HC-SR04 Ultrasonic Sensor Test")
    print("=" * 50)
    print("\nTroubleshooting ECHO stuck HIGH:")
    print("1. Check voltage divider (ECHO outputs 5V!)")
    print("2. Verify wiring connections")
    print("3. Check power supply (sensor needs stable 5V)")
    print("4. Try different GPIO pins")
    print("\nPress Ctrl+C to exit\n")
    
    setup_sensor()
    
    successful = 0
    failed = 0
    
    try:
        while True:
            distance = measure_distance()
            
            if distance is not None:
                successful += 1
                print(f"✓ Distance: {distance:.1f} cm | Success: {successful}, Failed: {failed}")
            else:
                failed += 1
                print(f"✗ Read failed | Success: {successful}, Failed: {failed}")
            
            time.sleep(0.3)  # Wait between measurements
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        GPIO.cleanup()
        print("GPIO cleaned up")
        print(f"\nTotal: {successful} successful, {failed} failed")

if __name__ == "__main__":
    main()
