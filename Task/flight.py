# # Your code here

# Airport = [1,2]

'''
This contains the opencv, line following stuff. This will make the decisions for the robot based on what the camera sees and 
send commands to the controller.
'''
import cv2
import numpy as np
import json
import os
from control import Control
import time
import threading
from sensor import Camera

class Brain:
    CONFIG_FILE = "tuning_params.json"

    # --- Default tuning ---
    DEFAULT_TUNING = {
        "GRAY_THRESH": 190,
        "BLUR_SIZE_K": 7,     # Trackbar value, real kernel is k*2+1 (15)
        "MORPH_SIZE_K": 7,    # Trackbar value, real kernel is k*2+1 (15)
        "KP": 40,             # Divided by 100 -> 0.4
        "KI": 0,              # Divided by 100 -> 0.0
        "KD": 10,             # Divided by 100 -> 0.1
        "K_YAW_RATE": 120,    # Divided by 100 -> 1.2
        "FORWARD_SPEED": 15,  # Divided by 100 -> 0.15
        "MAX_LATERAL_SPEED": 3 # Divided by 100 -> 0.03
    }

    # --- Minimum contour area to count as a line (filters noise) ---
    MIN_LINE_AREA = 500

    # --- Yellow mask thresholds (HSV) for color input ---
    YELLOW_LOWER = np.array([18, 80, 80], dtype=np.uint8)
    YELLOW_UPPER = np.array([40, 255, 255], dtype=np.uint8)

    # --- AprilTag / turn settings ---
    TOTAL_TAGS = 4            # number of AprilTags on the course (last one = land)

    def __init__(self):
        self.control = Control()
        self.camera = Camera()

        # Load configuration
        self.tuning = self.DEFAULT_TUNING.copy()
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    saved = json.load(f)
                    self.tuning.update(saved)
                print("Loaded saved tuning params.")
            except Exception as e:
                print(f"Could not load tuning params: {e}")

        # Shared state between camera thread and control loop
        self._lock = threading.Lock()
        self._line_detected = False
        self._error = 0.0          # normalised error  (-1 .. +1)
        self._frame_width = 1      # updated on first frame

        # PID state
        self._prev_error = 0.0
        self._integral = 0.0

        # AprilTag detection state
        self._tag_detected = False
        self._tag_id = -1
        self._last_seen_tag_id = -1   # to avoid re-triggering on same tag
        self._tags_seen_count = 0

        # AprilTag detector (OpenCV ArUco with AprilTag dictionary)
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._aruco_detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)

    # -----------------------------------------------------------------
    # Frame processing (runs on the camera thread)
    # -----------------------------------------------------------------
    def process_frame(self, frame):
        """
        Process camera frame to detect the path. 
        Uses Otsu's thresholding for robust grayscale segmentation, or HSV for color.
        """
        h, w = frame.shape[:2]
        is_color = len(frame.shape) == 3 and frame.shape[2] >= 3
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if is_color else frame

        # --- Interactive Tuning UI ---
        if not hasattr(self, '_tuning_init'):
            cv2.namedWindow("Tuning", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tuning", 400, 450)
            
            # Create trackbars set to currently loaded tuning values
            cv2.createTrackbar("Thresh(0=Auto)", "Tuning", self.tuning["GRAY_THRESH"], 255, lambda x: None)
            cv2.createTrackbar("Blur(x2+1)", "Tuning", self.tuning["BLUR_SIZE_K"], 20, lambda x: None)
            cv2.createTrackbar("Morph(x2+1)", "Tuning", self.tuning["MORPH_SIZE_K"], 20, lambda x: None)
            
            # PID & Speed tuning (Values multiplied by 100 for trackbar)
            cv2.createTrackbar("KP (/100)", "Tuning", self.tuning["KP"], 200, lambda x: None)
            cv2.createTrackbar("KI (/100)", "Tuning", self.tuning["KI"], 100, lambda x: None)
            cv2.createTrackbar("KD (/100)", "Tuning", self.tuning["KD"], 200, lambda x: None)
            cv2.createTrackbar("YawRate (/100)", "Tuning", self.tuning["K_YAW_RATE"], 300, lambda x: None)
            cv2.createTrackbar("FwdSpd (/100)", "Tuning", self.tuning["FORWARD_SPEED"], 50, lambda x: None)
            cv2.createTrackbar("LatSpd (/100)", "Tuning", self.tuning["MAX_LATERAL_SPEED"], 20, lambda x: None)
            
            self._tuning_init = True
            
        # Read current trackbar values
        new_tuning = {
            "GRAY_THRESH": cv2.getTrackbarPos("Thresh(0=Auto)", "Tuning"),
            "BLUR_SIZE_K": cv2.getTrackbarPos("Blur(x2+1)", "Tuning"),
            "MORPH_SIZE_K": cv2.getTrackbarPos("Morph(x2+1)", "Tuning"),
            "KP": cv2.getTrackbarPos("KP (/100)", "Tuning"),
            "KI": cv2.getTrackbarPos("KI (/100)", "Tuning"),
            "KD": cv2.getTrackbarPos("KD (/100)", "Tuning"),
            "K_YAW_RATE": cv2.getTrackbarPos("YawRate (/100)", "Tuning"),
            "FORWARD_SPEED": cv2.getTrackbarPos("FwdSpd (/100)", "Tuning"),
            "MAX_LATERAL_SPEED": cv2.getTrackbarPos("LatSpd (/100)", "Tuning")
        }

        # Save if changed
        if new_tuning != self.tuning:
            self.tuning = new_tuning
            try:
                with open(self.CONFIG_FILE, 'w') as f:
                    json.dump(self.tuning, f, indent=4)
            except Exception as e:
                print(f"Error saving tuning params: {e}")

        # Derive actual usage values
        thresh_val = self.tuning["GRAY_THRESH"]
        blur_val = max(1, self.tuning["BLUR_SIZE_K"] * 2 + 1)
        morph_val = max(1, self.tuning["MORPH_SIZE_K"] * 2 + 1)

        # 1. AprilTag detection
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            tag_id = int(ids[0][0])
            with self._lock:
                if tag_id != self._last_seen_tag_id:
                    self._tag_detected = True
                    self._tag_id = tag_id

        # 2. Line Binary Mask Creation
        if is_color:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            binary = cv2.inRange(hsv, self.YELLOW_LOWER, self.YELLOW_UPPER)
        else:
            # Grayscale: heavily blur texture and isolate bright path
            blurred = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)
            
            # If Threshold is 0, fallback to Otsu auto-calculation. Otherwise, use explicit threshold.
            if thresh_val == 0:
                _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

        # 3. Morphology cleanup (merges noise) & Lookahead
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_val, morph_val))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary[int(h * 0.6):, :] = 0  # Ignore bottom 40% (Lookahead)

        # 4. Extract Line Centroid
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug = frame.copy() if is_color else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        with self._lock:
            self._line_detected = False
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                cv2.drawContours(debug, [largest], -1, (0, 255, 0), 2)
                
                if area >= self.MIN_LINE_AREA:
                    M = cv2.moments(largest)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        self._line_detected = True
                        self._error = (cx - w / 2) / (w / 2)
                        self._frame_width = w
                        
                        # Draw visuals
                        cv2.circle(debug, (cx, cy), 6, (0, 0, 255), -1)
                        cv2.line(debug, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
                        cv2.putText(debug, f"err={self._error:+.2f} area={area:.0f}", 
                                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(debug, f"TOO SMALL ({area:.0f})", (5, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                cv2.putText(debug, "NO LINE", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Overlay current tuned values for visibility
        if not is_color:
            tuning_text = f"Tuning -> Thresh: {thresh_val} | Blur(k): {blur_val} | Morph(k): {morph_val}"
            cv2.putText(debug, tuning_text, (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 255), 1)

        self._show_debug(debug, binary)

    @staticmethod
    def _show_debug(debug_frame, binary_frame):
        """Show the debug windows. Call from process_frame only."""
        cv2.imshow("Camera Feed (debug)", debug_frame)
        cv2.imshow("Binary Mask", binary_frame)
        cv2.waitKey(1)

    # -----------------------------------------------------------------
    # Line-following control loop (runs on the main thread)
    # -----------------------------------------------------------------
    def line_follow(self):
        """
        Main line-following logic using PID control.
        Moves forward at a constant speed while correcting laterally
        to keep the yellow line centred in the frame.
        When an AprilTag is detected, turn 90° right and continue.
        On the last AprilTag, land.
        """
        print("Starting yellow-line following…")
        self._integral = 0.0
        self._prev_error = 0.0
        dt = 0.05  # control loop period (s)

        while True:
            # --- Check for AprilTag event first ---
            with self._lock:
                tag_event = self._tag_detected
                tag_id = self._tag_id

            if tag_event:
                self._tags_seen_count += 1
                print(f"[TAG] AprilTag #{tag_id} detected! (tag {self._tags_seen_count}/{self.TOTAL_TAGS})")

                # Acknowledge tag so we don't re-trigger
                with self._lock:
                    self._tag_detected = False
                    self._last_seen_tag_id = tag_id

                # Last tag → land
                if self._tags_seen_count == 2:
                    print("[TAG] Last AprilTag reached — landing!")
                    self.control.land()
                    return

                # Otherwise turn 90° right and keep following
                print("[TAG] Turning 90° right…")
                self.control.turn_yaw(90)
                # Reset PID state after turn
                self._integral = 0.0
                self._prev_error = 0.0
                continue

            with self._lock:
                detected = self._line_detected
                error = self._error

            if detected:
                # Retrieve tuned values from dict & convert from trackbar int ranges
                kp = self.tuning["KP"] / 100.0
                ki = self.tuning["KI"] / 100.0
                kd = self.tuning["KD"] / 100.0
                k_yaw_rate = self.tuning["K_YAW_RATE"] / 100.0
                max_lat_spd = self.tuning["MAX_LATERAL_SPEED"] / 100.0
                fwd_spd = self.tuning["FORWARD_SPEED"] / 100.0

                # PID calculation
                self._integral += error * dt
                derivative = (error - self._prev_error) / dt
                correction = (kp * error +
                              ki * self._integral +
                              kd * derivative)
                self._prev_error = error

                # Clamp lateral speed
                vy = max(-max_lat_spd,
                         min(max_lat_spd, correction * 0.5))

                # Control yaw rate to turn into the curve instead of just sliding sideways
                MAX_YAW_RATE = 0.8
                yaw_rate = k_yaw_rate * correction
                yaw_rate = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, yaw_rate))

                print(f"[FOLLOW] error={error:+.2f}  vy={vy:+.3f}  yaw={yaw_rate:+.2f} vx={fwd_spd}")

                # Move: forward (vx) + lateral correction (vy) + yaw (turn into curves)
                # vz = 0 → maintain altitude
                self.control.move_with_velocity(
                    vx=fwd_spd,
                    vy=vy,
                    vz=0,
                    duration=dt,
                    dt=dt,
                    yaw_rate=yaw_rate
                )
            else:
                # No line seen — hover in place and wait
                print("[FOLLOW] no line detected — hovering")
                self.control.move_with_velocity(0, 0, 0, duration=dt, dt=dt)

            time.sleep(dt)

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------
    def start(self):
        """Start the processing"""
        self.camera.start_thread(self.process_frame)
        self.control.set_mode('GUIDED')
        self.control.arm_motors()
        self.control.takeoff(1)

        # Nudge forward so the downward camera can see the line
        print("Moving forward to find the line…")
        self.control.move_with_velocity(vx=0.2, vy=0, vz=0, duration=3, dt=0.1)

        # Begin following the yellow line
        try:
            self.line_follow()
        except KeyboardInterrupt:
            print("Line follow interrupted.")
            self.control.land()

    def __del__(self):
        """Destructor to ensure threads are stopped"""
        self.camera.stop_thread()


if __name__ == "__main__":
    brain = Brain()
    try:
        brain.start()
    except KeyboardInterrupt:
        print("Stopping brain...")
    finally:
        del brain