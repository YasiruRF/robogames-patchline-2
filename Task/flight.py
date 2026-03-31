# # Your code here

Airports = [1, 2]

'''
This contains the opencv, line following stuff. This will make the decisions for the robot based on what the camera sees and 
send commands to the controller.
'''
import cv2
import numpy as np
import json
import os
import random
from control import Control
import time
import threading
from sensor import Camera

class Brain:
    # --- Default tuning ---
    DEFAULT_TUNING = {
        "GRAY_THRESH": 140,
        "BLUR_SIZE_K": 8,     # Trackbar value, real kernel is k*2+1
        "MORPH_SIZE_K": 7,    # Trackbar value, real kernel is k*2+1
        "KP": 40,             # Divided by 100 -> 0.4
        "KI": 0,              # Divided by 100 -> 0.0
        "KD": 10,             # Divided by 100 -> 0.1
        "K_YAW_RATE": 120,    # Divided by 100 -> 1.2
        "FORWARD_SPEED": 13,  # Divided by 100 -> 0.13
        "MAX_LATERAL_SPEED": 3 # Divided by 100 -> 0.03
    }

    # --- Minimum contour area to count as a line (filters noise) ---
    MIN_LINE_AREA = 500

    # --- Yellow mask thresholds (HSV) for color input ---
    YELLOW_LOWER = np.array([18, 80, 80], dtype=np.uint8)
    YELLOW_UPPER = np.array([40, 255, 255], dtype=np.uint8)

    # --- AprilTag / turn settings ---
    TOTAL_TAGS = 5            # number of AprilTags on the course (last one = land)

    def __init__(self):
        self.control = Control()
        self.camera = Camera()

        # Load configuration
        self.tuning = self.DEFAULT_TUNING.copy()

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
        
        self._junction_decision = 'straight'
        self._junction_timeout = 0

        # AprilTag detector (OpenCV ArUco with AprilTag dictionary)
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self._aruco_params = cv2.aruco.DetectorParameters()
        
        # Optimize ArUco parameters for more robust, forgiving detection
        self._aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._aruco_params.adaptiveThreshWinSizeMin = 3
        self._aruco_params.adaptiveThreshWinSizeMax = 53
        self._aruco_params.adaptiveThreshWinSizeStep = 10
        self._aruco_params.minMarkerPerimeterRate = 0.02
        self._aruco_params.maxErroneousBitsInBorderRate = 0.5
        self._aruco_params.errorCorrectionRate = 1.0  # Max out error correction
        self._aruco_params.polygonalApproxAccuracyRate = 0.08  # More forgiving polygon shape
        self._aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.3
        
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
        
        # Lookahead: Ignore the bottom 15% instead of 40% so it can see 90-degree lines exiting 
        # horizontally off the bottom corners of the camera view
        binary[int(h * 0.85):, :] = 0 

        # 4. Extract Line Centroid
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug = frame.copy() if is_color else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        with self._lock:
            self._line_detected = False
            
            valid_contours = [c for c in contours if cv2.contourArea(c) >= self.MIN_LINE_AREA]
            
            if valid_contours:
                # If there are multiple paths (e.g. at a junction), pick based on our junction choice.
                # First, sort contours left-to-right based on their centroid X coordinates.
                def get_cx(c):
                    M = cv2.moments(c)
                    return int(M["m10"] / M["m00"]) if M["m00"] > 0 else 0
                
                valid_contours.sort(key=get_cx)
                
                # Default is the largest contour, but if we expect a multiple-path junction:
                if len(valid_contours) > 1 and getattr(self, '_junction_choice', -1) != -1:
                    # Pick a branch randomly assigned earlier, modding by length to be safe.
                    idx = min(self._junction_choice, len(valid_contours) - 1)
                    chosen = valid_contours[idx]
                else:
                    chosen = max(valid_contours, key=cv2.contourArea)
                
                area = cv2.contourArea(chosen)
                cv2.drawContours(debug, [chosen], -1, (0, 255, 0), 2)
                
                M = cv2.moments(chosen)
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
        Searches the graph of airports to fulfill the sequence of Targets.
        """
        print("Starting yellow-line following…")
        
        # Filter valid target destinations and start queue
        target_airports = [a for a in Airports if a != 0]
        current_target_idx = 0
        
        if len(target_airports) == 0:
            print("No valid airports defined in 'Airports' array. Exiting.")
            self.control.land()
            return
            
        print(f"Target destinations sequence: {target_airports}")
        current_target = target_airports[current_target_idx]
        
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
                
                # Convert the tag_id to a 3-character string and separate it into 3 digits
                # padding with zeros if necessary (e.g. 15 -> "015")
                tag_str = str(tag_id).zfill(3)
                
                country_code = int(tag_str[0])
                status = int(tag_str[1])
                reachable = int(tag_str[2])

                print(f"[TAG] Airport Detected! Tag ID: {tag_id} ({self._tags_seen_count}/{self.TOTAL_TAGS})")
                print(f"      - Country Code: {country_code}")
                print(f"      - Status: {'OK to Land' if status == 1 else 'Cannot Land'}")
                print(f"      - Reachable Airports: {reachable}")

                # If there are multiple branches from this node, pick one at random
                with self._lock:
                    if reachable > 1:
                        self._junction_choice = random.randint(0, reachable - 1)
                        print(f"[TAG] Node has {reachable} branches. Setting junction logic to pick branch #{self._junction_choice}.")
                    else:
                        self._junction_choice = -1 # Default behavior

                # Acknowledge tag so we don't re-trigger
                with self._lock:
                    self._tag_detected = False
                    self._last_seen_tag_id = tag_id

                # Evaluate if this is the currently required valid airport
                if country_code == current_target and status == 1:
                    print(f"[TAG] Valid airport found for target country {current_target}! Landing on platform...")
                    
                    # Land descending slowly (positive vz is down in NED frame)
                    self.control.move_with_velocity(vx=0, vy=0, vz=0.5, duration=2.5, dt=0.1)

                    # Stay in the landing platform for 4 seconds, don't turn off motors
                    print("[TAG] Waiting on platform for 4 seconds...")
                    self.control.move_with_velocity(vx=0, vy=0, vz=0, duration=4, dt=0.1)
                    
                    # Lift back up to ~1m (negative vz is up in NED frame)
                    print("[TAG] Lifting back to 1m...")
                    self.control.move_with_velocity(vx=0, vy=0, vz=-0.5, duration=2.5, dt=0.1)
                    
                    # Stabilize hover before continuing
                    self.control.move_with_velocity(vx=0, vy=0, vz=0, duration=1, dt=0.1)
                    
                    # Advance to next target in sequence
                    current_target_idx += 1
                    if current_target_idx >= len(target_airports):
                        print("[MISSION] Successfully visited all target airports. Mission Complete!")
                        self.control.land()
                        return
                    
                    # More targets remaining -> resume flying!
                    current_target = target_airports[current_target_idx]
                    print(f"[MISSION] Next target country objective is: {current_target}")
                    print("[MISSION] Resuming search from platform...")
                    
                    print("[MISSION] Airborne. Re-aligning with line...")
                    self.control.move_with_velocity(vx=0.2, vy=0, vz=0, duration=3, dt=0.1)
                    
                    self._integral = 0.0
                    self._prev_error = 0.0
                    self._tags_seen_count = 0  # Reset graph counter since we are at a new node
                    continue

                # Last tag fallback → land
                if self._tags_seen_count >= self.TOTAL_TAGS:
                    print("[TAG] Visited max node limit without finding valid target — landing anyway.")
                    self.control.land()
                    return

                # Otherwise keep following the line straight to the next airport
                print(f"[TAG] Condition does not match (needs Country {current_target} & OK to Land).")
                print(f"[TAG] Continuing search through connected nodes...")
                
                # We do NOT turn 90° anymore! Just follow the line.
                # Reset PID state to prevent sudden jerks, but keep _prev_error in memory
                # so the drone knows which way to turn if it temporarily loses the line at 90-deg junctions!
                self._integral = 0.0
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
                # No line seen — attempt recovery
                # Rotate in the direction of the last known error to search for the line
                # This specifically handles 90-degree corners where the drone might initially fly past the track
                spin_speed = 0.5  # rad/s max spin
                
                # Make the drone rotate in place towards the last seen side
                direction = 1 if self._prev_error > 0 else -1
                recovery_yaw = spin_speed * direction
                
                print(f"[FOLLOW] no line detected — recovering! yaw={recovery_yaw:+.2f}")
                
                # Cut forward velocity, just spin in place
                self.control.move_with_velocity(
                    vx=0.0,
                    vy=0.0,
                    vz=0,
                    duration=dt,
                    dt=dt,
                    yaw_rate=recovery_yaw
                )

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