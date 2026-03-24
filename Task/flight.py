# # Your code here

# Airport = [1,2]

'''
This contains the opencv, line following stuff. This will make the decisions for the robot based on what the camera sees and 
send commands to the controller.
'''
import cv2
import numpy as np
from control import Control
import time
import threading
from sensor import Camera

class Brain:
    # --- Adaptive threshold settings ---
    # Block size for adaptive threshold (must be odd). Larger = considers
    # a wider neighbourhood when deciding what is "different".
    ADAPT_BLOCK = 51
    # Constant subtracted from the local mean.  Positive values mean a
    # pixel must be noticeably *brighter* than its surroundings to pass.
    ADAPT_C = -15

    # --- PID gains for lateral steering ---
    KP = 0.4   # proportional
    KI = 0.0   # integral
    KD = 0.1   # derivative

    # --- Speed settings (m/s) ---
    FORWARD_SPEED = 0.15    # constant forward velocity
    MAX_LATERAL_SPEED = 0.03   # clamp for sideways correction

    # --- Minimum contour area to count as a line (filters noise) ---
    MIN_LINE_AREA = 80

    # --- Yellow mask thresholds (HSV) for color input ---
    YELLOW_LOWER = np.array([18, 80, 80], dtype=np.uint8)
    YELLOW_UPPER = np.array([40, 255, 255], dtype=np.uint8)

    # --- AprilTag / turn settings ---
    TOTAL_TAGS = 4            # number of AprilTags on the course (last one = land)

    def __init__(self):
        self.control = Control()
        self.camera = Camera()

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
        Process a grayscale camera frame to detect the line by local
        contrast.  The yellow strip has a different grey-level from
        the surrounding ground; adaptive thresholding picks it up
        regardless of its absolute brightness.
        """
        h, w = frame.shape[:2]
        is_color = len(frame.shape) == 3 and frame.shape[2] >= 3

        # --- AprilTag detection (works on grayscale) ---
        gray_for_tag = frame if not is_color else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._aruco_detector.detectMarkers(gray_for_tag)
        if ids is not None and len(ids) > 0:
            tag_id = int(ids[0][0])
            with self._lock:
                if tag_id != self._last_seen_tag_id:
                    self._tag_detected = True
                    self._tag_id = tag_id

        if is_color:
            # Prefer color-based yellow extraction when RGB/BGR frames are available.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            binary = cv2.inRange(hsv, self.YELLOW_LOWER, self.YELLOW_UPPER)
        else:
            # Fallback for grayscale camera stream.
            blurred = cv2.GaussianBlur(frame, (7, 7), 0)
            binary = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.ADAPT_BLOCK,
                self.ADAPT_C,
            )

        # --- Step 3: Clean up with morphology ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # --- Step 4: Find contours ---
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # --- Debug visualisation ---
        debug = frame.copy() if is_color else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if contours:
            # Pick the largest contour (most likely the line)
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            # Draw all contours (blue) and largest (green)
            cv2.drawContours(debug, contours, -1, (255, 0, 0), 1)
            cv2.drawContours(debug, [largest], -1, (0, 255, 0), 2)

            if area < self.MIN_LINE_AREA:
                with self._lock:
                    self._line_detected = False
                cv2.putText(debug, f"TOO SMALL ({area:.0f})", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                self._show_debug(debug, binary)
                return

            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                error = (cx - w / 2) / (w / 2)
                with self._lock:
                    self._line_detected = True
                    self._error = error
                    self._frame_width = w

                # Draw centroid, centre reference, info
                cv2.circle(debug, (cx, cy), 6, (0, 0, 255), -1)
                cv2.line(debug, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
                cv2.putText(debug, f"err={error:+.2f}  area={area:.0f}",
                            (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                self._show_debug(debug, binary)
                return

        # Nothing found
        with self._lock:
            self._line_detected = False
        cv2.putText(debug, "NO LINE", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
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
                # PID calculation
                self._integral += error * dt
                derivative = (error - self._prev_error) / dt
                correction = (self.KP * error +
                              self.KI * self._integral +
                              self.KD * derivative)
                self._prev_error = error

                # Clamp lateral speed
                vy = max(-self.MAX_LATERAL_SPEED,
                         min(self.MAX_LATERAL_SPEED, correction))

                print(f"[FOLLOW] error={error:+.2f}  vy={vy:+.3f}  vx={self.FORWARD_SPEED}")

                # Move: forward (vx) + lateral correction (vy)
                # vz = 0 → maintain altitude
                self.control.move_with_velocity(
                    vx=self.FORWARD_SPEED,
                    vy=vy,
                    vz=0,
                    duration=dt,
                    dt=dt,
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