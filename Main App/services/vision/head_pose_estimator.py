import cv2
import numpy as np
import time


class HeadPoseEstimator:
    """
    Estimates 3D Head Pose (Pitch, Yaw, Roll) using 6 standard facial landmarks
    and OpenCV solvePnP with iterative Levenberg-Marquardt optimization.
    Also handles neutral position calibration and 3D orientation axis drawing.
    """
    # 3D Canonical Model Points (in mm)
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),             # 1: Nose tip
        (0.0, -330.0, -65.0),        # 152: Chin
        (-225.0, 170.0, -135.0),     # 33: Left eye outer corner
        (225.0, 170.0, -135.0),      # 263: Right eye outer corner
        (-150.0, -150.0, -125.0),    # 61: Left mouth corner
        (150.0, -150.0, -125.0)      # 291: Right mouth corner
    ], dtype=np.float64)

    # MediaPipe Face Landmark Indices corresponding to MODEL_POINTS
    LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

    def __init__(self):
        self.is_calibrated = False
        self.calibrating = True
        self.calibration_samples = []
        self.max_calibration_samples = 25  # ~0.8-1 second of frames

        # Calibrated baseline angles
        self.base_pitch = 0.0
        self.base_yaw = 0.0
        self.base_roll = 0.0

        # Smoothed angles to reduce webcam jitter
        self.smooth_pitch = 0.0
        self.smooth_yaw = 0.0
        self.smooth_roll = 0.0
        self.alpha = 0.35  # Smoothing factor (0 = full smooth, 1 = raw)

        # Dist coeffs (assumes pinhole camera without severe distortion)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def start_calibration(self):
        """Reset calibration and start collecting neutral reference frames."""
        self.is_calibrated = False
        self.calibrating = True
        self.calibration_samples.clear()

    @staticmethod
    def _angle_diff(target, base):
        """Calculate shortest angular difference between target and base in [-180, 180]."""
        return ((target - base + 180.0) % 360.0) - 180.0

    def estimate(self, face_landmarks, frame_w, frame_h):
        """
        Estimate pitch, yaw, roll from MediaPipe face landmarks.
        Returns a dictionary with raw and relative angles, rvec, tvec, and calibration status.
        """
        if not face_landmarks:
            return None

        # Extract 2D image coordinates for the 6 key points
        image_points = []
        for idx in self.LANDMARK_INDICES:
            lm = face_landmarks[idx]
            image_points.append((lm.x * frame_w, lm.y * frame_h))
        image_points = np.array(image_points, dtype=np.float64)

        # Camera Intrinsic Matrix
        focal_length = frame_w
        center = (frame_w / 2.0, frame_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Solve Perspective-n-Point
        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS,
            image_points,
            camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose rotation matrix into Euler angles (Pitch, Yaw, Roll in degrees)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        raw_pitch = float(angles[0])
        raw_yaw = float(angles[1])
        raw_roll = float(angles[2])

        # Calibration handling
        if self.calibrating:
            self.calibration_samples.append((raw_pitch, raw_yaw, raw_roll))
            if len(self.calibration_samples) >= self.max_calibration_samples:
                samples = np.array(self.calibration_samples)
                self.base_pitch = float(np.median(samples[:, 0]))
                self.base_yaw = float(np.median(samples[:, 1]))
                self.base_roll = float(np.median(samples[:, 2]))
                self.is_calibrated = True
                self.calibrating = False

        # Calculate relative angles from neutral baseline
        if self.is_calibrated:
            rel_pitch = self._angle_diff(raw_pitch, self.base_pitch)
            rel_yaw = self._angle_diff(raw_yaw, self.base_yaw)
            rel_roll = self._angle_diff(raw_roll, self.base_roll)
        else:
            rel_pitch = 0.0
            rel_yaw = 0.0
            rel_roll = 0.0

        # Apply exponential moving average smoothing
        if self.is_calibrated:
            self.smooth_pitch = self.alpha * rel_pitch + (1 - self.alpha) * self.smooth_pitch
            self.smooth_yaw = self.alpha * rel_yaw + (1 - self.alpha) * self.smooth_yaw
            self.smooth_roll = self.alpha * rel_roll + (1 - self.alpha) * self.smooth_roll
        else:
            self.smooth_pitch = 0.0
            self.smooth_yaw = 0.0
            self.smooth_roll = 0.0

        nose_point = (int(image_points[0][0]), int(image_points[0][1]))

        return {
            "pitch": self.smooth_pitch,      # (+) Down / Flexion, (-) Up / Extension
            "yaw": self.smooth_yaw,          # (+) Turn Left, (-) Turn Right
            "roll": self.smooth_roll,        # (-) Tilt Left, (+) Tilt Right
            "raw_pitch": raw_pitch,
            "raw_yaw": raw_yaw,
            "raw_roll": raw_roll,
            "is_calibrated": self.is_calibrated,
            "calibrating": self.calibrating,
            "calibration_progress": len(self.calibration_samples) / float(self.max_calibration_samples),
            "rvec": rvec,
            "tvec": tvec,
            "camera_matrix": camera_matrix,
            "nose_point": nose_point,
            "image_points": image_points
        }

    def draw_axes(self, img, pose_data, length=50):
        """
        Draw 3D coordinate axes projecting from the nose tip:
        - X axis (Pitch): Red
        - Y axis (Yaw): Green
        - Z axis (Roll / Forward vector): Blue
        """
        if not pose_data:
            return

        rvec = pose_data["rvec"]
        tvec = pose_data["tvec"]
        camera_matrix = pose_data["camera_matrix"]
        nose_pt = pose_data["nose_point"]

        axis_points = np.array([
            (length, 0.0, 0.0),   # X axis (Pitch)
            (0.0, length, 0.0),   # Y axis (Yaw)
            (0.0, 0.0, length)    # Z axis (Roll)
        ], dtype=np.float64)

        proj_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, self.dist_coeffs)

        pt_x = tuple(proj_points[0].ravel().astype(int))
        pt_y = tuple(proj_points[1].ravel().astype(int))
        pt_z = tuple(proj_points[2].ravel().astype(int))

        # Red for X (Pitch axis)
        cv2.line(img, nose_pt, pt_x, (0, 0, 255), 3, cv2.LINE_AA)
        # Green for Y (Yaw axis)
        cv2.line(img, nose_pt, pt_y, (0, 255, 0), 3, cv2.LINE_AA)
        # Blue for Z (Roll axis)
        cv2.line(img, nose_pt, pt_z, (255, 0, 0), 3, cv2.LINE_AA)

        # Highlight nose point
        cv2.circle(img, nose_pt, 4, (0, 255, 255), -1)

    @staticmethod
    def draw_facial_mesh_outline(img, face_landmarks):
        """Draw minimal facial contour for visual feedback."""
        if not face_landmarks:
            return

        h, w = img.shape[:2]
        # Draw key feature points: eyes, mouth, nose, chin
        key_indices = [
            1, 152, 10, 33, 133, 263, 362, 61, 291, 0, 17, 234, 454
        ]
        for idx in key_indices:
            lm = face_landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 2, (0, 255, 200), -1)
