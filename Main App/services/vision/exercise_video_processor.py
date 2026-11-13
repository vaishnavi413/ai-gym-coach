import os
import cv2
from pathlib import Path
import av
import numpy as np
import mediapipe as mp
import threading
from streamlit_webrtc import VideoProcessorBase
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from detectors.squat import SquatDetector
from detectors.pushup import PushUpDetector
from detectors.biceps_curl import BicepsCurlDetector
from detectors.shoulder_press import ShoulderPressDetector
from detectors.lunges import LungesDetector
from detectors.neck_rotation import NeckRotationDetector
from detectors.neck_side_bending import NeckSideBendingDetector
from detectors.neck_flexion import NeckFlexionDetector
from detectors.neck_extension import NeckExtensionDetector

from services.config.workout_config import POSE_CONNECTIONS, NECK_EXERCISES
from services.vision.head_pose_estimator import HeadPoseEstimator


class VideoProcessorClass(VideoProcessorBase):
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_metrics = None
        self._exercise_type = "Squats"
        self._frame_timestamps_ms = 0

        # --- 1. Pose Landmarker (for Strength Exercises) ---
        pose_model_path = os.path.join(os.getcwd(), "ml_models", "pose_landmarker_full.task")
        if not os.path.exists(pose_model_path):
            pose_model_path = str(Path(__file__).resolve().parent.parent.parent / "ml_models" / "pose_landmarker_full.task")

        try:
            pose_base_opt = python.BaseOptions(model_asset_path=pose_model_path)
            pose_opts = vision.PoseLandmarkerOptions(
                base_options=pose_base_opt,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.7,
                min_pose_presence_confidence=0.7,
                min_tracking_confidence=0.7,
                output_segmentation_masks=False
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(pose_opts)
        except Exception as e:
            print(f"PoseLandmarker load error: {e}")
            self._landmarker = None

        # --- 2. Face Landmarker (for Neck Mobility Exercises) ---
        face_model_path = os.path.join(os.getcwd(), "ml_models", "face_landmarker.task")
        if not os.path.exists(face_model_path):
            face_model_path = str(Path(__file__).resolve().parent.parent.parent / "ml_models" / "face_landmarker.task")

        try:
            face_base_opt = python.BaseOptions(model_asset_path=face_model_path)
            face_opts = vision.FaceLandmarkerOptions(
                base_options=face_base_opt,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_facial_transformation_matrixes=True
            )
            self._face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)
        except Exception as e:
            print(f"FaceLandmarker load error: {e}")
            self._face_landmarker = None

        # --- 3. Head Pose Estimator with Neutral Calibration ---
        self._head_pose_estimator = HeadPoseEstimator()

        # --- 4. All Detectors (5 Strength + 4 Neck Mobility) ---
        self._detectors = {
            "Squats": SquatDetector(),
            "Push-ups": PushUpDetector(),
            "Biceps Curls (Dumbbell)": BicepsCurlDetector(),
            "Shoulder Press": ShoulderPressDetector(),
            "Lunges": LungesDetector(),
            "Neck Rotation": NeckRotationDetector(),
            "Neck Side Bending": NeckSideBendingDetector(),
            "Neck Flexion": NeckFlexionDetector(),
            "Neck Extension": NeckExtensionDetector(),
        }

    def set_latest_metrics(self, metrics):
        with self._lock:
            self._latest_metrics = metrics.copy()

    def get_latest_metrics(self):
        with self._lock:
            return None if self._latest_metrics is None else self._latest_metrics.copy()

    def set_exercise(self, exercise_type):
        with self._lock:
            if self._exercise_type != exercise_type:
                self._exercise_type = exercise_type
                detector = self._detectors.get(exercise_type)
                if detector:
                    detector.reset()
                if exercise_type in NECK_EXERCISES:
                    self._head_pose_estimator.start_calibration()

    def get_exercise(self):
        with self._lock:
            return self._exercise_type

    def recalibrate_neck(self):
        """Allow manual trigger to recalibrate neutral head position."""
        with self._lock:
            self._head_pose_estimator.start_calibration()

    # ---------------- SKELETON & STRENGTH OVERLAYS ----------------

    def _draw_skeleton(self, img, landmarks):
        h, w = img.shape[:2]

        for start_idx, end_idx in POSE_CONNECTIONS:
            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]

            if p1.visibility > 0.7 and p2.visibility > 0.7:
                cv2.line(
                    img,
                    (int(p1.x * w), int(p1.y * h)),
                    (int(p2.x * w), int(p2.y * h)),
                    (0, 255, 0),
                    8
                )

        for lm in landmarks:
            if lm.visibility > 0.7:
                cv2.circle(
                    img,
                    (int(lm.x * w), int(lm.y * h)),
                    8,
                    (255, 0, 0),
                    -1
                )

    def _draw_no_pose_warnings(self, img):
        cv2.putText(img, "NO POSE DETECTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "PLEASE FACE THE CAMERA", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    def _draw_no_face_warnings(self, img):
        cv2.putText(img, "FACE NOT DETECTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "PLEASE FACE THE CAMERA DIRECTLY", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    def _draw_overlays(self, img, metrics, ex_type):
        if ex_type == "Squats":
            self._draw_squats_overlays(img, metrics)
        elif ex_type == "Push-ups":
            self._draw_pushup_overlays(img, metrics)
        elif ex_type == "Biceps Curls (Dumbbell)":
            self._draw_curl_overlays(img, metrics)
        elif ex_type == "Shoulder Press":
            self._draw_press_overlays(img, metrics)
        elif ex_type == "Lunges":
            self._draw_lunge_overlays(img, metrics)

    def _draw_squats_overlays(self, img, metrics):
        h, _ = img.shape[:2]
        cv2.putText(img, f"DEPTH: {metrics['depth_status']}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def _draw_pushup_overlays(self, img, metrics):
        h, _ = img.shape[:2]
        cv2.putText(img, f"BODY: {metrics['body_alignment']} | HIP: {metrics['hip_status']}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def _draw_curl_overlays(self, img, metrics):
        h, _ = img.shape[:2]
        cv2.putText(img, f"SWING: {metrics['swing_status']}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def _draw_press_overlays(self, img, metrics):
        h, _ = img.shape[:2]
        cv2.putText(img, f"EXT: {metrics['extension_status']} | BACK: {metrics['back_arch_status']}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def _draw_lunge_overlays(self, img, metrics):
        h, _ = img.shape[:2]
        cv2.putText(img, f"BALANCE: {metrics['balance_status']}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ---------------- NECK MOBILITY OVERLAYS ----------------

    def _draw_neck_overlays(self, img, metrics, ex_type):
        h, w = img.shape[:2]

        # Top Bar: Calibration status
        is_calibrated = metrics.get("is_calibrated", False)
        calibrating = metrics.get("calibrating", False)

        if calibrating:
            calib_box_color = (0, 165, 255)  # Orange
            status_text = "CALIBRATING: LOOK STRAIGHT AT CAMERA..."
        elif is_calibrated:
            calib_box_color = (0, 200, 0)    # Green
            status_text = "NEUTRAL POSITION CALIBRATED"
        else:
            calib_box_color = (100, 100, 100)
            status_text = "CALIBRATION PENDING"

        # Background badge for status
        cv2.rectangle(img, (15, 15), (w - 15, 55), (20, 20, 20), -1)
        cv2.rectangle(img, (15, 15), (w - 15, 55), calib_box_color, 2)
        cv2.putText(img, status_text, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, calib_box_color, 2, cv2.LINE_AA)

        # Bottom Box: Exercise movement status & current angle
        cv2.rectangle(img, (15, h - 65), (w - 15, h - 15), (20, 20, 20), -1)
        cv2.rectangle(img, (15, h - 65), (w - 15, h - 15), (0, 255, 200), 2)

        if ex_type == "Neck Rotation":
            yaw = metrics.get("yaw", 0)
            status = metrics.get("turn_status", "")
            cv2.putText(img, f"ROTATION: {status.upper()}", (30, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"Yaw: {yaw:+d} deg", (w - 190, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

        elif ex_type == "Neck Side Bending":
            roll = metrics.get("roll", 0)
            status = metrics.get("tilt_status", "")
            cv2.putText(img, f"SIDE BEND: {status.upper()}", (30, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"Roll: {roll:+d} deg", (w - 190, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

        elif ex_type == "Neck Flexion":
            pitch = metrics.get("pitch", 0)
            status = metrics.get("flexion_status", "")
            cv2.putText(img, f"FLEXION: {status.upper()}", (30, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"Pitch: {pitch:+d} deg", (w - 200, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

        elif ex_type == "Neck Extension":
            pitch = metrics.get("pitch", 0)
            status = metrics.get("extension_status", "")
            cv2.putText(img, f"EXTENSION: {status.upper()}", (30, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"Pitch: {pitch:+d} deg", (w - 200, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    # ---------------- MAIN RECV LOOP ----------------

    def recv(self, frame):
        image = np.asarray(cv2.flip(frame.to_ndarray(format="bgr24"), 1), dtype=np.uint8)
        h, w = image.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        self._frame_timestamps_ms += 33  # ~30 FPS timestamp progression

        ex_type = self.get_exercise()
        is_neck = ex_type in NECK_EXERCISES

        if is_neck:
            # === Process Neck Mobility with Face Landmarker ===
            if self._face_landmarker is None:
                cv2.putText(image, "Face Landmarker Model Not Loaded", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(image, format="bgr24")

            try:
                face_result = self._face_landmarker.detect_for_video(mp_image, self._frame_timestamps_ms)
            except Exception as e:
                face_result = None

            if face_result and face_result.face_landmarks:
                landmarks = face_result.face_landmarks[0]

                # Estimate 3D Head Pose
                pose_data = self._head_pose_estimator.estimate(landmarks, w, h)

                if pose_data:
                    # Draw 3D coordinate axes from nose tip (Red: Pitch, Green: Yaw, Blue: Roll)
                    self._head_pose_estimator.draw_axes(image, pose_data, length=50)
                    # Draw facial feature points outline
                    self._head_pose_estimator.draw_facial_mesh_outline(image, landmarks)

                    detector = self._detectors.get(ex_type)
                    if detector:
                        metrics = detector.process(landmarks, head_pose=pose_data)
                        metrics["face_detected"] = True
                        metrics["pose_detected"] = True
                        metrics["is_calibrated"] = pose_data["is_calibrated"]
                        metrics["calibrating"] = pose_data["calibrating"]
                        metrics["pitch"] = int(pose_data["pitch"])
                        metrics["yaw"] = int(pose_data["yaw"])
                        metrics["roll"] = int(pose_data["roll"])

                        self._draw_neck_overlays(image, metrics, ex_type)
                        self.set_latest_metrics(metrics)
            else:
                self._draw_no_face_warnings(image)
                with self._lock:
                    if self._latest_metrics is not None:
                        self._latest_metrics["face_detected"] = False
                        self._latest_metrics["pose_detected"] = False
                    else:
                        self._latest_metrics = {"face_detected": False, "pose_detected": False}

        else:
            # === Process Strength Exercises with Pose Landmarker ===
            if self._landmarker is None:
                cv2.putText(image, "Pose Landmarker Model Not Loaded", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(image, format="bgr24")

            try:
                result = self._landmarker.detect_for_video(mp_image, self._frame_timestamps_ms)
            except Exception as e:
                result = None

            if result and result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                self._draw_skeleton(image, landmarks)

                detector = self._detectors.get(ex_type)
                if detector:
                    metrics = detector.process(landmarks)
                    metrics["pose_detected"] = True
                    metrics["face_detected"] = True

                    self._draw_overlays(image, metrics, ex_type)
                    self.set_latest_metrics(metrics)
            else:
                self._draw_no_pose_warnings(image)
                with self._lock:
                    if self._latest_metrics is not None:
                        self._latest_metrics["pose_detected"] = False
                    else:
                        self._latest_metrics = {"pose_detected": False}

        return av.VideoFrame.from_ndarray(image, format="bgr24")
