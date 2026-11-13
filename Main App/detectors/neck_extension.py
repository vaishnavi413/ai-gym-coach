import time
from core.base_exercise import BaseExercise


class NeckExtensionDetector(BaseExercise):
    """
    Neck Extension Detector
    Movement: NEUTRAL -> LOOK SLIGHTLY UPWARDS -> RETURN TO NEUTRAL = 1 REP
    """
    PITCH_UP_THRESHOLD = -10.0      # Degrees upward pitch (looking slightly up)
    PITCH_NEUTRAL_THRESHOLD = -4.0  # Degrees within neutral
    EXCESSIVE_EXTENSION_LIMIT = -30.0 # Alert if neck hyperextended backwards

    def __init__(self):
        super().__init__()
        self.stage = "NEUTRAL"
        self.last_rep_time = 0.0
        self.min_rep_interval = 0.8

    def reset(self):
        self.reps = 0
        self.stage = "NEUTRAL"
        self.last_rep_time = 0.0

    def process(self, landmarks, head_pose=None):
        if not head_pose or not head_pose.get("is_calibrated"):
            return {
                "reps": self.reps,
                "pitch": 0,
                "stage": "CALIBRATING",
                "extension_status": "LOOK STRAIGHT (CALIBRATING)",
                "form_issue": None
            }

        pitch = head_pose["pitch"]
        now = time.time()
        form_issue = None

        if pitch < self.EXCESSIVE_EXTENSION_LIMIT:
            form_issue = "Excessive extension detected — look up gently, do not strain backwards."

        # State machine: NEUTRAL -> UP -> NEUTRAL
        if self.stage == "NEUTRAL":
            if pitch <= self.PITCH_UP_THRESHOLD:
                self.stage = "UP"

        elif self.stage == "UP":
            if pitch >= self.PITCH_NEUTRAL_THRESHOLD:
                if now - self.last_rep_time >= self.min_rep_interval:
                    self.reps += 1
                    self.last_rep_time = now
                self.stage = "NEUTRAL"

        status_map = {
            "NEUTRAL": "Neutral Position (Looking Ahead)",
            "UP": "Looking Upwards (Extension)",
        }
        extension_status = status_map.get(self.stage, "Neutral")

        return {
            "reps": self.reps,
            "pitch": int(pitch),
            "stage": self.stage,
            "extension_status": extension_status,
            "form_issue": form_issue
        }
