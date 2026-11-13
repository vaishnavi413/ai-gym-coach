import time
from core.base_exercise import BaseExercise


class NeckFlexionDetector(BaseExercise):
    """
    Neck Flexion Detector
    Movement: NEUTRAL -> LOWER CHIN DOWN -> RETURN TO NEUTRAL = 1 REP
    """
    PITCH_DOWN_THRESHOLD = 11.0   # Degrees downward pitch (chin toward chest)
    PITCH_NEUTRAL_THRESHOLD = 5.0  # Degrees within neutral
    EXCESSIVE_FLEXION_LIMIT = 35.0 # Alert if head bent too far forward

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
                "flexion_status": "LOOK STRAIGHT (CALIBRATING)",
                "form_issue": None
            }

        pitch = head_pose["pitch"]
        now = time.time()
        form_issue = None

        if pitch > self.EXCESSIVE_FLEXION_LIMIT:
            form_issue = "Excessive flexion detected — lower chin gently without forcing."

        # State machine: NEUTRAL -> DOWN -> NEUTRAL
        if self.stage == "NEUTRAL":
            if pitch >= self.PITCH_DOWN_THRESHOLD:
                self.stage = "DOWN"

        elif self.stage == "DOWN":
            if pitch <= self.PITCH_NEUTRAL_THRESHOLD:
                if now - self.last_rep_time >= self.min_rep_interval:
                    self.reps += 1
                    self.last_rep_time = now
                self.stage = "NEUTRAL"

        status_map = {
            "NEUTRAL": "Neutral Position (Looking Ahead)",
            "DOWN": "Lowering Chin (Flexion)",
        }
        flexion_status = status_map.get(self.stage, "Neutral")

        return {
            "reps": self.reps,
            "pitch": int(pitch),
            "stage": self.stage,
            "flexion_status": flexion_status,
            "form_issue": form_issue
        }
