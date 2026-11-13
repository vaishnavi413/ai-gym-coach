import time
from core.base_exercise import BaseExercise


class NeckSideBendingDetector(BaseExercise):
    """
    Neck Side Bending (Lateral Flexion) Detector
    Movement: CENTER -> LEFT TILT -> CENTER -> RIGHT TILT -> CENTER = 1 REP
    (Also supports starting with RIGHT TILT first)
    """
    ROLL_TILT_THRESHOLD = 9.0    # Degrees to count as an intentional gentle lateral tilt
    ROLL_CENTER_THRESHOLD = 5.0   # Degrees within which the head is upright
    EXCESSIVE_TILT_LIMIT = 35.0   # Alert if user tilts too aggressively

    def __init__(self):
        super().__init__()
        self.stage = "CENTER"
        self.last_rep_time = 0.0
        self.min_rep_interval = 0.8

    def reset(self):
        self.reps = 0
        self.stage = "CENTER"
        self.last_rep_time = 0.0

    def process(self, landmarks, head_pose=None):
        if not head_pose or not head_pose.get("is_calibrated"):
            return {
                "reps": self.reps,
                "roll": 0,
                "stage": "CALIBRATING",
                "tilt_status": "LOOK STRAIGHT (CALIBRATING)",
                "form_issue": None
            }

        roll = head_pose["roll"]
        abs_roll = abs(roll)
        now = time.time()
        form_issue = None

        if abs_roll > self.EXCESSIVE_TILT_LIMIT:
            form_issue = "Excessive neck tilt detected — keep the movement gentle."

        # State machine
        # Negative roll = Left tilt, Positive roll = Right tilt (or vice-versa, handled symmetrically)
        if self.stage == "CENTER":
            if roll <= -self.ROLL_TILT_THRESHOLD:
                self.stage = "LEFT_TILT_REACHED"
            elif roll >= self.ROLL_TILT_THRESHOLD:
                self.stage = "RIGHT_TILT_REACHED"

        elif self.stage == "LEFT_TILT_REACHED":
            if abs_roll <= self.ROLL_CENTER_THRESHOLD:
                self.stage = "LEFT_RETURNED"

        elif self.stage == "LEFT_RETURNED":
            if roll >= self.ROLL_TILT_THRESHOLD:
                self.stage = "RIGHT_AFTER_LEFT"

        elif self.stage == "RIGHT_AFTER_LEFT":
            if abs_roll <= self.ROLL_CENTER_THRESHOLD:
                if now - self.last_rep_time >= self.min_rep_interval:
                    self.reps += 1
                    self.last_rep_time = now
                self.stage = "CENTER"

        # Symmetrical support: Right tilt first
        elif self.stage == "RIGHT_TILT_REACHED":
            if abs_roll <= self.ROLL_CENTER_THRESHOLD:
                self.stage = "RIGHT_RETURNED"

        elif self.stage == "RIGHT_RETURNED":
            if roll <= -self.ROLL_TILT_THRESHOLD:
                self.stage = "LEFT_AFTER_RIGHT"

        elif self.stage == "LEFT_AFTER_RIGHT":
            if abs_roll <= self.ROLL_CENTER_THRESHOLD:
                if now - self.last_rep_time >= self.min_rep_interval:
                    self.reps += 1
                    self.last_rep_time = now
                self.stage = "CENTER"

        status_map = {
            "CENTER": "Head Upright (Center)",
            "LEFT_TILT_REACHED": "Tilting Left (Toward Shoulder)",
            "LEFT_RETURNED": "Left Done — Now Tilt Right",
            "RIGHT_AFTER_LEFT": "Tilting Right (Toward Shoulder)",
            "RIGHT_TILT_REACHED": "Tilting Right (Toward Shoulder)",
            "RIGHT_RETURNED": "Right Done — Now Tilt Left",
            "LEFT_AFTER_RIGHT": "Tilting Left (Toward Shoulder)",
        }
        tilt_status = status_map.get(self.stage, "Upright")

        return {
            "reps": self.reps,
            "roll": int(roll),
            "stage": self.stage,
            "tilt_status": tilt_status,
            "form_issue": form_issue
        }
