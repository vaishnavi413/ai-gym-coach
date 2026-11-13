import time
from core.base_exercise import BaseExercise


class NeckRotationDetector(BaseExercise):
    """
    Neck Rotation Exercise Detector
    Movement: CENTER -> LEFT -> CENTER -> RIGHT -> CENTER = 1 REP
    (Also supports starting with RIGHT first)
    """
    YAW_TURN_THRESHOLD = 13.0   # Degrees to count as an intentional gentle turn
    YAW_CENTER_THRESHOLD = 7.0   # Degrees within which the head is considered at center
    EXCESSIVE_TURN_LIMIT = 45.0  # Alert if user turns too aggressively

    def __init__(self):
        super().__init__()
        self.stage = "CENTER"
        self.last_rep_time = 0.0
        self.min_rep_interval = 0.8  # seconds

    def reset(self):
        self.reps = 0
        self.stage = "CENTER"
        self.last_rep_time = 0.0

    def process(self, landmarks, head_pose=None):
        if not head_pose or not head_pose.get("is_calibrated"):
            return {
                "reps": self.reps,
                "yaw": 0,
                "stage": "CALIBRATING",
                "turn_status": "LOOK STRAIGHT (CALIBRATING)",
                "form_issue": None
            }

        yaw = head_pose["yaw"]
        abs_yaw = abs(yaw)
        now = time.time()
        form_issue = None

        # Safety check: avoid excessive end-range forcing
        if abs_yaw > self.EXCESSIVE_TURN_LIMIT:
            form_issue = "Excessive rotation detected — keep the turn gentle and comfortable."

        # State machine
        # Case A: Left first (yaw > +THRESHOLD)
        if self.stage == "CENTER":
            if yaw >= self.YAW_TURN_THRESHOLD:
                self.stage = "LEFT_REACHED"
            elif yaw <= -self.YAW_TURN_THRESHOLD:
                self.stage = "RIGHT_REACHED"

        elif self.stage == "LEFT_REACHED":
            # Must return to center before proceeding to the other side
            if abs_yaw <= self.YAW_CENTER_THRESHOLD:
                self.stage = "LEFT_RETURNED"

        elif self.stage == "LEFT_RETURNED":
            # Now turn right to complete the cycle
            if yaw <= -self.YAW_TURN_THRESHOLD:
                self.stage = "RIGHT_AFTER_LEFT"

        elif self.stage == "RIGHT_AFTER_LEFT":
            # Return to center finishes 1 complete rep!
            if abs_yaw <= self.YAW_CENTER_THRESHOLD:
                if now - self.last_rep_time >= self.min_rep_interval:
                    self.reps += 1
                    self.last_rep_time = now
                self.stage = "CENTER"

        # Case B: Symmetrical support if user starts by turning right first
        elif self.stage == "RIGHT_REACHED":
            if abs_yaw <= self.YAW_CENTER_THRESHOLD:
                self.stage = "RIGHT_RETURNED"

        elif self.stage == "RIGHT_RETURNED":
            if yaw >= self.YAW_TURN_THRESHOLD:
                self.stage = "LEFT_AFTER_RIGHT"

        elif self.stage == "LEFT_AFTER_RIGHT":
            if abs_yaw <= self.YAW_CENTER_THRESHOLD:
                if now - self.last_rep_time >= self.min_rep_interval:
                    self.reps += 1
                    self.last_rep_time = now
                self.stage = "CENTER"

        # Human-readable movement status for UI
        status_map = {
            "CENTER": "Looking Forward (Center)",
            "LEFT_REACHED": "Turning Left",
            "LEFT_RETURNED": "Left Done — Now Turn Right",
            "RIGHT_AFTER_LEFT": "Turning Right",
            "RIGHT_REACHED": "Turning Right",
            "RIGHT_RETURNED": "Right Done — Now Turn Left",
            "LEFT_AFTER_RIGHT": "Turning Left",
        }
        turn_status = status_map.get(self.stage, "Centered")

        return {
            "reps": self.reps,
            "yaw": int(yaw),
            "stage": self.stage,
            "turn_status": turn_status,
            "form_issue": form_issue
        }
