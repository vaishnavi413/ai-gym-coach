import streamlit as st


def initial_session_defaults():
    defaults = {
        "reps": 0,
        "target_sets": 3,
        "reps_per_set": 10,
        "sets_completed": 0,
        "current_set_reps": 0,
        "workout_complete": False,
        "last_notified_sets_completed": 0,
        "last_notified_workout_complete": False,
        "last_saved_sets_completed": 0,
        "last_saved_reps": 0,
        "set_cycle_started_at": 0.0,
        "last_exercise_type": "Squats",

        # Category and Demo Mode
        "exercise_category": "Strength Exercises",
        "quick_demo_mode": False,

        # Workout plan (set before starting)
        "workout_started": False,
        "plan_exercise": "Squats",
        "plan_sets": 3,
        "plan_reps": 10,

        # Strength Angles & Statuses
        "knee_angle": 0,
        "back_angle": 0,
        "elbow_angle": 0,
        "front_knee_angle": 0,
        "torso_angle": 0,
        "depth_status": "N/A",
        "body_alignment": "N/A",
        "hip_status": "N/A",
        "shoulder_status": "N/A",
        "swing_status": "N/A",
        "extension_status": "N/A",
        "back_arch_status": "N/A",
        "balance_status": "N/A",

        # Neck Angles & Statuses
        "yaw": 0,
        "pitch": 0,
        "roll": 0,
        "stage": "CENTER",
        "turn_status": "Looking Forward (Center)",
        "tilt_status": "Head Upright (Center)",
        "flexion_status": "Neutral Position",
        "neck_status": "Neutral Position",
        "is_calibrated": False,
        "calibrating": False,
        "face_detected": True,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
