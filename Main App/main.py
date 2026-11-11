import streamlit as st
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure the app directory is in sys.path so imports work from any working directory
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Load environment variables from .env file (supports .env in Main App or project root)
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv()

import pandas as pd
from services.auth.login_wall import render_login_wall
from services.state.session_defaults import initial_session_defaults
from services.config.workout_config import EXERCISE_OPTIONS
from services.ui.style_loader import load_css, inject_local_font, inject_webrtc_styles
from services.persistence.exercise_repository import init_db
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from services.vision.exercise_video_processor import VideoProcessorClass
from services.tracking.metrics import sync_metrics_update
from services.persistence.exercise_repository import get_users_exercises, add_exercise
from groq import Groq
from services.coaching.llm import LLMCoach
from services.coaching.tts import TextToSpeech
from services.coaching.voice_pipeline import VoicePipeline, autoplay_audio

  
@st.fragment(run_every="1s")
def live_sync_widget(context):
    if context and hasattr(context, "state") and context.state.playing:
        sync_metrics_update(context)
        st.success("🟢 Camera Active • Tracking & Auto-Saving Reps in Real Time")
    elif context and hasattr(context, "state") and not context.state.playing:
        st.info("💡 Click the **START** button above to activate the camera and start tracking.")

    total_reps = st.session_state.get("reps", 0)
    reps_per_set = st.session_state.get("reps_per_set", 10)
    current_set_reps = st.session_state.get("current_set_reps", 0)
    sets_done = st.session_state.get("sets_completed", 0)
    target_sets = st.session_state.get("target_sets", 3)

    col1, col2, col3 = st.columns(3)
    col1.metric("🏋️ Total Reps", f"{total_reps}")
    col2.metric("🎯 Current Set", f"{current_set_reps} / {reps_per_set}")
    col3.metric("🏆 Sets Done", f"{sets_done} / {target_sets}")


def main():
    st.set_page_config(
        page_icon="🏋️‍♀️",
        page_title="AI Real-time GYM Coach",
        initial_sidebar_state="expanded",
        layout="centered"
    )

    css_path = BASE_DIR / "static" / "style.css"
    if not css_path.exists():
        css_path = Path(os.getcwd()) / "static" / "style.css"
    load_css(str(css_path))

    font_path = BASE_DIR / "static" / "AdobeClean.otf"
    if not font_path.exists():
        font_path = Path(os.getcwd()) / "static" / "AdobeClean.otf"
    inject_local_font(str(font_path), "AdobeClean")

    init_db()

    if not render_login_wall():
        return 

    initial_session_defaults()

    if "voice_pipeline" not in st.session_state:
        try:
            api_key = os.environ.get("GROQ_API_KEY", "").strip()

            if not api_key and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                api_key = str(st.secrets["GROQ_API_KEY"]).strip()

            if api_key and len(api_key) > 10 and not api_key.startswith("your_"):
                groq_client = Groq(api_key=api_key)
                llm_coach = LLMCoach(groq_client)
                tts = TextToSpeech()
                st.session_state.voice_pipeline = VoicePipeline(llm_coach, tts)
            else:
                st.session_state.voice_pipeline = None
        except Exception as e:
            st.session_state.voice_pipeline = None

    workout_started = st.session_state.get("workout_started", False)
    
    with st.sidebar:
        st.title("🏋️‍♂️ Apna AI Coach")

        if st.session_state.username:
            st.caption(f"👤 Logged in as **{st.session_state.username}**")
        if st.button("Log Out / Switch User", key="logout_btn"):
            st.session_state["user_id"] = None
            st.session_state["username"] = None
            st.session_state["workout_started"] = False
            st.rerun()

        st.divider()

        st.subheader("Workout Plan")

        if not workout_started:
            plan_exercise = st.selectbox("Exercise", options=EXERCISE_OPTIONS, key="plan_exercise")

            plan_sets = st.number_input("Sets", min_value=1, max_value=50, value=3, key="plan_sets", step=1)

            plan_reps = st.number_input("Reps per Set", min_value=1, max_value=50, value=10, key="plan_reps", step=1)

            st.markdown("")

            start_session_button = st.button("Start Workout", width="stretch", key="start_session_button")

            if start_session_button:
                st.session_state.exercise_type = plan_exercise
                st.session_state.target_sets = int(plan_sets)
                st.session_state.reps_per_set = int(plan_reps)
                st.session_state.reps = 0
                st.session_state.sets_completed = 0
                st.session_state.current_set_reps = 0
                st.session_state.last_saved_reps = 0
                st.session_state.last_saved_sets_completed = 0
                st.session_state.workout_started = True
                st.session_state.set_cycle_started_at = time.time()

                if st.session_state.voice_pipeline:
                    try:
                        result = st.session_state.voice_pipeline.process_event(
                            event="workout_started",
                            exercise=plan_exercise,
                            metrics={}
                        )
                        if result:
                            st.session_state.audio_to_play, st.session_state.coach_feedback = result
                    except Exception:
                        pass

                st.session_state.last_notified_sets_completed = 0
                st.session_state.last_notified_workout_complete = False
                st.rerun()
        else:
            exercise = st.session_state.get("exercise_type")
            sets = st.session_state.get("target_sets")
            reps = st.session_state.get("reps_per_set")

            st.info(f"**{exercise}** — {sets} Sets / {reps} Reps Target")

            end_session_button = st.button("💾 End & Save Workout", key="end_session_button", width="stretch")

            if end_session_button:
                st.session_state.workout_started = False
                # Save any remaining unsaved progress to SQLite
                user_id = st.session_state.get("user_id", 0)
                reps_done = st.session_state.get("reps", 0)
                sets_done = st.session_state.get("sets_completed", 0)
                last_saved_reps = st.session_state.get("last_saved_reps", 0)
                last_saved_sets = st.session_state.get("last_saved_sets_completed", 0)
                
                if reps_done > last_saved_reps and user_id and exercise:
                    now_ts = time.time()
                    started_at = st.session_state.get("set_cycle_started_at", now_ts)
                    time_taken = max(1, int(now_ts - started_at))
                    unsaved_reps = reps_done - last_saved_reps
                    unsaved_sets = max(0, sets_done - last_saved_sets)
                    add_exercise(user_id, exercise, unsaved_reps, unsaved_sets, time_taken)
                    st.session_state["last_saved_reps"] = reps_done
                    st.session_state["last_saved_sets_completed"] = sets_done
                
                if st.session_state.voice_pipeline:
                    try:
                        result = st.session_state.voice_pipeline.process_event(
                            event="workout_completed",
                            exercise=exercise,
                            metrics={}
                        )
                        if result:
                            st.session_state.audio_to_play, st.session_state.coach_feedback = result
                    except Exception:
                        pass

                st.success("Workout saved to history!")
                st.rerun()

        if workout_started:
            st.divider()

            exercise = st.session_state.get("exercise_type")
            st.subheader("Live Exercise Metrics")

            if exercise == "Squats":
                st.metric("Knee Angle", f"{st.session_state.get('knee_angle', 0)}°")
                st.metric("Back Angle", f"{st.session_state.get('back_angle', 0)}°")
                st.metric("Depth Status", st.session_state.get("depth_status", "N/A"))

            elif exercise == "Push-ups":
                st.metric("Elbow Angle", f"{st.session_state.get('elbow_angle', 0)}°")
                st.metric("Body Alignment", st.session_state.get("body_alignment", "N/A"))
                st.metric("Hip Position", st.session_state.get("hip_status", "N/A"))

            elif exercise == "Biceps Curls (Dumbbell)":
                st.metric("Elbow Angle", f"{st.session_state.get('elbow_angle', 0)}°")
                st.metric("Shoulder Stability", st.session_state.get("shoulder_status", "N/A"))
                st.metric("Swing Detection", st.session_state.get("swing_status", "N/A"))

            elif exercise == "Shoulder Press":
                st.metric("Elbow Angle", f"{st.session_state.get('elbow_angle', 0)}°")
                st.metric("Arm Extension", st.session_state.get("extension_status", "N/A"))
                st.metric("Back Arch", st.session_state.get("back_arch_status", "N/A"))

            elif exercise == "Lunges":
                st.metric("Front Knee Angle", f"{st.session_state.get('front_knee_angle', 0)}°")
                st.metric("Torso Angle", f"{st.session_state.get('torso_angle', 0)}°")
                st.metric("Balance Status", st.session_state.get("balance_status", "N/A"))

    st.title("AI Real-time GYM Coach")
    st.markdown("#### Real-time pose detection with proactive AI voice coaching")
 
    if st.session_state.get("audio_to_play"):
        autoplay_audio(st.session_state.audio_to_play)

    if st.session_state.get("coach_feedback"):
        st.markdown("")
        st.success(f"🤖 **Coach:** {st.session_state.coach_feedback}")

    if not workout_started:
        st.markdown(
            """
            <div style="
                border: 2px dashed #444;
                border-radius: 8px;
                padding: 40px 24px;
                text-align: center;
                color: #aaa;
                margin-top: 24px;
                margin-bottom: 24px;
                background-color: rgba(255, 255, 255, 0.02);
            ">
                <h3 style="color:#eee; margin-bottom:8px;">👈 Set your workout plan in the sidebar</h3>
                <p style="font-size:1.05rem; color:#888;">
                    Choose your exercise, sets and reps, then click <strong>Start Workout</strong> to activate your camera and AI trainer.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        ice_servers = [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:global.stun.twilio.com:3478"]},
            {"urls": ["stun:stun.cloudflare.com:3478"]},
        ]
        try:
            if hasattr(st, "secrets") and "webrtc" in st.secrets:
                turn_cfg = st.secrets["webrtc"]
                if "turn_url" in turn_cfg:
                    ice_servers.append({
                        "urls": [turn_cfg["turn_url"]],
                        "username": turn_cfg.get("turn_username", ""),
                        "credential": turn_cfg.get("turn_password", ""),
                    })
        except Exception:
            pass

        context = webrtc_streamer(
            key="exercise-analysis",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessorClass,
            rtc_configuration={"iceServers": ice_servers},
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 30}
                },
                "audio": False
            },
            async_processing=True
        )

        sync_metrics_update(context)
        live_sync_widget(context)

        inject_webrtc_styles()

    st.divider()

    # Workout History Section
    st.markdown("### 📜 Workout History")

    user_id = st.session_state.get("user_id", 0)

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 Refresh History", key="refresh_hist_btn"):
            st.rerun()

    if isinstance(user_id, int) and user_id > 0:
        history_rows = get_users_exercises(user_id)

        arr = [
            {
                "Exercise": row['exercise_name'],
                "Reps": row['reps'],
                "Sets": row['sets'],
                "Time (sec)": row['time'],
                "Date": row['created_at']
            }
            for row in history_rows
        ]

        df = pd.DataFrame(arr)

        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            agg_df = df.groupby(["Exercise", "Date"]).agg({
                "Reps": 'sum',
                "Sets": "sum",
                "Time (sec)": "sum"
            }).reset_index()
            agg_df.index += 1
            st.table(agg_df, border="horizontal")
        else:
            st.info("No workout history found yet. Start an exercise above and your reps will automatically save here!")
    else:
        st.info("Please log in to view and save your workout history.")


if __name__ == "__main__":
    main()
