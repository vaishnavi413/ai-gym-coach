import streamlit as st
from services.persistence.exercise_repository import get_or_create_user


def render_login_wall():
    if st.session_state.get("user_id") is not None:
        return True
    
    st.title("🏋️ AI Real-time GYM Trainer")
    st.markdown("### Login or Register New User")
    st.caption("Enter your username below. If you are a new user, your account and progress history will be created automatically.")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", placeholder="e.g. vaishnavi, alex, rahul").strip()
        submit_button = st.form_submit_button("Start Session", width="stretch")

    if submit_button:
        if not username:
            st.error("Username cannot be empty.")
            return False
        
        user = get_or_create_user(username)
    
        st.session_state["user_id"] = user["id"]
        st.session_state["username"] = user["username"]

        st.rerun()

    return False
