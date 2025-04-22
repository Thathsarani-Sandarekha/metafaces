import streamlit as st
import json

# Load credentials
with open("/teamspace/studios/this_studio/metafaces_UI/credentials.json", "r") as f:
    credentials = json.load(f)

st.set_page_config(page_title="Login", layout="centered")

st.title("🔐 Login to Story Generator")

username = st.text_input("Username")
password = st.text_input("Password", type="password")
login_btn = st.button("Login")

if login_btn:
    if username == credentials["user"]["username"] and password == credentials["user"]["password"]:
        st.success("Logged in as User")
        st.switch_page("pages/user_ui.py")
    elif username == credentials["engineer"]["username"] and password == credentials["engineer"]["password"]:
        st.success("Logged in as ML Engineer")
        st.switch_page("/teamspace/studios/this_studio/metafaces_UI/engineer_ui.py")
    else:
        st.error("Invalid username or password")
