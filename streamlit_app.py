import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Streamlit page config
st.set_page_config(layout="wide", page_title="🎥 Articulation Meter")

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawing = mp.solutions.drawing_utils

# Data storage
MAX_LEN = 100
left_hand_data = deque(maxlen=MAX_LEN)
right_hand_data = deque(maxlen=MAX_LEN)
shoulder_mid_data = deque(maxlen=MAX_LEN)
head_angle_data = deque(maxlen=MAX_LEN)
engagement_levels = deque(maxlen=MAX_LEN)
frame_numbers = deque(maxlen=MAX_LEN)

# Logic functions
def calculate_movement(landmarks):
    try:
        lh = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        rh = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh_xy = np.array([lh.x, lh.y])
        rh_xy = np.array([rh.x, rh.y])
        ls_xy = np.array([ls.x, ls.y])
        rs_xy = np.array([rs.x, rs.y])
        shoulder_mid = ((ls_xy + rs_xy) / 2)[0]
        head_angle = np.arctan2(ls.y - rs.y, ls.x - rs.x)
        return lh_xy[0], rh_xy[0], shoulder_mid, head_angle
    except:
        return None, None, None, None

def classify_engagement(l, r, s, h):
    motion = np.std([l, r, s, h])
    if motion > 0.02:
        return 2
    elif motion > 0.01:
        return 1
    else:
        return 0

def get_engagement_label(level):
    return ['🟦 Attentive', '🟧 Interactive', '🟥 Inspired'][level]

# Sidebar controls
st.sidebar.title("🎛 Control Panel")
st.sidebar.markdown("Use this panel to control the session.")
stop_stream = st.sidebar.button("⛔ Stop Stream")

# Title
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🎥 Real-Time Articulation Meter</h1>", unsafe_allow_html=True)

# Layout columns
video_col, stat_col = st.columns([1.2, 1])

# Streamlit containers
frame_placeholder = video_col.empty()
status_box = stat_col.container()
chart_box = stat_col.container()

# Webcam init
cap = cv2.VideoCapture(0)
frame_idx = 0

# Processing loop
with st.spinner("🔄 Initializing camera..."):
    while cap.isOpened():
        if stop_stream:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        label = "❌ No Pose Detected"

        if result.pose_landmarks:
            drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = result.pose_landmarks.landmark
            l, r, s, h = calculate_movement(landmarks)
            if None not in [l, r, s, h]:
                left_hand_data.append(l)
                right_hand_data.append(r)
                shoulder_mid_data.append(s)
                head_angle_data.append(h)
                frame_numbers.append(frame_idx)
                level = classify_engagement(l, r, s, h)
                engagement_levels.append(level)
                label = get_engagement_label(level)

        # Display webcam
        frame = cv2.putText(frame, f"Engagement: {label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)

        # Engagement status
        with status_box:
            st.markdown(f"### 🧠 Current Engagement: `{label}`")

        # Graphs
        if len(frame_numbers) > 5:
            fig, axs = plt.subplots(4, 1, figsize=(6, 10))
            axs[0].plot(frame_numbers, left_hand_data, label="Left Hand", color="blue")
            axs[0].plot(frame_numbers, right_hand_data, label="Right Hand", color="red")
            axs[0].set_title("🤲 Hand Movements"); axs[0].legend()

            axs[1].plot(frame_numbers, shoulder_mid_data, color="orange")
            axs[1].set_title("💪 Shoulder Midpoint")

            axs[2].plot(frame_numbers, head_angle_data, color="purple")
            axs[2].set_title("🧍 Head Turn Angle")

            axs[3].step(frame_numbers, engagement_levels, where='mid', color="green")
            axs[3].set_yticks([0, 1, 2])
            axs[3].set_yticklabels(['Attentive', 'Interactive', 'Inspired'])
            axs[3].set_title("📈 Engagement Level")

            for ax in axs:
                ax.set_xlabel("Frame")
                ax.grid(True)

            with chart_box:
                st.pyplot(fig)

        frame_idx += 1

cap.release()
