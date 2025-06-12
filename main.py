import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import time

# --- Setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Data
left_hand_data, right_hand_data = [], []
shoulder_mid_data, head_angle_data = [], []
engagement_levels, frame_count = [], []

# --- Functions ---
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
        return 2  # Inspired
    elif motion > 0.01:
        return 1  # Interactive
    else:
        return 0  # Attentive

def get_engagement_label(level):
    return ['Attentive', 'Interactive', 'Inspired'][level]

def get_engagement_color(level):
    return [(255, 255, 0), (0, 165, 255), (0, 0, 255)][level]  # yellow, orange, red

# --- Matplotlib setup ---
plt.ion()
fig, axs = plt.subplots(4, 1, figsize=(10, 8))
titles = ['Hand Movements', 'Shoulder Midpoint', 'Head Turn Angle', 'Engagement Level']

# --- Main Loop ---
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    label = "No Pose"
    color = (0, 0, 255)

    if result.pose_landmarks:
        drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark
        l, r, s, h = calculate_movement(landmarks)
        if None not in [l, r, s, h]:
            left_hand_data.append(l)
            right_hand_data.append(r)
            shoulder_mid_data.append(s)
            head_angle_data.append(h)
            engagement = classify_engagement(l, r, s, h)
            engagement_levels.append(engagement)
            frame_count.append(frame_idx)
            label = get_engagement_label(engagement)
            color = get_engagement_color(engagement)

    # Display engagement label on frame
    cv2.putText(frame, f"Engagement: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Live Feed", frame)

    # Plot updates every 10 frames for performance
    if frame_idx % 10 == 0 and len(frame_count) > 5:
        for ax in axs:
            ax.clear()

        axs[0].plot(frame_count, left_hand_data, label="Left", color="blue")
        axs[0].plot(frame_count, right_hand_data, label="Right", color="red")
        axs[0].legend()
        axs[0].set_title("Hand Movements")

        axs[1].plot(frame_count, shoulder_mid_data, color="orange")
        axs[1].set_title("Shoulder Midpoint")

        axs[2].plot(frame_count, head_angle_data, color="purple")
        axs[2].set_title("Head Turn Angle")

        axs[3].step(frame_count, engagement_levels, where='mid', color="green")
        axs[3].set_yticks([0, 1, 2])
        axs[3].set_yticklabels(['Attentive', 'Interactive', 'Inspired'])
        axs[3].set_title("Engagement Level")

        plt.tight_layout()
        plt.pause(0.01)

    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
