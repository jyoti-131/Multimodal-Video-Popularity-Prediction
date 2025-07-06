# 🎥 Multimodal Engagement Meter

> 💡 Real-Time Engagement Detection using Pose Estimation & Motion Analysis

![engagement](https://img.shields.io/badge/AI-ComputerVision-blueviolet?style=flat-square)
![built-with](https://img.shields.io/badge/Built%20With-Streamlit%20%7C%20OpenCV%20%7C%20MediaPipe-lightgrey?style=flat-square)

---

## 🚀 Overview

**Multimodal Engagement Meter** is a real-time system that tracks user body language via webcam 📸 to analyze **engagement levels** during video content or live sessions. It combines **MediaPipe Pose Detection**, **OpenCV**, and **matplotlib** to predict whether a user is:

* 🟦 **Attentive**
* 🟧 **Interactive**
* 🟥 **Inspired**

Great for educational tech, streaming analytics, productivity tracking, and video popularity research!

---

## 🔍 Key Features

✨ **Real-Time Pose Detection**
✨ **Engagement Classification via Motion Variance**
✨ **Streamlit Interface for Intuitive Display**
✨ **Multimodal Visualization: Hands, Head Angle, Shoulders**
✨ **Auto-Updating Graphs of Movement + Engagement Trends**

---

## 🧠 Tech Stack

| Technology   | Purpose                                 |
| ------------ | --------------------------------------- |
| `MediaPipe`  | Pose & landmark detection 🧍            |
| `OpenCV`     | Webcam input + frame rendering 🎥       |
| `Streamlit`  | Interactive UI & real-time dashboard 🌐 |
| `Matplotlib` | Visual graphs for motion tracking 📊    |
| `NumPy`      | Vector & motion calculation 📐          |

---

## 🛠️ How it Works

📌 **Landmark Tracking**

* Extracts key points: 👋 hands, 💪 shoulders, 🧍‍♂️ head tilt

📌 **Movement Analysis**

* Calculates standard deviation of tracked points

📌 **Engagement Classification**

* 📉 Low motion → Attentive
* ⚖️ Medium motion → Interactive
* 📈 High motion → Inspired

📌 **Visualization**

* Plots live graphs of motion over time with color-coded engagement 📊

---

## 📸 Preview

| Streamlit UI              | Engagement Plots           |
| ------------------------- | -------------------------- |
| 🧑‍💻 Live Pose Detection | 📈 Graphs of Movements     |
| 🧠 Engagement Prediction  | 🟦🟧🟥 Based Trend Display |

---

## 🎯 Use Cases

* 🧑‍🏫 **Online Learning Platforms**: Track student attention
* 📺 **Content Creators**: Measure viewer reactions
* 👩‍💻 **Interview Simulators**: Feedback on articulation
* 🎮 **Game Testing**: Record excitement via motion


