import streamlit as st
import cv2
from PIL import Image
import tempfile
from ultralytics import YOLO
from collections import deque
import numpy as np
from pygame import mixer
import time
import os
import plotly.graph_objects as go

# Load YOLO model and class labels
model = YOLO('yolov5s.pt')
with open("coco.txt", "r") as f:
    class_list = f.read().strip().split("\n")

# Initialize alert sound
mixer.init()
mixer.music.load("alert.mp3")

# Streamlit dashboard config
st.set_page_config(page_title="Safe Manasik", layout="wide", page_icon="🕋")
st.markdown("""
    <h1 style='text-align: center; color: #104E8B;'>🕋 Safe Manasik</h1>
    <h4 style='text-align: center; color: #1E90FF;'>Smart crowd management during Hajj and Umrah</h4>
""", unsafe_allow_html=True)

# Sidebar: video source selector
source = st.sidebar.radio("Select Video Source:", ["📁 Upload Video", "📷 Laptop Camera", "📷 External Camera"])
target_count = 60

# Sidebar: lost person image upload
st.sidebar.markdown("---")
uploaded_image = st.sidebar.file_uploader("🔍 Upload image of missing person", type=["jpg", "png", "jpeg"])
if uploaded_image:
    st.sidebar.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

# Cards row layout
with st.container():
    stats = st.columns(3)
    people_placeholder = stats[0].empty()
    status_placeholder = stats[1].empty()
    time_placeholder = stats[2].empty()

# Simple Tracker
class Tracker:
    def __init__(self):
        self.id_count = 0
        self.tracks = {}

    def update(self, detections):
        updated_tracks = []
        for det in detections:
            matched = False
            for track_id, track in self.tracks.items():
                if self._iou(det, track) > 0.3:
                    updated_tracks.append([*det, track_id])
                    self.tracks[track_id] = det
                    matched = True
                    break
            if not matched:
                self.id_count += 1
                self.tracks[self.id_count] = det
                updated_tracks.append([*det, self.id_count])
        self.tracks = {track_id: track for track_id, track in self.tracks.items() if track_id in [t[-1] for t in updated_tracks]}
        return updated_tracks

    def _iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        return inter_area / (box1_area + box2_area - inter_area + 1e-5)

# Video processing function
def process_video(video_path):
    stframe = st.empty()
    graph_placeholder = st.empty()
    tracker = Tracker()
    counter = deque(maxlen=1000)
    line_position = 380
    offset = 6
    alert_played = False
    people_history = []
    start_time = time.time()
    last_people_count = 0

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame, verbose=False)
        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            if class_list[int(cls)] == "person":
                detections.append([int(x1), int(y1), int(x2), int(y2)])

        tracked_objects = tracker.update(detections)

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if line_position - offset < cy < line_position + offset and obj_id not in counter:
                counter.append(obj_id)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        people_count = len(counter)
        elapsed = int(time.time() - start_time)

        # Calculate percentage change
        percentage_change = 0
        if last_people_count > 0:
            percentage_change = ((people_count - last_people_count) / last_people_count) * 100

        # Update cards with dynamic styling and layout
        people_placeholder.markdown(
            f"""
            <div style="background-color: #007BFF; color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                <h3>👥 Current Count</h3>
                <h2>{people_count}</h2>
                <p>📈 Change: {percentage_change:.2f}%</p>
            </div>
            """, unsafe_allow_html=True
        )

        time_placeholder.markdown(
            f"""
            <div style="background-color: #28A745; color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                <h3>⏱️ Time Elapsed</h3>
                <h2>{elapsed // 60:02}:{elapsed % 60:02}</h2>
            </div>
            """, unsafe_allow_html=True
        )

        status = "Overcrowded ⚠️" if people_count >= target_count else "Normal ✅"
        status_placeholder.markdown(
            f"""
            <div style="background-color: #FFC107; color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                <h3>📊 Crowd Status</h3>
                <h2>{status}</h2>
            </div>
            """, unsafe_allow_html=True
        )

        # Draw line and people count on video
        cv2.line(frame, (0, line_position), (1020, line_position), (0, 255, 0), 2)
        cv2.putText(frame, f"People Count: {people_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Play alert and warning if threshold is reached
        if people_count >= target_count:
            if not alert_played:
                mixer.music.play()
                alert_played = True
            cv2.putText(frame, "⚠️ Warning: Overcrowding!", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Display frame
        stframe.image(frame, channels="BGR")

        # Append and plot crowd graph
        people_history.append(people_count)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=people_history, mode='lines+markers', name='People Count'))
        fig.update_layout(title="Crowd Trend", xaxis_title="Frame", yaxis_title="Count")
        graph_placeholder.plotly_chart(fig, use_container_width=True, key=f"plot_{len(people_history)}")

        # Update the last people count
        last_people_count = people_count

    cap.release()

# Handle video input source
if source == "📁 Upload Video":
    uploaded_file = st.file_uploader("Select a video file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        process_video(tfile.name)

elif source == "📷 Laptop Camera":
    process_video(0)

elif source == "📷 External Camera":
    process_video(1)
