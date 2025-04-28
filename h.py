import streamlit as st
import cv2
from PIL import Image
import tempfile
from ultralytics import YOLO
from collections import deque
import numpy as np
import time
import os
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import threading

# Load YOLO model and class list
model = YOLO('yolov5s.pt')
with open("COCO.txt", "r") as f:
    class_list = f.read().strip().split("\n")

# Alert sound URL
alert_url = "https://raw.githubusercontent.com/Hanan71/MansakAmin_modul/main/alert.mp3"

# Streamlit page setup
st.set_page_config(page_title="Mansak Amin", layout="wide", page_icon="🕋")
st.markdown("""
    <h1 style='text-align: center; color: #104E8B;'>🕋 Mansak Amin</h1>
    <h4 style='text-align: center; color: #1E90FF;'>Smart crowd management during Hajj and Umrah</h4>
""", unsafe_allow_html=True)

# Sidebar controls
source = st.sidebar.radio("Select Video Source:", ["📁 Upload Video", "📷 Laptop Camera", "📷 External Camera"])
target_count = 60
update_interval = 1
uploaded_image = st.sidebar.file_uploader("🔍 Upload image of missing person", type=["jpg", "png", "jpeg"])
if uploaded_image:
    st.sidebar.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

# Dashboard placeholders
with st.container():
    stats = st.columns(4)
    people_placeholder = stats[0].empty()
    status_placeholder = stats[1].empty()
    time_placeholder = stats[2].empty()
    accuracy_placeholder = stats[3].empty()

# Session state for graphs
if 'minute_data' not in st.session_state:
    st.session_state.minute_data = {'timestamps': [], 'people_counts': [], 'avg_accuracies': [], 'start_time': datetime.now()}

# Camera reading thread
class CameraThread(threading.Thread):
    def __init__(self, src=0):
        super().__init__()
        self.src = src
        self.frame = None
        self.running = False

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            st.error("Failed to open camera!")
            self.running = False
            return
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def stop(self):
        self.running = False
        self.join()

# Simple tracker
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

# Update graphs
def update_minute_data(current_count, current_accuracy):
    now = datetime.now()
    if (now - st.session_state.minute_data['start_time']) >= timedelta(minutes=update_interval):
        st.session_state.minute_data['timestamps'].append(now.strftime("%H:%M"))
        st.session_state.minute_data['people_counts'].append(current_count)
        st.session_state.minute_data['avg_accuracies'].append(current_accuracy)
        st.session_state.minute_data['start_time'] = now

# Main video processing function
def process_video(video_path):
    stframe = st.empty()
    tracker = Tracker()
    counter = deque(maxlen=1000)
    alert_played = False
    start_time = time.time()
    last_people_count = 0
    line_position = 380

    if isinstance(video_path, int):
        cam_thread = CameraThread(video_path)
        cam_thread.start()
        time.sleep(2)
    else:
        cap = cv2.VideoCapture(video_path)

    while True:
        if isinstance(video_path, int):
            if cam_thread.frame is None:
                continue
            frame = cam_thread.frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                break

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame, verbose=False)

        detections, confidences = [], []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            if class_list[int(cls)] == "person":
                detections.append([int(x1), int(y1), int(x2), int(y2)])
                confidences.append(conf)

        avg_accuracy = np.mean(confidences) if confidences else 0
        tracked_objects = tracker.update(detections)

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if line_position - 6 < cy < line_position + 6 and obj_id not in counter:
                counter.append(obj_id)

        people_count = len(counter)
        elapsed_seconds = int(time.time() - start_time)
        update_minute_data(people_count, avg_accuracy)

        # Dashboard Updates
        people_placeholder.metric("👥 Current Count", people_count)
        time_placeholder.metric("⏱️ Time Elapsed", f"{elapsed_seconds//60:02}:{elapsed_seconds%60:02}")
        status_text = "Overcrowded ⚠️" if people_count >= target_count else "Normal ✅"
        status_placeholder.metric("📊 Crowd Status", status_text)
        accuracy_placeholder.metric("🎯 Detection Accuracy", f"{avg_accuracy:.2%}")

        # Audio Alert
        if people_count >= target_count and not alert_played:
            st.audio(alert_url, format='audio/mp3')
            alert_played = True
        elif people_count < target_count:
            alert_played = False

        # Draw Line
        cv2.line(frame, (0, line_position), (1020, line_position), (0, 255, 0), 2)
        stframe.image(frame, channels="BGR")

    if isinstance(video_path, int):
        cam_thread.stop()
    else:
        cap.release()

# Graph plotting
if len(st.session_state.minute_data['timestamps']) > 0:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state.minute_data['timestamps'], y=st.session_state.minute_data['people_counts'], mode='lines+markers', name='People Count'))
    fig.add_trace(go.Scatter(x=st.session_state.minute_data['timestamps'], y=[target_count]*len(st.session_state.minute_data['timestamps']), mode='lines', name='Threshold', line=dict(dash='dash', color='red')))
    fig.update_layout(title="Crowd Trend", xaxis_title="Time", yaxis_title="People Count")
    st.plotly_chart(fig)

# Source selector
if source == "📁 Upload Video":
    uploaded_file = st.file_uploader("Select a video file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            temp_video_path = tfile.name
        process_video(temp_video_path)
        os.unlink(temp_video_path)

elif source == "📷 Laptop Camera":
    process_video(0)

elif source == "📷 External Camera":
    process_video(1)

