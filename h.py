import streamlit as st
import cv2
from PIL import Image
import tempfile
from ultralytics import YOLO
from collections import deque
import numpy as np
import time
from datetime import timedelta
import os
import streamlit as st
import requests
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import plotly.express as px
import plotly.graph_objects as go
import threading
from datetime import datetime, timedelta


model = YOLO('yolov5s.pt')
with open("COCO.txt", "r") as f:
    class_list = f.read().strip().split("\n")
    

st.set_page_config(page_title="Mansak Amin", layout="wide", page_icon="🕋")
st.markdown("""
    <h1 style='text-align: center; color: #104E8B;'>🕋 Mansak Amin</h1>
    <h4 style='text-align: center; color: #1E90FF;'>Smart crowd management during Hajj and Umrah</h4>
""", unsafe_allow_html=True)

source = st.sidebar.radio("Select Video Source:", ["📁 Upload Video", "📷 Laptop Camera", "📷 External Camera"])
target_count = 60
update_interval = 1

st.sidebar.markdown("---")
uploaded_image = st.sidebar.file_uploader("🔍 Upload image of missing person", type=["jpg", "png", "jpeg"])
if uploaded_image:
    st.sidebar.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

with st.container():
    stats = st.columns(4)
    people_placeholder = stats[0].empty()
    status_placeholder = stats[1].empty()
    time_placeholder = stats[2].empty()
    accuracy_placeholder = stats[3].empty()

if 'minute_data' not in st.session_state:
    st.session_state.minute_data = {
        'timestamps': [],
        'people_counts': [],
        'avg_accuracies': [],
        'start_time': datetime.now()
    }

class CameraThread(threading.Thread):
    def __init__(self, src=0):
        super().__init__()
        self.src = src
        self.frame = None
        self.running = False
        self.backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
        
    def run(self):
        self.running = True
        cap = None
        for backend in self.backends:
            cap = cv2.VideoCapture(self.src, backend)
            if cap.isOpened():
                break
                
        if not cap or not cap.isOpened():
            st.error("Failed to open camera!")
            self.running = False
            return
            
        while self.running:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if np.mean(frame[:,:,1]) > np.mean(frame[:,:,0]) + 20:
                    frame[:,:,1] = cv2.multiply(frame[:,:,1], 0.7)
                    frame[:,:,0] = cv2.multiply(frame[:,:,0], 1.1)
                    frame[:,:,2] = cv2.multiply(frame[:,:,2], 1.1)
                self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
    def stop(self):
        self.running = False
        self.join()

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

def update_minute_data(current_count, current_accuracy):
    now = datetime.now()
    elapsed = now - st.session_state.minute_data['start_time']
    
    if elapsed >= timedelta(minutes=update_interval):
        st.session_state.minute_data['timestamps'].append(now.strftime("%H:%M"))
        st.session_state.minute_data['people_counts'].append(current_count)
        st.session_state.minute_data['avg_accuracies'].append(current_accuracy)
        st.session_state.minute_data['start_time'] = now

def process_video(video_path):
    stframe = st.empty()
    graph_placeholder = st.empty()
    tracker = Tracker()
    counter = deque(maxlen=1000)
    line_position = 380
    offset = 6
    alert_played = False
    start_time = time.time()
    last_people_count = 0

    if isinstance(video_path, int):
        cam_thread = CameraThread(video_path)
        cam_thread.start()
        time.sleep(2)
    else:
        cap = cv2.VideoCapture(video_path)

    color_correction = st.sidebar.checkbox("Enable Advanced Color Correction", value=True)

    while True:
        if isinstance(video_path, int):
            if cam_thread.frame is None:
                continue
            frame = cam_thread.frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                break

        if color_correction and isinstance(video_path, int):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame[:,:,1] = cv2.multiply(frame[:,:,1], 0.8)
            frame[:,:,0] = cv2.multiply(frame[:,:,0], 1.1)
            frame[:,:,2] = cv2.multiply(frame[:,:,2], 1.1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame, verbose=False)
        detections = []
        confidences = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            if class_list[int(cls)] == "person":
                detections.append([int(x1), int(y1), int(x2), int(y2)])
                confidences.append(conf)

        avg_accuracy = np.mean(confidences) if len(confidences) > 0 else 0
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
        elapsed_seconds = int(time.time() - start_time)
        update_minute_data(people_count, avg_accuracy)

        percentage_change = 0
        if last_people_count > 0:
            percentage_change = ((people_count - last_people_count) / last_people_count) * 100

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
                <h2>{elapsed_seconds // 60:02}:{elapsed_seconds % 60:02}</h2>
            </div>
            """, unsafe_allow_html=True
        )

        status = "Overcrowded ⚠️" if people_count >= target_count else "Normal ✅"
        status_color = "#FFC107" if status == "Normal ✅" else "#DC3545"
        status_placeholder.markdown(
            f"""
            <div style="background-color: {status_color}; color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                <h3>📊 Crowd Status</h3>
                <h2>{status}</h2>
            </div>
            """, unsafe_allow_html=True
        )

        accuracy_placeholder.markdown(
            f"""
            <div style="background-color: #6F42C1; color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                <h3>🎯 Detection Accuracy</h3>
                <h2>{avg_accuracy:.2%}</h2>
                <p>Based on {len(confidences)} detections</p>
            </div>
            """, unsafe_allow_html=True
        )
        
        st.markdown(
    f"""
    <audio autoplay>
        <source src="{alert_url}" type="audio/mp3">
    </audio>
    """,
    unsafe_allow_html=True
)

      

        cv2.line(frame, (0, line_position), (1020, line_position), (0, 255, 0), 2)
        cv2.putText(frame, f"People Count: {people_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Accuracy: {avg_accuracy:.2%}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        


alert_url = "https://raw.githubusercontent.com/Hanan71/MansakAmin_modul/main/alert.mp3"
alert_played = False

if people_count >= target_count:
    if not alert_played:
        st.audio(alert_url, format='audio/mp3')  # Play sound via Streamlit
        alert_played = True
    cv2.putText(frame, "⚠️ Warning: Overcrowding!", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
else:
    alert_played = False

stframe.image(frame, channels="BGR")

        if len(st.session_state.minute_data['timestamps']) > 0:
            fig_crowd = go.Figure()
            fig_crowd.add_trace(go.Scatter(
                x=st.session_state.minute_data['timestamps'],
                y=st.session_state.minute_data['people_counts'],
                mode='lines+markers',
                name='People Count',
                line=dict(color='blue')
            ))
            fig_crowd.add_trace(go.Scatter(
                x=st.session_state.minute_data['timestamps'],
                y=[target_count]*len(st.session_state.minute_data['timestamps']),
                mode='lines',
                name='Threshold',
                line=dict(color='red', dash='dash')
            ))
            fig_crowd.update_layout(
                title="Crowd Trend (Updated every minute)",
                xaxis_title="Time",
                yaxis_title="People Count"
            )
            

        last_people_count = people_count

    if isinstance(video_path, int):
        cam_thread.stop()
    else:
        cap.release()

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
