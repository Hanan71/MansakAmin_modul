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
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import plotly.express as px
import plotly.graph_objects as go

# Load model and labels
model = YOLO('yolov5s.pt')
with open("COCO.txt", "r") as f:
    class_list = f.read().strip().split("\n")

# Initialize alert sound
mixer.init()
mixer.music.load("alert.mp3")

# Dashboard state
st.set_page_config(page_title="Safe Manasik", layout="wide", page_icon="🕋")
st.markdown("""
    <h1 style='text-align: center; color: #104E8B;'>🕋 Safe Manasik</h1>
    <h4 style='text-align: center; color: #1E90FF;'>Smart system for crowd management during Hajj and Umrah seasons</h4>
""", unsafe_allow_html=True)

# Sidebar for source selection
source = st.sidebar.radio("Select Video Source:", ["📁 Upload Video", "📷 Laptop Camera", "📷 External Camera"])
target_count = st.sidebar.slider("🚨 Crowd Threshold", 20, 200, 60, 5)

# Upload image for lost person
st.sidebar.markdown("---")
uploaded_image = st.sidebar.file_uploader("🔍 Upload image to search for lost person", type=["jpg", "png", "jpeg"])
if uploaded_image:
    lost_person = Image.open(uploaded_image).convert("RGB")
    st.sidebar.image(lost_person, caption="Uploaded Image", use_container_width=True)
else:
    lost_person = None

# Placeholders for live dashboard values
col1, col2, col3, col4 = st.columns(4)
current_count_box = col1.empty()
crowd_threshold_box = col2.empty()
alerts_box = col3.empty()
tracked_box = col4.empty()

current_count_box.metric("Current Count", 0)
crowd_threshold_box.metric("Crowd Threshold", target_count)
alerts_box.metric("Alerts Triggered", 0)
tracked_box.metric("People Tracked", 0)

# Chart + Radar layout
chart_placeholder = st.container()
radar_placeholder = st.container()

# Tracker class
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

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.tracker = Tracker()
        self.counter = deque(maxlen=1000)
        self.line_position = 380
        self.offset = 6
        self.alert_played = False
        self.alerts_triggered = 0
        self.frame_count = 0
        self.graph_data = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame = cv2.resize(img, (1020, 500))
        results = model.predict(frame, verbose=False)
        detections = []

        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                if class_list[cls] == "person":
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    detections.append([int(x1), int(y1), int(x2), int(y2)])
                    if lost_person is not None:
                        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        crop_pil = crop_pil.resize(lost_person.size)
                        diff = np.mean(np.abs(np.array(crop_pil) - np.array(lost_person)))
                        if diff < 25:
                            cv2.putText(frame, "🔎 Match Found!", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        tracked_objects = self.tracker.update(detections)
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if self.line_position - self.offset < cy < self.line_position + self.offset and obj_id not in self.counter:
                self.counter.append(obj_id)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        people_count = len(self.counter)
        total_tracked = len(set(self.counter))

        cv2.line(frame, (0, self.line_position), (1020, self.line_position), (0, 255, 0), 2)
        cv2.putText(frame, f"People Count: {people_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if people_count >= target_count and not self.alert_played:
            mixer.music.play()
            self.alert_played = True
            self.alerts_triggered += 1
            cv2.putText(frame, "⚠️ Warning: Overcrowding!", (300, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Update live metrics
        current_count_box.metric("Current Count", people_count)
        alerts_box.metric("Alerts Triggered", self.alerts_triggered)
        tracked_box.metric("People Tracked", total_tracked)

        # Update chart
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            self.graph_data.append({"Frame": self.frame_count, "People": people_count})
            graph_df = self.graph_data[-30:]
            fig = px.line(graph_df, x="Frame", y="People", title="📈 Live Crowd Trend", markers=True)
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            # Radar Chart (dynamic)
            radar_fig = go.Figure()
            radar_fig.add_trace(go.Scatterpolar(
                r=[people_count, total_tracked, self.alerts_triggered, target_count],
                theta=["Current", "Tracked", "Alerts", "Threshold"],
                fill='toself',
                name='Live Metrics'
            ))
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max(200, people_count + 10)])),
                showlegend=False,
                height=300,
                title="📊 Radar View of Key Stats"
            )
            radar_placeholder.plotly_chart(radar_fig, use_container_width=True)

        return frame

# Source handling
if source == "📁 Upload Video":
    st.warning("Video upload is not supported in this mode with dynamic chart.")
else:
    device_index = 0 if source == "📷 Laptop Camera" else 1
    webrtc_streamer(
        key="camera",
        video_processor_factory=VideoTransformer,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": {"deviceId": {"exact": device_index}}}
    )


import cv2
print("OpenCV version:", cv2.__version__)
