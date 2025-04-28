import cv2
import torch
import numpy as np
import streamlit as st
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime
from collections import deque
import tempfile
from threading import Thread

# تحميل نموذج YOLOv5
model_path = "crowdhajjbest.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# إعداد واجهة Streamlit
st.set_page_config(page_title="Crowd Detection", layout="wide")
st.title("Crowd Detection using YOLOv5")

# مساحة للعرض
stframe = st.empty()
graph_placeholder = st.empty()

# متغيرات لإنذار الازدحام
alert_played = False
alert_url = "https://upload.wikimedia.org/wikipedia/commons/4/4e/Beep-09.mp3"

# منطقة خط العد
line_position = 400

# إعدادات الهدف
st.sidebar.header("Settings")
target_count = st.sidebar.slider("Target People Count", 10, 100, 50)
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5)

# تكوين بيانات للمتابعة
if 'minute_data' not in st.session_state:
    st.session_state.minute_data = {'timestamps': deque(maxlen=30), 'people_counts': deque(maxlen=30)}

# كلاس لمعالجة الكاميرا
class CameraCapture:
    def __init__(self):
        self.running = True
        self.capture = None
        self.frame = None
        self.thread = None

    def start(self, src):
        self.capture = cv2.VideoCapture(src)
        self.thread = Thread(target=self.update, args=())
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.capture.release()

# دالة لمعالجة الفيديو
def process_video(video_path):
    global alert_played
    people_count = 0
    last_people_count = 0
    cam_thread = None

    if isinstance(video_path, int):
        cam_thread = CameraCapture()
        cam_thread.start(video_path)
    else:
        cap = cv2.VideoCapture(video_path)

    while True:
        frame = cam_thread.read() if isinstance(video_path, int) else cap.read()[1]
        if frame is None:
            break

        frame = cv2.resize(frame, (1020, 500))
        results = model(frame)
        detections = results.xyxy[0]

        current_count = 0
        confidences = []

        for *box, conf, cls in detections:
            if conf > confidence_threshold:
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if cy < line_position + 10 and cy > line_position - 10:
                    current_count += 1

                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                confidences.append(float(conf))

        if confidences:
            avg_accuracy = sum(confidences) / len(confidences)
        else:
            avg_accuracy = 0.0

        if current_count != last_people_count:
            st.session_state.minute_data['timestamps'].append(datetime.now().strftime('%H:%M:%S'))
            st.session_state.minute_data['people_counts'].append(current_count)

        # رسم خط ونص على الفيديو
        cv2.line(frame, (0, line_position), (1020, line_position), (0, 255, 0), 2)
        cv2.putText(frame, f"People Count: {current_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Accuracy: {avg_accuracy:.2%}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if current_count >= target_count:
            if not alert_played:
                st.audio(alert_url, format='audio/mp3')
                alert_played = True
            cv2.putText(frame, "\u26a0\ufe0f Warning: Overcrowding!", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
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
                y=[target_count] * len(st.session_state.minute_data['timestamps']),
                mode='lines',
                name='Threshold',
                line=dict(color='red', dash='dash')
            ))
            fig_crowd.update_layout(
                title="Crowd Trend (Updated every minute)",
                xaxis_title="Time",
                yaxis_title="People Count"
            )
            graph_placeholder.plotly_chart(fig_crowd, use_container_width=True)

        last_people_count = current_count

    if isinstance(video_path, int):
        cam_thread.stop()
    else:
        cap.release()

# اختيار مصدر الفيديو
source = st.sidebar.radio("Select Source", ('Webcam', 'Video File'))

if source == 'Webcam':
    if st.button('Start Webcam'):
        process_video(0)
else:
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        process_video(tfile.name)

