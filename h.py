import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from ultralytics import YOLO
from collections import deque
from datetime import datetime, timedelta
import plotly.graph_objects as go
import threading
import pygame  # Added pygame for the audio alert

# Initialize pygame mixer for audio
pygame.mixer.init()

# إعداد المودل
model = YOLO('yolov8n.pt')  # تأكد أن المسار صحيح

# تحميل أسماء الكلاسات
class_list = []
with open("COCO.txt", "r") as f:
    class_list = f.read().strip().split("\n")

# رابط الصوت
alert_url = "https://raw.githubusercontent.com/Hanan71/MansakAmin_modul/main/alert.mp3"
# Download and load the alert sound
alert_sound_file = "alert_sound.mp3"
import requests
with open(alert_sound_file, "wb") as f:
    f.write(requests.get(alert_url).content)
pygame.mixer.music.load(alert_sound_file)

# إعداد صفحة Streamlit
st.set_page_config(page_title="Mansak Amin", layout="wide", page_icon="🕋")
st.markdown("""
    <h1 style='text-align: center; color: #104E8B;'>🕋 Mansak Amin</h1>
    <h4 style='text-align: center; color: #1E90FF;'>Smart Crowd Management during Hajj and Umrah</h4>
""", unsafe_allow_html=True)

# اختيار المصدر
source = st.sidebar.radio("Select Video Source:", ["📁 Upload Video", "📷 Laptop Camera", "📷 External Camera"])
target_count = 60  # عدد الأشخاص الذي نعتبره خطراً
update_interval = 1  # كل دقيقة

# رفع صورة شخص مفقود
st.sidebar.markdown("---")
uploaded_image = st.sidebar.file_uploader("🔍 Upload image of missing person", type=["jpg", "png", "jpeg"])
if uploaded_image:
    st.sidebar.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

# عناصر الواجهة الرئيسية
with st.container():
    stats = st.columns(4)
    people_placeholder = stats[0].empty()
    status_placeholder = stats[1].empty()
    time_placeholder = stats[2].empty()
    accuracy_placeholder = stats[3].empty()

# حفظ بيانات الجلسة
if 'minute_data' not in st.session_state:
    st.session_state.minute_data = {
        'timestamps': [],
        'people_counts': [],
        'avg_accuracies': [],
        'start_time': datetime.now()
    }

# كلاس لتشغيل الكاميرا في ثريد
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
                self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
    def stop(self):
        self.running = False
        self.join()

# التراكينغ للأشخاص
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
        self.tracks = {tid: tr for tid, tr in self.tracks.items() if tid in [t[-1] for t in updated_tracks]}
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

# Function to calculate percentage change safely
def safe_percentage_change(current, previous):
    if previous == 0:
        return 0  # No change percentage if previous was 0
    return ((current - previous) / previous * 100)

# تحديث بيانات الرسم البياني
def update_minute_data(current_count, current_accuracy):
    now = datetime.now()
    elapsed = now - st.session_state.minute_data['start_time']
    if elapsed >= timedelta(minutes=update_interval):
        st.session_state.minute_data['timestamps'].append(now.strftime("%H:%M"))
        st.session_state.minute_data['people_counts'].append(current_count)
        st.session_state.minute_data['avg_accuracies'].append(current_accuracy)
        st.session_state.minute_data['start_time'] = now

# معالجة الفيديو أو الكاميرا
def process_video(video_path):
    stframe = st.empty()
    tracker = Tracker()
    counter = deque(maxlen=1000)
    line_position = 380
    offset = 6
    alert_played = False
    start_time = time.time()

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
        results = model(frame, verbose=False)
        detections = []
        confidences = []

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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if line_position - offset < cy < line_position + offset and obj_id not in counter:
                counter.append(obj_id)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        people_count = len(counter)
        elapsed_seconds = int(time.time() - start_time)
        update_minute_data(people_count, avg_accuracy)

        # Calculate percentage change safely
        previous_count = st.session_state.minute_data['people_counts'][-2] if len(st.session_state.minute_data['people_counts']) > 1 else people_count
        percent_change = safe_percentage_change(people_count, previous_count)

        # تحديث الواجهة
        people_placeholder.markdown(
            f"""
            <div style="background-color: #007BFF; color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                <h3>👥 Current Count</h3>
                <h2 style="display: inline-block; border-bottom: 4px solid #FFD700;">{people_count} ➤</h2>
                <p>📈 Change: {percent_change:.2f}%</p>
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

        # رسم الخط الأخضر
        cv2.line(frame, (0, line_position), (1020, line_position), (0, 255, 0), 2)
        
        # إضافة عداد الأشخاص ونسبة الدقة على الإطار بلون أصفر
        cv2.putText(frame, f"People Count: {people_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Accuracy: {avg_accuracy:.2%}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # إضافة حالة المكان على الإطار
        status_text = "Status: " + ("Overcrowded ⚠️" if people_count >= target_count else "Normal ✅")
        status_color = (0, 0, 255) if people_count >= target_count else (0, 255, 0)  # أحمر للازدحام، أخضر للوضع الطبيعي
        cv2.putText(frame, status_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # إضافة الوقت المنقضي على الإطار
        time_text = f"Time: {elapsed_seconds // 60:02}:{elapsed_seconds % 60:02}"
        cv2.putText(frame, time_text, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # تحذير صوتي لو ازدحام
        if people_count >= target_count and not alert_played:
            st.audio(alert_url, format='audio/mp3')
            alert_played = True
            # إضافة تحذير على الإطار عند الازدحام
            cv2.putText(frame, "⚠️ Warning: Overcrowding!", (300, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # إضافة مستطيل تحذير في الخلفية
            overlay = frame.copy()
            cv2.rectangle(overlay, (290, 5), (790, 50), (0, 0, 200), -1)
            alpha = 0.6  # معامل الشفافية
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.putText(frame, "⚠️ WARNING: OVERCROWDING! ⚠️", (300, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        elif people_count < target_count:
            alert_played = False
            
        # Play sound alert using pygame
        if people_count >= target_count:
            if not alert_played:
                pygame.mixer.music.play()
                alert_played = True
            cv2.putText(frame, "⚠️ Warning: Overcrowding!", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            alert_played = False

        # عرض الفريم
        stframe.image(frame, channels="BGR")

    if isinstance(video_path, int):
        cam_thread.stop()
    else:
        cap.release()


# تشغيل المصدر المختار
if source == "📁 Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.write(uploaded_file.read())
            temp_path = tmpfile.name
        process_video(temp_path)
        os.unlink(temp_path)

elif source == "📷 Laptop Camera":
    process_video(0)

elif source == "📷 External Camera":
    process_video(1)

# Clean up temporary files when app closes
try:
    os.remove(alert_sound_file)
except:
    pass
