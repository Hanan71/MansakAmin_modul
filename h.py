import cv2
import streamlit as st
import numpy as np
from PIL import Image
import time

# تحميل النموذج المدرب لـ YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# فتح الكاميرا
cap = cv2.VideoCapture(0)

# تحقق إذا كانت الكاميرا تعمل
if not cap.isOpened():
    st.error("Could not open video device")
else:
    st.title("Live Video Feed with YOLO Detection")
    
    # Streamlit loop for continuous video processing
    while True:
        ret, frame = cap.read()  # قراءة الإطار من الكاميرا
        if not ret:
            break
        
        # تحويل الإطار إلى RGB
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        # تحليل النتائج التي يتم إرجاعها من YOLO
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # الحد الأدنى للثقة (يمكنك تغييره حسب احتياجك)
                    # استخراج الإحداثيات الخاصة بالـ bounding box
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    # رسم المستطيل على الشخص المكتشف
                    cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2), 
                                  (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)
                    cv2.putText(frame, str(class_id), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # عرض الإطار باستخدام Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # التأخير قليلاً لضمان سلاسة العرض
        time.sleep(0.1)

    cap.release()  # إغلاق الكاميرا بعد الانتهاء
