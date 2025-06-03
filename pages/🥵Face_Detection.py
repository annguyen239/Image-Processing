import streamlit as st
import cv2 as cv
import numpy as np
import tempfile
import joblib
import time

# Hide menu items in sidebar
hide_menu_items = """
    <style>
    /* Hide the list of pages / radio buttons in the sidebar */
    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
"""
st.markdown(hide_menu_items, unsafe_allow_html=True)

# Sidebar navigation
col1,col2 = st.columns(2)
with col1:
    st.page_link("Main.py", label = "Back",icon = ":material/arrow_back:",use_container_width=True)
with col2:
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
    with c1:
        if st.button("🥵Face Detection"):
            st.switch_page("pages/🥵Face_Detection.py")
    with c2:
        if st.button("🔬Object Detection"):
            st.switch_page("pages/🔬Object_Detection.py")
    with c3:
        if st.button("🔶Shape Detection"):
            st.switch_page("pages/🔶Shape_Detection.py")
    with c4:
        if st.button("📸Image Processing"):
            st.switch_page("pages/📸Image_Processing.py")
    with c5:
        if st.button("👾Xtra"):
            st.switch_page("pages/👾Xtra.py")

image = st.sidebar.image("images/sidebar_image/hcmute.png")
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["Home","About", "Contact"], key="main_nav") # Added key   

# Load models
face_detector = cv.FaceDetectorYN.create("model/face_detection_yunet_2023mar.onnx", "", (320, 320), 0.9, 0.3, 5000)                 
face_recognizer = cv.FaceRecognizerSF.create("model/face_recognition_sface_2021dec.onnx", "")
svc = joblib.load("model/svc.pkl")
mydict = ['DuyTruong', 'NgocAn', 'TanDuong','VanTien','VanVan']
colors = {
    'AnVuong': (0, 255, 0),
    'DuyTruong': (0, 0, 255),
    'TanDuong': (255, 0, 0),
    'VanTien': (255, 255, 0),
    'VanVan': (255, 0, 255),
}

# Helper function
def recognize_faces(frame):
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    face_detector.setInputSize((w, h))
    faces = face_detector.detect(frame)
    if faces[1] is not None:
        for face in faces[1]:
            aligned_face = face_recognizer.alignCrop(frame, face)
            feature = face_recognizer.feature(aligned_face)
            prediction = svc.predict(feature)
            name = mydict[prediction[0]]
            color = colors.get(name, (255, 255, 255))
            coords = face[:4].astype(np.int32)
            cv.rectangle(frame_copy, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), color, 2)
            cv.putText(frame_copy, name, (coords[0], coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame_copy

# Streamlit UI
st.title("🤬:blue[Face Detection]")
option = st.selectbox("Choose Mode", ["Detect With Webcam", "Detect Video"])

if option == "Detect With Webcam":
    run = st.button("Detect")
    
    FRAME_WINDOW = st.image([])
    
    cap = cv.VideoCapture(0)
    if run:
        progress_text = st.empty()
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.02)
            my_bar.progress(percent_complete + 1)
            progress_text.text(f"Loading... {percent_complete}%")
        time.sleep(0.5)
        my_bar.empty()
        progress_text.success("Predict Completed!")
        time.sleep(1)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Don't Run Webcam")
            break
        
        result = recognize_faces(frame)
        FRAME_WINDOW.image(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    
    cap.release()

elif option == "Detect Video":
    uploaded_file = st.file_uploader("Choose video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        if st.button("Detect"):
            cap = cv.VideoCapture(video_path)
            progress_text = st.empty()
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.02)
                my_bar.progress(percent_complete + 1)
                progress_text.text(f"Loading... {percent_complete}%")
            time.sleep(0.5)
            my_bar.empty()
            progress_text.success("Predict Completed!")
            time.sleep(1)
            FRAME_WINDOW = st.image([])
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                result = recognize_faces(frame)
                FRAME_WINDOW.image(cv.cvtColor(result, cv.COLOR_BGR2RGB))
            cap.release()