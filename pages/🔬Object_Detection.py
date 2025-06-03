import streamlit as st
import cv2
import numpy as np
import os
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

# For PT detection using YOLOv8 (ultralytics)
# Check if ultralytics package is installed
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

# Function to run object detection using YOLOv8
def run_detection(image, model_path, conf_threshold=0.5):
    if not HAS_ULTRALYTICS:
        st.error("Ultralytics package not installed. Please run 'pip install ultralytics'")
        return image
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found!")
        return image
    model = YOLO(model_path)
    results = model.predict(image, conf=conf_threshold,device='cuda')
    annotated_image = results[0].plot()  # Returns an image (BGR format)
    return annotated_image
def object_detection_page():
    st.title("⏲:green[Object Detection]")
    object = st.selectbox('Select type of object',['Multiple objects', 'Fruits'])
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
    col_input, col_output = st.columns(2)
    with col_input:
        st.subheader("Input Image")
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
        else:
            image = None
    predict_clicked = st.button("Predict")
    with col_output:
        st.subheader("Detection Result")
        result_placeholder = st.empty()

    if image is not None and predict_clicked:
        # Simulated progress bar with custom styling
        progress_text = st.empty()
        my_bar = st.progress(0)
        st.markdown(
            """
            <style>
                .stProgress > div > div > div > div {
                    background-color: cyan;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        if object == "Multiple objects":
            result_img = run_detection(image.copy(), "model/yolo11n.pt", conf_threshold=0.5)
            result_display = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        else:
            #result_img = run_detection(image.copy(), "model/yolov8n.pt", conf_threshold=0.5)
            result_img = run_detection(image.copy(), "model/yolov8n_traicay.pt", conf_threshold=0.5)
            result_display = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        for percent_complete in range(100):
            time.sleep(0.02)
            my_bar.progress(percent_complete + 1)
            progress_text.text(f"Loading... {percent_complete}%")
        time.sleep(0.5)
        my_bar.empty()
        progress_text.success("Predict Completed!")
        time.sleep(1)
        result_placeholder.image(result_display, caption="Detection Result", use_container_width=True)
        time.sleep(1)
        progress_text.empty()
    if image is None and predict_clicked:
        st.write("No image to predict !!!")

# Main function to run the app
if __name__ == '__main__':
    object_detection_page()
