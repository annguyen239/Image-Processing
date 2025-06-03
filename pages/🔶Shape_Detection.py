import streamlit as st
import time # Make sure time is imported if needed for effects
import os
import cv2 as cv
import numpy as np
from part import phan_nguong as pn

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
          
# Function to predict shape based on Hu moments
def shape_predict(imgin):
    # Check if the image is already grayscale
    if len(imgin.shape) == 3:
        gray = cv.cvtColor(imgin, cv.COLOR_BGR2GRAY)
    else:
        gray = imgin.copy()  # Already grayscale, so just copy it

    temp = pn.phan_nguong(gray)
    m = cv.moments(temp)
    hu = cv.HuMoments(m)
    if 0.000620 <= hu[0, 0] <= 0.000630:
        s = 'Hinh Tron'
    elif 0.000644 <= hu[0, 0] <= 0.000668:
        s = 'Hinh Vuong'
    elif 0.000725 <= hu[0, 0] <= 0.000747:
        s = 'Hinh Tam Giac'
    else:
        s = 'Unknown'
    
    # Convert grayscale back to BGR for visualization and put text on it
    imgout = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    cv.putText(imgout, s, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    return imgout

def shape_detection_page():
    st.title("💠:red[Shape Detection]")
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png", "bmp"])
    col_input, col_output = st.columns(2)

    with col_input:
        st.subheader("Input Image")
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            st.image(cv.cvtColor(image, cv.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
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
        result_img = shape_predict(image.copy())
        result_display = result_img  # Already in BGR format
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
    shape_detection_page()
    