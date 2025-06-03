import streamlit as st
import cv2 as cv
import numpy as np
import time 
from part import chapter3 as ch3
from part import chapter4 as ch4
from part import chapter5 as ch5
from part import chapter9 as ch9

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


st.title("📷:orange[Image Processing]")
         
# Function to process and display the image
def image_processing_page():
    col1,col2 = st.columns([1,2])
    with col1:
        chapter = st.radio("Select chapter", ("Chapter 3", "Chapter 4", "Chapter 5", "Chapter 9"))
    with col2:
        if chapter == "Chapter 3":
            process = st.selectbox('Select process', [
                'Negative image', 'Negative color image', 'Log transform',
                'Power transform', 'Piecewise linear transform', 'Show histogram',
                'Histogram equalize', 'Histogram equalizer - openCV', 'Local histogram processing',
                'Histogram statistics', 'Smooth box filter', 'Box filter - openCV', 'Median blur',
                'Smooth Gaussian filter', 'Gaussian blur', 'Sharpen image', 'Gradient'
            ])
        elif chapter == "Chapter 4":
            process = st.selectbox('Select process', [
                'Spectrum','Remove period noise', 'Draw notch filter','Draw period noise filter','Spectrum - frequency domain',
                'Remove Moiré - frequency domain', 'Remove interference - frequency domain'
            ])
        elif chapter == "Chapter 5":
            process = st.selectbox('Select process', [
                'Create motion', 'Demotion', 'Demotion noise'
            ])
        else:
            process = st.selectbox('Select process', [
                'Erosion', 'Dilation', 'Boundary','Contour','Convex Hull','Defect detect'
            ])
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
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

    predict_clicked = st.button("Processing")

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
        match process:
            case "Negative image":
                result_img = ch3.Negative(image)
            case "Negative color image":
                result_img = ch3.NegativeColor(image)
            case "Log transform":
                result_img = ch3.Log(image)
            case "Power transform":
                result_img = ch3.Power(image)
            case "Piecewise linear transform":
                result_img = ch3.PiecewiseLine(image)
            case "Show histogram":
                result_img = ch3.Histogram(image)
            case "Histogram equalize":
                result_img = ch3.HistEqual(image)
            case "Histogram equalizer - openCV":
                if image_gray is None:
                    raise ValueError("Grayscale image required for OpenCV equalization")
                result_img = cv.equalizeHist(image_gray)
            case "Local histogram processing":
                result_img = ch3.LocalHist(image)
            case "Histogram statistics":
                result_img = ch3.HistStat(image)
            case "Smooth box filter":
                result_img = ch3.BoxFilter(image)
            case "Box filter - openCV":
                result_img = cv.boxFilter(image, cv.CV_8UC1, (21, 21))
            case "Median blur":
                result_img = cv.medianBlur(image, 7)
            case "Smooth Gaussian filter":
                result_img = ch3.smoothGauss(image)
            case "Gaussian blur":
                result_img = cv.GaussianBlur(image, (43, 43), 7.0)
            case "Gradient":
                result_img = cv.Laplacian(cv.GaussianBlur(image, (43, 43), 7.0), cv.CV_64F)
            case "Sharpen image":
                result_img = ch3.Sharp(image)

            case "Spectrum":
                result_img = ch4.Spectrum(image)
            case "Remove period noise":
                result_img = ch4.RemovePeriodNoise(image)
            case "Draw notch filter":
                result_img = ch4.DrawNotchFilter(image)
            case "Draw period noise filter":
                result_img = ch4.DrawNotchPeriodFilter(image)
            case "Spectrum - frequency domain":
                result_img = ch4.Spec(image)
            case "Remove Moiré - frequency domain":
                result_img = ch4.RemoveMoireFreq(image)
            case "Remove interference - frequency domain":
                result_img = ch4.RemoveInterferenceFreq(image)

            case "Create motion":
                result_img = ch5.CreateMotion(image)
            case "Demotion":
                result_img = ch5.DeMotion(image)
            case "Demotion noise":
                blurred = cv.medianBlur(image, 5)
                result_img = ch5.DeMotion(blurred)

            case "Erosion":
                result_img = ch9.Erosion(image)
            case "Dilation":
                result_img = ch9.Dilation(image)
            case "Boundary":
                result_img = ch9.Boundary(image)
            case "Contour":
                result_img = ch9.Contour(image)
            case "Convex Hull":
                result_img = ch9.ConvexHull(image)
            case "Defect detect":
                result_img = ch9.Defectdetect(image)
            case "Connected components":
                result_img = ch9.ConnectedComponents(image)
        if len(result_img.shape) == 2:
            result_img = cv.cvtColor(result_img, cv.COLOR_GRAY2BGR)
        for percent_complete in range(100):
            time.sleep(0.02)
            my_bar.progress(percent_complete + 1)
            progress_text.text(f"Loading... {percent_complete}%")
        time.sleep(0.5)
        my_bar.empty()
        progress_text.success("Predict Completed!")
        time.sleep(1)
        result_placeholder.image(result_img, caption="Detection Result", use_container_width=True)
        time.sleep(1)
        progress_text.empty()
    if image is None and predict_clicked:
        st.write("No image to predict !!!")

# Main function to run the app
if __name__ == '__main__':
    image_processing_page()
