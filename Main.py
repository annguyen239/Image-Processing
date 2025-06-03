import streamlit as st
from streamlit_lottie import st_lottie 
import requests
import datetime
import time # Make sure time is imported if needed for effects
import os
import pandas as pd
import plotly.express as px
        
# --- Functipn to load Lottie animations from URL ---
# Improved error handling
# https://lottiefiles.com/free-animation/technology-isometric-ai-robot-brain-2vX5SyCjFX
# lottie_hello_url = "https://assets2.lottiefiles.com/packages/lf20_2vX5SyCjFX.json
# lottie_hello = load_lottieurl(lottie_hello_url)
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=10) # Add timeout
        r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return r.json()
    except requests.exceptions.RequestException as e:
        # Handle specific request errors
        st.error(f"Error loading Lottie animation from {url}: {e}")
        return None
    except Exception as e:
        # Handle other potential errors like JSON decoding
        st.error(f"An unexpected error occurred loading Lottie: {e}")
        return None

# --- Page configuration ---
st.set_page_config(
    page_title="Computer Vision", # Updated title
    page_icon="👁️",         # Updated icon
    layout="wide",
    initial_sidebar_state="expanded",
)
hide_menu_items = """
    <style>
      /* Hide the list of pages / radio buttons in the sidebar */
      [data-testid="stSidebarNav"] {
        display: none;
      }
    </style>
"""
st.markdown(hide_menu_items, unsafe_allow_html=True)
# --- Load Global Assets ---
lottie_ai_url = "https://lottie.host/957718f1-7d87-44b2-9d34-c194a07a2484/v6iSCPW5DN.json"
lottie_ai_brain_url = "https://lottie.host/1c2226d4-4c24-4b92-a664-7275b82c0971/TtuFsoBD8X.json" # Example: AI Brain
lottie_hello_url = "https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json"

# --- Sidebar Navigation ---
image = st.sidebar.image("images/sidebar_image/hcmute.png")
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["Home","About", "Contact"], key="main_nav") # Added key

if page == "Home":
    st.header("🏠 Home Page")
    st.title("Welcome to the Computer Vision")
    
    c1,c2 = st.columns(2)
    with c1:
        lottie_ai_url = "https://lottie.host/957718f1-7d87-44b2-9d34-c194a07a2484/v6iSCPW5DN.json"
        lottie_ai = load_lottieurl(lottie_ai_url)
        if lottie_ai:
            # Add a unique key to avoid conflicts if st_lottie is used elsewhere
                st_lottie(lottie_ai, height=480, key="global_hero_lottie")
    with c2:
        st.subheader("MODE")
        col1, col2 = st.columns(2)
        with col1:
            st.page_link("pages/🥵Face_Detection.py", label = ":blue[Face Detection]",icon = ":material/face_retouching_natural:",use_container_width=True)
            st.page_link("pages/🔬Object_Detection.py", label = ":green[Object Detection]",icon = ":material/laptop:",use_container_width=True)
            st.page_link("pages/🔶Shape_Detection.py", label = ":red[Shape Detection]",icon = ":material/change_history:",use_container_width=True)
            
        with col2:
            st.page_link("pages/📸Image_Processing.py", label = ":orange[Image Processing]",icon = ":material/image:",use_container_width=True)
            st.page_link("pages/👾Xtra.py", label = ":violet[Bonus]",icon = ":material/thumb_up:",use_container_width=True)
# --- About Page (REPLACED CONTENT) ---
elif page == "About":
    # --- Load Lottie animation specific to About page ---

    # --- Page Header ---
    st.header("🤖 About Us")
    # --- Use Tabs for Organization ---
    tab1, tab3 = st.tabs(["🎯 Overview", "📊 Project Timeline"])

    with tab1:
        
        # First person
        col1,col2 = st.columns(2)  # Adjust ratio as needed
        with col1:
            lottie_ai_brain_url = "https://lottie.host/1c2226d4-4c24-4b92-a664-7275b82c0971/TtuFsoBD8X.json" # Example: AI Brain
            lottie_ai_brain = load_lottieurl(lottie_ai_brain_url)
            if lottie_ai_brain:
            # Add a unique key to avoid conflicts if st_lottie is used elsewhere
                st_lottie(lottie_ai_brain, height=480, key="global_hero_lottie")
        with col2:
            col1,col2 = st.columns([1,2])
            with col1: 
                st.image("images/profile image/an.jpg", width=210)
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.image("images/profile image/tien.jpg", width=210)
            with col2:
                st.markdown("""
                    <div style="display: flex; align-items: center; height: 120%;">
                        <div>
                            <p><strong>NAME:</strong> NGUYỄN NGỌC AN</p>
                            <p><strong>DoB:</strong> 23/09/2004</p>
                            <p><strong>Student ID:</strong> 22146259</p>
                            <p><strong>Major:</strong> MECHATRONICS ENGINEERING</p>
                            <p><strong>Year of study:</strong> Junior</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("""
                <div style="display: flex; align-items: center; height: 120%;">
                    <div>
                        <p><strong>NAME:</strong> TRẦN VĂN TIẾN</p>
                        <p><strong>Date of Birth:</strong> 20/05/2004</p>
                        <p><strong>Student ID:</strong> 22146417</p>
                        <p><strong>Major:</strong> MECHATRONICS ENGINEERING</p>
                        <p><strong>Year of study:</strong> Junior</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    with tab3:
    # Prepare data directly as a list of dictionaries
        roadmap_data = [
            dict(Task="💡Ideation & Planning         ", Start="2025-04-01", Finish="2025-04-07"),
            dict(Task="💻UI/UX & Appearance          ", Start="2025-04-08", Finish="2025-04-14"),
            dict(Task="🔎Detection & Image Processing          ", Start="2025-04-15", Finish="2025-04-27"),
            dict(Task="👾Extra Features & Polish          ", Start="2025-04-28", Finish="2025-05-09"),
        ]

        # Draw timeline using the list of dictionaries directly
        fig = px.timeline(
            data_frame=roadmap_data,
            x_start="Start",
            x_end="Finish",
            y="Task",
            color="Task",
            color_discrete_sequence=["#DAF7A6","#FFC300","#FF5733","#900C3F"]
        )

        # Configure Y-axis (Note: autorange=None doesn't reverse; use "reversed" for chronological top-to-bottom)
        fig.update_yaxes(autorange=None, title_text="") # Keeps original order unless data is sorted differently

        # Update layout for aesthetics
        fig.update_layout(
            title_text="Image Processing Web Development Roadmap",
            title_x=0.5,
            bargap=0.1,
            showlegend=False,
            plot_bgcolor="#879ee7",
            xaxis=dict(showgrid=True, gridwidth=3, gridcolor="lightgray",tickfont=dict(size=16),tickformat="%d %b %Y"),
            barcornerradius=20,  # Rounded corners for bars
            yaxis=dict(
                tickfont=dict(size=16) # Increase font size of Y-axis labels
            )
        )
        # Display the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)
# --- Contact Page ---
elif page == "Contact":
    # --- Define variables locally for this page section ---
    # (Better practice might be to define all URLs/contacts globally once at the top)
    fb_url_an = "https://www.facebook.com/canocsociu2309"
    fb_url_tien = "https://www.facebook.com/tran.tien.615618"
    gh_url_an = "https://github.com/annguyen239"
    gh_url_tien = "https://github.com/TranTien2oo4"
    an_phone_number_disp = "Nguyễn Ngọc An: (+84) 785907478" # Use distinct name if global exists
    an_zalo_number_disp = "Nguyễn Ngọc An: 0785907478"      # Use distinct name if global exists
    tien_phone_number_disp = "Trần Văn Tiến: (+84) 399013031" # Use distinct name if global exists
    tien_zalo_number_disp = "Trần Văn Tiến: 0399013031"# Use distinct name if global exists
    zalo_icon_url_contact = "https://cdn.simpleicons.org/zalo/0068FF" # Use distinct name

    # --- Inject Font Awesome CSS & Custom Styles (Needed for icons on this page) ---
    st.markdown("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
        <style>
            .icon-link i, .icon-link img {
                vertical-align: middle;
                margin-right: 6px;
                font-size: 1.1em;
                width: 1.1em;
                height: 1.1em;
            }
            .contact-info span, .contact-info a {
                 vertical-align: middle;
            }
            .contact-info {
                margin-bottom: 8px;
                line-height: 1.6;
            }
            /* Style form */
            .stTextInput label, .stTextArea label {
                font-weight: bold;
            }
        </style>
        """, unsafe_allow_html=True)

    # --- Page Content ---
    st.header("📧 Contact Us") # Added emoji
    # --- Email Form ---
    st.subheader("Email contact form")
    # Use unique keys for form elements
    col1,col2 = st.columns([1,2])
    with col1:
        lottie_hello_url = "https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json"
        lottie_hello = load_lottieurl(lottie_hello_url)   
        if lottie_hello:
        # Add a unique key to avoid conflicts if st_lottie is used elsewhere
            st_lottie(lottie_hello, height=480, key="global_hero_lottie")
    with col2:
        with st.form(key='contact_form'):
            col1_form, col2_form = st.columns(2)
            with col1_form:
                name = st.text_input("First Name*", key="form_first_name")
            with col2_form:
                email = st.text_input("Last Name*", key="form_last_name")
            email = st.text_input("Email*", key="form_email") 
            message = st.text_area("Message*", height=150, key="form_message_contact")
            submit_button = st.form_submit_button(label="Send Message")
            if submit_button:
                if name and email and message: # Basic validation
                    # In a real app, add email sending logic here
                    st.success(f"Thank you, {name}! Your message has been sent.")
                else:
                    st.warning("Please fill in all required fields (*).")

    st.markdown("---") # Divider

    # --- Social Media and Hotline (Using Icons) ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Social Media")
        col3, col4 = st.columns(2)  # Adjust ratio as needed
        with col3:
            st.markdown(f"""
                <div class="icon-link contact-info">
                    <i class="fab fa-facebook"></i> <a href="{fb_url_an}" target="_blank">Nguyễn An</a>
                </div>
                <div class="icon-link contact-info">
                    <i class="fab fa-github"></i> <a href="{gh_url_an}" target="_blank">Nguyễn Ngọc An</a>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="icon-link contact-info">
                    <i class="fab fa-facebook"></i> <a href="{fb_url_tien}" target="_blank">Trần Tiến</a>
                </div>
                <div class="icon-link contact-info">
                    <i class="fab fa-github"></i> <a href="{gh_url_tien}" target="_blank">Trần Văn Tiến</a>
                </div>
            """, unsafe_allow_html=True)
            

    with col2:
        st.subheader("Hotline Contact")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"""
                <div class="icon-link contact-info">
                    <i class="fas fa-phone"></i> <span>{an_phone_number_disp}</span>
                </div>
                <div class="icon-link contact-info">
                    <img src="{zalo_icon_url_contact}" alt="Zalo"> <span>{an_zalo_number_disp}</span>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="icon-link contact-info">
                    <i class="fas fa-phone"></i> <span>{tien_phone_number_disp}</span>
                </div>
                <div class="icon-link contact-info">
                    <img src="{zalo_icon_url_contact}" alt="Zalo"> <span>{tien_zalo_number_disp}</span>
                </div>
            """, unsafe_allow_html=True)
            

    st.markdown("---") # Divider
    current_year = datetime.datetime.now().year
    st.caption(f"© {current_year} - All rights reserved.")

