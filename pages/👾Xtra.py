import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- CONFIG ---
MODEL_PATH = "model/best.pt"
CONFIDENCE = 0.5
PRICE_FILE = "part/prices.xlsx"
CAMERA_URL = "http://192.168.1.8:2309/video"
IMG_SIZE   = 320

# --- PAGE SETUP ---
st.set_page_config(page_title="Object Detection + Billing", layout="wide")
col_1,col_2 = st.columns(2)
with col_1:
    st.page_link("Main.py", label = "Back",icon = ":material/arrow_back:",use_container_width=True)
with col_2:
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
st.title("📸 Object Detection with Price Lookup")
hide_menu_items = """
    <style>
      /* Hide the list of pages / radio buttons in the sidebar */
      [data-testid="stSidebarNav"] {
        display: none;
      }
    </style>
"""
st.markdown(hide_menu_items, unsafe_allow_html=True)

# --- ONE-TIME LOADS ---
model      = YOLO(MODEL_PATH)
tracker    = DeepSort(max_age=30, max_cosine_distance=0.3)
price_df   = pd.read_excel(PRICE_FILE)
price_df.columns = [c.strip().lower() for c in price_df.columns]
price_map  = dict(zip(price_df["labels"].astype(str), price_df["price"].astype(float)))

# --- UI SETUP ---
col1, col2= st.columns([1,3])
with col1:
    if st.button("▶️ Start Detection"):
        st.session_state.running = True
with col2:
    if st.button("⏹️ Stop Detection"):
        st.session_state.running = False


col_bill, col_video = st.columns([2,3])
col_bill.subheader("Price List")
col_bill.dataframe(price_df, use_container_width=True)

# placeholders
frame_pl = col_video.empty()
bill_pl  = col_bill.empty()

# session defaults
st.session_state.setdefault("running", False)
st.session_state.setdefault("billed_ids", set())
st.session_state.setdefault("bill_rows", {})
st.session_state.setdefault("last_bill_len", 0)

if st.session_state.running:
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        st.error("Cannot open camera")
        st.session_state.running = False
    else:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Frame read failed")
                break

            # --- detection & tracking ---
            results = model.predict(frame, conf=CONFIDENCE, device="cuda", imgsz=IMG_SIZE)[0]
            annotated = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)
            frame_pl.image(annotated, use_container_width=True)

            dets = []
            for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                dets.append(([x1, y1, x2 - x1, y2 - y1], float(conf), int(cls)))
            tracks = tracker.update_tracks(dets, frame=frame)

            # --- billing logic ---
            for t in tracks:
                if not t.is_confirmed(): continue
                tid   = t.track_id
                label = model.names[t.get_det_class()]
                if tid not in st.session_state.billed_ids:
                    st.session_state.billed_ids.add(tid)
                    st.session_state.bill_rows[tid] = {
                        "label":  label,
                        "price":  price_map.get(label, 0.0),
                        "amount": 1
                    }

            # --- update billing table only if new rows appeared ---
            bill_len = len(st.session_state.bill_rows)
            if bill_len != st.session_state.last_bill_len:
                rows = []
                for tid, info in st.session_state.bill_rows.items():
                    total = info["price"] * info["amount"]
                    rows.append({
                        "ID":         tid,
                        "Label":      info["label"],
                        "Unit Price": info["price"],
                        "Qty":        info["amount"],
                        "Total":      total
                    })
                df_bill = pd.DataFrame(rows)
                bill_pl.table(df_bill)
                bill_pl.markdown(f"**Grand Total:** {df_bill['Total'].sum()} VND")
                st.session_state.last_bill_len = bill_len

        cap.release()
        frame_pl.empty()
        bill_pl.empty()
        st.success("Detection stopped.")
else:
    col_video.write("▶️ Click **Start Detection** to begin.")
