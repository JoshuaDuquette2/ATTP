import streamlit as st
from streamlit_webrtc import VideoProcessorBase, RTCConfiguration, WebRtcMode, webrtc_streamer
import av
import cv2
import torch
from ultralytics import YOLO

DETECTION_MODEL_LIST = [
    "last.pt",
    "GunV3.pt"]

# 1.
confidence = st.sidebar.slider("Select Model Confidence", 30, 100, 50) / 100

# 2.
weights_option = st.sidebar.selectbox("Select Model", DETECTION_MODEL_LIST)
model = YOLO(weights_option)


class YOLOv8VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Resize the image to a standard size
        #img = cv2.resize(img, (720, int(720 * (9 / 16))))

        # Predict the objects in the image using YOLOv8 model
        conf_threshold = confidence  # You can adjust the confidence threshold here
        results = model.predict(img, conf=conf_threshold)

        # Plot the detected objects on the image
        for result in results:
            img = result.plot()

        # Convert the image back to a VideoFrame
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")

        return frame


#st.markdown("<h1 style='text-align: center;'>ATTP</h1>", unsafe_allow_html=True)

import base64
# Load the image
with open('img.png', 'rb') as f:
    image = f.read()

# Convert the image to Base64
image_base64 = base64.b64encode(image).decode()

# Display the image
st.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{image_base64}' width='150'></div>", unsafe_allow_html=True)



# ... [Your previous code here] ...

# Create two columns for the left and right video streams
col1, col2 = st.columns(2)

# Display the first video stream in the left column
with col1:
    webrtc_streamer(key="example1", mode=WebRtcMode.SENDRECV, video_processor_factory=YOLOv8VideoProcessor)

# Display the second video stream in the right column
with col2:
    webrtc_streamer(key="example2", mode=WebRtcMode.SENDRECV, video_processor_factory=YOLOv8VideoProcessor)


