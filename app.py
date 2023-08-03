import streamlit as st
from streamlit_webrtc import VideoProcessorBase, RTCConfiguration, WebRtcMode, webrtc_streamer
import av
import cv2
import torch
from ultralytics import YOLO

model = YOLO("GunV3.pt")


class YOLOv8VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Resize the image to a standard size
        #img = cv2.resize(img, (720, int(720 * (9 / 16))))

        # Predict the objects in the image using YOLOv8 model
        conf_threshold = 0.5  # You can adjust the confidence threshold here
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



webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_processor_factory=YOLOv8VideoProcessor)
