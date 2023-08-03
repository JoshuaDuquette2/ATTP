import streamlit as st
from streamlit_webrtc import VideoProcessorBase, RTCConfiguration, WebRtcMode, webrtc_streamer
import av
import cv2
import torch

from ultralytics import YOLO

# Determine the device to use
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the YOLO model onto the device
model = YOLO("GunV3.pt").to(device)

class YOLOv8VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Resize the image to a lower resolution while keeping the original aspect ratio
        height, width = img.shape[:2]
        new_width = 640
        new_height = int(height * (new_width / width))
        img = cv2.resize(img, (new_width, new_height))

        # Move the image data onto the device
        img = torch.from_numpy(img).to(device)

        # Predict the objects in the image using YOLOv8 model
        conf_threshold = 0.5  # You can adjust the confidence threshold here
        results = model.predict(img, conf=conf_threshold)

        # Plot the detected objects on the image
        for result in results:
            img = result.plot()

        # Convert the image back to a VideoFrame
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")

        return frame

st.markdown("<h1 style='text-align: center;'>ATTP</h1>", unsafe_allow_html=True)
webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_processor_factory=YOLOv8VideoProcessor)
