import streamlit as st
import openai
import base64
import cv2
import os
import requests

# OpenAI API Key
api_key = os.environ["OPENAI_API_KEY"]

st.title('Environment Describer')
st.sidebar.header("Webcam")
run_webcam = st.sidebar.button('Start Webcam')
FRAME_WINDOW = st.image([])
placeholder = st.empty()


if run_webcam:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (512, 512))
        FRAME_WINDOW.image(frame)
        _, buffer = cv2.imencode(".jpg", frame)
        image = base64.b64encode(buffer).decode("utf-8")
        headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
                }
                        
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
              {
                "role": "user",
                "content": [
                  {
                    "type": "text",
                    "text": "Describe This image to persian and english"
                  },
                  {
                    "type": "image_url",
                    "image_url": {
                      "url": f"data:image/jpeg;base64,{image}"
                    }
                  }
                ]
              }
            ],
            "max_tokens": 300
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)  
        result = response.json()
        with placeholder.container():
            st.write(result['choices'][0]['message']['content'],
                    style={'font-size': '20px', 'line-height': '1.5em'})            

    cap.release()
