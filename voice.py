import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import requests
from dotenv import load_dotenv 

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
audio_path = "/Users/vivek.singh/AI-Frameworks/audio_video_analyzer/harvard.wav"
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=open(audio_path, "rb"),
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
    {"role": "system", "content":"""Here is a candidate who is presenting something and wants to know their feedback. Provide how he can work on his presentation skills by judging out pros and cons. Respond in Markdown."""},
    {"role": "user", "content": [
        {"type": "text", "text": f"The audio transcription is: {transcription.text}"}
        ],
    }
    ],
    temperature=0,
)
print(response.choices[0].message.content)
