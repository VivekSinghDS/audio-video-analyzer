import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import requests
from dotenv import load_dotenv 
import time 

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

start = time.time()
video = cv2.VideoCapture("/Users/vivek.singh/AI-Frameworks/audio_video_analyzer/DF Imp intro.mov")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")
print(time.time() - start, ' TIME FOR VIDEO PROCESSING')

start = time.time()
PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "Here is a candidate who is presenting something and wants to know their feedback, and how they could improve upon their posture and stance to engage audiences further. The feedback should be generated based on all the available frames, and not very particular ones",
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::200]),
        ],
    },
]
params = {
    "model": "gpt-4o",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 300,
}

result = client.chat.completions.create(**params)
print(result.choices[0].message.content)
print(time.time() - start, ' TIME FOR GPT CALL ONLY')

####### for Voice #####

# Transcribe the audio
# audio_path = "path/to/audio.mp3"
# transcription = client.audio.transcriptions.create(
#     model="whisper-1",
#     file=open(audio_path, "rb"),
# )

# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#     {"role": "system", "content":"""Here is a candidate who is presenting something and wants to know their feedback. Provide how he can work on his presentation skills by judging out pros and cons. Respond in Markdown."""},
#     {"role": "user", "content": [
#         {"type": "text", "text": f"The audio transcription is: {transcription.text}"}
#         ],
#     }
#     ],
#     temperature=0,
# )
# print(response.choices[0].message.content)
