import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk
import time
import pyttsx3
from twilio.rest import Client
import threading
import csv
from datetime import datetime
import geocoder
import webbrowser
import os
from queue import Queue

# Twilio setup---------------+918489616411
TWILIO_ACCOUNT_SID = 
TWILIO_AUTH_TOKEN =
TWILIO_PHONE_NUMBER = 
EMERGENCY_CONTACT = "000000000"

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Load Fall Detection Model
model = tf.keras.models.load_model("fall_detection_model.h5")

# CSV Logging
CSV_FILE = "fall_logs.csv"
queue = Queue()

# Flags
fall_detected = False
fall_time = None
sound_played = False
sms_sent = False
fall_logged = False
hospital_page_opened = False

# ------------------------------------------> CSV file function
def write_to_csv():
    global fall_logged
    if not fall_logged:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        with open(CSV_FILE, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([date_str, time_str, "FALL DETECTED"])
        
        fall_logged = True 

# ------------------------------------------> Get location function
def get_location():
    try:
        g = geocoder.ip("me")
        if g.latlng:
            return g.latlng
        else:
            return None
    except Exception as e:
        print("Location Fetch Failed:", e)
        return None  

# ------------------------------------------> SMS function
def send_sms():
    global sms_sent
    if not sms_sent:
        def twilio_thread():
            try:
                location = get_location()
                if location:
                    lat, lon = location
                    maps_link = f"https://www.google.com/maps?q={lat},{lon}"
                    sms_body = f"⚠️ Fall Detected! Please check immediately.\n📍 Location: {maps_link}"
                else:
                    sms_body = "⚠️ Fall Detected! Please check immediately.\n(Location not available)"

                message = twilio_client.messages.create(
                    body=sms_body,
                    from_=TWILIO_PHONE_NUMBER,
                    to=EMERGENCY_CONTACT
                )
                print("✅ SMS Sent with Location!")
            except Exception as e:
                print("❌ Twilio SMS Failed:", e)

        threading.Thread(target=twilio_thread).start()
        sms_sent = True

# ------------------------------------------> Frame Preprocessing for ML Model
def preprocess_frame(frame):
    image = cv2.resize(frame, (150, 150))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ------------------------------------------> ML-Based Fall Detection Function
def detect_fall(frame):
    global fall_detected, fall_time, sound_played, sms_sent, fall_logged, hospital_page_opened
    processed_image = preprocess_frame(frame)
    prediction = model.predict(processed_image)
    confidence = float(prediction[0][0])

    print(f"🔍 Model Prediction Confidence: {confidence:.4f}")

    if confidence > 0.88 and not fall_detected:
        fall_detected = True
        fall_time = time.time()
        sound_played = False
        sms_sent = False
        fall_logged = False
        hospital_page_opened = False
        return True

    return False

# ------------------------------------------> Voice alert function
def play_fall_alert():
    global sound_played
    if not sound_played:
        queue.put("Fall detected! Please check immediately.")
        sound_played = True

def process_queue():
    if not queue.empty():
        text = queue.get()
        engine.say(text)
        engine.runAndWait()

# ------------------------------------------> Tkinter UI
root = tk.Tk()
root.title("Fall Detection System")
root.geometry("800x600")
root.configure(bg="#ffffff")

status_label = tk.Label(root, text="Monitoring...", font=("times new roman", 24, "bold"), bg="white", fg="black")
status_label.pack(pady=20)

canvas = tk.Canvas(root, width=640, height=480, bg="black")
canvas.pack()

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FPS, 30)

def update_frame():
    global fall_detected, fall_time, sound_played, sms_sent, fall_logged, hospital_page_opened

    ret, frame = video.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)

    if detect_fall(frame):
        root.configure(bg="#ff4d4d")
        status_label.config(text="⚠️ FALL DETECTED!", bg="#ff4d4d", fg="white")
        play_fall_alert()
        send_sms()
        write_to_csv()
        
        if not hospital_page_opened:
            html_path = os.path.join(os.path.dirname(__file__), 'web.html')
            webbrowser.open_new(f'file:///{html_path}')
            hospital_page_opened = True  

    if fall_detected and time.time() - fall_time > 5:  
        fall_detected = False
        fall_logged = False

    else:
        root.configure(bg="white")
        status_label.config(text="Monitoring...", bg="white", fg="black")
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame).resize((640, 480))
    img_tk = ImageTk.PhotoImage(img)
    
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.img_tk = img_tk

    process_queue()
    root.after(33, update_frame)

# ------------------------------------------> Exit function
def close_app():
    video.release()
    cv2.destroyAllWindows()
    root.quit()

exit_button = tk.Button(root, text="EXIT", command=close_app, font=("times new roman", 18, "bold"), bg="black", fg="white", relief="raised", padx=20, pady=10)
exit_button.pack(pady=20)

try:
    with open(CSV_FILE, "x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Time", "Status"])
except FileExistsError:
    pass

update_frame()
root.mainloop()
