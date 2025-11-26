import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import os

# Constants
EAR_THRESHOLD = 0.25
DROWSY_DURATION = 4  # seconds

# Flags and timers
alarm_on = False
start_drowsy_time = None

# Initialize pygame mixer safely
try:
    pygame.mixer.init()
    alarm_file = r"C:\Users\anjan\OneDrive\Desktop\drowsy\alarm.wav"
    if os.path.exists(alarm_file):
        pygame.mixer.music.load(alarm_file)
    else:
        print(f"[WARNING] '{alarm_file}' not found. Sound will not play.")
        alarm_file = None
except pygame.error as e:
    print(f"[ERROR] Failed to initialize sound system: {e}")
    alarm_file = None

# Alarm trigger function
def sound_alarm():
    if alarm_file and not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Eye landmark indices from MediaPipe Face Mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# EAR calculation function
def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Start webcam feed
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            left_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE])
            right_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE])

            # Draw landmarks
            for x, y in np.concatenate((left_eye, right_eye)):
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            # EAR calculation
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Display EAR on screen
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Drowsiness logic
            if ear < EAR_THRESHOLD:
                if start_drowsy_time is None:
                    start_drowsy_time = time.time()
                elif time.time() - start_drowsy_time > DROWSY_DURATION:
                    sound_alarm()
                    cv2.putText(frame, "DROWSY ALERT!", (30, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                start_drowsy_time = None
                alarm_on = False
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()

    # Show the frame
    cv2.imshow("Drowsiness Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
