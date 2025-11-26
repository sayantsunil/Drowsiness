#  Driver Drowsiness Detection System

A real-time computer vision system to detect driver drowsiness using **Eye Aspect Ratio (EAR)** via **MediaPipe Face Mesh** and alert the driver with a sound using **Pygame**.

#  Features

- Real-time video processing with OpenCV
- Eye landmark detection using MediaPipe's Face Mesh
- EAR (Eye Aspect Ratio) calculation to detect closed eyes
- Triggers alert if eyes remain closed for a set duration
- Plays an alarm sound using Pygame to wake the driver

## 🔧 Technologies Used

  Library                    Purpose                         

   OpenCV  -           Real-time video feed & drawing  
   MediaPipe-          Facial landmarks detection     
   NumPy -             Efficient numerical operations  
   Pygame -            Audio alert system                 
   Threading -         Run alarm without freezing UI   
   time -              Timing drowsy intervals         


## ⚙️ How It Works

1. The webcam captures frames of the driver's face.
2. MediaPipe detects 3D facial landmarks, especially the eyes.
3. The **Eye Aspect Ratio (EAR)** is calculated:
   - Low EAR = Eyes are closed
   - If low EAR persists > 4 seconds → Drowsiness detected
4. An alarm is triggered using a sound to alert the driver.
   

##  Eye Aspect Ratio (EAR)

The EAR is calculated using the distance between six specific points around each eye:
     A = ||p2 - p6||
     B = ||p3 - p5||
     C = ||p1 - p4||
   
# EAR = (A + B) / (2.0 * C)

If the EAR falls below a set threshold (e.g., 0.25) for more than 4 seconds, the system activates the alert.




