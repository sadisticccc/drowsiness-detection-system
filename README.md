# Real-Time Driver Drowsiness Detection System (DrowsGuard)

## Overview

This project presents a real-time driver drowsiness detection system developed using computer vision techniques and pre-trained machine learning models. The system monitors a driver through a webcam feed and identifies signs of fatigue based on eye and mouth movement patterns.

The objective is to provide a non-intrusive and accessible solution that contributes to road safety by detecting early indicators of drowsiness and issuing timely alerts.

---

## Problem Statement

Driver fatigue is a significant factor in road accidents worldwide. Traditional detection methods often rely on physiological sensors or vehicle-based metrics, which may be intrusive, expensive, or limited in reliability.

This project explores a computer vision-based solution that uses facial feature analysis to detect drowsiness in real time without requiring additional hardware.

---

## Project Scope

This project is implemented as a prototype standalone Python application. It demonstrates real-time drowsiness detection using a webcam along with basic analytics and logging capabilities.

The system includes:

* Real-time monitoring
* Alert generation
* Session logging
* Basic analytics dashboard
* Simulation integration

Future improvements may include mobile deployment, cloud analytics, and enhanced UI/UX.

---

## Methodology

The system operates using a real-time video processing pipeline combined with facial landmark analysis.

Key steps include:

* Face detection using Haar Cascade classifiers
* Facial landmark detection using a pre-trained dlib 68-point model
* Extraction of eye and mouth regions from facial landmarks
* Calculation of Eye Aspect Ratio (EAR)
* Calculation of Mouth Aspect Ratio (MAR)
* Monitoring EAR and MAR values across frames
* Detection of prolonged eye closure and yawning
* Triggering alerts when drowsiness conditions are met

EAR is used to detect eye closure, while MAR helps identify yawning patterns.

---

## System Architecture

The system follows a continuous pipeline:

* Webcam input acquisition
* Frame preprocessing (resize and grayscale)
* Face detection
* Facial landmark extraction
* Feature computation (EAR and MAR)
* Threshold-based decision logic
* Alert generation (visual + audio)
* Data logging (sessions and alerts)
* Dashboard analytics and simulation integration

---

## Technologies Used

* Python 3
* OpenCV
* dlib (pre-trained facial landmark model)
* NumPy
* SciPy
* imutils
* Flask (for dashboard)

---

## Features

* Real-time face detection
* Facial landmark tracking
* Eye Aspect Ratio (EAR) computation
* Mouth Aspect Ratio (MAR) computation
* Threshold-based drowsiness detection
* Visual and audio alert system
* Session and alert logging (SQLite database)
* Dashboard for analytics and reporting
* Ride simulation integration
* FPS and performance tracking

---

## Project Structure

Implementation/
└── Source_Code/
├── Database/
│ ├── drowsiness.db
│ ├── schema.sql
│ └── sample_queries.sql
│
├── templates/
│ └── index.html
│
├── main.py
├── dashboard.py
├── accuracy_test.py
├── ride_simulation.py
│
├── shape_predictor_68_face_landmarks.dat
│
├── fps_log.txt
├── latency_log.txt
│
├── requirements.txt
└── README.md

---

## Setup and Installation

1. Clone the repository:
   git clone https://github.com/sadisticccc/drowsiness-detection-system.git
   cd drowsiness-detection-system

2. Install dependencies:
   pip install -r requirements.txt

3. Download required model:
   Download shape_predictor_68_face_landmarks.dat from the official dlib repository and place it inside the Source_Code folder.

---

## Usage

### Run Main Detection System

python main.py

* Webcam activates automatically
* Real-time monitoring begins
* Press Q or ESC to quit

---

### Run Dashboard

python dashboard.py

* Open browser:
  http://127.0.0.1:5000

* View:

  * Sessions
  * Alerts
  * Risk levels
  * Analytics reports

---

### Run Simulation

python ride_simulation.py

* Simulates driver safety system integration

---

### Run Accuracy / Performance Test

python accuracy_test.py

* Displays EAR and MAR values
* Measures detection behaviour and system performance

---

## Database

The system uses SQLite to store:

* Session data
* Alert events
* EAR/MAR values

Additional files:

* schema.sql → database structure
* sample_queries.sql → example queries

---

## Controls

| Key     | Function         |
| ------- | ---------------- |
| Q / ESC | Exit application |
| C       | Switch camera    |

---

## Limitations

* Accuracy depends on lighting conditions
* Requires clear visibility of face
* Thresholds may vary per individual
* Not tested on large-scale datasets

---

## Future Work

* Machine learning-based adaptive thresholds
* Mobile app integration
* Cloud-based analytics
* Multi-driver tracking
* Improved UI/UX design

---

## Conclusion

This project demonstrates a practical and non-intrusive approach to driver drowsiness detection using computer vision techniques. By combining real-time monitoring, alert generation, and basic analytics, the system provides a strong foundation for further development into a full-scale intelligent safety system.

---

## Author

Sadikshya Kunwar
