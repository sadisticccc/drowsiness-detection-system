# Real-Time Driver Drowsiness Detection System

## Overview

This project presents a real-time driver drowsiness detection system developed using computer vision techniques and pre-trained machine learning models. The system monitors a driver through a webcam feed and identifies signs of fatigue based on eye movement patterns.

The objective is to provide a non-intrusive and accessible solution that contributes to road safety by detecting early indicators of drowsiness and issuing alerts.

---

## Problem Statement

Driver fatigue is a significant factor in road accidents worldwide. Traditional detection methods often rely on physiological sensors or vehicle-based metrics, which may be intrusive or limited in reliability.

This project investigates a computer vision-based approach that uses facial feature analysis to detect drowsiness in real time without requiring additional hardware.

---

## Project Scope

This project is implemented as a prototype standalone Python application. It focuses on demonstrating the core functionality of real-time drowsiness detection using a webcam.

The system performs facial detection, landmark extraction, and eye state analysis to determine drowsiness conditions. Future work includes extending the system into a full application with enhanced interfaces and analytics capabilities.

---

## Methodology

The system operates using a real-time video processing pipeline combined with facial landmark analysis.

Key steps include:

1. Face detection using Haar Cascade classifiers
2. Facial landmark detection using a pre-trained dlib 68-point model
3. Extraction of eye regions from facial landmarks
4. Calculation of Eye Aspect Ratio (EAR)
5. Monitoring EAR values across consecutive frames
6. Detection of prolonged eye closure
7. Triggering alerts when drowsiness conditions are met

The Eye Aspect Ratio (EAR) is used as the primary metric. A sustained decrease in EAR below a defined threshold indicates potential drowsiness.

---

## System Architecture

The system follows a continuous processing pipeline:

- Webcam input acquisition
- Frame preprocessing (resizing and grayscale conversion)
- Face detection
- Facial landmark extraction
- Feature computation (EAR)
- Threshold-based decision logic
- Alert generation
- Real-time visualization

---

## Technologies Used

- Python 3
- OpenCV
- dlib (pre-trained facial landmark model)
- NumPy
- SciPy
- imutils

---

## Features

- Real-time face detection
- Facial landmark tracking
- Eye Aspect Ratio (EAR) computation
- Threshold-based drowsiness detection
- Visual alert system
- Real-time display of detection metrics

---

## Setup and Installation

1. Clone the repository:

```bash
git clone https://github.com/sadisticccc/drowsiness-detection-system.git
cd drowsiness-detection-system


2. Install dependencies:

pip install opencv-python dlib imutils scipy numpy

3. Download the required model:

The file shape_predictor_68_face_landmarks.dat must be downloaded separately from the official dlib repository.

4. Usage

Run the application:

python main.py

The webcam will activate and begin real-time monitoring. Press Q to terminate the program.


Usage

Run the application:

python main.py

The webcam will activate and begin real-time monitoring. Press Q to terminate the program.