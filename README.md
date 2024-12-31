# Smart_Attendance_System
An automated attendance System

## Overview
The Smart Attendance System is an innovative solution designed to automate attendance marking from a group photo. Using advanced face recognition technologies, it identifies individuals in the photo and records their attendance in a CSV file. The system also provides visualizations of attendance data for easy analysis.

**Note** -
This project is currently in its initial phase and has been tested with a limited dataset. Extensive testing with larger and more diverse datasets is required to ensure robustness and reliability in real-world scenarios.

## Features
- Automatic face detection and recognition from group photos.
- Attendance marking with timestamp in a CSV file.
- Support for individual photo datasets for identification.
- Visualization of attendance records using Matplotlib.

## Prerequisites
#### Ensure you have the following installed:
- Python 3.8+
- Virtual environment (venv)
- Required libraries (see requirements.txt)

## Installation
- Step 1: Clone the Repository
- Step 2: Set Up Virtual Environment
- Step 3: Install Dependencies

## Dependencies
- TensorFlow
- Keras
- OpenCV
- MTCNN
- Pandas
- Matplotlib

Install all dependencies using the **requirements.txt** file.

## Usage
- Step 1: Prepare Dataset
- Step 2: Run the Program
  - python Attendance_System.py
- Output

## How it Works
1. Face Detection:
   MTCNN is used to detect faces in the group photo.
2. Face Recognition:
   FaceNet generates embeddings for faces in the group photo and individual photos. Matches embeddings to identify individuals.
3. Attendance Marking:
   Recognized individuals are logged in the attendance_records.csv file with their name, date, and time.
4.Visualization:
   Attendance data is visualized using Matplotlib for better analysis.

## Acknowledgments
- FaceNet for face embeddings.
- MTCNN for face detection.
- Kaggle community for inspiration.

## Conclusion
The Smart Attendance System is an innovative solution that leverages facial recognition technology to streamline the attendance marking process. By automating the identification of individuals in group photos and maintaining accurate attendance records, this project demonstrates the practical applications of computer vision and machine learning in everyday tasks. This project serves as a foundation for further enhancements, such as real-time video processing, cloud-based data storage, or integration with other systems like educational or organizational management platforms. Contributions and suggestions for improvement are welcome to take this project to the next level.
