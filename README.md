# Barbell-Trajectory-Tracking-for-Performance-Analysis-During-Snatch-Movement-in-Weightlifting

# 🏋️ Robust Barbell Trajectory Tracking and Classification for Weightlifting Analysis

**This project leverages computer vision and machine learning to track barbell motion, classify trajectory types, and extract kinematic parameters to assist athletes and coaches in performance analysis and injury prevention.**

---

## 📌 Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Tools and Dependencies](#tools-and-dependencies)
- [Results and Evaluation](#results-and-evaluation)
- [Limitations and Future Work](#limitations-and-future-work)
---

## Introduction

Weightlifting, especially during snatch lifts, requires precise technique to optimize performance and minimize injury risk. Automatic analysis of barbell trajectories can assist athletes and coaches in technique refinement and injury prevention. Traditional methods face challenges such as occlusion, background clutter, and camera variability.  

This project proposes a robust, camera viewpoint-invariant pipeline employing computer vision and deep learning techniques for accurate tracking and classification of barbell trajectories.

---

## Project Overview

This project develops a computer vision framework to:
- Automatically track the barbell during weightlifting movements
- Classify trajectory types based on movement patterns
- Extract kinematic parameters (e.g., height, velocity) for analytics
- Provide performance metrics to evaluate lift quality  

Key innovations include:
- Using MedianFlow tracker for stability
- Camera viewpoint invariance techniques
- Quantitative assessments based on recorded kinematic variables

---

## Objectives

- Automate barbell motion tracking during weightlifting
- Classify different trajectory types accurately
- Extract meaningful kinematic features
- Develop a performance metric for technique evaluation
- 
---

## Dataset

- Collected during regional U.S. weightlifting competitions in April 2025
- 200+ trials, over 100,000 video frames
- Videos recorded with GoPro HERO10 Black at 2160p (~30 fps), placed ~3.35 meters from lifting platform
- Participants: 44 athletes (28 males, 16 females), diverse skill levels
- Annotation: Trajectory types (Type 1-4) based on guidelines

---

## Methodology

### Data Acquisition & Preprocessing
- Videos resized to 1920×1080 pixels
- Trimmed to focus on lift motion
- Standardized lateral plane camera placement  

### Tracking Algorithm
- MedianFlow tracker for robustness against occlusion and speed variation
- Other trackers evaluated: Boosting, MIL, KCF, TLD, DlibTracker, CamShift, Template Matching  

### Trajectory Classification
- Based on horizontal displacement pattern and crossing of vertical reference line
- Achieved ~70% accuracy across 10 tested videos  

### Kinematic Parameter Extraction
- Pixel measurements mapped to real-world metrics (average height ~173.33 cm)
- Parameters: maximum height, velocity, displacement  
- Used for holistic performance scoring  

### Performance Metrics
- Combines classification accuracy + kinematic consistency
- Supports technique assessment and injury risk analysis  

---

## Tools and Dependencies

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy, SciPy
- Matplotlib, Seaborn
- TensorFlow / PyTorch (future pose estimation extension)

---

## Results and Evaluation

- Achieved approximately **70% accuracy** in trajectory classification.
- **MedianFlow tracker** provided stable tracking performance even in the presence of occlusion and speed variations.
- Successfully extracted **kinematic parameters** (e.g., height, velocity, displacement) to enable performance assessment.

  ### Classification and Barbell Kinematic Parameters Validation

| Video | Type Given | Type Predicted | Height of Athlete (cm) | Ymax (cm) | X1 (cm) | Bar Drop (cm) | Score (4) |
|--------|------------|----------------|-----------------------|-----------|---------|---------------|-----------|
| 1      | 1          | 2              | 165                   | 129       | 5       | 8             | NA        |
| 2      | 2          | 1              | 170                   | 133.5     | 1.5     | 9.5           | NA        |
| 3      | 3          | 3              | 174                   | 137       | 3.5     | 14            | 3         |
| 4      | 3          | 3              | 179                   | 139.5     | 8       | 4.5           | 4         |
| 5      | 3          | 3              | 185                   | 139.5     | 11      | 6             | 3         |
| 6      | 4          | –              | –                     | –         | –       | –             | NA        |
| 7      | 2          | 2              | 170                   | 133.5     | 3       | 6.5           | 3         |
| 8      | 1          | 1              | 177                   | 139       | 5       | 6             | 4         |
| 9      | 1          | 1              | 178                   | 136       | 3.5     | 7             | 4         |
| 10     | 1          | 1              | 162                   | 126       | 5.5     | 8.5           | 4         |

---

## Limitations and Future Work

- Current classification accuracy can be improved with additional data and model refinement.
- Camera viewpoint invariance is limited by specific calibration settings.

### Future Directions

- Automate **pixel-to-centimeter conversion** for real-world measurement consistency.
- Implement **camera calibration** to correct affine distortions.
- Develop **automatic barbell center detection** to improve tracking precision.
- Extend the framework to cover **clean and jerk** lifts in addition to snatch lifts.
- Deploy as a **mobile app** for real-time athlete feedback.
- Incorporate **velocity and power metrics** into performance evaluation.
