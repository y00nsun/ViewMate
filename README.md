# ViewMate: AI-Based Convenience Store Assistant for the Visually Impaired

ViewMate is an AI-powered assistance system designed to help visually impaired users shop independently in convenience stores.  
The system supports **real-time corner recognition** and **product information reading** by combining object detection, OCR, and text-to-speech technologies.

---

## 🚀 Overview

ViewMate enables users to:

- Identify convenience store corners (snacks, beverages, ramen, instant meals) using **real-time vision detection**
- Read product information such as **name, nutrition facts, and expiration date**
- Receive all information through **clear audio guidance**

The goal is to improve **accessibility, independence, and usability** during everyday shopping.


## 🌟 Key Features

### 1. Corner Recognition Mode

Detects store corners and informs the user with audio feedback.

**How it works:**
1. The user points the camera toward store shelves (auto or manual trigger).  
2. YOLOv8n detects corner categories.  
3. The system analyzes bounding box positions to determine left / right / front.  
4. TTS provides real-time guidance.  
   - *“Left: Beverage corner. Right: Snack corner.”*



### 2. Product Information Reading Mode

Extracts product information while the user rotates the item.

**Process:**
1. User activates product mode via voice command or button.  
2. System prompts: *“Please rotate the product slowly.”*  
3. YOLOv8n detects product tag, nutrition facts, and date regions.  
4. Detected boxes are cropped and processed using Google Vision OCR.  
5. TTS outputs the extracted information.  

**Example Output:**  
*“Product: Chilsung Cider, 250ml, 120kcal. Expiration date: December 28, 2024.”*



## 🧠 Model Architecture

- **YOLOv8n** — real-time object detection  
- **Google Cloud Vision API** — OCR for text extraction  
- **Text-to-Speech (TTS)** — audio guidance  
- **Custom data pipeline** featuring:
  - In-store image data
  - Augmentation to enhance robustness
  - Balanced training/validation splits

**Performance:**

- Average F1-score **90%+**
- Evaluation metrics:
  - mAP, class loss, box loss, DFL loss  
  - Confusion matrix  
  - F1-score per class



## 📐 System Workflow

### Corner Recognition
Camera → YOLOv8n → Corner Classification → Position Analysis → Audio Output


### Product Information Reading
Camera → YOLOv8n (tag/nutrition/date) → Crop → OCR → Text Parsing → Audio Output




## 🎥 Demo Video

The demo includes:

- Corner recognition in a simulated convenience-store environment  
- Product information reading with real packaged goods  
- Robust performance under diverse lighting and angles

(Insert video link here)

---

## 🛠️ Tech Stack

- Python  
- YOLOv8n (Ultralytics)  
- Google Cloud Vision API  
- TTS (Text-To-Speech)  
- OpenCV  
- Custom data collection & preprocessing pipeline


## Project Duration  
October 2024 – December 2024
