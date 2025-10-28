# Image-Analyzer-with-Object-Detection

# 🧠 Image Intelligence System
**Object Detection and Dominant Color Extraction from Images (with PostgreSQL Storage)**

This Python project loads an image from a URL, performs object detection using a pre-trained Faster R-CNN model, extracts the dominant color using OpenCV + KMeans, and stores the results in a PostgreSQL database.

---

## 📂 Project Structure

```
project-root/
├── assets/                  # Optional folder for local images
├── .env                    # Environment variables (DB credentials)
├── .gitignore              # Git ignore file
├── main.py                 # Main entry script
├── README.md               # This file
└── .venv/                  # Python virtual environment (not pushed to GitHub)
```

---

## 📦 Features

- ✅ Load and process any image via URL
- 🔍 Detect objects using Faster R-CNN (COCO dataset)
- 🎨 Extract dominant color using OpenCV and KMeans
- 💾 Save image metadata and detection results to PostgreSQL
- 🔒 Store DB credentials securely using `.env` file
- 🌍 Compatible with ngrok for remote PostgreSQL access
- 🖼️ Visualize detection results using matplotlib

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/image-analyzer.git
cd image-analyzer
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install torch torchvision numpy opencv-python pillow matplotlib requests scikit-learn psycopg2-binary python-dotenv
```

---

## ⚙️ Configuration (.env)

Create a `.env` file in the root directory:

```
DB_HOST=your_ngrok_or_localhost
DB_PORT=5432
DB_NAME=your_db
DB_USER=your_user
DB_PASSWORD=your_password
```

---

## 🚀 Usage

```bash
python main.py
```

This will:
- Load the image from a URL
- Detect objects and extract the dominant color
- Save all results into PostgreSQL
- Visualize the image with bounding boxes and color info

---

## 📊 Example Output

- Detected objects (with bounding boxes)
- Dominant color (hex code like #ff0000)
- Results saved in `images` and `tags` tables in PostgreSQL

---

## 📈 PostgreSQL Tables

- `images`: Image path, width, height, dominant color
- `tags`: Object label, score, bounding box coordinates

---

## 🔐 Security

- The `.env` file is used to hide sensitive credentials.
- Add `.env` and `.venv/` to `.gitignore`.

---

## 📌 Notes

- You can change the `IMAGE_URL` to a dynamic or rotating image provider if needed.
- For remote DB access, use `ngrok tcp 5432` and update your `.env`.

---

## ✨ Future Ideas

- Add multiple dominant colors
- Web interface with Flask or Streamlit
- Store image result as base64 in DB
- Docker support

---
