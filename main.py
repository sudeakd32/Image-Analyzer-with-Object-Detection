import io
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import functional as F
from sklearn.cluster import KMeans
from dotenv import load_dotenv
import os
import psycopg2

# --- PART 1: IMAGE ANALYZER CLASS ---
# This class will handle all Computer Vision operations.
class ImageAnalyzer:

    def __init__(self):
        # Load a powerful, pre-trained object detection model from Torchvision.
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        # Set the model to evaluation mode.
        self.model.eval()

        # The list of object names that the model can recognize (COCO dataset).
        self.COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def load_image_from_url(self, url):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            return image
        except requests.exceptions.RequestException as e:
            print(f"Error: Could not load image from URL. {e}")
            return None

    def analyze_dominant_color(self, image: Image.Image, k=1) -> str:

        # Convert PIL image to OpenCV format (numpy array)
        open_cv_image = np.array(image)
        # Convert color channels from RGB to BGR (used by OpenCV)
        img_bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # Resize the image for faster processing
        img_small = cv2.resize(img_bgr, (100, 100))

        # Reshape the image to be a list of pixels (each row is a pixel's BGR value)
        pixels = img_small.reshape((-1, 3))

        # Find the most dominant color (cluster center) using KMeans algorithm
        clt = KMeans(n_clusters=k, n_init='auto', random_state=42)
        clt.fit(pixels)
        dominant_color_bgr = clt.cluster_centers_[0].astype(int)

        # Convert the color back from BGR to RGB and then to a hex string
        dominant_color_hex = "#{:02x}{:02x}{:02x}".format(dominant_color_bgr[2], dominant_color_bgr[1], dominant_color_bgr[0])
        return dominant_color_hex

    def detect_objects(self, image: Image.Image, score_threshold=0.5):

        # Convert the image to a format the model understands (Tensor)
        image_tensor = F.to_tensor(image).unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Convert the model's output to a more user-friendly format
        results = []
        for i in range(len(predictions[0]['labels'])):
            score = predictions[0]['scores'][i].item()
            if score > score_threshold:
                label_id = predictions[0]['labels'][i].item()
                label_name = self.COCO_INSTANCE_CATEGORY_NAMES[label_id]
                bbox = predictions[0]['boxes'][i].cpu().numpy().tolist()
                results.append({
                    "label": label_name,
                    "score": score,
                    "bbox": [int(coord) for coord in bbox] # [x1, y1, x2, y2]
                })
        return results

    @staticmethod
    def draw_detections(image: Image.Image, detections):

        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw, "RGBA")
        try:
            font = ImageFont.truetype("Arial.ttf", size=max(15, img_draw.width // 50))
        except IOError:
            font = ImageFont.load_default()

        line_width = max(2, img_draw.width // 250)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            score = det['score']
            text = f"{label} {score:.2f}"

            # Draw the bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)

            # Draw the background box and text for the label
            text_bbox = draw.textbbox((x1, y1 - 2), text, font=font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.rectangle([x1, y1 - text_h - 5, x1 + text_w + 4, y1], fill=(255, 0, 0, 180))
            draw.text((x1 + 2, y1 - text_h - 3), text, fill="white", font=font)

        return img_draw

# --- PART 2: DATABASE MANAGER CLASS ---
# This class manages all PostgreSQL operations.
class DatabaseManager:

    def __init__(self, db_params):
        self.db_params = db_params
        self.connection = None
        self.cursor = None

    def connect(self):
        try:
            self.connection = psycopg2.connect(**self.db_params)
            self.cursor = self.connection.cursor()
            print(f"Successfully connected to the PostgreSQL database!")
            return True
        except psycopg2.OperationalError as e:
            print(f"ERROR: Could not connect to the database. Please check your credentials.")
            print(f"Detail: {e}")
            return False

    def disconnect(self):
        """Closes the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("🔌 Database connection closed.")

    def setup_tables(self):
        commands = (
            """
            CREATE TABLE IF NOT EXISTS images (
                id SERIAL PRIMARY KEY,
                file_path VARCHAR(255) NOT NULL,
                width INTEGER,
                height INTEGER,
                dominant_color_hex VARCHAR(7),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS tags (
                id SERIAL PRIMARY KEY,
                image_id INTEGER NOT NULL,
                tag VARCHAR(50) NOT NULL,
                score REAL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                FOREIGN KEY (image_id)
                    REFERENCES images (id)
                    ON DELETE CASCADE
            )
            """
        )
        try:
            for command in commands:
                self.cursor.execute(command)
            self.connection.commit()
            print(f"Tables created successfully or already exist.")
        except Exception as e:
            print(f"ERROR: An error occurred while creating tables: {e}")
            self.connection.rollback()

    def insert_image_data(self, image_path, width, height, dominant_color, detections):
        """Inserts the analyzed image data and its tags into the database."""
        sql_insert_image = """
            INSERT INTO images (file_path, width, height, dominant_color_hex)
            VALUES (%s, %s, %s, %s) RETURNING id;
        """
        sql_insert_tag = """
            INSERT INTO tags (image_id, tag, score, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        try:
            # 1. Insert image info and get the new 'id'
            self.cursor.execute(sql_insert_image, (image_path, width, height, dominant_color))
            image_id = self.cursor.fetchone()[0]

            # 2. Insert each detection (tag) into the 'tags' table
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                tag_data = (image_id, det['label'], det['score'], x1, y1, x2, y2)
                self.cursor.execute(sql_insert_tag, tag_data)

            # Commit all transactions
            self.connection.commit()
            print(f" Image and {len(detections)} tags were successfully added to the database (ID: {image_id}).")

        except Exception as e:
            print(f" ERROR: An error occurred during database insertion: {e}")
            self.connection.rollback() # Roll back all changes if an error occurs

# --- PART 3: MAIN EXECUTION BLOCK ---
def main():

    IMAGE_URL = 'https://picsum.photos/600/400'

    load_dotenv()

    DB_PARAMS = {
        "host": os.getenv('DB_HOST'),
        "port": os.getenv('DB_PORT'),
        "dbname": os.getenv('DB_NAME'),
        "user": os.getenv('DB_USER'),
        "password": os.getenv('DB_PASSWORD')
    }

    # 1. Initialize the Analyzer and Database Manager classes
    analyzer = ImageAnalyzer()
    db_manager = DatabaseManager(db_params=DB_PARAMS)

    # 2. Load the image
    print(f"Loading image: {IMAGE_URL}")
    image = analyzer.load_image_from_url(IMAGE_URL)
    if image is None:
        return # Stop the program if the image cannot be loaded

    # 3. Analyze the image
    print("Analyzing dominant color...")
    dominant_color = analyzer.analyze_dominant_color(image)
    print(f"-> Dominant Color: {dominant_color}")

    print("Detecting objects...")
    detections = analyzer.detect_objects(image, score_threshold=0.6)
    print(f"-> Found {len(detections)} objects.")

    # 4. Connect to the database and save the data
    if db_manager.connect():
        db_manager.setup_tables()
        db_manager.insert_image_data(
            image_path=IMAGE_URL,
            width=image.width,
            height=image.height,
            dominant_color=dominant_color,
            detections=detections
        )
        db_manager.disconnect()

    # 5. Visualize the results
    print("Generating result image...")
    result_image = ImageAnalyzer.draw_detections(image, detections)

    plt.figure(figsize=(12, 8))
    plt.imshow(result_image)
    plt.title(f"Detected Objects & Dominant Color: {dominant_color}")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()