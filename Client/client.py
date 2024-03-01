import requests
from PIL import Image
import io
import os
import time
url = 'http://127.0.0.1:8000/predict/'
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
def find_jpg_files():
    folder_path = "../Training/DATASET/asl_alphabet_train"
    jpg_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".jpg"):
                jpg_files.append(os.path.join(root, file))
    return jpg_files
def get_parent_folder(file_path):
    parent_folder_path = os.path.dirname(file_path)
    parent_folder_name = os.path.basename(parent_folder_path)
    return parent_folder_name

def preprocess_image(image_path):
    image = Image.open(image_path)
    image_resized = image.resize((64, 64))
    image_io = io.BytesIO()
    image_resized.save(image_io, format='JPEG', quality=70)
    image_io.seek(0)
    return image_io

total_response_time = 0
num_requests = 0

all_files = find_jpg_files()

count = 0
for file in all_files:
    processed_image = preprocess_image(file)
    start_time = time.time()
    response = requests.post(url, files={'file': processed_image})
    end_time = time.time()
    response_time = end_time - start_time
    total_response_time += response_time
    num_requests += 1
    print(f"Response time for {file}: {response_time} seconds")
    print(response)
    break
