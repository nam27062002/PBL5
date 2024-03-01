import requests
from PIL import Image
import io

url = 'http://127.0.0.1:8000/predict/'
def preprocess_image(image_path):
    image = Image.open(image_path)
    image_resized = image.resize((64, 64))
    image_io = io.BytesIO()
    image_resized.save(image_io, format='JPEG', quality=70)
    image_io.seek(0)
    return image_io

processed_image = preprocess_image('../Training/DATASET/asl_alphabet_test/D_test.jpg')
response = requests.post(url, files={'file': processed_image})
print(response.json())
