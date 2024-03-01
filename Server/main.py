from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = FastAPI()

model = load_model('asl_vgg16_best_weights.h5')

def preprocess_image(image):
    img = image.resize((64, 64))
    img = img_to_array(img) / 255.
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/api/v1/predict-sign-language/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    processed_image = preprocess_image(image)
    predictions = await get_predictions(processed_image)
    predicted_label = np.argmax(predictions[0])
    return {"predicted_label": int(predicted_label)}

async def get_predictions(image):
    return model.predict(image)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=8)
