const express = require('express');
const multer = require('multer');
const fs = require('fs').promises; // Import fs with promises
const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');

const app = express();
const upload = multer();

// Define a top-level async function to load the model
const loadModel = async () => {
    // Load the model
    const model = await tf.loadLayersModel('file://model.json');
    return model;
};

// Preprocess image
const preprocessImage = async (file) => {
    const buffer = await sharp(file.buffer)
        .resize(64, 64)
        .toBuffer();
    const img = tf.node.decodeImage(buffer, 3);
    return img.toFloat().div(tf.scalar(255));
};

// Predict route
app.post('/predict', upload.single('file'), async (req, res) => {
    try {
        // Load the model
        const model = await loadModel();

        // Preprocess image
        const image = await preprocessImage(req.file);

        // Make predictions
        const predictions = model.predict(image.expandDims());
        const predictedLabel = predictions.argMax(1).dataSync()[0];

        // Send response
        res.json({ predicted_label: predictedLabel });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'An error occurred' });
    }
});

// Start server
const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
