let model;

// Load the TensorFlow Lite model and update the progress bar
async function loadModel() {
    const modelUrl = 'https://raw.githubusercontent.com/shuvo153853/skin-disease-classification/main/model.tflite';
    const progressFill = document.getElementById('progressFill');
    
    try {
        const response = await fetch(modelUrl);
        if (!response.ok) {
            throw new Error(`Failed to fetch model: ${response.statusText}`);
        }
        
        const reader = response.body.getReader();
        const contentLength = +response.headers.get('Content-Length');
        let receivedLength = 0;
        const chunks = [];
        
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            chunks.push(value);
            receivedLength += value.length;
            
            const progress = (receivedLength / contentLength) * 100;
            progressFill.style.width = `${progress}%`;
            console.log(`Loading model: ${progress.toFixed(2)}%`);
        }
        
        const arrayBuffer = new Uint8Array(receivedLength);
        let position = 0;
        for (let chunk of chunks) {
            arrayBuffer.set(chunk, position);
            position += chunk.length;
        }
        
        model = new tflite.Interpreter(arrayBuffer.buffer);
        await model.allocateTensors();
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

// Elements
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const openCamera = document.getElementById('openCamera');
const cameraInput = document.getElementById('cameraInput');
const fileInput = document.getElementById('fileInput');
const selectedImage = document.getElementById('selectedImage');
const classifyButton = document.getElementById('classifyButton');
const result = document.getElementById('result');
const loadingIndicator = document.createElement('div');

// Open the camera
openCamera.addEventListener('click', () => {
    cameraInput.click();
});

// Handle camera input
cameraInput.addEventListener('change', (event) => {
    handleImageInput(event);
});

// Handle file input
fileInput.addEventListener('change', (event) => {
    handleImageInput(event);
});

// Handle image input (camera or file)
function handleImageInput(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            selectedImage.src = e.target.result;
            selectedImage.style.display = 'block';
            classifyButton.style.display = 'block';
            console.log('Image loaded successfully');
        };
        reader.readAsDataURL(file);
    }
}

// Classify the image using the TensorFlow Lite model
classifyButton.addEventListener('click', () => {
    canvas.width = selectedImage.width;
    canvas.height = selectedImage.height;
    context.drawImage(selectedImage, 0, 0, selectedImage.width, selectedImage.height);
    
    // Show loading indicator
    loadingIndicator.innerText = 'Classifying...';
    result.appendChild(loadingIndicator);

    classifyImage(canvas); // Pass the image on the canvas for classification
});

async function classifyImage(image) {
    console.log('Classifying image...');
    try {
        const input = tf.browser.fromPixels(image).resizeNearestNeighbor([224, 224]).toFloat().expandDims(0);
        console.log('Input tensor shape:', input.shape);

        console.log('Running model...');
        model.run(input);
        console.log('Model run complete.');

        const outputTensor = model.getOutputTensor(0);
        const predictions = outputTensor.dataSync(); // Get the prediction data
        console.log('Predictions:', predictions);

        displayResult(predictions);
    } catch (error) {
        console.error('Error during classification:', error);
        result.innerText = 'An error occurred during classification. Please try again.';
    } finally {
        // Remove loading indicator
        loadingIndicator.remove();
    }
}

// Display the classification result
function displayResult(predictions) {
    const classLabels = [
        "Eczema", "Warts Molluscum", "Melanoma", "Basal Cell Carcinoma",
        "Melanocytic Nevi (NV)", "Benign Keratosis-like Lesions (BKL)",
        "Psoriasis pictures Lichen Planus", "Seborrheic Keratoses and other Benign Tumor",
        "Tinea Ringworm Candidiasis and other Fungal"
    ];
    const maxIndex = predictions.indexOf(Math.max(...predictions));
    const prediction = classLabels[maxIndex];
    const probability = predictions[maxIndex];

    const threshold = 0.6; // Define a threshold for confident predictions
    if (probability < threshold) {
        result.innerText = `The input image doesn't match any known categories.`;
    } else {
        result.innerText = `Prediction: ${prediction} (Confidence: ${(probability * 100).toFixed(2)}%)`;
    }
    result.style.display = 'block';
    console.log(`Prediction: ${prediction}, Probability: ${probability}`);
}

// Load the model on page load
window.onload = () => {
    loadModel();
};
