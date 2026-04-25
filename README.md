# pure-numpy-nn 🔍

A handwritten digit recognizer built **entirely from scratch using only NumPy** — no PyTorch, no TensorFlow, no deep learning frameworks of any kind. Every component including forward propagation, backpropagation, and weight updates is implemented manually.

The project includes a **Streamlit web app** called DigitDetective where users can draw a digit on a canvas and the model predicts what number it is.

---

## Demo

Draw any digit (0–9) on the canvas and hit **Predict**. The app shows the predicted digit along with confidence scores for all 10 classes.

---

## How It Works

The neural network is a fully connected feedforward network trained on the EMNIST Digits dataset:

- **Input layer** — 784 neurons (28×28 flattened image)
- **Hidden layers** — 2 layers of 128 neurons each with ReLU activation
- **Output layer** — 10 neurons with Softmax activation (one per digit)

### Training Details
- **Dataset** — EMNIST Digits (280,000 samples)
- **Loss function** — Cross-entropy loss
- **Optimizer** — Mini-batch SGD with batch size 128
- **Weight initialization** — He initialization
- **Test accuracy** — 97.8%

### What's Implemented From Scratch
- Forward pass with ReLU and numerically stable Softmax
- Backpropagation with ReLU gradient gating
- Cross-entropy loss
- Mini-batch gradient descent with shuffling
- He weight initialization
- Weight save/load using NumPy's `.npz` format

---

## Image Preprocessing Pipeline

Raw canvas drawings are preprocessed to match EMNIST format before prediction:

1. Convert RGBA canvas output to grayscale
2. Binary threshold to pure black and white
3. Find largest contour (the drawn digit)
4. Crop to bounding box
5. Add 20% padding on all sides
6. Resize to 28×28 using `cv2.INTER_AREA`
7. Threshold again to remove resize artifacts
8. Normalize pixel values to 0–1
9. Flatten to `(1, 784)`

---

## Project Structure

```
pure-numpy-nn/
├── app.py            ← Streamlit web app (DigitDetective)
├── neuralnet.py      ← Neural network class (pure NumPy)
├── neuralnet.ipynb   ← Training notebook
├── weights.npz       ← Saved trained weights
├── biases.npz        ← Saved trained biases
└── README.md
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Neer-17/pure-numpy-nn.git
cd pure-numpy-nn
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app loads the pretrained weights automatically — no training required.

---

## Training Your Own Model

If you want to retrain from scratch, download the EMNIST Digits dataset from [Kaggle](https://www.kaggle.com/datasets/crawford/emnist) and run the `neuralnet.ipynb` notebook. The CSV format is expected with 785 columns — first column is the label, remaining 784 are pixel values.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Everything — the entire neural network |
| `opencv-python` | Image preprocessing |
| `streamlit` | Web app framework |
| `streamlit-drawable-canvas` | Drawing canvas component |

---

## Key Takeaway

This project proves that modern deep learning fundamentals don't require a framework. Every gradient, every weight update, every activation function is just mathematics expressed in NumPy.