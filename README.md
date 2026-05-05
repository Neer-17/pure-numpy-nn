# Pytorch Version 🔍

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
- **Optimizer** — Mini-batch SGD with batch size 128/256
- **Weight initialization** — He initialization
- **Test accuracy** — 92.6%


## Project Structure

```
pure-numpy-nn/
├── app.py            ← Streamlit web app (DigitDetective)
├── neuralnet.py      ← Neural network class 
├── neuralnet.ipynb   ← Training notebook
├── model_emnist.pth       ← Saved trained weights
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

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Neural Network |
| `streamlit` | Web app framework |
| `streamlit-drawable-canvas` | Drawing canvas component |

---
