# Hum to Search 🎵🔍

This repository contains an end-to-end Machine Learning system designed to accurately match a raw, human audio recording (a hum, whistle, or singing) to the original music track. This backend is primarily powered by **PyTorch** and served dynamically via **FastAPI**.

## 🎥 Proof of Work / Demo

Watch the full system in action (Frontend + WebGL + Backend + PyTorch):

![Demo Video](./proof_demo.gif)

*(Note: The above is a GIF preview. **[Click Here to watch the full 1080p Video WITH Audio!](./proof_of_completion.mp4)**)*

## 🧠 Deep Learning Architecture

The core of the "Hum to Search" capability revolves around a **Siamese Neural Network**. Because human humming lacks the instrumental complexity of a studio track but shares the same underlying melody and rhythm, the network maps these two vastly different acoustic environments into a shared continuous latent space.

### 1. Feature Engineering (`src/data/preprocess.py`)
Raw audio waveforms (sampled at 22050 Hz) are mathematically transformed into two distinct feature representations before entering the neural network:
*   **Mel-Spectrograms:** Captures the rich, frequency-domain energy of the audio (spatial representation).
*   **Pitch Contours (F0):** Extracted using Librosa's `pyin` algorithm to isolate the raw vocal melody and temporal time-series progression, actively filtering out noise.

### 2. Network Topology (`src/models/`)
The underlying feature extractors utilize a dual-branch configuration:
*   **CNN Branch (`cnn_branch.py`):** A Convolutional Neural Network analyzes the 2D feature maps from the Mel-spectrograms to isolate local acoustic textures.
*   **LSTM Branch (`lstm_branch.py`):** Recurrent layers digest the sequential pitch tracking over time, observing the length and intervals between musical notes.
*   **Siamese Fusion (`siamese.py`):** The isolated embeddings from the CNN and LSTM branches are fused and passed through linear projection layers to output a singular, high-dimensional vector.

### 3. Training & Embedding
The model learns via **Metric Learning**. During the forward pass (`train.py`), matching pairs of (Hum $\leftrightarrow$ Song) are pulled closer together in the vector space, while mismatching pairs are pushed apart (usually utilizing Contrastive Loss or Triplet Margin Loss). Over time, the model ignores instrumentation and specifically zeroes in on melodic shape.

---

## ⚡ Inference & Vector Search

Instead of classifying audio natively, the system utilizes a **k-NN (k-Nearest Neighbors) Vector Search** approach:
1. **Pre-Indexing (`create_index.py`):** All original reference songs are chunked into 16-second segments, passed through the model, and stored as a tensor database dictionary (`data/song_index.pt`).
2. **Matching (`inference.py`):** During inference, a user's hum is embedded by the model. We perform a highly optimized **Cosine Similarity** operation using PyTorch against thousands of pre-calculated song chunks to find the closest angular distance.

---

## 🚀 Fast API Production Subsystem

To make this model interactable from a user interface, it is wrapped in an async **FastAPI** environment. 

### Core features:
*   **Hardware Agnostic:** Automatically checks `torch.cuda.is_available()` to seamlessly route matrix multiplications on a GPU if detected.
*   **Stateless Inference:** The network's frozen weights (`models/best_hum_model.pth`) are kept resident in memory.
*   **CORS & Blob Handling:** Handles raw binary chunking, storing `multipart/form-data` uploads into the temporary files instantly passed to the Librosa audio pipeline.

### API Routing
*   `POST /` : Accepts an `UploadFile` (the `.wav` hum audio blob). Runs the Siamese model -> calculates cosine threshold -> returns `{"prediction": "Song Title"}`.
*   `GET /` : Lightweight cache fetch to retrieve the `last_prediction`.

---

## 📦 Setup & Execution

### Installation
Ensure you have Python 3.9+ installed and navigate to the backend directory:
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Running the API
```bash
uvicorn app:app --port 8000 --reload
```
Once booted, the backend operates as a headless microservice, passively listening for audio blobs to embed and search.
