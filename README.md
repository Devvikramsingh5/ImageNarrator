# ImageNarrator 🖼️

An image captioning system that automatically generates natural language descriptions for images. Built with a CNN + LSTM architecture — Xception extracts visual features, and an LSTM-based decoder generates the caption word by word.

---

## How It Works

1. **Feature Extraction** — Xception (pretrained on ImageNet, no top layer) processes each image into a 2048-dimensional feature vector.
2. **Caption Generation** — An LSTM decoder takes the image features and a partial caption sequence as input, predicting the next word at each step.
3. **Inference** — Given a new image, the model generates a caption token by token, starting from `<start>` and stopping at `<end>`.

### Model Architecture

```
Image (2048,)  →  Dropout → Dense(256)  ─┐
                                          ├─ Add → Dense(256) → Dense(vocab_size, softmax)
Sequence       →  Embedding → Dropout → LSTM(256) ─┘
```

---

## Dataset

Trained on the [Flickr8k dataset](https://illinois.edu/fb/sec/1713398) — 8,091 images each paired with 5 human-written captions.

> The dataset files (`Flicker8k_Dataset/`, `Flickr8k_text/`) are not included in this repo due to size. Download them separately and place them in the project root.

---

## Project Structure

```
ImageNarrator/
├── main.py             # Data preprocessing, feature extraction, model training
├── test.py             # Inference script — generate a caption for any image
├── descriptions.txt    # Cleaned captions used for training
├── tokenizer.pkl       # Fitted Keras tokenizer (saved vocabulary)
├── models/             # Saved model weights per epoch (not tracked in git)
└── .gitignore
```

---

## Setup

### Prerequisites

- Python 3.10+
- TensorFlow 2.x

### Install dependencies

```bash
python -m venv tf-env
tf-env\Scripts\activate        # Windows
pip install tensorflow pillow tqdm matplotlib
```

---

## Training

1. Download the Flickr8k dataset and place `Flicker8k_Dataset/` and `Flickr8k_text/` in the project root.
2. Run `main.py` to preprocess captions, extract features, and train the model:

```bash
python main.py
```

Model weights are saved after each epoch to `models/model_0.h5` through `models/model_9.h5`.

---

## Inference

Generate a caption for any image using the trained model:

```bash
python test.py --image path/to/your/image.jpg
```

Example output:
```
a dog is running through the grass
```

---

## Tech Stack

| Component | Library |
|---|---|
| Image feature extraction | Xception (Keras) |
| Sequence modeling | LSTM (Keras) |
| Deep learning framework | TensorFlow 2.x |
| Image processing | Pillow |

---

## Notes

- Trained for 10 epochs with 50 steps per epoch.
- Vocabulary size and max caption length are derived automatically from the training set.
- Model weights (~57MB each) are excluded from the repo. Host them on Google Drive or Hugging Face and link here if sharing.
