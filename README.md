---
title: 'PicMatch: Your Visual Search Companion'
emoji: 📷🔍
colorFrom: blue
colorTo: green
sdk: gradio
python_version: 3.9
sdk_version: 4.39.0
suggested_hardware: t4-small
suggested_storage: medium
app_file: app.py
fullWidth: true
header: mini
short_description: Search images using text or other images as queries.
models:
- wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M
- Salesforce/blip-image-captioning-base


tags:
- image search
- visual search
- image processing
- CLIP
- image captioning
thumbnail: https://example.com/thumbnail.png
pinned: true
hf_oauth: false
disable_embedding: false
startup_duration_timeout: 30m
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin

---

# 📸 PicMatch: Your Visual Search Companion 🔍

PicMatch lets you effortlessly search through your image archive using either a text description or another image as your query.  Find those needle-in-a-haystack photos in a flash! ✨

Try PicMatch image search with 25,000 Unsplash images on this [🤗 Space](https://huggingface.co/spaces/satishjasthij/PicMatch)

## 🚀 Getting Started:

1. **Prerequisites:** Ensure you have Python 3.9 or higher installed on your system. 🐍

2. **Create a Virtual Environment:**
   ```bash
   python -m venv env
   ```

3. **Activate the Environment:**
   ```bash
   source ./venv/bin/activate 
   ```

4. **Install Dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   ```

5. **Start the App (with Sample Data):**
   ```bash
   python app.py
   ```

6. **Open Your Browser:**  Head to `localhost:7860` to access the PicMatch interface. 🌐

## 📂 Data: Organize Your Visual Treasures 

Make sure you have the following folders in your project's root directory:

```
data
├── images   
└── features
```

## 🛠️ Image Pipeline: Download & Process with Speed ⚡

The `engine/download_data.py` Python script streamlines downloading and processing images from a list of URLs. It's designed for performance and reliability:

- **Async Operations:**  Uses `asyncio` for concurrent image downloading and processing. ⏩
- **Rate Limiting:**  Follows API usage rules to prevent blocks with a `RateLimiter`. 🚦
- **Parallel Resizing:**  Employs a `ProcessPoolExecutor` for fast image resizing. ⚙️
- **State Management:**  Saves progress in a JSON file so you can resume later. 💾

### 🗝️ Key Components:

- **`ImagePipeline` Class:** Manages the entire pipeline, its state, and rate limiting. 🎛️
- **Functions:** Handle URL feeding (`url_feeder`), downloading (`image_downloader`), and processing (`image_processor`). 📥
- **`ImageSaver` Class:** Defines how images are processed and saved. 🖼️
- **`resize_image` Function:**  Ensures image resizing maintains the correct aspect ratio. 📏

### 🏃 How it Works:

1. **Start:** Configure the pipeline with your URL list, download limits, and rate settings. 
2. **Feed URLs:** Asynchronously read URLs from your file. 
3. **Download:** Download images concurrently while respecting rate limits. 
4. **Process:** Save the original images and resize them in parallel. 
5. **Save State:**  Regularly save progress to avoid starting over if interrupted. 

To get the sample data run the command
```bash
cd engine && python download_data.py
```

## ✨ Feature Creation: Making Your Images Searchable ✨

This step prepares your images for searching.  We generate two types of embeddings:

- **Visual Embeddings (CLIP):** Capture the visual content of your images. 👁️‍🗨️ 
- **Textual Embeddings:** Create embeddings from image captions for text-based search. 💬

To generate these features run the command 
```bash
cd engine && python generate_features.py
```
This process uses these awesome models from Hugging Face:

- TinyCLIP: `wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M` 
- BLIP Image Captioning: `Salesforce/blip-image-captioning-base`
- SentenceTransformer: `all-MiniLM-L6-v2` 

## ⚡ Asynchronous Feature Extraction: Supercharge Your Process ⚡

This script extracts image features (both visual and textual) efficiently:

- **Asynchronous:**  Loads images, extracts features, and saves them concurrently. ⚡
- **Dual Embeddings:** Creates both CLIP (visual) and caption (textual) embeddings. 🖼️📝
- **Checkpoints:** Keeps track of progress and allows resuming from interruptions. 🔄
- **Parallel:** Uses multiple CPU cores for feature extraction. ⚙️


## 📊 Vector Database Module: Milvus for Fast Search 🚤

This module connects to the Milvus vector database to store and search your image embeddings:

- **Milvus:**  A high-performance database built for handling vector data. 📊
- **Easy Interface:**  Provides a simple way to manage embeddings and perform searches. 🔍
- **Single Server:**  Ensures only one Milvus server is running for efficiency. 
- **Indexing:** Automatically creates an index to speed up your searches. 🚀
- **Similarity Search:** Find the most similar images using cosine similarity. 💯



## 📚 References: The Brains Behind PicMatch 🧠

PicMatch leverages these incredible open-source projects:

- **TinyCLIP:**  The visual powerhouse for understanding your images.  
  - 👉 [https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M](https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M)

- **Image Captioning:** The wordsmith that describes your photos in detail. 
  - 👉 [https://huggingface.co/Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)

- **Sentence Transformers:** Turns captions into embeddings for text-based search. 
  - 👉 [https://sbert.net](https://sbert.net)

- **Unsplash:** Images used were taken from Unsplash's open source data
   - 👉 [https://github.com/unsplash/datasets](https://github.com/unsplash/datasets)

Let's give credit where credit is due! 🙌 These projects make PicMatch smarter and more capable. 
