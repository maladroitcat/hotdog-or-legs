# 🌭 Hotdogs or Legs? Full-Stack ML Pipeline

This repo implements an end-to-end image classification system that answers the burning question:

> **Is this a picture of hotdogs or legs?**

The project covers:

- Data collection from the web (Bing Image Search)  
- Image cleaning & preprocessing  
- Hosting cleaned data in **Google Cloud Storage (GCS)**  
- A reproducible ML pipeline with **fastai** & **PyTorch**  
- Experiment tracking with **MLflow**  
- A containerized **FastAPI** model server (Docker)  
- Deployment to **Google Cloud Run**  
- A **Streamlit** front-end that calls the live Cloud Run API  

---

## 🗺️ Project Structure

```text
.
├─ README.md                    # This file
├─ requirements.txt             # Python dependencies
├─ config.yaml                  # Central config for data + training + paths
├─ Dockerfile                   # Container for training & serving
├─ create_local_dataset.py      # Helper to stage the dataset locally
├─ artifacts/
│   └─ hotdog_or_legs.pkl       # Exported fastai model (committed for reuse)
├─ data/
│   ├─ hotdog_legs_dataset.zip  # Downloaded archive from cloud storage
│   └─ hotdog_or_legs/          # Extracted dataset used for training
└─ src/
    ├─ api/
    │   └─ main.py              # FastAPI app (/health, /predict)
    ├─ data/
    │   └─ download_data.py     # Downloads dataset from GCS to local data dir
    ├─ frontend/
    │   └─ app.py               # Streamlit front-end
    ├─ models/
    │   ├─ train.py             # Training script (fastai + MLflow)
    │   └─ predict.py           # Model loading + predict_from_url helper
    └─ utils/
        └─ config.py            # Shared config helpers
````

---

## 1. Project Overview & Goals

This project builds a ResNet-18 image classifier that predicts whether an image from a public URL is a pair of hotdogs or a pair of legs. While we're all wondering *"Why hasn't this been picke up by the big MAANG companies yet!?"*, I must admit I created this model with the ulterior motive of **learning** (*gasp!*). So, while this technology is bound to change the world, feel free to use it as a learning tool yourself! It focuses on creating a reproducible ML pipeline, deploying a FastAPI model to Cloud Run, and integrating a Streamlit front-end; all things that will soon become obsolete after my state-of-the-art, hotdog-detecting technology goes full SKYNET...but it's still a fun little project to tool around with until then.

**Goals:** 
* Practice real-world data sourcing and cleaning 
* Host final data in **GCS** and ingest it programmatically 
* Build a **reproducible ML pipeline** controlled by config.yaml 
* Track experiments using **MLflow** 
* Containerize training and serving with **Containerization** (Docker) 
* Deploy a FastAPI model server to **Cloud Run** 
* Build a **Streamlit** front-end that calls the deployed API

---

## 2. Dataset: “Hotdog vs Legs”

### 2.1. Data Collection via Bing

Because there was no “hotdog vs legs” dataset publicly available (yes I am just as shocked as you are), I created a dataset from web images using Bing Image Search. The script in the repo root (`build_bing_dataset.py`) does the following:

* Uses `bing_image_downloader` to query Bing with separate query lists for:

  * `hotdog` (e.g. *“hotdogs that look like legs”*, *“tan hotdogs”*, etc.)
  * `legs` (e.g. *“legs sunbathing”*, *“legs on beach”*, etc.)

* Downloads raw images into:

  ```text
  local_data/hotdog_or_legs/raw_bing/<class>/
  ```

* Cleans & preprocesses images:

  * Converts all images to **RGB JPG**
  * Resizes images with `thumbnail` to a max edge size (e.g. 400px)
  * Drops corrupt / unreadable files

* Saves cleaned images into:

  ```text
  local_data/hotdog_or_legs/images/hotdog/*.jpg
  local_data/hotdog_or_legs/images/legs/*.jpg
  ```

To reconstruct the dataset locally:

```bash
python build_bing_dataset.py
```

> **Ethical note:** These images are scraped from public Bing search results. This dataset is used strictly for educational purposes. Some images may contain identifiable human legs (this was kind of hilarious to write but I'm sure you'd recognize your own legs if you saw them and wouldn't want them missused on the internet - so be responsible!); no images are redistributed as a separate dataset.

### 2.2. Uploading Clean Data to GCS

After building the cleaned dataset locally, I:

1. Zipped the cleaned folder:

   ```bash
   cd local_data
   zip -r hotdog_legs_dataset.zip hotdog_or_legs
   ```

2. Uploaded `hotdog_legs_dataset.zip` to a private GCS bucket:

   ```text
   gs://<PRIVATE_BUCKET>/hotdog_legs_dataset.zip
   ```

3. Configured `config.yaml` to point to this bucket and object.

If you want to use your own GCS bucket:

* Upload the zip to your bucket
* Update `config.yaml` with your bucket and blob name

**Alternatively** you can use the `hotdog_legs_dataset.zip` in this repo.

---

## 3. Reproducible ML Pipeline

### 3.1. Environment Setup

Clone the repo and create a virtual environment:

```bash
git clone https://github.com/maladroitcat/hotdog-or-legs.git
cd hotdot-or-legs

python -m venv env
source env/bin/activate  # (On Windows: env\Scripts\activate)
pip install -r requirements.txt
```

### 3.2. Data Ingestion from GCS

The data ingestion script reads `config.yaml`, downloads the zip from GCS (if needed), and extracts it to `data/hotdog_or_legs`.

Run:

```bash
python -m src.data.download_data
```

You should see logs like:

```text
[data] Downloading gs://<bucket>/hotdog_legs_dataset.zip -> data/hotdog_legs_dataset.zip
[data] Extracting data/hotdog_legs_dataset.zip -> data/hotdog_or_legs
[data] Extraction complete.
```

If you run it again, it will skip work if the file / extract dir already exists.

### 3.3. Training Script (fastai)

Training is handled by `src/models/train.py`. It:

1. Calls the data download helper (ensuring data is present)
2. Builds an `ImageDataLoaders` from folder structure (`hotdog` / `legs`)
3. Uses a **ResNet18** base model with transfer learning
4. Trains for `training.epochs` epochs
5. Evaluates accuracy on a validation set
6. Logs metrics & params to **MLflow**
7. Saves logs and metadata locally to the `./mlruns` directory
8. Exports the model to `artifacts/hotdog_or_legs.pkl`

Run:

```bash
python -m src.models.train
```

On success, you’ll see logs like:

```text
[train] Using images from: data/hotdog_or_legs/images
[train] Starting training for 3 epochs...
epoch     train_loss  valid_loss  accuracy  time
0         0.336583    0.385472    0.910714  ...
1         0.278272    0.507362    0.910714  ...
2         0.214624    0.482500    0.916667  ...
[train] Validation loss: 0.4825, accuracy: 0.9167
[train] Exported model to artifacts/hotdog_or_legs.pkl
```

---

## 4. Experiment Tracking with MLflow

This project uses **MLflow** with the file-based backend (local `./mlruns` directory).

Each training run logs:

* Params (batch size, epochs, image size, etc.)
* Metrics (validation loss, accuracy)
* The exported model artifact

### 4.1. Viewing MLflow UI

Run:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open:

```text
http://127.0.0.1:5000
```

> Note: `mlruns/` is **not** committed to git (it’s in `.gitignore`), since runs are reproducible and environment-specific.

---

## 5. Serving the Model as an API (FastAPI)

The model API is implemented in `src/api/main.py` with **FastAPI**.

### 5.1. Endpoints

* `GET /health`
  Returns `{"status": "ok"}` if the app is alive.

* `POST /predict`
  Request body:

  ```json
  {
    "image_url": "https://example.com/image.jpg"
  }
  ```

  Response body:

  ```json
  {
    "label": "hotdog",
    "probabilities": {
      "hotdog": 0.98,
      "legs": 0.02
    }
  }
  ```

Under the hood, `predict.py`:

* Loads `artifacts/hotdog_or_legs.pkl` with `fastai.learner.load_learner`
* Downloads the image URL
* Preprocesses it and calls `learn.predict`
* Returns the predicted label + class probabilities

### 5.2. Run the API Locally

From the project root:

```bash
uvicorn src.api.main:app --reload
```

Then visit:

* Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 5.3. Test `/predict` via curl

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://upload.wikimedia.org/wikipedia/commons/1/13/Hotdog_-_Evan_Swigart.jpg"}'
```

You should get back JSON with a label and probabilities.

---

## 6. Docker: Training & Serving

### 6.1. Dockerfile Overview

The `Dockerfile`:

* Uses `python:3.12-slim`
* Installs system deps (`build-essential`)
* Installs Python deps via `requirements.txt`
* Copies `config.yaml` and `src/`
* Copies `artifacts/` (so the model is baked into the image for serving)
* Sets `PYTHONPATH=/app`
* Default `CMD` runs uvicorn:

  ```dockerfile
  CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
  ```

### 6.2. Build the Image

```bash
docker build -t hotdog-legs:latest .
```

### 6.3. Run the API in Docker (local)

```bash
docker run -p 8000:8000 hotdog-legs:latest
```

Then test:

```bash
curl "http://127.0.0.1:8000/health"
```

and `/predict` as above.

### 6.4. Run Training in Docker

To run the training pipeline inside the same container image:

```bash
docker run hotdog-legs:latest python -m src.models.train
```

This will:

* Use `download_data.py` to pull data from GCS
* Train the model
* Log to MLflow (inside container)
* Export a model artifact in `/app/artifacts/hotdog_or_legs.pkl`

This gives you **entry points for both training and serving in Docker**.

---

## 7. Deployment to Google Cloud Run

### 7.1. Build & Push Image with Cloud Build

From project root:

```bash
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/<YOUR_PROJECT_ID>/<YOUR_REPO_NAME>/hotdog-legs:latest
```

Example:

```bash
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/hotdog-or-legs/hotdog-or-legs/hotdog-legs:latest
```

This:

* Zips your source
* Builds the Docker image on **Google’s amd64 infra**
* Pushes it to **Artifact Registry**

> This avoids ARM vs amd64 issues on Apple Silicon.

### 7.2. Deploy to Cloud Run

```bash
gcloud run deploy hotdog-legs-api \
  --image us-central1-docker.pkg.dev/<PROJECT_ID>/<REPO_NAME>/hotdog-legs:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory=1Gi
```

We explicitly set `--memory=1Gi` because PyTorch, fastai, and model load exceeded the default 512Mi.

On success, it prints a URL like:

```text
https://hotdog-legs-api-<random>.us-central1.run.app
```

### 7.3. Test the Live API

Health:

```bash
curl "https://hotdog-legs-api-<random>.us-central1.run.app/health"
```

Predict:

```bash
curl -X POST "https://hotdog-legs-api-<random>.us-central1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://upload.wikimedia.org/wikipedia/commons/1/13/Hotdog_-_Evan_Swigart.jpg"}'
```

---

## 8. Streamlit Front-End

The front-end is built with **Streamlit** in `frontend/app.py`.

### 8.1. Configure API URL

In `frontend/app.py`, set:

```python
API_BASE_URL = "https://hotdog-legs-api-<random>.us-central1.run.app"
PREDICT_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"
```

### 8.2. Run Front-End Locally

Install Streamlit if needed:

```bash
pip install streamlit
```

Then:

```bash
streamlit run frontend/app.py
```

This app:

* Checks the backend `/health`
* Lets you paste an image URL
* Shows an image preview
* Sends a POST request to `/predict`
* Displays the predicted label + probabilities

### 8.3. Deploy Front-End (Streamlit Cloud)

1. Push your repo to GitHub (with `frontend/app.py` and `streamlit` in `requirements.txt`).
2. Go to [https://share.streamlit.io](https://share.streamlit.io).
3. Create a new app:

   * Repo: `<your github repo>`
   * Branch: `main`
   * Main file: `frontend/app.py`
4. Deploy.

You’ll get a public URL like:

```text
https://hotdog-legs-frontend-<username>.streamlit.app
```

Add that link here:

* **Deployed API:** `https://hotdog-legs-api-<random>.us-central1.run.app`
* **Deployed Front-End:** `https://hotdog-legs-frontend-<username>.streamlit.app`

---

## 9. Troubleshooting Guide

### 9.1. Common Local Issues

**Issue:** `ModuleNotFoundError: No module named 'fastai'`
**Fix:** Ensure your venv is active and run:

```bash
pip install -r requirements.txt
```

---

**Issue:** `FileNotFoundError: Model file not found at artifacts/hotdog_or_legs.pkl`
**Fix:**

* Run training first: `python -m src.models.train`, OR
* Ensure `artifacts/hotdog_or_legs.pkl` is present (if you commit it).

---

**Issue:** `data/hotdog_or_legs/images` not found when training
**Fix:**

1. Run `python -m src.data.download_data`
2. Validate `config.yaml` paths match your bucket and local dirs.

---

### 9.2. Docker / Cloud Run Issues

**Issue:** `Container failed to start and listen on port 8080`

* Make sure the Docker `CMD` uses the Cloud Run `PORT` environment variable:

  ```dockerfile
  CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
  ```

* Ensure you **rebuild and push** the image after changing the Dockerfile.

---

**Issue:** `exec format error: failed to load /usr/bin/sh` on Cloud Run

* This is almost always an ARM vs amd64 issue on Apple Silicon.
* Fix: use **Cloud Build**:

  ```bash
  gcloud builds submit \
    --tag us-central1-docker.pkg.dev/<PROJECT_ID>/<REPO_NAME>/hotdog-legs:latest
  ```

This builds for the correct architecture automatically.

---

**Issue:** `Memory limit of 512 MiB exceeded`

* Increase memory on Cloud Run:

  ```bash
  gcloud run deploy hotdog-legs-api \
    --image us-central1-docker.pkg.dev/<PROJECT_ID>/<REPO_NAME>/hotdog-legs:latest \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --memory=1Gi
  ```

---

**Issue (frontend):** `Read timed out` when calling backend from Streamlit

* Backend might be cold-starting and taking longer than 5 seconds

* Increase timeouts in `frontend/app.py`:

  ```python
  resp = requests.get(HEALTH_URL, timeout=15)
  resp = requests.post(PREDICT_URL, json=payload, timeout=30)
  ```

* Also test backend directly with `curl` to confirm it’s healthy.

---

## 10. Limitations & Ethical Considerations

* The dataset is scraped from public image search; labels are **noisy** and may contain biases.
* Images of human legs may involve identifiable individuals; this project is used strictly for academic purposes.
* Model performance is not the primary goal; the focus is on **MLOps, reproducibility, and deployment**.
* No fairness guarantees are provided; this classifier should **not** be used in any real-world decision-making.

---

## 11. AI Assistance Disclosure

Portions of this project, such as code debugging and documentation, including the formatting of this README, were developed with assistance from ChatGPT (gpt-5.1 Thinking). All generated code and text were reviewed, tested, and adapted for this specific assignment. Specifics about AI use are outlined below and can be found within code comments as well:

- `config.yaml`: AI aided in identifying the initial keys for formatting for the `training` section.

- `src/data/download_data.py` (lines 27–30): Formatting and debugging the download, bucket, and blob hadling in the `download-from-gcs` function.

- `src/models/train.py` (lines 54–60): Setup of the fastai `ImageDataLoaders` from the folder structure and configuration of the ResNet18-based learner using values read from `config.yaml`.

- `src/models/train.py` (lines 118–125): Debugged MLflow logging block that records hyperparameters, validation metrics (loss, accuracy), and logs the exported model artifact path.

- `src/models/predict.py` (lines 48–49): In `predict_from_url` formatting of a dict with `pred_label` and `probs`.

- `src/api/main.py` (lines 1–5): The `/health` endpoint returning a simple JSON status was generated by AI.

- `Dockerfile` (line 26): Formatted final `CMD` definition using `uvicorn src.api.main:app` and honoring the `PORT` environment variable with `${PORT:-8000}` to be compatible with Cloud Run.

- `frontend/app.py` (e.g., lines 10–14): Streamlit page layout, including `st.set_page_config`, the title, and initial health check to the `/health` endpoint of the Cloud Run service.

- `frontend/app.py` : Emoji generation and text formatting into markdown were performed by AI. The content of the text, however, is original. I am nothing if not a prolific goober so it will be a warm day in Antarctica before I let AI take credit for my jokes.
