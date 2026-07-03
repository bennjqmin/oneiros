# Oneiros

Oneiros is a visual machine learning IDE. Build neural networks on a node canvas, preprocess tabular and image datasets, train models (PyTorch or XGBoost), and export results — all from the browser.

## Features

- **Model editor** — drag-and-drop layers (CNN, RNN, LSTM, Transformer, pretrained backbones, and more)
- **Training** — live metrics over WebSocket, learning-rate schedulers, model export (`.pt`, ONNX)
- **Datasets** — CSV/JSON tabular pipelines, EDF biosignal processing (MNE), image folders (ZIP) with augmentation
- **XGBoost** — classification and regression from the Model page
- **Projects** — multiple saved projects in the browser

---

## Prerequisites

| Tool | Version |
|------|---------|
| Node.js | 20+ recommended |
| npm | 10+ (comes with Node) |
| Python | 3.11+ recommended |
| pip | latest |

Optional:

- **CUDA** — GPU training (PyTorch uses CPU if no GPU is available)
- **MNE-Python** — EDF / EEG import (`pip install mne`)
- **OpenAI API key** — AI assistant in the Model page (set in the UI or via env when using Docker)

---

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/oneiros.git
cd oneiros
```

### 2. Frontend

```bash
npm install
```

### 3. Backend

Create a virtual environment (recommended), then install dependencies:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Optional extras**

```bash
pip install mne          # EDF biosignal support
```

The first training run may download built-in datasets (MNIST, CIFAR-10, etc.) into `backend/data/`.

### 4. Run the app

From the repo root, start both services:

```bash
npm run dev:all
```

Or run them separately in two terminals:

```bash
# Terminal 1 — frontend (Vite)
npm run dev

# Terminal 2 — backend (FastAPI)
npm run dev:backend
```

| Service | URL |
|---------|-----|
| UI | http://localhost:5173 |
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |

The frontend expects the API at `http://localhost:8000`. CORS is configured for `http://localhost:5173`.

### 5. Production build (frontend only)

```bash
npm run build
npm run preview
```

You still need the backend running separately for training and dataset features.

---

## Project layout

```
oneiros/
├── src/                 # React + TypeScript UI
├── backend/
│   ├── main.py          # FastAPI entrypoint
│   ├── trainer.py       # PyTorch training loop
│   ├── compiler.py      # Graph → nn.Module
│   ├── datasets.py      # MNIST / CIFAR / custom loaders
│   ├── cv_dataset.py    # Image folder + augmentation
│   ├── edf_processor.py # EDF / MNE pipeline
│   └── requirements.txt
└── package.json
```

---

## Example Docker Compose (reference only)

This repository does **not** ship a `docker-compose.yml`. The example below is for reference — copy and adapt it if you want containerized development or deployment.

You will also need Dockerfiles (examples included). Adjust paths, env vars, and CORS in `backend/main.py` if the UI is served from a host other than `localhost:5173`.

### Example `docker-compose.yml`

```yaml
services:
  api:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
    volumes:
      - ./backend/data:/app/data
      - cv_uploads:/tmp/oneiros_cv_
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/datasets"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  ui:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "5173:5173"
    environment:
      - VITE_API_URL=http://localhost:8000
    depends_on:
      api:
        condition: service_healthy

volumes:
  cv_uploads:
```

### Example `backend/Dockerfile`

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir mne

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Example `Dockerfile.frontend` (repo root)

```dockerfile
FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
```

### Using the example

```bash
# From repo root (after creating the Dockerfiles above)
docker compose up --build
```

Then open http://localhost:5173. The API is at http://localhost:8000.

**Notes**

- PyTorch wheels are large; the first `docker compose build` can take several minutes.
- GPU training inside Docker requires NVIDIA Container Toolkit and a CUDA-enabled base image — not shown in this minimal example.
- If you change the UI port or hostname, update `allow_origins` in `backend/main.py` and the hardcoded `API_BASE` values in the frontend stores (or refactor to use `import.meta.env.VITE_API_URL`).
- Dataset and model state are in-memory on the API process; restarting the container clears active training sessions.

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| UI loads but training fails | Ensure the backend is running on port 8000 (`npm run dev:backend`) |
| MNIST / CIFAR download SSL errors | Run downloads manually or fix system CA certs; use a custom CSV/image dataset instead |
| EDF import fails | `pip install mne` in the backend venv |
| Shape mismatch when training | Match Input node `channels` × `height` × `width` to your dataset (CSV: `channels=N`, `H=1`, `W=1`) |
| Plot errors | Confirm `matplotlib` and `seaborn` are installed in the backend environment |

---

## License

See repository license file if present.
