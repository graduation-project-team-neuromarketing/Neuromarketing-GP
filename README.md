# Neuromarketing-GP: EEG-Based Emotion Classification System

A web-based system for emotion classification from EEG signals, built with a FastAPI backend (integrating a PyTorch deep learning model) and an HTML/CSS/JS frontend.

## Project Structure

```
Neuromarketing-Website/
├── Backend/      # FastAPI app + AI model + database
└── Frontend/     # HTML/CSS/JS pages (admin, company, user)
```

## How to Run

### 1. Backend

Step 1 — Open a terminal inside the `Backend/` folder:
```bash
cd Backend
```

Step 2 — Install the required Python packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

Step 3 — Run the FastAPI server:
```bash
python -m uvicorn main:app --reload
```

Step 4 — The API will be available at:
- `http://127.0.0.1:8000`
- Interactive docs: `http://127.0.0.1:8000/docs`

### 2. Frontend

Step 1 — Open the `Frontend/` folder.

Step 2 — Open `sign_in-out.html` directly in your browser (double-click it, or right-click → Open with → your browser).

> Make sure the Backend server (Step 1 above) is already running before using the Frontend, since every page calls the API.

## Tech Stack

- **Backend:** FastAPI, SQLAlchemy, PyTorch, Pydantic, JWT Auth
- **Frontend:** HTML, CSS, JavaScript