# Handwriting Graphology & Letter Trait Analyzer

End-to-end Python project for graphology and handwriting analysis based on letter structures (A-Z).

## Project Structure
- `app.py`: Streamlit web interface with interactive letter selection & image upload.
- `graphology_db.py`: Dictionary mapping letters A-Z to detailed graphology traits.
- `model.py`: PyTorch CNN Model & PCA dimensionality reduction pipeline.
- `dataset.py`: Synthetic dataset generator engine.
- `requirements.txt`: Python package dependencies.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Web Application:
   ```bash
   streamlit run app.py
   ```
