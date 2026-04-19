# Image To Graphic Programs AI (End-to-End CPU Optimized)

This project trains an AI model to convert a hand-drawn geometric image into a sequence of drawing commands (`line`, `circle`, `rectangle`, `triangle`, `ellipse`, `polygon`).

Everything is heavily optimized for a standard Core i5/i6 CPU environments. Model avoids Transformers and instead relies on CNN Adaptive Pooling attached to an Attention-based GRU decoder.

## How To Install & Run (Step-by-Step Guide)

**Step 1: Install Dependencies**
Pehle environment tayar karein. CMD ya Terminal open karke yeh command type karein:
```bash
pip install -r requirements.txt
```

**Step 2: Train the Model**
Agar aap model ko zero se naye images pe train karna chahte hain toh `train.py` run karein. Is code mein 'on-the-fly' dataset banta hai (images save nahi hoti bulke RAM mein banti aur augmentation se imperfect hoti hain).
```bash
python train.py
```
*Note: Jab model loss minimize kardega tab wo automatically aap kay folder mein `best_model.pth` ki file bana dega.*

**Step 3: Test Engine (Optional)**
Agar aap individually inference system ka command text base check karna chahte hain, bina web app lagaye, tau `predict.py` run karein.
```bash
python predict.py
```

**Step 4: Start Graphic Decoder App (Streamlit Frontend)**
Sab set hone ke baad final application kholne ke liye ye likhein:
```bash
streamlit run app.py
```
Isme browser me ek tab khulegi jahan aap directly shape ko draw karke ya image file upload karke uski exact code sequence nikalwa saktay hain.

## File Hierarchy Overview
* `dataset_generator.py` - Early experiment utility to check images.
* `dataset.py` - Final PyTorch Dataset module with dynamic random parameter injections.
* `model.py` - Heart of the system (CNN Encoder -> Adaptive Pooling -> Decoder GRU with Beam Search Attention).
* `train.py` - Automated training loop with Early Stopping & dynamic Learning Rates.
* `predict.py` - Beam Search inferencing with aggressive Post-Processing for coordinate overlaps.
* `app.py` - User-facing Interface Streamlit App.
