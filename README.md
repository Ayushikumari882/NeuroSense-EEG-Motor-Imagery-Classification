# 🧠 NeuroSense – EEG Motor Imagery Classification

NeuroSense is a full-stack EEG motor imagery classification system that identifies **left-hand vs right-hand imagined movement** from EEG signals. The project combines **EEG preprocessing**, **Common Spatial Patterns (CSP)** feature extraction, **machine learning classification**, and an interactive **web dashboard** for analysis and visualization.

This version is built using:

- **Backend:** Flask, MNE, NumPy, SciPy, Scikit-learn
- **Frontend:** HTML, CSS, JavaScript, Plotly.js

The system supports both:

- **EDF files**
- **MAT files**

and provides outputs such as:

- predicted class
- confidence score
- accuracy
- precision / recall / F1-score
- ROC-AUC
- confusion matrix
- EEG waveform visualization
- spectrogram
- band power
- CSP pattern map
- classifier benchmark comparison

Standalone Flask dashboard for EEG motor imagery analysis with:

- PhysioNet EEGBCI loading through MNE
- Multi-format uploads for EDF, BDF, GDF, MAT, CSV, SET, FIF, and ZIP bundles
- 8-30 Hz preprocessing and epoch extraction
- CSP feature extraction
- Calibrated linear SVM classification
- Premium dark clinical dashboard UI with Plotly visualizations


## 📁 Project Structure

```text
NeuroSense/
├── app.py
├── config.json
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── saved_models/
├── .venv/
└── __pycache__/
🚀 Features
Load PhysioNet EDF EEG motor imagery data
Upload EDF and MAT EEG datasets
Preprocess EEG using filtering and epoch extraction
Extract spatial features using CSP
Classify motor imagery using SVM
Benchmark against LDA
Compute validation metrics:
accuracy
precision
recall
F1-score
ROC-AUC
cross-validation score
Visualize:
EEG waveform monitor
spectrogram
band power
confusion matrix
classifier benchmark chart
CSP pattern heatmap
Export results as CSV
Save and reload previous report state
Support subject-based and multi-subject evaluation
⚙️ Installation
1. Clone the repository
git clone <your-repository-url>
cd NeuroSense
2. Create and activate virtual environment
Windows
python -m venv .venv
.venv\Scripts\activate
Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
▶️ Run the Project
python app.py
Then open:

http://127.0.0.1:5000

If python does not work, use:

.venv\Scripts\python.exe app.py
🧪 How the Project Works
The project follows this pipeline:

EEG Dataset (EDF / MAT)
        │
        ▼
Data Loading
        │
        ▼
Preprocessing
- channel standardization
- 8–30 Hz bandpass filter
- epoch extraction
        │
        ▼
Feature Extraction
- Common Spatial Patterns (CSP)
        │
        ▼
Classification
- SVM (main classifier)
- LDA (benchmark classifier)
        │
        ▼
Evaluation
- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- confusion matrix
- cross-validation
        │
        ▼
Interactive Dashboard Output
🔬 Supported Input Formats
EDF
EDF mode is intended for raw EEG recordings that contain:

EEG channel signals
event annotations for motor imagery
left/right cue markers such as T1 and T2
MAT
MAT mode supports MATLAB EEG structures such as:

imagery_left, imagery_right, srate
data, labels, sfreq
session.data, session.labels, session.sfreq
BNCI-like trial structures
🧠 Algorithms Used
Core Algorithms
Bandpass Filtering
Epoch Extraction
Common Spatial Patterns (CSP)
Support Vector Machine (SVM)
Linear Discriminant Analysis (LDA)
K-Fold Cross-Validation
Main Pipeline
EEG preprocessing
CSP feature extraction
SVM classification
LDA benchmarking
validation metrics
