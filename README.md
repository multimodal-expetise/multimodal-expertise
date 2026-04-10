
## Multimodal Expertise: Machine Learning Based Data Fusion

## Overview
This repository contains the code and data used in the research paper "Multimodal Expertise: Machine Learning Based Data Fusion". 

## Installation
Before running the scripts, ensure you have the following dependencies installed:
- Python 3.x
- PyTorch
- Pandas
- Numpy
- Pickle
- Statsmodels
- Scikit-learn

You can install these packages using pip:
```bash
pip install torch pandas statsmodels numpy scikit-learn
```
--

## Dataset
- Due to GitHub's storage limitations, the dataset has been uploaded to Google Drive.
- You can access the dataset using the following link:  https://drive.google.com/file/d/16TMd4gMGT3QDuvJrJ1qVCg1-LRxDfdUy/view?usp=drive_link
- After download, save it in path: data

### Multimodal Features 
#### Verbal Features
- **Tool Used**: BERT.
- **Features**: 768 dimensions.

#### Vocal Features
- **Tool Used**: Covarep.
- **Features**: 74 dimensions.

#### Visual Features
- **Tool Used**: OpenFace 2.0.
- **Features**: 35 dimensions.


## Usage
To run the analysis, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/multimodal-expertise/multimoda-expertise.
   ```
2. Navigate to the cloned directory.
3. Run the main script:
   ```bash
   python main_run_evaluation_pretrained_model.py
   ```


## Model Description
This repository implements and evaluates 11 models structured into four categories:

 **1. No Fusion + Unimodal Data:**
   - `A1_LSTM_l.py`: LSTM model for text data.
   - `A2_LSTM_a.py`: LSTM model for audio data.
   - `A3_LSTM_v.py`: LSTM model for image data.
 
 **2. Partial Fusion + Bimodal Data:**
   - `A4_LF_LSTM_la.py`: LF_LSTM model for text and audio data.
   - `A5_LF_LSTM_lv.py`: LF_LSTM model for text and image data.
   - `A6_LF_LSTM_av.py`: LF_LSTM model for audio and image data.

 **3.Partial Fusion + Trimodal Data:**
   - `A7_LF_LSTM_lav.py`: LF_LSTM model for text, audio, and image data.
   - `A8_LF_MLP_lav.py`:  LF_MLP model for text, audio, and image data.

 **4.Full Fusion + Trimodal Data:**
   - `A9_EF_MLP_lav.py`:   EF_MLP model for text, audio, and image data.
   - `A10_EF_LSTM_lav.py`: EF_LSTM model for text, audio, and image data.
   - `A11_MFN_lav.py`:     MFN model for text, audio, and image data.


## Evaluation Metrics
The code includes functions for evaluating the model performance:

- Accuracy
- F1 Score
- Mean Absolute Error (MAE)
- Correlation Coefficient

