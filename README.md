# EEG Attention Classifier

This project uses the **EEG Eye State dataset** to classify attentiveness:  
- `0` = Eye Open (Attentive)  
- `1` = Eye Closed (Distracted)  

---

## Dataset
- Source: [UCI Machine Learning Repository – EEG Eye State](https://archive.ics.uci.edu/dataset/264/eeg+eye+state)  
- 14 EEG channels (~15,000 samples)  
- Labels: 0 = Eye Open, 1 = Eye Closed  

---

## Methods
1. Cleaned dataset (removed `b'0'` / `b'1'` labels)  
2. Standardized EEG features  
3. Trained ML models:  
   - ✅ Random Forest (best: ~93% accuracy)  
   - ❌ SVM (baseline: ~63% accuracy)  
4. Evaluated with accuracy, confusion matrix, feature importance plots  

---

## Results
- **Random Forest Accuracy:** ~93%  
- **Key Features:** Occipital/Parietal channels (O1, P7) are most important  
- **Confusion Matrix:** Balanced performance on both attentive & distracted states  

---

## How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/chaitanyalogin/sample-EEG-Project.git
