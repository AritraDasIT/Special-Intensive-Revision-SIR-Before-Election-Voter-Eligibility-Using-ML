# 🗳️ SIR: Special Intensive Revision - Voter Eligibility Predictor

**SIR (Special Intensive Revision)** is a machine learning system designed to predict voter eligibility fairly and transparently, using hierarchical document verification and government-aligned rules. This project is aimed at ensuring that **only eligible voters cast their votes** before elections.

---

## 🔹 Key Features

- **Hierarchical Document Verification**  
  Checks in the following order:  
  Voter → Father → Grandfather → 10th/12th → Birth Certificate → Aadhaar/Passport

- **Hard vs Soft Rules**  
  - Hard rules (must pass): Age ≥ 18, Citizenship, Valid Voter ID (EPIC)  
  - Soft rules (advisory): Duplicate entries, Mobile validity, Address validity

- **3-Class Prediction**  
  - ✅ Valid  
  - ⚠️ Needs Verification  
  - ❌ Invalid  

- **Probability Visualization**  
  - Interactive bar chart showing the probability for each class  
  - Clear reasoning for predictions  

- **Strong ML Models**  
  - Random Forest  
  - Gradient Boosting Classifier  

---

🔹 Dataset

The project uses a hypothetical dataset simulating voter records.

It includes demographic data, voter ID status, document verification, and advisory fields.

Dataset can be found in voter_hierarchical_dataset.csv.

🔹 Model Evaluation

The dataset is split into training and testing sets (80/20).

Confusion matrix and classification reports are generated to evaluate model performance.

Both Random Forest and Gradient Boosting models are included for comparison.

---

