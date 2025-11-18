
# Voter Eligibility Predictor Project 


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Step 1: Generate Hypothetical Dataset (Adjusted Probabilities)
# -------------------------------

np.random.seed(42)
n = 1000

data = pd.DataFrame({
    # Demographics
    'age': np.random.randint(16, 70, n),  # some <18
    'gender': np.random.choice(['Male','Female','Other'], n, p=[0.48,0.48,0.04]),
    'state': np.random.choice(['West Bengal'], n),
    'district': np.random.choice(['Kolkata','Howrah','Darjeeling','South 24 Pgs','North 24 Pgs'], n),
    'urban_rural': np.random.choice(['Urban','Rural'], n, p=[0.6,0.4]),

    # Voter Record Checks
    'has_epic': np.random.choice([0,1], n, p=[0.2,0.8]),        # 20% no EPIC
    'epic_valid': np.random.choice([0,1], n, p=[0.1,0.9]),      # 10% invalid EPIC
    'citizenship': np.random.choice([0,1], n, p=[0.05,0.95]),   # 5% non-Indian
    'dup_flag': np.random.choice([0,1], n, p=[0.85,0.15]),      
    'address_valid': np.random.choice([0,1], n, p=[0.1,0.9]),
    'mobile_valid': np.random.choice([0,1], n, p=[0.1,0.9]),

    # Document Verification (hierarchical)
    'listed_2002': np.random.choice([0,1], n, p=[0.5,0.5]),
    'father_listed_2002': np.random.choice([0,1], n, p=[0.6,0.4]),
    'grandfather_listed_2002': np.random.choice([0,1], n, p=[0.7,0.3]),
    'has_10th_admit': np.random.choice([0,1], n, p=[0.7,0.3]),
    'has_12th_marksheet': np.random.choice([0,1], n, p=[0.7,0.3]),
    'has_birth_cert': np.random.choice([0,1], n, p=[0.8,0.2]),
    'has_aadhaar': np.random.choice([0,1], n, p=[0.7,0.3]),
    'has_passport': np.random.choice([0,1], n, p=[0.8,0.2]),
})

# -------------------------------
# Step 2: Hierarchical Document Verification
# -------------------------------

def hierarchical_doc_check(row):
    if row['listed_2002'] == 1:
        return 1
    elif row['father_listed_2002'] == 1:
        return 1
    elif row['grandfather_listed_2002'] == 1:
        return 1
    elif row['has_10th_admit'] == 1:
        return 1
    elif row['has_12th_marksheet'] == 1:
        return 1
    elif row['has_birth_cert'] == 1:
        return 1
    elif row['has_aadhaar'] == 1 or row['has_passport'] == 1:
        return 1
    else:
        return 0

data['doc_verification_ok'] = data.apply(hierarchical_doc_check, axis=1)
doc_cols = ['listed_2002','father_listed_2002','grandfather_listed_2002',
            'has_10th_admit','has_12th_marksheet','has_birth_cert','has_aadhaar','has_passport']
data['num_valid_docs'] = data[doc_cols].sum(axis=1)

# -------------------------------
# Step 3: Define 3-Class Outcome
# -------------------------------

def voter_status(row):
    # Hard rules
    if row['age'] < 18 or row['citizenship']==0 or row['has_epic']==0 or row['epic_valid']==0:
        return 'Invalid'
    # Document verification
    elif row['doc_verification_ok']==0:
        return 'Needs Verification'
    else:
        return 'Valid'

data['record_status'] = data.apply(voter_status, axis=1)
print("Class distribution:\n", data['record_status'].value_counts())

# -------------------------------
# Step 4: Prepare Data for ML
# -------------------------------

feature_cols = [
    'age','has_epic','epic_valid','citizenship','dup_flag','address_valid','mobile_valid',
    'listed_2002','father_listed_2002','grandfather_listed_2002',
    'has_10th_admit','has_12th_marksheet','has_birth_cert','has_aadhaar','has_passport',
    'doc_verification_ok','num_valid_docs'
]

X = data[feature_cols]
y = data['record_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------------
# Step 5: Train ML Models
# -------------------------------

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# -------------------------------
# Step 6: Evaluate Models
# -------------------------------

def evaluate_model(y_test, y_pred, model_name):
    print(f"\n--- {model_name} ---")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=['Valid','Needs Verification','Invalid'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Valid','Needs Verification','Invalid'],
                yticklabels=['Valid','Needs Verification','Invalid'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)

evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_gb, "Gradient Boosting")

# -------------------------------
# Step 7: Interactive Prediction
# -------------------------------

def get_input(prompt, type_=int, allowed=[0,1]):
    while True:
        try:
            value = type_(input(prompt))
            if allowed and value not in allowed:
                print(f"Enter {allowed} only!")
                continue
            return value
        except:
            print(f"Invalid input. Please enter a {type_.__name__} value.")

print("\nEnter new voter record details for prediction:")

new_entry = {}
new_entry['age'] = int(input("Age: "))
new_entry['has_epic'] = get_input("Has voter ID? (1=Yes,0=No): ")
new_entry['epic_valid'] = get_input("Voter ID valid? (1=Yes,0=No): ")
new_entry['citizenship'] = get_input("Indian citizenship? (1=Yes,0=No): ")
new_entry['dup_flag'] = get_input("Duplicate entry? (1=Yes,0=No): ")
new_entry['address_valid'] = get_input("Address valid? (1=Yes,0=No): ")
new_entry['mobile_valid'] = get_input("Mobile valid? (1=Yes,0=No): ")
new_entry['listed_2002'] = get_input("Listed in 2002 voter list? (1=Yes,0=No): ")
new_entry['father_listed_2002'] = get_input("Father listed in 2002 voter list? (1=Yes,0=No): ")
new_entry['grandfather_listed_2002'] = get_input("Grandfather listed in 2002 voter list? (1=Yes,0=No): ")
new_entry['has_10th_admit'] = get_input("Has 10th admit card? (1=Yes,0=No): ")
new_entry['has_12th_marksheet'] = get_input("Has 12th marksheet? (1=Yes,0=No): ")
new_entry['has_birth_cert'] = get_input("Has birth certificate? (1=Yes,0=No): ")
new_entry['has_aadhaar'] = get_input("Has Aadhaar? (1=Yes,0=No): ")
new_entry['has_passport'] = get_input("Has Passport? (1=Yes,0=No): ")

# Hierarchical document check
def hierarchical_doc_check_single(entry):
    if entry['listed_2002'] == 1:
        return 1
    elif entry['father_listed_2002'] == 1:
        return 1
    elif entry['grandfather_listed_2002'] == 1:
        return 1
    elif entry['has_10th_admit'] == 1:
        return 1
    elif entry['has_12th_marksheet'] == 1:
        return 1
    elif entry['has_birth_cert'] == 1:
        return 1
    elif entry['has_aadhaar'] == 1 or entry['has_passport'] == 1:
        return 1
    else:
        return 0

new_entry['doc_verification_ok'] = hierarchical_doc_check_single(new_entry)
new_entry['num_valid_docs'] = sum([new_entry[c] for c in doc_cols])
new_df = pd.DataFrame([new_entry])

prediction = rf_model.predict(new_df)[0]
proba = rf_model.predict_proba(new_df)[0]

# Bar Graph
plt.figure(figsize=(6,4))
colors = ['green','orange','red']
sns.barplot(x=rf_model.classes_, y=proba, palette=colors)
plt.ylim(0,1)
plt.title("Prediction Probability")
plt.ylabel("Probability")
plt.show()

# Reasoning
reasons = []
# Hard rules
if new_entry['age'] < 18:
    reasons.append("Underage (<18)")
if new_entry['citizenship']==0:
    reasons.append("Non-Indian citizen")
if new_entry['has_epic']==0:
    reasons.append("No voter ID")
if new_entry['epic_valid']==0:
    reasons.append("Voter ID invalid format")

# Document verification
if new_entry['doc_verification_ok']==0:
    reasons.append("No valid document found (voter/father/grandfather/10th/12th/Birth/Aadhaar/Passport)")

# Soft rules
soft_issues = []
if new_entry['dup_flag']==1:
    soft_issues.append("Duplicate entry (needs verification)")
if new_entry['address_valid']==0:
    soft_issues.append("Invalid address (needs verification)")
if new_entry['mobile_valid']==0:
    soft_issues.append("Mobile invalid (needs verification)")

# Print outcome
print(f"\nPrediction: {prediction}")
if prediction == 'Valid':
    print("Reason: All hard rules passed and document verification satisfied.")
elif prediction == 'Needs Verification':
    print("Reason: Hard rules passed but the following require verification:")
    for r in soft_issues:
        print(f"- {r}")
else:
    print("Reason: Failed hard eligibility rules:")
    for r in reasons:
        print(f"- {r}")
