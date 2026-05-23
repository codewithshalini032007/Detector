# ================================
# PHISHING EMAIL DETECTION MODEL
# ================================

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ================================
# STEP 1: LOAD DATASET
# ================================

# Create sample dataset
data = {
    'email': [
        'Congratulations! You won a free iPhone. Click here now',
        'Your bank account has been suspended. Verify immediately',
        'Meeting scheduled tomorrow at 10 AM',
        'Project report attached for review',
        'Claim your lottery prize now',
        'Important security update required',
        'Lunch meeting with team today',
        'Invoice attached please check'
    ],
    
    # phishing = 1, safe = 0
    'label': [1, 1, 0, 0, 1, 1, 0, 0]
}

# Convert into DataFrame
df = pd.DataFrame(data)

print("Dataset:")
print(df)

# ================================
# STEP 2: SPLIT DATA
# ================================

X = df['email']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

# ================================
# STEP 3: TEXT VECTORIZATION
# ================================

vectorizer = TfidfVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# ================================
# STEP 4: TRAIN MODEL
# ================================

model = MultinomialNB()

model.fit(X_train_vectorized, y_train)

# ================================
# STEP 5: PREDICTION
# ================================

y_pred = model.predict(X_test_vectorized)

# ================================
# STEP 6: EVALUATION
# ================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ================================
# STEP 7: CONFUSION MATRIX
# ================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Safe', 'Phishing'],
            yticklabels=['Safe', 'Phishing'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ================================
# STEP 8: TEST WITH NEW EMAIL
# ================================

new_email = ["Urgent! Your account password expired. Click link now"]

new_email_vectorized = vectorizer.transform(new_email)

prediction = model.predict(new_email_vectorized)

if prediction[0] == 1:
    print("\nThis email is PHISHING")
else:
    print("\nThis email is SAFE")