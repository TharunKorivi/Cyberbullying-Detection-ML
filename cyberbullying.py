# Step 1: Import necessary libraries
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import re   # <-- Add this line
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')



# Step 2: Load and combine datasets
# Load Cyberbullying_tweets.csv (95 records)
cyberbullying_df = pd.read_csv('kaggle_cyberbullying_tweets.csv', names=['Text', 'CB_Label'], header=0)
cyberbullying_df = cyberbullying_df.rename(columns={'Text': 'tweet_text', 'CB_Label': 'cyberbullying_type'})
cyberbullying_df['cyberbullying_type'] = cyberbullying_df['cyberbullying_type'].map({0: 'not_cyberbullying', 1: 'cyberbullying'})

# Load cyberbullying_tweets.csv (66 records)
additional_df = pd.read_csv('cyberbullying_tweets.csv', names=['tweet_text', 'cyberbullying_type'], header=0)
additional_df['cyberbullying_type'] = additional_df['cyberbullying_type'].map({
    'not_cyberbullying': 'not_cyberbullying',
    'ethnicity': 'cyberbullying',  # Treat ethnicity as a form of cyberbullying
    # Add other mappings if more categories exist (e.g., 'religion' → 'cyberbullying')
})

# Combine datasets
combined_df = pd.concat([cyberbullying_df, additional_df], ignore_index=True)
print(f"Total records before preprocessing: {len(combined_df)}")  # Expected ~161 records


# Step 3: Preprocessing
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    stop_words = {'the', 'and', 'is', 'in', 'it', 'to', 'a', 'of'}  # Basic stop words
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

combined_df['clean_text'] = combined_df['tweet_text'].apply(preprocess_text)
print(f"Sample preprocessed text: {combined_df['clean_text'].iloc[0]}")


# Step 4: Handle labels and oversampling
# Encode multi-class labels (not_cyberbullying, cyberbullying, ethnicity)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
label_encoder = LabelEncoder()
combined_df['encoded_label'] = label_encoder.fit_transform(combined_df['cyberbullying_type'])
print(f"Classes: {list(label_encoder.classes_)}")

# Oversample to balance classes
X = combined_df['clean_text'].values.reshape(-1, 1)
y = combined_df['encoded_label'].values
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
combined_df_resampled = pd.DataFrame({'clean_text': X_resampled.flatten(), 'encoded_label': y_resampled})
print(f"Records after oversampling: {len(combined_df_resampled)}")


# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    combined_df_resampled['clean_text'], combined_df_resampled['encoded_label'],
    test_size=0.3, random_state=42, stratify=combined_df_resampled['encoded_label']
)
print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")



# Step 6: Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', min_df=2, max_df=0.8)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")



# Step 6: Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', min_df=2, max_df=0.8)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")

# Step 7: Train and evaluate ML models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(tree_method='hist', device='cuda', random_state=42),
    'LinearSVC': LinearSVC(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted'),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }
    print(f"\n{name} Results:")
    print(f"Accuracy: {results[name]['Accuracy']:.4f}")
    print(f"Precision: {results[name]['Precision']:.4f}")
    print(f"Recall: {results[name]['Recall']:.4f}")
    print(f"F1-Score: {results[name]['F1-Score']:.4f}")
    print(f"Confusion Matrix:\n{results[name]['Confusion Matrix']}")
    
    
# Step 8: Display best model (highest F1-Score)
best_model_name = max(results, key=lambda x: results[x]['F1-Score'])
print(f"\nBest Model: {best_model_name} with F1-Score: {results[best_model_name]['F1-Score']:.4f}")
