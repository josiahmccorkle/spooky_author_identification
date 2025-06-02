import argparse
import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser(description="passing in test flag to run predictions")
# Add a flag (boolean)
parser.add_argument("--test", action="store_true", help="Load and run predictions on test.csv instead of training data")
# Parse the arguments
args = parser.parse_args()


# --------------------------------
#  Load Data
# --------------------------------
filePath = './data/'
files = [f for f in os.listdir(filePath) if f.endswith('.csv')]
print(files)
expected_files = {"train.csv", "test.csv"}
dfs = {file: pd.read_csv(os.path.join(filePath, file)) for file in files if file in expected_files}

try:
    df = dfs["test.csv"] if args.test else dfs["train.csv"]
    print(df.head())

    if not args.test:
        print(df["author"].value_counts())   # Check label distribution
        print(df["text"].apply(len).describe())  # Check text length stats

        # --------------------------------
        #  Encode labels (for training only)
        # --------------------------------
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df["author"])
        print(f"label_encoder.classes_:           {label_encoder.classes_}")
except KeyError as e:
    raise FileNotFoundError(f"Missing file: {e.args[0]} in './data/'")


# --------------------------------
#  Prepare input features
# --------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

# --------------------------------
#  Define output paths
# --------------------------------
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

vectorizer_path = os.path.join(output_dir, "vectorizer.pkl")
model_path = os.path.join(output_dir, "model.pkl")
encoder_path = os.path.join(output_dir, "encoder.pkl")

if not args.test:
    X = vectorizer.fit_transform(df["text"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(label_encoder, encoder_path)
    print(f"y_val_pred:          {y_val_pred}")
    print(f"✅ Model and components saved to {output_dir}")
else:
    vectorizer = joblib.load(vectorizer_path)
    clf = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    X = vectorizer.transform(df["text"])
    y_pred = clf.predict(X)
    y_labels = label_encoder.inverse_transform(y_pred)


    # id,       EAP,    HPL,    MWS
    # id07943,  0.33,   0.33,   0.33
    submission = pd.DataFrame({
        "id": df["id"],
        "author": y_labels
    })


    # Get probabilities for each class
    y_pred_proba = clf.predict_proba(X)  # shape: (n_samples, 3)

    # Create submission DataFrame with proper column names
    submission = pd.DataFrame(y_pred_proba, columns=label_encoder.classes_)
    submission.insert(0, "id", df["id"])  # insert ID column at the front
    submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
    print(f"✅ Submission saved to {output_dir}/submission.csv")