import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.io_utils import output_dir


def generate_submission(model, vectorizer: TfidfVectorizer, test_df, label_encoder, output_path):
    X_test = vectorizer.transform(test_df["text"])
    proba = model.predict_proba(X_test)
    submission = pd.DataFrame(proba, columns=label_encoder.classes_)
    submission.insert(0, "id", test_df["id"])
    submission.to_csv(output_path, index=False)

    print(f"✅ Submission saved to {output_dir}/submission.csv")