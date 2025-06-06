import argparse
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from models.logistic import train_logistic_regression
from models.naive_bayes import train_naive_bayes
from utils.evaluation import generate_submission
from utils.io_utils import load_data, output_dir
from utils.preprocess import encode_labels
from contexts.training_context import TrainingContext


parser = argparse.ArgumentParser(description="passing in test flag to run predictions")
parser.add_argument("--test", action="store_true", help="Load and run predictions on test.csv instead of training data")
parser.add_argument("--model", type=str, choices=["logistic", "naive_bayes"], default="logistic", help="Choose model to train (default: logistic)")
args = parser.parse_args()

# --------------------------------
#  Load Data
# --------------------------------
df = load_data(args.test)
encoder_path = os.path.join(output_dir, "encoder.pkl")

if not args.test:
    label_encoder, y = encode_labels(df["author"])
    print(f"label_encoder.classes_:  {label_encoder.classes_}")
else:
    label_encoder = joblib.load(encoder_path)

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

os.makedirs(output_dir, exist_ok=True)

vectorizer_path = os.path.join(output_dir, "vectorizer.pkl")
model_path = os.path.join(output_dir, "model.pkl")
encoder_path = os.path.join(output_dir, "encoder.pkl")

trainingContext = TrainingContext(
    df=df,
    label_encoder=label_encoder,
    vectorizer=vectorizer,
    output_dir=output_dir,
    model_path=model_path,
    vectorizer_path=vectorizer_path,
    encoder_path=encoder_path
)

if args.test:

    vectorizer = joblib.load(vectorizer_path)
    clf = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    generate_submission(clf, vectorizer, df, label_encoder, os.path.join(output_dir, "submission.csv"))

elif args.model == "logistic" and not args.test:
    clf = train_logistic_regression(trainingContext)

elif args.model == "naive_bayes":
    clr = train_naive_bayes(trainingContext)