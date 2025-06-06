
import joblib
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from contexts.training_context import TrainingContext

def train_logistic_regression(trainingContext: TrainingContext):

    df = trainingContext.df
    label_encoder = trainingContext.label_encoder
    vectorizer = trainingContext.vectorizer
    model_path = trainingContext.model_path
    vectorizer_path = trainingContext.vectorizer_path
    encoder_path = trainingContext.encoder_path


    y = label_encoder.fit_transform(df["author"])
    X = vectorizer.fit_transform(df["text"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train, y_train)

    # printing out accuracy
    y_val_pred = clf.predict(X_val)
    print("LogisticRegression Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))

    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(label_encoder, encoder_path)

    return clf