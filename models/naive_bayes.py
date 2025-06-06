import joblib
from sklearn.naive_bayes import MultinomialNB
from contexts.training_context import TrainingContext
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_naive_bayes(trainingContext: TrainingContext):
    df = trainingContext.df
    vectorizer = trainingContext.vectorizer
    label_encoder = trainingContext.label_encoder
    model_path = trainingContext.model_path
    vectorizer_path = trainingContext.vectorizer_path
    encoder_path = trainingContext.encoder_path

    X = vectorizer.fit_transform(df["text"])
    y = label_encoder.fit_transform(df["author"])
    X_train, X_val, y_train, y_val  = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)


    # printing out accuracy
    y_val_pred = clf.predict(X_val)
    print("VMultinomialNB alidation Accuracy:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))


    # storing results
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(label_encoder, encoder_path)

    return clf
