from sklearn.calibration import LabelEncoder

# --------------------------------
#  Encode labels (for training only)
# --------------------------------
def encode_labels(y_series):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_series)
    return label_encoder, y_encoded
