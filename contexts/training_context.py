from typing import Literal, LiteralString
from pandas import DataFrame
from sklearn.calibration import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

class TrainingContext:
    def __init__(self, df: DataFrame, label_encoder: LabelEncoder, vectorizer: TfidfVectorizer, output_dir: Literal['./outputs'],
        model_path: LiteralString, vectorizer_path: LiteralString, encoder_path: LiteralString):
        self.df = df
        self.label_encoder = label_encoder
        self.vectorizer = vectorizer
        self.output_dir = output_dir
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.encoder_path = encoder_path