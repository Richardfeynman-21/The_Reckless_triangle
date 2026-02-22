import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Paths to datasets
PUBG_DATA_PATH = r"f:\noobathon.bgmi\pubg-finish-placement-prediction\train_V2.csv"
JIGSAW_DATA_PATH = r"f:\noobathon.bgmi\archive\train.csv"

def load_and_preprocess_pubg(sample_size=None):
    if not os.path.exists(PUBG_DATA_PATH):
        raise FileNotFoundError(f"PUBG dataset not found at {PUBG_DATA_PATH}")

    print("Loading PUBG data...")
    if sample_size:
        df = pd.read_csv(PUBG_DATA_PATH, nrows=sample_size)
    else:
        df = pd.read_csv(PUBG_DATA_PATH)
    
    # Handle missing values
    df.dropna(subset=['winPlacePerc'], inplace=True)
    
    # Feature Engineering
    df['totalDistance'] = df['rideDistance'] + df['walkDistance'] + df['swimDistance']
    df['healsAndBoosts'] = df['heals'] + df['boosts']
    df['headshotRate'] = df['headshotKills'] / df['kills'].replace(0, 1)
    
    # Drop IDs and irrelevant categorical features
    drop_cols = ['Id', 'groupId', 'matchId', 'matchType']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    X = df.drop(columns=['winPlacePerc']).values
    y = df['winPlacePerc'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def load_and_preprocess_pubg_classification(sample_size=None):
    """
    Loads PUBG data and bins winPlacePerc into 4 classes.
    0: Early Elimination (0.0 - 0.25)
    1: Mid Game (0.25 - 0.5)
    2: Late Game (0.5 - 0.8)
    3: Top / Winner (0.8 - 1.0)
    """
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess_pubg(sample_size)
    
    def get_class(y):
        if y <= 0.25: return 0
        elif y <= 0.5: return 1
        elif y <= 0.8: return 2
        else: return 3
        
    y_train_class = np.array([get_class(val) for val in y_train])
    y_test_class = np.array([get_class(val) for val in y_test])
    
    return X_train_scaled, X_test_scaled, y_train_class, y_test_class, scaler

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class SimpleTokenizer:
    def __init__(self, max_vocab=20000, max_len=100):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def fit(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())
            
        common_words = counter.most_common(self.max_vocab - 2) # reserved 0 for pad, 1 for unk
        
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        
        for idx, (word, _) in enumerate(common_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        self.vocab_size = len(self.word2idx)
        
    def transform(self, texts):
        sequences = []
        for text in texts:
            tokens = text.split()
            seq = [self.word2idx.get(t, 1) for t in tokens][:self.max_len]
            # pad
            if len(seq) < self.max_len:
                seq.extend([0] * (self.max_len - len(seq)))
            sequences.append(seq)
        return np.array(sequences)


def load_and_preprocess_jigsaw(sample_size=None, max_vocab=20000, max_len=100):
    if not os.path.exists(JIGSAW_DATA_PATH):
        raise FileNotFoundError(f"Jigsaw dataset not found at {JIGSAW_DATA_PATH}")

    print("Loading Jigsaw toxicity data...")
    if sample_size:
        df = pd.read_csv(JIGSAW_DATA_PATH, nrows=sample_size)
    else:
        df = pd.read_csv(JIGSAW_DATA_PATH)

    df['comment_text'] = df['comment_text'].fillna("").apply(clean_text)
    
    X = df['comment_text'].values
    y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tokenizer = SimpleTokenizer(max_vocab=max_vocab, max_len=max_len)
    tokenizer.fit(X_train)
    
    X_train_vec = tokenizer.transform(X_train)
    X_test_vec = tokenizer.transform(X_test)
    
    return X_train_vec, X_test_vec, y_train, y_test, tokenizer
