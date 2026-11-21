
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
from transformers import BertTokenizer


# Tạo đặc trưng TF-IDF
def create_tfidf_features(X_train, X_test, max_features=50000):

    tfidf = TfidfVectorizer(max_features=max_features)

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    return X_train_tfidf, X_test_tfidf, tfidf


# Huấn luyện mô hình Word2Vec
def train_word2vec(sentences, vector_size=300, window=5, min_count=2):

    tokenized = [text.split() for text in sentences]

    model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4
    )

    return model


# Chuyển văn bản thành vector Word2Vec
def get_word2vec_embeddings(model, texts, vector_size=300):

    embeddings = []

    for text in texts:

        vectors = [
            model.wv[word]
            for word in text.split()
            if word in model.wv
        ]

        if vectors:
            embeddings.append(np.mean(vectors, axis=0))
        else:
            embeddings.append(np.zeros(vector_size))

    return np.array(embeddings)


# Khởi tạo bộ Tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Tách từ bằng BERT
def tokenize_for_bert(texts, max_length=512):

    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return encodings['input_ids'], encodings['attention_mask']
