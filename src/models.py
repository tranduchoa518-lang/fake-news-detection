
from sklearn.naive_bayes import MultinomialNB

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    LSTM,
    Dropout
)

from transformers import TFBertForSequenceClassification


# Xây dựng mô hình Naive Bayes
def build_naive_bayes(alpha=0.1):

    return MultinomialNB(alpha=alpha)


# Xây dựng mô hình CNN
def build_cnn(
    vocab_size=50000,
    embedding_dim=100,
    max_length=512
):

    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length
        ),

        Conv1D(128, 5, activation='relu'),

        GlobalMaxPooling1D(),

        Dense(64, activation='relu'),

        Dropout(0.5),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# Xây dựng mô hình LSTM
def build_lstm(
    vocab_size=50000,
    embedding_dim=100,
    max_length=512
):

    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length
        ),

        LSTM(
            64,
            dropout=0.3,
            recurrent_dropout=0.3
        ),

        Dense(32, activation='relu'),

        Dropout(0.5),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# Xây dựng mô hình BERT
def build_bert_model(num_labels=2):

    model = TFBertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=2e-5
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        metrics=['accuracy']
    )

    return model
