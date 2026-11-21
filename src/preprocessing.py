
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Tải dữ liệu cần thiết của NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Khởi tạo danh sách từ dừng
stop_words = set(stopwords.words('english'))

# Khởi tạo bộ Lemmatizer
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    # Chuyển văn bản thành chữ thường
    text = str(text).lower()

    # Xóa thẻ HTML
    text = re.sub(r'<.*?>', '', text)

    # Xóa đường dẫn URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Chỉ giữ lại chữ cái và khoảng trắng
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tách từ
    tokens = word_tokenize(text)

    # Xóa từ dừng và chuẩn hóa từ
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    # Ghép các từ thành văn bản
    return " ".join(tokens)
