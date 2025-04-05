import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import warnings
import re
import pickle
from sklearn.preprocessing import LabelEncoder



# Load the saved model
model = load_model("LSTM_youtube.h5")
tokenizer=Tokenizer()
with open('/content/label_encoder_youtube.pkl', 'rb') as file:
    le=pickle.load(file)

max_length=200
# Function to preprocess input text
def preprocess_text(text, tokenizer, max_length):
    text = text.lower()
    text = re.sub('[^a-z A-Z 0-9]+', " ", text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    text = " ".join(words)
    text_sequence = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(text_sequence, maxlen=max_length)
    return text_padded

# Example usage
input_text = "he is a very bad creater. his content is also very bad and disgusting"
processed_text = preprocess_text(input_text, tokenizer, max_length)

# Generate prediction
prediction = model.predict(processed_text)
predicted_label = np.argmax(prediction)

# Decode predicted label
predicted_genre = le.inverse_transform([predicted_label])[0]


print(f"Predicted Genre: {predicted_genre}")