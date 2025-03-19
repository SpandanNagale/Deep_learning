from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import sequence




model=load_model("/content/simpleRNN_IMDB.h5")
def decoded_review(encoded_review):
  return " ".join( reverse_index.get(i-3,"?") for i in encoded_review)

def preproessing(text):
  words=text.lower().split()
  encoded_review=[index.get(word,2)+3 for word in words]
  padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
  return padded_review

def prediction(review):
  preprocessed_input=preproessing(review)
  prediction=model.predict(preprocessed_input)
  sentiment= "positive" if prediction[0][0]>0.5 else "negative"
  return sentiment , prediction[0][0]

index=imdb.get_word_index()
reverse_index={value:key for (key,value) in index.items()}
decoded_review=" ".join(reverse_index.get(i-3,"?") for i in sample_review)

example="the movie was bad. the plot and the characters were bad"
prediction(example)