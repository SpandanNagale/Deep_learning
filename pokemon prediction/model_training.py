import pandas as pd 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

df=pd.read_csv("/content/pokedex.csv")


df['info']=df['info'].astype(str).str.lower()
df['info']=df['info'].apply(lambda x:re.sub('[^a-z A-Z 0-9]+', " ",x))
df['info']=df['info'].apply(lambda x:" ".join([y for y in x.split() if y not in stopwords.words('english')]))


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['name']=le.fit_transform(df['name'])
df['name']


from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
lemmatizer=WordNetLemmatizer()
def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text
df['info'] = df['info'].apply(lemmatize_text)



tokenizer = Tokenizer(num_words=5000) # Limit vocabulary size
tokenizer.fit_on_texts(df['info'])
sequences = tokenizer.texts_to_sequences(df['info'])

# Pad sequences to a reasonable length
max_length = 30 # Set a reasonable maximum length
padded_sequences = pad_sequences(sequences, maxlen=max_length)



from sklearn.preprocessing import StandardScaler
numeric_cols = ['height', 'weight', 'hp', 'attack', 'defense', 's_attack', 's_defense', 'speed']
scaler = StandardScaler()
scaled_numerics = scaler.fit_transform(df[numeric_cols])


X_text = padded_sequences
X_num = scaled_numerics
y = df['name'].values

num_classes=len(le.classes_)


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate

# Set hyperparameters
vocab_size = 5000
embedding_dim = 64
max_length = 30
num_numerical_features = X_num.shape[1]

# Text input branch
text_input = Input(shape=(max_length,), name='text_input')
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_input)
lstm = LSTM(64)(embedding)

# Numerical input branch
num_input = Input(shape=(num_numerical_features,), name='num_input')

# Combine
combined = Concatenate()([lstm, num_input])
dense1 = Dense(128, activation='relu')(combined)
dense2 = Dense(64, activation='relu')(dense1)
output = Dense(num_classes, activation='softmax')(dense2)

# Build model
model = Model(inputs=[text_input, num_input], outputs=output)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Summary
model.summary()

history=model.fit(
    [X_text,X_num],y,validation_split=0.2,epochs=50,batch_size=32
)