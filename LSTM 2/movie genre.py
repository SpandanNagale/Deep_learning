import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import warnings

warnings.filterwarnings('ignore')

tokenizer=Tokenizer()

df1 = pd.read_csv("/content/test_data_solution.txt", sep=':::' , names=['names','genre','DESCR'] )
df = df1.sample(n=10000, random_state=42)

df['DESCR']=df['DESCR'].str.lower()
df['DESCR']=df['DESCR'].apply(lambda x:re.sub('[^a-z A-Z 0-9]+', " ",x))
df['DESCR']=df['DESCR'].apply(lambda x:" ".join([y for y in x.split() if y not in stopwords.words('english')]))


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['genre']=le.fit_transform(df['genre'])
df['genre']


from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
lemmatizer=WordNetLemmatizer()
def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

df['DESCR'] = df['DESCR'].apply(lemmatize_text)

tokenizer = Tokenizer(num_words=5000) # Limit vocabulary size
tokenizer.fit_on_texts(df['DESCR'])
sequences = tokenizer.texts_to_sequences(df['DESCR'])

# Pad sequences to a reasonable length
max_length = 100 # Set a reasonable maximum length
padded_sequences = pad_sequences(sequences, maxlen=max_length)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# The change is on this line: Using padded_sequences instead of df['DESCR']
x_train,x_test,y_train,y_test=train_test_split(padded_sequences,df['genre'],test_size=0.2,random_state=42)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping



model=Sequential()
model.add(Embedding(input_dim=5000,output_dim=128,input_length=max_length))
model.add(LSTM(150,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(len(le.classes_ ),activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


history=model.fit(x_train,y_train,epochs=50,validation_split=0.25,verbose=1)

model.save("LSTM2.h5")