from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def prediction_next_word(model,tokenizer,text,max_sequence_len):
  token_list=tokenizer.texts_to_sequences([line])[0]
  if len(token_list)>=max_sequence_len:
    token_list=token_list[-(max_sequence_len-1):]
  token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
  prediction=model.predict(token_list,verbose=0)
  predicted_word_index=np.argmax(prediction, axis=1)
  for word , index in tokenizer.word_index.items():
    if index==predicted_word_index:
      return word
  return None


with open("shakespeare.txt","r") as f:
  data=f.read().lower()


tokenizer=Tokenizer()
tokenizer.fit_on_texts([data])
total_words=len(tokenizer.word_index)+1
total_words
 
input_sequences=[]
for line in data.split('\n'):
  token_list=tokenizer.texts_to_sequences([line])[0]
  for i in range(1,len(token_list)):
   n_gram_sequence=token_list[:i+1]
   input_sequences.append(n_gram_sequence)  

max_sequence_len=max([len(x) for x in input_sequences])
input_sequences=np.array(pad_sequences(input_sequences,maxlen=max_sequence_len,padding='pre'))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

x,y=input_sequences[:,:-1],input_sequences[:,-1]
y=tf.keras.utils.to_categorical(y,num_classes=total_words)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


model=Sequential()
model.add(Embedding(total_words,100,input_length=max_sequence_len-1))
model.add(LSTM(150,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history=model.fit(x_train,y_train,epochs=50,validation_data=(x_test,y_test),verbose=1)

input_text="anything "
max_sequence_len=model.input_shape[1]+1
next_word=prediction_next_word(model,tokenizer,input_text,max_sequence_len)
print(next_word)


#this model is not accurate try running it on higher epochs , it has only 47% accuraccy 