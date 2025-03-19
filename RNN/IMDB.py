import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping

max_feature=10000
(X_train,Y_train),(X_test,Y_test)=imdb.load_data(num_words=max_feature)


sample_review=X_train[100]

index=imdb.get_word_index()
reverse_index={value:key for (key,value) in index.items()}
decoded_review=" ".join(reverse_index.get(i-3,"?") for i in sample_review)
decoded_review

max_len=500
X_train=sequence.pad_sequences(X_train,maxlen=max_len)
X_test=sequence.pad_sequences(X_test,maxlen=max_len)


model=Sequential()
model.add(Embedding(max_feature,128))
model.add(SimpleRNN(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


early_stopping=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)

history=model.fit(
    X_train,Y_train,validation_split=0.2,epochs=10,batch_size=32,callbacks=[early_stopping]
)

model.save("simpleRNN_IMDB.h5")