import tensorflow as tf
import tensorflow_datasets as tfds

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
import numpy as np
train_data, test_data = imdb['train'], imdb['test']

training_sentences=[]
training_labels=[]
testing_sentences=[]
testing_labels=[]


# s,l are stored as tensors in data, so calling tonumpy()/numpy() will extract their values only.
for s,l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())
for s,l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())

training_labels_final=np.array(training_labels)
testing_labels_final=np.array(testing_labels)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size=10000
embed_dim=16
max_len=120
trunk_type='post'
oov_tok='<OOV>'

tokenizer=Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index=tokenizer.word_index
sequences=tokenizer.texts_to_sequences(training_sentences)
padded=pad_sequences(sequences,maxlen=max_len,truncating=trunk_type)

testing_sequences=tokenizer.texts_to_sequences(testing_sentences)
testing_padded=pad_sequences(testing_sequences,maxlen=max_len)

# result of embedding layer will be a 2d array with length of sentence and embedding dimension.
model=tf.keras.Sequential([
                           tf.keras.layers.Embedding(vocab_size,embed_dim,input_length=max_len),
                           tf.keras.layers.Flatten(),
                           tf.keras.layers.Dense(6,activation='relu'),
                           tf.keras.layers.Dense(1,activation='sigmoid')
                          ])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(padded,
          training_labels_final,
          epochs=25,
          validation_data=(testing_padded,testing_labels_final))
model.summary()


#visualizing embeddings 3d dimensions:-
#reversing word index

reversed_word_index=dict([(value,key) for (key,value) in word_index.items()])
