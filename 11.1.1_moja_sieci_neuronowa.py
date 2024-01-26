#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Przykładowy tekst do treningu modelu
with open('bulhakow-fatalne-jaja.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenizacja tekstu
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Przygotowanie danych treningowych
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

X, y = input_sequences[:,:-1],input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Budowa modelu LSTM
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_sequence_length-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Trening modelu
model.fit(X, y, epochs=25, verbose=1)

# Funkcja do generowania tekstu
def generate_text(seed_text, next_words, model, max_sequence_length):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        #na podstawie predykcji ma wygenerować kolejne słowa
        
        # Wybieranie indeksu z najwyższym prawdopodobieństwem
        predicted_index = tf.argmax(predicted_probs, axis=-1).numpy()
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Generowanie tekstu z seed_text o długości 10 słów
generated_text = generate_text("Twój model treninguje tekst", 10, model, max_sequence_length)
print(generated_text)

model.save('moj_model.h5')

with open("wynik.txt", 'w', encoding='utf-8') as file:
    file.write(generated_text)
    #zapisuje wygenerowany tekst; wartość tej zmiennej 
    #bo system kolejkowy nie wyrzuca informacji, tylko działa w tle; więc wszystkie outputy trzeba sobie dodatkowo zapisać

model.save('moj_model.h5')
# In[ ]:




