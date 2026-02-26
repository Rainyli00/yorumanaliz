import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- VERİ AYARLARI  ---
MAX_UZUNLUK = 150    
KELIME_SAYISI = 50000 
EPOCH_SAYISI = 20     
BATCH_SIZE = 1024     
EMBEDDING_DIM = 128   

print(f"🔵 RNN EĞİTİMİ BAŞLIYOR")

try:
    data = pd.read_csv('hazir_veri.csv')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except:
    print("HATA: Dosyalar eksik.")
    sys.exit()

# Veriyi Sayısal Hale Getirir
data['Text'] = data['Text'].astype(str)
X = tokenizer.texts_to_sequences(data['Text'].values)
X = pad_sequences(X, maxlen=MAX_UZUNLUK)
Y = data['label'].values

print(f"🥊 Model {len(X)} satır veriyle eğitiliyor...")

# Model Mimarisi
model = Sequential()
model.add(Embedding(KELIME_SAYISI, EMBEDDING_DIM))
model.add(SimpleRNN(128, dropout=0.3, recurrent_dropout=0.3)) 
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

history = model.fit(X, Y, epochs=EPOCH_SAYISI, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1, callbacks=[early_stop])

# Kaydeder
model.save('rnn_model.keras')
with open('rnn_history.pickle', 'wb') as f: pickle.dump(history.history, f)

print("✅ RNN EĞİTİMİ BİTTİ!")