import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
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

print("🟢 LSTM EĞİTİMİ BAŞLIYOR")
try:
    data = pd.read_csv('hazir_veri.csv')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except:
    print("HATA: Dosyalar eksik.")
    sys.exit()

data['Text'] = data['Text'].astype(str)
X = tokenizer.texts_to_sequences(data['Text'].values)
X = pad_sequences(X, maxlen=MAX_UZUNLUK)
Y = data['label'].values

print(f"🧠 LSTM Model hazırlanıyor...")

# --- LSTM MİMARİSİ ---
model = Sequential()
model.add(Embedding(KELIME_SAYISI, EMBEDDING_DIM))
model.add(SpatialDropout1D(0.4)) 
model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3)) 
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- EARLY STOPPING ---
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

history = model.fit(X, Y, 
                    epochs=EPOCH_SAYISI, 
                    batch_size=BATCH_SIZE, 
                    validation_split=0.2, 
                    verbose=1, 
                    callbacks=[early_stop])

# Kaydet
model.save('lstm_model.keras')
with open('lstm_history.pickle', 'wb') as f: pickle.dump(history.history, f)

print("✅ LSTM EĞİTİMİ BİTTİ!")