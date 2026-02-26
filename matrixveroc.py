import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import io
import time

# Türkçe karakter ayarı
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- VERİ AYARLARI ---
MAX_UZUNLUK = 150  
BATCH_SIZE = 1024  

print("📊 GÖRSEL ANALİZ MODU BAŞLIYOR...")
print("⏳ Veriler ve Modeller yükleniyor (RAM'i hazırla)...")

try:
    data = pd.read_csv('hazir_veri.csv')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    rnn_model = load_model('rnn_model.keras')
    lstm_model = load_model('lstm_model.keras')
    print("✅ Dosyalar yüklendi!")
except Exception as e:
    print(f"HATA: Dosyalar eksik! Detay: {e}")
    sys.exit()

# Veri Hazırlığı
data['Text'] = data['Text'].astype(str)
X = tokenizer.texts_to_sequences(data['Text'].values)
X = pad_sequences(X, maxlen=MAX_UZUNLUK)
Y = data['label'].values

# %20 GİZLİ TEST VERİSİNİ AYIRIR
print("🧪 Test verileri ayrıştırılıyor...")
_, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"🔎 {len(X_test)} adet GİZLİ TEST sorusu üzerinde görsel analiz yapılıyor...")

# --- TAHMİNLER ---
print("🤖 RNN Tahmin Ediyor...")
y_pred_rnn_prob = rnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1).ravel()
y_pred_rnn = (y_pred_rnn_prob > 0.5).astype(int)

print("🧠 LSTM Tahmin Ediyor...")
y_pred_lstm_prob = lstm_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1).ravel()
y_pred_lstm = (y_pred_lstm_prob > 0.5).astype(int)

# --- GRAFİK 1: CONFUSION MATRIX ---

def plot_cm(y_true, y_pred, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'], 
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Gerçek Durum')
    ax.set_xlabel('Model Tahmini')

print("🎨 Grafikler çiziliyor...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_cm(Y_test, y_pred_rnn, "RNN Başarısı", axes[0])
plot_cm(Y_test, y_pred_lstm, "LSTM Başarısı", axes[1])
plt.tight_layout()
plt.show()

# --- GRAFİK 2: ROC EĞRİSİ ---
fpr_rnn, tpr_rnn, _ = roc_curve(Y_test, y_pred_rnn_prob)
auc_rnn = auc(fpr_rnn, tpr_rnn)

fpr_lstm, tpr_lstm, _ = roc_curve(Y_test, y_pred_lstm_prob)
auc_lstm = auc(fpr_lstm, tpr_lstm)

plt.figure(figsize=(9, 7))
plt.plot(fpr_rnn, tpr_rnn, color='red', lw=2, label=f'RNN (Puan: {auc_rnn:.3f})')
plt.plot(fpr_lstm, tpr_lstm, color='green', lw=2, label=f'LSTM (Puan: {auc_lstm:.3f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Rastgele Tahmin')

plt.title('RNN vs LSTM: Zeka Karşılaştırması (ROC)', fontsize=15, fontweight='bold')
plt.xlabel('Hata Oranı ', fontsize=12)
plt.ylabel('Başarı Oranı', fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

print("✅ BİTTİ! Ekran görüntülerini al.")