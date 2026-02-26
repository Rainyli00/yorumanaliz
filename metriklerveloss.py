import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, roc_curve
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import io
import time

# Türkçe karakter ayarı
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- AYARLAR ---
MAX_UZUNLUK = 150
BATCH_SIZE = 1024 

print("🚀 MEGA SUNUM MODU: Türkçe proje analiz ediliyor...")

# --- VERİ VE MODELLERİ YÜKLER ---
try:
    print("📂 Veriler okunuyor...")
    data = pd.read_csv('hazir_veri.csv')
    
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    print("⏳ Modeller ve Geçmiş Kayıtlar yükleniyor...")
    rnn_model = load_model('rnn_model.keras')
    lstm_model = load_model('lstm_model.keras')
    
    
    with open('rnn_history.pickle', 'rb') as f: rnn_hist = pickle.load(f)
    with open('lstm_history.pickle', 'rb') as f: lstm_hist = pickle.load(f)
    
except Exception as e:
    print(f"❌ HATA: Dosyalar eksik! Detay: {e}")
    sys.exit()

# Veri Hazırlığı
data['Text'] = data['Text'].astype(str)
X = tokenizer.texts_to_sequences(data['Text'].values)
X = pad_sequences(X, maxlen=MAX_UZUNLUK)
Y = data['label'].values

# Test Verisi Ayırır (%20)
_, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"🧪 Toplam {len(X_test)} adet GİZLİ TEST VERİSİ üzerinde sınav yapılıyor...")

# --- TAHMİNLER  ---
print("🤖 RNN Tahmin Ediyor...")
start = time.time()
y_pred_rnn_prob = rnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1).ravel()
rnn_time = time.time() - start
y_pred_rnn = (y_pred_rnn_prob > 0.5).astype(int)

print("🧠 LSTM Tahmin Ediyor...")
start = time.time()
y_pred_lstm_prob = lstm_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1).ravel()
lstm_time = time.time() - start
y_pred_lstm = (y_pred_lstm_prob > 0.5).astype(int)

# --- METRİK HESAPLAMA FONKSİYONU ---
def skorlari_al(y_true, y_pred, y_prob):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob),
        'MCC': matthews_corrcoef(y_true, y_pred) 
    }

rnn_scores = skorlari_al(Y_test, y_pred_rnn, y_pred_rnn_prob)
lstm_scores = skorlari_al(Y_test, y_pred_lstm, y_pred_lstm_prob)

# --- KONSOLA YAZDIRMA ---
print("\n" + "="*60)
print(f"{'METRİK':<15} | {'RNN':<10} | {'LSTM':<10} | {'FARK':<10}")
print("="*60)
for k in rnn_scores.keys():
    fark = lstm_scores[k] - rnn_scores[k]
    print(f"{k:<15} | {rnn_scores[k]:.4f}     | {lstm_scores[k]:.4f}     | {fark:+.4f}")
print("="*60)
print(f"{'Hız (Saniye)':<15} | {rnn_time:.2f} sn    | {lstm_time:.2f} sn")


# ==========================================
# GRAFİK 1: METRİKLER KARŞILAŞTIRMASI GRAFİĞİ
# ==========================================
metrikler = ['Accuracy', 'F1-Score', 'Recall', 'AUC']
rnn_vals = [rnn_scores[m] for m in metrikler]
lstm_vals = [lstm_scores[m] for m in metrikler]

x = np.arange(len(metrikler))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, rnn_vals, width, label='RNN', color='#ff9999') # Kırmızımsı
rects2 = ax.bar(x + width/2, lstm_vals, width, label='LSTM', color='#66b3ff') # Mavimsi

ax.set_ylabel('Puan')
ax.set_title('RNN vs LSTM')
ax.set_xticks(x)
ax.set_xticklabels(metrikler)
ax.legend()
ax.set_ylim(0.5, 1.05) 
plt.grid(axis='y', linestyle='--', alpha=0.7)

def etiket_yaz(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

etiket_yaz(rects1)
etiket_yaz(rects2)
plt.show()

# ==========================================
# GRAFİK 2: LOSS (HATA) EĞRİLERİ 
# ==========================================
plt.figure(figsize=(10, 6))
# RNN
plt.plot(rnn_hist['loss'], label='RNN Eğitim Hatası', color='salmon', linestyle='--')
plt.plot(rnn_hist['val_loss'], label='RNN Test Hatası', color='red', linewidth=2)
# LSTM
plt.plot(lstm_hist['loss'], label='LSTM Eğitim Hatası', color='lightgreen', linestyle='--')
plt.plot(lstm_hist['val_loss'], label='LSTM Test Hatası', color='green', linewidth=2)

plt.title('Hata (Loss) Değişimi')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# GRAFİK 3: ACCURACY (DOĞRULUK) EĞRİLERİ 
# ==========================================
plt.figure(figsize=(10, 6))
# RNN
plt.plot(rnn_hist['accuracy'], label='RNN Eğitim Başarısı', color='salmon', linestyle='--')
plt.plot(rnn_hist['val_accuracy'], label='RNN Test Başarısı', color='red', linewidth=2)
# LSTM
plt.plot(lstm_hist['accuracy'], label='LSTM Eğitim Başarısı', color='lightgreen', linestyle='--')
plt.plot(lstm_hist['val_accuracy'], label='LSTM Test Başarısı', color='green', linewidth=2)

plt.title('Doğruluk (Accuracy) Değişimi')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.ylim(0.8, 1.0) 
plt.show()

