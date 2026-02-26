import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import sys
import io

# Türkçe karakter ayarı
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')



# --- VERİYİ YÜKLER ---
try:
    df = pd.read_csv('hazir_veri.csv')
    df['Text'] = df['Text'].astype(str) 
    print(f" Veri seti yüklendi: {len(df)} satır")
except:
    print("❌ HATA: 'hazir_veri.csv' bulunamadı.")
    sys.exit()

# --- AYARLAR (Görsel Kalite) ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# ==========================================
# 1. YORUM DAĞILIMI 
# ==========================================

plt.figure(figsize=(8, 8))

# Sayım yap
counts = df['label'].value_counts()
labels = ['Negatif (0)', 'Pozitif (1)']
colors = ['#ff4d4d', '#00ff88'] 

# Pasta Grafiği
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, 
        explode=(0.05, 0), shadow=True, textprops={'fontsize': 14, 'weight': 'bold'})
plt.title('Veri Seti Dengesi: Negatif vs Pozitif', fontsize=16)

plt.savefig('grafik_1_dagilim.png') 
print(" 'grafik_1_dagilim.png' kaydedildi.")
plt.close()

# ==========================================
# 2. YORUM UZUNLUKLARI DAĞILIMI
# ==========================================
print("📏 2. Grafik: Yorum Uzunlukları Hesaplanıyor...")

df['kelime_sayisi'] = df['Text'].apply(lambda x: len(x.split()))

plt.figure(figsize=(12, 6))
sns.histplot(df['kelime_sayisi'], bins=50, kde=True, color='purple')
plt.title('Yorum Uzunluk Dağılımı (Kelime Sayısı)', fontsize=16)
plt.xlabel('Kelime Sayısı')
plt.ylabel('Yorum Adedi')
plt.xlim(0, 100) 
plt.savefig('grafik_2_uzunluk.png')
print(" 'grafik_2_uzunluk.png' kaydedildi.")
plt.close()
