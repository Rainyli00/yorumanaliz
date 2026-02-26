import pandas as pd
import pickle
import sys
import io
from tensorflow.keras.preprocessing.text import Tokenizer 

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- AYARLAR ---
KELIME_SAYISI = 50000  # En sık kullanılan 50.000 kelimeyi öğrenir

print(f"🚀 HAZIRLIK SÜRECİ BAŞLIYOR...")

try:
    print("⏳ 'dengeli_veri.csv' okunuyor...")
    df = pd.read_csv('dengeli_veri.csv')
    
    # Metinleri String'e çevirir sıkıntı olmaması için
    df['Text'] = df['Text'].astype(str)
    
except FileNotFoundError:
    print("❌ HATA: 'dengeli_veri.csv' yok!")
    sys.exit()

# Tokenizer Eğitir
print(f"📚 Sözlük oluşturuluyor (Bu işlem veri boyutuna göre 1-2 dk sürebilir)...")
tokenizer = Tokenizer(num_words=KELIME_SAYISI, oov_token="<OOV>") 
tokenizer.fit_on_texts(df['Text'].values)

print(f"✅ Sözlük tamamlandı! Toplam kelime hazinesi: {len(tokenizer.word_index)}")

# Kaydet
print("💾 'tokenizer.pickle' ve 'hazir_veri.csv' kaydediliyor...")
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

df.to_csv('hazir_veri.csv', index=False, encoding='utf-8')

print("\n🎉 HAZIRLIK BİTTİ! Şimdi 02 ve 03 numaralı eğitim kodlarını çalıştır.")