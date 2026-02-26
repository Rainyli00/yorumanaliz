import pandas as pd
import sys
import io

# Türkçe karakter ayarı
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- AYARLAR ---

DOSYA_ADI = 'veri.csv' 


try:
    print(f"⏳ '{DOSYA_ADI}' okunuyor...")
    df = pd.read_csv(DOSYA_ADI)
    
    print(f"📄 Sütunlar: {df.columns.tolist()}")
    

    # Yorum sütununu string yapar ve Text olarak atar
    df['Text'] = df['Yorum'].astype(str)
    
    # Boş yorum varsa  onları temizler
    df = df[df['Text'].str.len() > 2] # 2 harften kısa yorumları atar
    
    # 2. PUANLARI DÖNÜŞTÜR (20-100 -> 0-1)
    print("⚙️ Puanlar (0-1) etiketine çevriliyor...")
    
    def puan_donustur(puan):
        try:
            p = int(puan)
            if p >= 80: return 1  # 80 ve 100 -> Pozitif
            elif p <= 40: return 0 # 20 ve 40 -> Negatif
            else: return -1        # 60 vs -> Nötr
        except:
            return -1

    df['label'] = df['Puan'].apply(puan_donustur)
    
    # Nötrleri (-1) atar
    df = df[df['label'] != -1]
    
    # 3. DENGELEME İŞLEMİ
    negatifler = df[df['label'] == 0]
    pozitifler = df[df['label'] == 1]
    
    sayi_neg = len(negatifler)
    sayi_poz = len(pozitifler)
    
    print(f"\n📊 DURUM:")
    print(f"🔴 Negatif Sayısı: {sayi_neg}")
    print(f"🟢 Pozitif Sayısı: {sayi_poz}")
    
    # Eşitleme
    limit = min(sayi_neg, sayi_poz)
    print(f"\n✂️ Dengeleme: Her iki taraftan {limit} adet alınıyor...")
    
    secilen_neg = negatifler.sample(n=limit, random_state=42)
    secilen_poz = pozitifler.sample(n=limit, random_state=42)
    
    # Birleştir
    df_dengeli = pd.concat([secilen_neg, secilen_poz])
    df_dengeli = df_dengeli.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Sadece lazım olanlar
    df_dengeli = df_dengeli[['label', 'Text']]
    
    # Kaydet
    df_dengeli.to_csv('dengeli_veri.csv', index=False, encoding='utf-8')
    
    print(f"\n✅ İŞLEM TAMAM! 'dengeli_veri.csv' hazır.")
    print(f"Toplam Satır: {len(df_dengeli)}")
    print("👉 Sıradaki adım: 01_hazirlik.py")

except Exception as e:
    print(f"❌ HATA: {e}")
    print("Dosya adını kontrol et ve sütun isminin 'Yorum' (büyük harfle) olduğundan emin ol.")