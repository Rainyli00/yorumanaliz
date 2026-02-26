import pandas as pd
import sys
import io

# Türkçe karakter ayarı
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("📊 veri.csv ANALİZİ BAŞLIYOR...")

try:
    # 1. Dosyayı Oku
    # Eğer hata verirse encoding='utf-16' veya 'latin-1' denenebilir.
    df = pd.read_csv('veri.csv')
    
    print(f"📂 'veri.csv' Yüklendi: {len(df)} Satır")
    
    # 2. Puanları Etikete Çevir
    # 80 ve 100 -> POZİTİF
    # 20 ve 40  -> NEGATİF
    # 60        -> NÖTR (Ara değer)
    
    def puan_cevir(puan):
        try:
            puan = int(puan) # Garanti olsun diye sayıya çevir
            if puan >= 80: return "POZİTİF"
            elif puan <= 40: return "NEGATİF"
            else: return "NÖTR"
        except:
            return "HATALI SATIR"

    # 'Durum' diye yeni bir sütun açıp etiketleri yazıyoruz
    df['Durum'] = df['Puan'].apply(puan_cevir)

    # 3. Sayım Yap
    sayim = df['Durum'].value_counts()
    
    negatif_sayisi = sayim.get('NEGATİF', 0)
    pozitif_sayisi = sayim.get('POZİTİF', 0)
    notr_sayisi    = sayim.get('NÖTR', 0)
    toplam         = len(df)
    
    # 4. Raporu Bas
    print("\n" + "="*35)
    print("📈 PUAN VE DUYGU DAĞILIMI")
    print("="*35)
    print(f"🟢 POZİTİF Yorumlar : {pozitif_sayisi:,}".replace(',', '.'))
    print(f"🔴 NEGATİF Yorumlar : {negatif_sayisi:,}".replace(',', '.'))
    print(f"⚪ NÖTR Yorumlar    : {notr_sayisi:,}".replace(',', '.'))
    print("-" * 35)
    print(f"📦 TOPLAM SATIR     : {toplam:,}".replace(',', '.'))
    print("="*35)
    
    # İstersen oranları da görebilirsin
    if toplam > 0:
        print(f"📊 Oran: %{pozitif_sayisi/toplam*100:.1f} Pozitif - %{negatif_sayisi/toplam*100:.1f} Negatif")

except FileNotFoundError:
    print("❌ HATA: 'veri.csv' dosyası bulunamadı! Lütfen dosya adını kontrol et.")
except Exception as e:
    print(f"❌ BEKLENMEDİK HATA: {e}")