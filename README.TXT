Mehmet Emin Fındıkçı - Veri Madenciliğine Giriş Dersi
23103021104

Yorum Analizi Projesi

Türkçe yorumları analiz eden bir uygulama yaptım. RNN ve LSTM modelleri kullanarak yorumların pozitif mi negatif mi olduğunu anlıyor.

Ne yapıyor bu proje?

- Kullanıcıların yazdığı yorumları alıyor
- İki farklı model (RNN ve LSTM) ile analiz ediyor
- Pozitif, negatif veya kararsız diyor
- Güven yüzdesi de veriyor
- Streamlit ile bir web arayüzü var

Kullandığım Teknolojiler

- Python
- TensorFlow (makine öğrenmesi için)
- Streamlit (web sitesi için)
- Pandas ve NumPy (veri işleme)
- Plotly (grafikler için)

Dosyalar ne işe yarıyor?

- app.py - Ana uygulama, web sitesi
- RnnEgitim.py - RNN modelini eğitiyorum
- LstmEgitim.py - LSTM modelini eğitiyorum
- verihazırlık1.py ve verihazırlık2.py - Veriyi temizleyip hazır ediyorum
- veri.csv - Ham yorum verisi
- hazir_veri.csv - Temizlenmiş veri
- rnn_model.keras ve lstm_model.keras - Eğitilmiş modeller

Grafikleri görmek için:
- metriklerveloss.py - Eğitim sırasında nasıl öğrendiğini gösteriyor
- lstmrnngrafik.py - İki modeli karşılaştırıyor
- matrixveroc.py - Ne kadar doğru tahmin ettiğini gösteriyor



Nasıl çalıştırılır?

Not: Model eğitimi sıfırdan gerçekleştirilmeye gerek yok. Eğitilmiş modeller proje dosyasında var. İsterseniz modeller yeniden eğitilebilir.

1. Önce bilgisayarına Python kur
2. Gerekli paketleri yükle:
   pip install streamlit pandas numpy tensorflow scikit-learn plotly
3. Veriyi hazırla:
   python verihazırlık1.py
   python verihazırlık2.py
4. Modelleri eğit:
   python RnnEgitim.py
   python LstmEgitim.py
5. Uygulamayı başlat:
   streamlit run app.py  

Tarayıcıda açılır

Modeller hakkında

RNN ve LSTM diye iki model kullandım. İkisi de metinleri anlamaya çalışıyor. RNN daha basit, LSTM daha gelişmiş. Hangisi daha iyi diye karşılaştırdım.

Veri setim yaklaşık 380000 yorumdan oluşuyor. Ürün yorumları, puanlara göre pozitif ve negatif diye ayırdım.

