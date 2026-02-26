import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Yorum Analizi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SABİTLER ---
MAX_UZUNLUK = 150

# --- CSS TASARIM (DARK & GLASS) ---
st.markdown("""
    <style>
    /* Arka Plan */
    .stApp {
        background: linear-gradient(to bottom right, #141E30, #243B55);
        color: white;
    }
    
    /* Cam Efektli Kartlar */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Butonlar */
    .stButton>button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
        color: white;
        font-weight: bold;
        border: none;
        height: 50px;
        border-radius: 8px;
        font-size: 16px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(255, 65, 108, 0.7);
    }
    
    /* Input Alanı */
    .stTextArea textarea {
        background-color: rgba(0,0,0,0.3);
        color: white;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .score-text { font-size: 28px; font-weight: 800; margin: 10px 0; }
    .model-label { font-size: 18px; opacity: 0.8; letter-spacing: 1px; }
    
    </style>
    """, unsafe_allow_html=True)

# --- YÜKLEME FONKSİYONLARI ---
@st.cache_resource
def modelleri_yukle():
    try:
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        rnn = load_model('rnn_model.keras')
        lstm = load_model('lstm_model.keras')
        
        # Geçmiş kayıtları (Grafikler için)
        with open('rnn_history.pickle', 'rb') as f: rnn_hist = pickle.load(f)
        with open('lstm_history.pickle', 'rb') as f: lstm_hist = pickle.load(f)
        
        return tokenizer, rnn, lstm, rnn_hist, lstm_hist
    except Exception as e:
        return None, None, None, None, None

# Yüklemeyi Başlat
tokenizer, rnn_model, lstm_model, rnn_hist, lstm_hist = modelleri_yukle()

if not tokenizer:
    st.error("🚨 SİSTEM HATASI: Model dosyaları bulunamadı! Lütfen eğitim kodlarını (02 ve 03) çalıştırın.")
    st.stop()

# --- ANA EKRAN ---
st.markdown("<h1 style='text-align: center;'>Yorum Analizi</h1>", unsafe_allow_html=True)

tab1 = st.tabs(["🧪 CANLI TEST "])[0]

# CANLI TEST

with tab1:
    col_input, col_results = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("📝 Yorum Girişi")
        user_input = st.text_area("Analiz edilecek yorumu yazın:", height=150, placeholder="Yorumunuzu buraya girin...")
        
        analyze_btn = st.button("🚀 ANALİZ ET")

    with col_results:
        st.subheader("🔍 Yapay Zeka Sonuçları")
        
        if analyze_btn and user_input:
            with st.spinner("Modeller düşünüyor..."):
                time.sleep(0.3) 
                
                # Tokenizer ve Pad
                seq = tokenizer.texts_to_sequences([user_input])
                pad = pad_sequences(seq, maxlen=MAX_UZUNLUK)
                
                # Tahminler (0 ile 1 arası olasılık)
                rnn_raw = rnn_model.predict(pad)[0][0]
                lstm_raw = lstm_model.predict(pad)[0][0]

            # --- GÜVEN PUANI HESAPLAMA  ---
            # Eğer değer 0.5'ten küçükse, tersini alıyoruz (1 - p)
            # Böylece %1.5 değil, %98.5 Negatif görüyoruz.
            
            def analiz_yap(raw_prob):
                if raw_prob > 0.5:
                    label = "POZİTİF"
                    icon = "😊"
                    color = "#00ff88" # Yeşil
                    confidence = raw_prob * 100
                else:
                    label = "NEGATİF"
                    icon = "😡"
                    color = "#ff4d4d" # Kırmızı
                    confidence = (1 - raw_prob) * 100 # Tersine çevir
                
                # Kararsızlık Bölgesi (%45 - %55 arası)
                if 45 < confidence < 55:
                    label = "KARARSIZ"
                    icon = "😐"
                    color = "#ffd700" # Sarı
                
                return label, icon, color, confidence

            r_lbl, r_icon, r_col, r_conf = analiz_yap(rnn_raw)
            l_lbl, l_icon, l_col, l_conf = analiz_yap(lstm_raw)
            
            # KARTLAR
            c_rnn, c_lstm = st.columns(2)
            
            # RNN
            with c_rnn:
                st.markdown(f"""
                <div class="glass-card" style="border-top: 5px solid {r_col};">
                    <div class="model-label" style="color:#ff9999;">🔴 RNN Modeli</div>
                    <div class="score-text" style="color:{r_col};">{r_lbl} {r_icon}</div>
                    <div style="font-size: 16px;">Güven: <b>%{r_conf:.2f}</b></div>
                </div>
                """, unsafe_allow_html=True)
                
            # LSTM
            with c_lstm:
                st.markdown(f"""
                <div class="glass-card" style="border-top: 5px solid {l_col};">
                    <div class="model-label" style="color:#99ff99;">🟢 LSTM Modeli</div>
                    <div class="score-text" style="color:{l_col};">{l_lbl} {l_icon}</div>
                    <div style="font-size: 16px;">Güven: <b>%{l_conf:.2f}</b></div>
                </div>
                """, unsafe_allow_html=True)