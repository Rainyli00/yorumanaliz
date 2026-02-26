import matplotlib.pyplot as plt
import pickle
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


try:
    with open('rnn_history.pickle', 'rb') as f: rnn = pickle.load(f)
    with open('lstm_history.pickle', 'rb') as f: lstm = pickle.load(f)
except:
    print("HATA: Pickle dosyaları yok.")
    sys.exit()

def get_best(history, metric='val_accuracy', mode='max'):
    vals = history[metric]
    best_val = max(vals) if mode == 'max' else min(vals)
    best_ep = vals.index(best_val) + 1
    return best_ep, best_val

# --- RNN ---
ep_rnn, val_rnn = get_best(rnn, 'val_accuracy', 'max')
ep_rnn_loss, val_rnn_loss = get_best(rnn, 'val_loss', 'min')
epochs = range(1, len(rnn['accuracy']) + 1)

plt.figure(figsize=(10, 8))
# Accuracy
plt.subplot(2, 1, 1)
plt.plot(epochs, rnn['accuracy'], label='Eğitim', color='salmon', marker='o')
plt.plot(epochs, rnn['val_accuracy'], label='Test', color='darkred', lw=3, marker='o')
plt.plot(ep_rnn, val_rnn, marker='*', color='gold', ms=20, mec='black', zorder=10)
plt.annotate(f'Zirve: %{val_rnn*100:.1f}', xy=(ep_rnn, val_rnn), xytext=(ep_rnn, val_rnn-0.08), arrowprops=dict(facecolor='black', shrink=0.05), ha='center', fontweight='bold')
plt.title(f'RNN BAŞARISI (Zirve: Epoch {ep_rnn})', fontweight='bold')
plt.ylabel('Doğruluk')
plt.xticks(epochs)
plt.legend()
plt.grid(True, alpha=0.3)

# Loss
plt.subplot(2, 1, 2)
plt.plot(epochs, rnn['loss'], label='Eğitim', color='salmon', marker='o')
plt.plot(epochs, rnn['val_loss'], label='Test', color='darkred', lw=3, marker='o')
plt.plot(ep_rnn_loss, val_rnn_loss, marker='*', color='gold', ms=20, mec='black', zorder=10)
plt.title('RNN HATA ORANI', fontweight='bold')
plt.ylabel('Hata')
plt.xlabel('Epoch')
plt.xticks(epochs)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# --- LSTM ---
ep_lstm, val_lstm = get_best(lstm, 'val_accuracy', 'max')
ep_lstm_loss, val_lstm_loss = get_best(lstm, 'val_loss', 'min')
epochs_l = range(1, len(lstm['accuracy']) + 1)

plt.figure(figsize=(10, 8))
# Accuracy
plt.subplot(2, 1, 1)
plt.plot(epochs_l, lstm['accuracy'], label='Eğitim', color='lightgreen', marker='o')
plt.plot(epochs_l, lstm['val_accuracy'], label='Test', color='darkgreen', lw=3, marker='o')
plt.plot(ep_lstm, val_lstm, marker='*', color='gold', ms=20, mec='black', zorder=10)
plt.annotate(f'Zirve: %{val_lstm*100:.1f}', xy=(ep_lstm, val_lstm), xytext=(ep_lstm, val_lstm-0.08), arrowprops=dict(facecolor='black', shrink=0.05), ha='center', fontweight='bold')
plt.title(f'LSTM BAŞARISI (Zirve: Epoch {ep_lstm})', fontweight='bold')
plt.ylabel('Doğruluk')
plt.xticks(epochs_l)
plt.legend()
plt.grid(True, alpha=0.3)

# Loss
plt.subplot(2, 1, 2)
plt.plot(epochs_l, lstm['loss'], label='Eğitim', color='lightgreen', marker='o')
plt.plot(epochs_l, lstm['val_loss'], label='Test', color='darkgreen', lw=3, marker='o')
plt.plot(ep_lstm_loss, val_lstm_loss, marker='*', color='gold', ms=20, mec='black', zorder=10)
plt.title('LSTM HATA ORANI', fontweight='bold')
plt.ylabel('Hata')
plt.xlabel('Epoch')
plt.xticks(epochs_l)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()