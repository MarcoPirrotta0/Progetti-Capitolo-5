import pandas as pd
import re
from textblob import TextBlob
from deep_translator import GoogleTranslator

feedbacks = pd.Series([
    "Ottimo prodotto! Super soddisfatto!!!",
    "Pessimo servizio, mai più.",
    "Qualità buona, ma spedizione lenta..."
])

# Pulizia base del testo
def pulisci_testo(testo):
    testo = re.sub(r'[^a-zA-Z0-9À-ÿ\s]', '', testo)
    testo = re.sub(r'\s+', ' ', testo)
    return testo.strip().lower()

# Pulizia
feedbacks_puliti = feedbacks.apply(pulisci_testo)

# Traduzione in inglese (source='auto' permette rilevamento automatico)
feedbacks_tradotti = feedbacks_puliti.apply(
    lambda x: GoogleTranslator(source='auto', target='en').translate(x)
)

# Analisi del sentiment con TextBlob
sentiments = feedbacks_tradotti.apply(lambda x: TextBlob(x).sentiment.polarity)

# Creazione DataFrame finale
df_feedback = pd.DataFrame({
    'Feedback Originale': feedbacks,
    'Feedback Pulito': feedbacks_puliti,
    'Tradotto (EN)': feedbacks_tradotti,
    'Sentiment': sentiments
})

print(df_feedback)
print("\nSentiment medio:", sentiments.mean())

# Esportazione
df_feedback.to_csv("feedback_analisi_tradotto.csv", index=False)
