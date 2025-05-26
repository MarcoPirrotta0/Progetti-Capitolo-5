import pandas as pd
import re
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

feedbacks = pd.Series([
    "Ottimo prodotto! Super soddisfatto!!!",
    "Pessimo servizio, mai più.",
    "Qualità buona, ma spedizione lenta..."
])

# 1) Pulizia base del testo
def pulisci_testo(testo):
    testo = re.sub(r'[^a-zA-Z0-9À-ÿ\s]', '', testo)
    testo = re.sub(r'\s+', ' ', testo)
    return testo.strip().lower()

feedbacks_puliti = feedbacks.apply(pulisci_testo)

# 2) Traduzione in inglese
feedbacks_tradotti = feedbacks_puliti.apply(
    lambda x: GoogleTranslator(source='auto', target='en').translate(x)
)

# 3) Vettorizzazione (Bag-of-Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(feedbacks_tradotti)

# 4) Etichette manuali (1=positivo, 0=negativo)
y = [1, 0, 1]

# 5) Alleniamo il modello
classifier = LogisticRegression()
classifier.fit(X, y)

# 6) Predizioni sullo stesso set (solo demo)
predictions = classifier.predict(X)

# 7) Assemblaggio DataFrame finale
df_feedback = pd.DataFrame({
    'Feedback Originale': feedbacks,
    'Feedback Pulito': feedbacks_puliti,
    'Tradotto (EN)': feedbacks_tradotti,
    'Predizione Sentiment (0/1)': predictions
})

print(df_feedback)

# 8) Esportazione con pandas
df_feedback.to_csv("feedback_analisi_lr.csv", index=False)
