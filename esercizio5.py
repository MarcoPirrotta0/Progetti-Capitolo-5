import pandas as pd
import re
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import defaultdict

# 1) Carica il dataset Amazon
df_amazon = pd.read_csv(
    'amazon_cells_labelled.txt',
    names=['review', 'sentiment'],
    sep='\t'
)

# 2) Split in training e test
X_train, X_test, y_train, y_test = train_test_split(
    df_amazon['review'],
    df_amazon['sentiment'],
    test_size=0.2,
    random_state=42
)

# 3) Vettorizzazione con Bag-of-Words
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4) Addestra il modello
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# 5) Valutazione
accuracy = clf.score(X_test_vec, y_test)
print(f"\nAccuracy sul test set: {accuracy:.3f}")

# 6) Feedback italiani
feedbacks = pd.Series([
    "Ottimo prodotto! Super soddisfatto!!!",
    "Pessimo servizio, mai più.",
    "Qualità buona, ma spedizione lenta...",
    "Consegna puntuale, prodotto in perfette condizioni.",
    "Non sono soddisfatto della qualità del materiale.",
    "Esperienza d'acquisto eccellente, tornerò sicuramente a comprare.",
    "Istruzioni poco chiare, ci ho messo troppo tempo a montare il prodotto.",
    "Ottimo rapporto qualità/prezzo, consigliato!"
])

# 7) Pulizia del testo
def pulisci(testo):
    testo = re.sub(r'[^a-zA-Z0-9À-ÿ\s]', '', testo)
    testo = re.sub(r'\s+', ' ', testo)
    return testo.strip().lower()

puliti = feedbacks.apply(pulisci)

# 8) Traduzione in inglese
tradotti = puliti.apply(lambda s: GoogleTranslator(source='auto', target='en').translate(s))

# 9) Vettorizzazione + predizione
X_new = vectorizer.transform(tradotti)
preds = clf.predict(X_new)
probs = clf.predict_proba(X_new)[:, 1]

# 10) Funzione per etichettare il sentiment
def sentiment_label(score):
    if score >= 0.66:
        return 'Positivo'
    elif score <= 0.33:
        return 'Negativo'
    else:
        return 'Neutro'

sentiments = [sentiment_label(p) for p in probs]

# 11) Estrai parole chiave (non-stopwords)
def estrai_keywords(text):
    return [w for w in text.lower().split() if w not in ENGLISH_STOP_WORDS]

keywords_list = tradotti.apply(estrai_keywords)

# 12) Calcola il sentiment medio
sentiment_medio = probs.mean()

# 13) Crea DataFrame dei risultati
df_out = pd.DataFrame({
    'Feedback Originale': feedbacks,
    'Feedback Pulito': puliti,
    'Tradotto (EN)': tradotti,
    'Parole Chiave': keywords_list.apply(lambda x: ', '.join(x)),
    'Predizione (0=Neg,1=Pos)': preds,
    'Score Positività': probs.round(3),
    'Sentiment': sentiments
})

# 14) Aggiungi riga con il sentiment medio
df_out.loc[len(df_out)] = [
    '[MEDIA]', '', '', '', '', round(sentiment_medio, 3), 'Sentiment Medio'
]

# 15) Salva su CSV
df_out.to_csv('feedback_analisi_completa.csv', index=False)
print("\nRisultati salvati in 'feedback_analisi_completa.csv'")

# 16) Stampa dettagliata nel terminale
print("\n--- Analisi Dettagliata dei Feedback ---")
for i, row in df_out.iterrows():
    if row['Feedback Originale'] == '[MEDIA]':
        continue

    index = i + 1
    keywords = estrai_keywords(row['Tradotto (EN)'])
    sentiment = row['Sentiment']
    score = row['Score Positività']
    original = row['Feedback Originale']

    print(f"\nComment #{index}")
    print(f"Keywords: {', '.join(keywords)}")
    print(f"Original: {original}")
    print(f"Sentiment: {sentiment} (Score: {score:.3f})")
