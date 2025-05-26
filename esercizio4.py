import pandas as pd

# Esempio di DataFrame recensioni
recensioni_film = pd.DataFrame({
    'ID Film': [1, 1, 2, 2, 2, 3],
    'Titolo Film': ['Inception', 'Inception', 'Matrix', 'Matrix', 'Matrix', 'Avatar'],
    'Punteggio': [9, 8, 10, 9, 8, 7]
})

# Raggruppamento per film
grouped = recensioni_film.groupby(['ID Film', 'Titolo Film'])

# Statistiche
media = grouped['Punteggio'].mean()
conteggio = grouped['Punteggio'].count()
max_punteggio = grouped['Punteggio'].max()
min_punteggio = grouped['Punteggio'].min()

# Risultati
print("Media punteggi per film:\n", media)
print("\nNumero recensioni per film:\n", conteggio)
print("\nPunteggio massimo per film:\n", max_punteggio)
print("\nPunteggio minimo per film:\n", min_punteggio)
