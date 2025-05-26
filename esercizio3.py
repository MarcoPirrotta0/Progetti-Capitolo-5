import pandas as pd

# Clienti
clienti = pd.DataFrame({
    'ID Cliente': [1, 2, 3],
    'Nome': ['Alice', 'Bob', 'Charlie']
})

# Ordini
ordini = pd.DataFrame({
    'ID Ordine': [101, 102, 103, 104],
    'ID Cliente': [1, 2, 4, 2],
    'Importo': [250, 150, 300, 180]
})

# Inner Join
inner = pd.merge(clienti, ordini, on='ID Cliente', how='inner')
print("INNER JOIN:")
print(inner)

# Left Join
left = pd.merge(clienti, ordini, on='ID Cliente', how='left')
print("\nLEFT JOIN:")
print(left)

# Right Join
right = pd.merge(clienti, ordini, on='ID Cliente', how='right')
print("\nRIGHT JOIN:")
print(right)
