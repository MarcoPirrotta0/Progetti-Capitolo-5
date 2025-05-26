
import pandas as pd

nomi_prodotti = pd.Series(['Laptop', 'Mouse', 'Tastiera'], index=['P001', 'P002', 'P003'])
quantita_disponibili = pd.Series([10, 50, 30], index=['P001', 'P002', 'P003'])

# Combinazione in un DataFrame
inventario_prodotti = pd.DataFrame({
    'Nome Prodotto': nomi_prodotti,
    'Quantità Disponibile': quantita_disponibili
})

print("Inventario Prodotti:")
print(inventario_prodotti)

# Accesso alla quantità disponibile per il prodotto con codice 'P002'
print("Quantità disponibile per P002:", inventario_prodotti.loc['P002', 'Quantità Disponibile'])
# loc is used for label-based indexing