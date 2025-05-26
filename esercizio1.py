
import numpy as np


sales_data = np.array([
    [15000, 18000, 17000, 16000],  # Regione A
    [20000, 21000, 19000, 22000],  # Regione B
    [13000, 14000, 13500, 15000]   # Regione C
])

# Vendite totali annuali per regione
total_per_region = np.sum(sales_data, axis=1)
print("Vendite annuali totali per regione:", total_per_region)

# Trimestre con vendite massime per ogni regione (1-based)
max_quarter_per_region = np.argmax(sales_data, axis=1)
print("Trimestre con vendite massime per regione:", max_quarter_per_region + 1)

# Media trimestrale complessiva
overall_avg = np.mean(sales_data)
print("Media trimestrale complessiva:", overall_avg)

# Vendita minima tra tutte le regioni e trimestri
min_sale = np.min(sales_data)
print("Vendita minima assoluta:", min_sale)
