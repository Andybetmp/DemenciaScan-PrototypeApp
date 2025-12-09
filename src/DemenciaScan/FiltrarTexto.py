import re

def riqueza_lexica(texto):
    palabras = texto.split()
    return len(set(palabras)) / len(palabras) if len(palabras) > 0 else 0

df['num_palabras'] = df['texto'].apply(lambda x: len(x.split()))
df['riqueza_lexica'] = df['texto'].apply(riqueza_lexica)

print(df[['num_palabras', 'riqueza_lexica']].describe())
