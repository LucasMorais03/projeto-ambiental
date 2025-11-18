# src/inspect_parquet.py
import pandas as pd
import os
p = "data/processed"
# procura o arquivo parquet mais recente que comece com 'autuacoes_processed'
files = [os.path.join(p,f) for f in os.listdir(p) if f.startswith("autuacoes_processed") and f.endswith(".parquet")]
if not files:
    raise SystemExit("Nenhum arquivo autuacoes_processed*.parquet encontrado em data/processed")
files = sorted(files)
latest = files[-1]
print("Arquivo lido:", latest)
df = pd.read_parquet(latest)
print("\nNúmero de linhas, colunas:", df.shape)
print("\nColunas:")
print(df.columns.tolist())
print("\nPrimeiras 5 linhas:")
print(df.head(5).to_string(index=False))
