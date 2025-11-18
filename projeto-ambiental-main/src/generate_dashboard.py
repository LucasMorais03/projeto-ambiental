# src/generate_dashboard.py
"""
Gera dashboard.html estático (offline) a partir de data/processed/sample_for_dashboard.parquet (ou CSV)
Corrigido para evitar conflitos de chaves em templates JS.
"""
import os, json
import pandas as pd
from datetime import datetime

BASE = os.getcwd()
PROC = os.path.join(BASE, "data", "processed")
PARQ_PATH = os.path.join(PROC, "sample_for_dashboard.parquet")
CSV_PATH = os.path.join(PROC, "sample_for_dashboard.csv")
OUT_HTML = os.path.join(BASE, "dashboard.html")

def read_sample():
    if os.path.exists(PARQ_PATH):
        try:
            df = pd.read_parquet(PARQ_PATH)
            print("Lido parquet:", PARQ_PATH)
            return df
        except Exception as e:
            print("Falha ao ler parquet:", e)
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, low_memory=False)
        print("Lido csv:", CSV_PATH)
        return df
    # try to find other sample files
    for f in os.listdir(PROC):
        if f.endswith(".parquet") and "sample" in f:
            try:
                df = pd.read_parquet(os.path.join(PROC,f))
                print("Lido (auto) parquet:", f)
                return df
            except Exception:
                pass
        if f.endswith(".csv") and "sample" in f:
            df = pd.read_csv(os.path.join(PROC,f), low_memory=False)
            print("Lido (auto) csv:", f)
            return df
    raise FileNotFoundError("Nenhum sample_for_dashboard.parquet/csv encontrado em data/processed.")

df = read_sample()
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# ensure lat/lon
if "num_latitude_auto" in df.columns:
    df["lat"] = pd.to_numeric(df["num_latitude_auto"], errors="coerce")
elif "lat" in df.columns:
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
else:
    df["lat"] = pd.NA

if "num_longitude_auto" in df.columns:
    df["lon"] = pd.to_numeric(df["num_longitude_auto"], errors="coerce")
elif "lon" in df.columns:
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
else:
    df["lon"] = pd.NA

# valor_multa normalization
if "valor_multa" not in df.columns and "val_auto_infracao" in df.columns:
    df["valor_multa"] = df["val_auto_infracao"].astype(str).apply(lambda s: s.replace(".", "").replace(",", "."))
    df["valor_multa"] = pd.to_numeric(df["valor_multa"], errors="coerce")
elif "valor_multa" in df.columns:
    df["valor_multa"] = pd.to_numeric(df["valor_multa"], errors="coerce")
else:
    df["valor_multa"] = pd.NA

# pred_risco fallback by quantiles
if "pred_risco" not in df.columns:
    try:
        q1 = df["valor_multa"].quantile(0.25)
        q3 = df["valor_multa"].quantile(0.75)
        def proxy(v):
            try:
                if pd.isna(v): return 1
                if v < q1: return 0
                if v >= q3: return 2
                return 1
            except:
                return 1
        df["pred_risco"] = df["valor_multa"].apply(proxy).astype(int)
    except Exception:
        df["pred_risco"] = 1
else:
    df["pred_risco"] = pd.to_numeric(df["pred_risco"], errors="coerce").fillna(1).astype(int)

# iso_flag fallback
if "iso_flag" in df.columns:
    df["iso_flag"] = df["iso_flag"].astype(bool)
else:
    df["iso_flag"] = False

# date handling
date_col = None
for c in ["dat_hora_auto_infracao","dt_fato_infracional","dt_lancamento"]:
    if c in df.columns:
        date_col = c
        break
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["year_month"] = df[date_col].dt.to_period("M").astype(str)
else:
    df["year_month"] = "unknown"

# sample reduce
MAX_ROWS = 50000
if len(df) > MAX_ROWS:
    df_small = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
else:
    df_small = df.reset_index(drop=True)

# robust points (safe keys)
name_cols = [c for c in ["nome_infrator","nome","infrator","nome_responsavel"] if c in df_small.columns]
name_col = name_cols[0] if name_cols else None

points = []
for _, r in df_small.iterrows():
    rec = {
        "seq_auto_infracao": r.get("seq_auto_infracao",""),
        "nome_infrator": r.get(name_col,"") if name_col else "",
        "municipio": r.get("municipio",""),
        "uf": r.get("uf",""),
        "valor_multa": float(r.get("valor_multa", 0)) if pd.notna(r.get("valor_multa", None)) else "",
        "pred_risco": int(r.get("pred_risco",1)) if pd.notna(r.get("pred_risco", None)) else 1,
        "iso_flag": bool(r.get("iso_flag", False)),
        "lat": float(r.get("lat")) if pd.notna(r.get("lat", None)) else None,
        "lon": float(r.get("lon")) if pd.notna(r.get("lon", None)) else None,
        "des_infracao": str(r.get("des_infracao","")),
        "year_month": r.get("year_month","")
    }
    points.append(rec)

# timeseries
if "year_month" in df.columns:
    ts = df.groupby("year_month").size().reset_index(name="count").sort_values("year_month")
    ts_list = ts.to_dict(orient="records")
else:
    ts_list = []

# top municipalities
if ("municipio" in df.columns) or ("uf" in df.columns):
    agg = df.groupby(["uf","municipio"], dropna=False).agg(
        qtd_autuacoes = ("seq_auto_infracao","count") if "seq_auto_infracao" in df.columns else ("municipio","size"),
        soma_multas = ("valor_multa","sum") if "valor_multa" in df.columns else ("seq_auto_infracao","count")
    ).reset_index()
    top_mun = agg.sort_values("qtd_autuacoes", ascending=False).head(15).to_dict(orient="records")
else:
    top_mun = []

# alerts
alerts = [p for p in points if p.get("iso_flag")]

# Convert to JSON for injection
POINTS_JSON = json.dumps(points, ensure_ascii=False)
TS_JSON = json.dumps(ts_list, ensure_ascii=False)
TOP_JSON = json.dumps(top_mun, ensure_ascii=False)
ALERTS_JSON = json.dumps(alerts, ensure_ascii=False)

# HTML template using str.format with placeholders (no f-string)
html_template = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Dashboard - Monitoramento Ambiental Industrial</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 10px; }}
    #map {{ height: 600px; border: 1px solid #ccc; }}
    .row {{ display:flex; gap:20px; }}
    .col {{ flex:1; min-width:300px; }}
    table {{ border-collapse: collapse; width:100%; }}
    th, td {{ text-align:left; padding:6px; border-bottom:1px solid #ddd; font-size:13px; }}
    h2 {{ margin-bottom:6px; }}
    .badge-low {{ color: white; background: green; padding:2px 6px; border-radius:4px; }}
    .badge-med {{ color: white; background: orange; padding:2px 6px; border-radius:4px; }}
    .badge-high {{ color: white; background: red; padding:2px 6px; border-radius:4px; }}
  </style>
</head>
<body>
  <h1>Monitoramento Ambiental Industrial — Dashboard</h1>
  <p>Gerado em: {gen_time} UTC</p>

  <div class="row">
    <div class="col">
      <h2>Mapa de autuações do IBGE em 2024</h2>
      <div id="map"></div>
    </div>
    <div class="col">
      <h2>Ranking de municípios (Top 15)</h2>
      <div id="bar"></div>
      <h2 style="margin-top:18px">Alertas — Anomalias</h2>
      <div id="alerts">
        <table>
          <thead><tr><th>Seq</th><th>Empresa</th><th>Município</th><th>UF</th><th>Valor (R$)</th><th>Risco</th></tr></thead>
          <tbody id="alerts_body"></tbody>
        </table>
      </div>
    </div>
  </div>

  <h2>Tendência temporal — Autuações por mês</h2>
  <div id="timeseries" style="height:320px;"></div>

<script>
const points = {points_json};
const ts = {ts_json};
const top_mun = {top_json};
const alerts = {alerts_json};

function riskLabel(r){{
  if(r===0) return '<span class="badge-low">Baixo</span>';
  if(r===1) return '<span class="badge-med">Médio</span>';
  return '<span class="badge-high">Alto</span>';
}}

// alerts table
const alertsBody = document.getElementById("alerts_body");
if(alerts.length===0){{
  alertsBody.innerHTML = '<tr><td colspan="6">Nenhuma anomalia detectada na amostra</td></tr>';
}} else {{
  alerts.forEach(a => {{
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${{a.seq_auto_infracao}}</td><td>${{a.nome_infrator}}</td><td>${{a.municipio}}</td><td>${{a.uf}}</td><td>${{a.valor_multa}}</td><td>${{riskLabel(a.pred_risco)}}</td>`;
    alertsBody.appendChild(tr);
  }});
}}

// bar chart
const barData = [{{
  x: top_mun.map(r=>r.municipio + " ("+r.uf+")"),
  y: top_mun.map(r=>r.qtd_autuacoes),
  type: 'bar'
}}];
const barLayout = {{ margin: {{t:30,l:40,r:20,b:150}}, height: 400 }};
Plotly.newPlot('bar', barData, barLayout);

// timeseries
const timesX = ts.map(r=>r.year_month);
const timesY = ts.map(r=>r.count);
const lineData = [{{
  x: timesX,
  y: timesY,
  type:'scatter',
  mode:'lines+markers',
  name:'Autuações'
}}];
const lineLayout = {{ margin: {{t:30,l:40,r:20,b:40}}, height:320 }};
Plotly.newPlot('timeseries', lineData, lineLayout);

// Leaflet map
const map = L.map('map').setView([ -15.0, -55.0 ], 4);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 18,
  attribution: '© OpenStreetMap'
}}).addTo(map);

function colorByRisk(r) {{
  if(r===0) return 'green';
  if(r===1) return 'orange';
  return 'red';
}}

points.forEach(p => {{
  try {{
    const lat = parseFloat(p.lat);
    const lon = parseFloat(p.lon);
    if(!isFinite(lat) || !isFinite(lon)) return;
    const marker = L.circleMarker([lat, lon], {{
      radius: p.iso_flag ? 7 : 5,
      color: colorByRisk(p.pred_risco),
      fillOpacity: 0.8
    }}).addTo(map);
    const popup = `<b>Empresa:</b> ${{p.nome_infrator}}<br><b>Mun:</b> ${{p.municipio}}/${{p.uf}}<br><b>Valor:</b> R$ ${{p.valor_multa}}<br><b>Risco:</b> ${{p.pred_risco}}<br><b>Descrição:</b> ${{(p.des_infracao||'').substring(0,200)}}`;
    marker.bindPopup(popup);
  }} catch(e){{console.warn(e)}}
}});

const coords = points.filter(p=>p.lat && p.lon).map(p=>[p.lat,p.lon]);
if(coords.length>0){{
  map.fitBounds(coords, {{maxZoom:10}});
}}

</script>
</body>
</html>
"""

html_out = html_template.format(
    gen_time = datetime.utcnow().isoformat(),
    points_json = POINTS_JSON,
    ts_json = TS_JSON,
    top_json = TOP_JSON,
    alerts_json = ALERTS_JSON
)

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_out)

print("Gerado:", OUT_HTML)
print("Abra o arquivo no navegador (duplo-clique).")
