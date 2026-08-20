import json
import requests
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# Substitua pela sua chave (Football-Data.org ou RapidAPI)
API_KEY = "3c1bbabef3074a03ac650f217eb4605f"

HEADERS = {
    'X-Auth-Token': API_KEY
}

# Treinando modelo base
dados_historicos = pd.DataFrame([
    {'diff_posicao': -18, 'vitorias_mandante': 5, 'vitorias_visitante': 0, 'resultado': 1},
    {'diff_posicao': -10, 'vitorias_mandante': 4, 'vitorias_visitante': 1, 'resultado': 1},
    {'diff_posicao': 12,  'vitorias_mandante': 1, 'vitorias_visitante': 4, 'resultado': 0},
    {'diff_posicao': 0,   'vitorias_mandante': 2, 'vitorias_visitante': 2, 'resultado': 0},
])

X_train = dados_historicos[['diff_posicao', 'vitorias_mandante', 'vitorias_visitante']]
y_train = dados_historicos['resultado']

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

resultados = []
hoje = datetime.now().strftime('%Y-%m-%d')
url = f"https://api.football-data.org/v4/matches?dateFrom={hoje}&dateTo={hoje}"

try:
    response = requests.get(url, headers=HEADERS, timeout=10)
    if response.status_code == 200:
        dados = response.json()
        for partida in dados.get("matches", []):
            casa = partida.get("homeTeam", {}).get("name", "Time Casa")
            fora = partida.get("awayTeam", {}).get("name", "Time Fora")
            
            prob = modelo.predict_proba([[-8, 4, 1]])[0][1]
            prob_percent = round(prob * 100, 1)

            cat = "95%" if prob_percent >= 90 else ("85%" if prob_percent >= 80 else "75%")

            resultados.append({
                'casa': casa,
                'fora': fora,
                'probabilidade': prob_percent,
                'categoria': cat,
                'palpite': f"Vitória do {casa}"
            })
except Exception as e:
    print(f"Aviso de execução: {e}")

# Garante a gravação do arquivo
with open('dados_apostas.json', 'w', encoding='utf-8') as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print(f"Concluído com {len(resultados)} jogos.")
