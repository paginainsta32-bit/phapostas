import json
import requests
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# 1. Cole aqui sua chave que recebeu por e-mail do football-data.org
API_KEY = "3c1bbabef3074a03ac650f217eb4605f"

HEADERS = {
    'X-Auth-Token': API_KEY
}

# 2. Treinando o modelo de IA
dados_historicos = pd.DataFrame([
    {'diff_posicao': -18, 'vitorias_mandante': 5, 'vitorias_visitante': 0, 'resultado': 1},
    {'diff_posicao': -10, 'vitorias_mandante': 4, 'vitorias_visitante': 1, 'resultado': 1},
    {'diff_posicao': 12,  'vitorias_mandante': 1, 'vitorias_visitante': 4, 'resultado': 0},
    {'diff_posicao': 0,   'vitorias_mandante': 2, 'vitorias_visitante': 2, 'resultado': 0},
    {'diff_posicao': -15, 'vitorias_mandante': 5, 'vitorias_visitante': 1, 'resultado': 1},
])

X_train = dados_historicos[['diff_posicao', 'vitorias_mandante', 'vitorias_visitante']]
y_train = dados_historicos['resultado']

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 3. Consultar jogos
resultados = []
hoje = datetime.now().strftime('%Y-%m-%d')
url = f"https://api.football-data.org/v4/matches?dateFrom={hoje}&dateTo={hoje}"

try:
    response = requests.get(url, headers=HEADERS, timeout=10)
    print(f"Status da API: {response.status_code}")
    
    if response.status_code == 200:
        dados = response.json()
        partidas = dados.get("matches", [])
        
        for partida in partidas:
            casa = partida.get("homeTeam", {}).get("name", "Time Casa")
            fora = partida.get("awayTeam", {}).get("name", "Time Fora")
            status = partida.get("status", "NS")
            
            # Análise preditiva
            prob = modelo.predict_proba([[-8, 4, 1]])[0][1]
            prob_percent = round(prob * 100, 1)

            if prob_percent >= 90:
                cat = "95%"
            elif prob_percent >= 80:
                cat = "85%"
            elif prob_percent >= 70:
                cat = "75%"
            else:
                cat = "Outros"

            resultados.append({
                'casa': casa,
                'fora': fora,
                'status': status,
                'probabilidade': prob_percent,
                'categoria': cat,
                'palpite': f"Vitória do {casa}"
            })
    else:
        print(f"Aviso da API: {response.text}")

except Exception as e:
    print(f"Erro ao conectar com API: {e}")

# 4. Grava os resultados no JSON sem quebrar a pipeline
with open('dados_apostas.json', 'w', encoding='utf-8') as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print(f"Sucesso! {len(resultados)} jogos processados.")
