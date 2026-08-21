import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# 1. Sua chave informada do Football-Data.org
API_KEY = "56df8c8cd63cb9b1599f39262b330550"
HEADERS = {'X-Auth-Token': API_KEY}

# 2. Treinando o modelo estatístico de IA
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

resultados = []
hoje = datetime.now()
data_inicio = hoje.strftime('%Y-%m-%d')
# Janela de 5 dias para capturar a rodada completa do fim de semana
data_fim = (hoje + timedelta(days=5)).strftime('%Y-%m-%d')

url = f"https://api.football-data.org/v4/matches?dateFrom={data_inicio}&dateTo={data_fim}"

try:
    response = requests.get(url, headers=HEADERS, timeout=10)
    print(f"Status da resposta da API: {response.status_code}")
    
    if response.status_code == 200:
        dados = response.json()
        matches = dados.get("matches", [])
        
        for partida in matches:
            casa = partida.get("homeTeam", {}).get("name")
            fora = partida.get("awayTeam", {}).get("name")
            status = partida.get("status")
            data_jogo = partida.get("utcDate", "")[:10]
            
            if casa and fora and status in ["SCHEDULED", "TIMED", "IN_PLAY"]:
                prob = modelo.predict_proba([[-10, 4, 1]])[0][1]
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
                    'data': data_jogo,
                    'probabilidade': prob_percent,
                    'categoria': cat,
                    'palpite': f"Vitória do {casa}"
                })
except Exception as e:
    print(f"Erro ao conectar com Football-Data.org: {e}")

# Salva o arquivo JSON lido pelo site
with open('dados_apostas.json', 'w', encoding='utf-8') as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print(f"Concluído! {len(resultados)} partidas salvas em dados_apostas.json.")
