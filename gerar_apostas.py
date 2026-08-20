import json
import requests
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# 1. Suas Credenciais do RapidAPI (extraídas do seu painel)
API_KEY = "abbe41afd2mshb8d62f7162bebd7p192b09jsn7e88190634c8"
HOST = "apifootball3.p.rapidapi.com"

HEADERS = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": HOST
}

# 2. Treinando o modelo simples de IA
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

# 3. Consultar jogos reais agendados para HOJE
hoje = datetime.now().strftime('%Y-%m-%d')
url = f"https://apifootball3.p.rapidapi.com/?action=get_events&from={hoje}&to={hoje}"

response = requests.get(url, headers=HEADERS)
dados_api = response.json()

resultados = []

if isinstance(dados_api, list):
    for partida in dados_api:
        casa = partida.get("match_hometeam_name", "Time Casa")
        fora = partida.get("match_awayteam_name", "Time Fora")
        status = partida.get("match_status", "NS")
        
        # Análise do Modelo
        diff_pos = -8
        vit_c = 4
        vit_v = 1
        
        prob = modelo.predict_proba([[diff_pos, vit_c, vit_v]])[0][1]
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

# 4. Salvar resultados no JSON lido pelo site
with open('dados_apostas.json', 'w', encoding='utf-8') as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print(f"Total de jogos reais processados para hoje ({hoje}): {len(resultados)}")
