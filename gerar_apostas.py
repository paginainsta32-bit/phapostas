import json
import requests
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# 1. Chave da API (Substitua pela sua chave do RapidAPI)
API_KEY = "SUA_CHAVE_RAPIDAPI_AQUI"
HEADERS = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": "v3.football.api-sports.io"
}

# 2. Treinando a IA com base estatística
# (Em produção, o modelo utiliza histórico de vitórias/posição)
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

# 3. Buscar jogos reais de HOJE na API
hoje = datetime.now().strftime('%Y-%m-%d')
url = f"https://v3.football.api-sports.io/fixtures?date={hoje}"

response = requests.get(url, headers=HEADERS)
dados_api = response.json()

resultados = []

if "response" in dados_api and len(dados_api["response"]) > 0:
    for partida in dados_api["response"]:
        casa = partida["teams"]["home"]["name"]
        fora = partida["teams"]["away"]["name"]
        status = partida["fixture"]["status"]["short"] # NS = Não iniciado, FT = Finalizado
        
        # Exemplo de extração de métricas (calculando diferencial de posições/forma)
        # Para demonstração ágil, simula o vetor de entrada
        diff_pos = -10 
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

# 4. Salvar resultados no JSON
with open('dados_apostas.json', 'w', encoding='utf-8') as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print(f"Total de jogos reais processados hoje ({hoje}): {len(resultados)}")
