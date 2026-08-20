import json
import requests
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# 1. Cole a sua chave da Football-Data.org aqui
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

# 3. Buscar os jogos reais de HOJE na API
hoje = datetime.now().strftime('%Y-%m-%d')
url = f"https://api.football-data.org/v4/matches?dateFrom={hoje}&dateTo={hoje}"

resultados = []

try:
    response = requests.get(url, headers=HEADERS)
    dados = response.json()

    if "matches" in dados and len(dados["matches"]) > 0:
        for partida in dados["matches"]:
            casa = partida["homeTeam"]["name"]
            fora = partida["awayTeam"]["name"]
            status = partida["status"]
            
            # Análise preditiva simples pela IA
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
    else:
        print("Nenhuma partida agendada encontrada para a data de hoje.")

except Exception as e:
    print(f"Erro na conexão com a API: {e}")

# 4. Grava no arquivo JSON consumido pela interface web
with open('dados_apostas.json', 'w', encoding='utf-8') as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print(f"Processamento concluído. Jogos salvos: {len(resultados)}")
