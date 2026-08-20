import json
import requests
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# 1. Treinando o modelo estatístico de IA
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

# ID das Ligas Principais no TheSportsDB
ligas = [
    {"id": "4351", "nome": "Brasileirão Série A"},
    {"id": "4328", "nome": "Premier League"},
    {"id": "4335", "nome": "La Liga"},
    {"id": "4332", "nome": "Serie A Italiana"},
    {"id": "4331", "nome": "Bundesliga"},
    {"id": "4480", "nome": "UEFA Champions League"}
]

print("Buscando próximas partidas nas principais ligas via TheSportsDB...")

for liga in ligas:
    try:
        # Usa a API pública oficial e gratuita
        url = f"https://www.thesportsdb.com/api/v1/json/3/eventsnextleague.php?id={liga['id']}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            dados = response.json()
            eventos = dados.get("events") or []
            
            for evento in eventos:
                casa = evento.get("strHomeTeam")
                fora = evento.get("strAwayTeam")
                data_jogo = evento.get("dateEvent")
                horario = evento.get("strTime", "")[:5]
                
                if casa and fora:
                    # Simulação de preditores estatísticos para a IA calcular
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
                        'liga': liga['nome'],
                        'data': data_jogo,
                        'horario': horario,
                        'probabilidade': prob_percent,
                        'categoria': cat,
                        'palpite': f"Vitória do {casa}"
                    })
    except Exception as e:
        print(f"Erro ao buscar liga {liga['nome']}: {e}")

# 2. Salva a lista processada no JSON consumido pelo site
with open('dados_apostas.json', 'w', encoding='utf-8') as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print(f"Sucesso total! {len(resultados)} partidas das maiores ligas salvas com sucesso.")
