import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Dados de treino históricos (A IA aprende com esses dados)
dados_treino = pd.DataFrame([
    {'posicao_mandante': 1, 'posicao_visitante': 18, 'vitorias_ultimos_5': 4, 'resultado': 1},
    {'posicao_mandante': 15, 'posicao_visitante': 2, 'vitorias_ultimos_5': 1, 'resultado': 0},
    {'posicao_mandante': 3, 'posicao_visitante': 12, 'vitorias_ultimos_5': 3, 'resultado': 1},
    {'posicao_mandante': 10, 'posicao_visitante': 9, 'vitorias_ultimos_5': 2, 'resultado': 0},
    {'posicao_mandante': 2, 'posicao_visitante': 20, 'vitorias_ultimos_5': 5, 'resultado': 1},
])

X_train = dados_treino[['posicao_mandante', 'posicao_visitante', 'vitorias_ultimos_5']]
y_train = dados_treino['resultado']

# Treinando o algoritmo de IA
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 2. Jogos do Dia para a IA analisar
jogos_hoje = pd.DataFrame([
    {'time_casa': 'Real Madrid', 'time_fora': 'Almería', 'posicao_mandante': 1, 'posicao_visitante': 19, 'vitorias_ultimos_5': 5},
    {'time_casa': 'Arsenal', 'time_fora': 'Everton', 'posicao_mandante': 2, 'posicao_visitante': 15, 'vitorias_ultimos_5': 4},
    {'time_casa': 'Sevilla', 'time_fora': 'Betis', 'posicao_mandante': 8, 'posicao_visitante': 7, 'vitorias_ultimos_5': 2},
    {'time_casa': 'Bayern Munique', 'time_fora': 'Bochum', 'posicao_mandante': 1, 'posicao_visitante': 17, 'vitorias_ultimos_5': 4},
])

X_hoje = jogos_hoje[['posicao_mandante', 'posicao_visitante', 'vitorias_ultimos_5']]
probabilidades = modelo.predict_proba(X_hoje)[:, 1] # Pega a probabilidade de vitória do mandante

# 3. Categorizando pelas faixas (95%, 85%, 75%)
resultados = []
for idx, row in jogos_hoje.iterrows():
    prob = round(probabilidades[idx] * 100, 1)
    
    if prob >= 90:
        cat = "95%"
    elif prob >= 80:
        cat = "85%"
    elif prob >= 70:
        cat = "75%"
    else:
        cat = "Outros"

    resultados.append({
        'casa': row['time_casa'],
        'fora': row['time_fora'],
        'probabilidade': prob,
        'categoria': cat,
        'palpite': f"Vitória de {row['time_casa']}"
    })

# Exporta em formato JSON para o site ler
with open('dados_apostas.json', 'w', encoding='utf-8') as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("Previsões geradas com sucesso!")