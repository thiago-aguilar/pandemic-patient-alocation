import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from pymoo.vendor.hv import HyperVolume

def print_pareto(df_pareto, title, utopico_point, antiutopico_point, decision=None):
    """
    Função para plotar a fronteira Pareto com pontos utópico, anti-utópico e a decisão destacada.

    Parâmetros:
    - df_pareto: DataFrame contendo os pontos da fronteira Pareto (com 'obj1' e 'obj2').
    - title: Título do gráfico.
    - utopico_point: Tupla ou lista com os valores do ponto utópico (priority, cost).
    - antiutopico_point: Tupla ou lista com os valores do ponto anti-utópico (priority, cost).
    - decision: Tupla ou lista com os valores do ponto de decisão (priority, cost).
    """
    plt.figure(figsize=(7, 5))
    
    # Plotando os pontos Pareto em azul
    plt.scatter(x=df_pareto['obj2'], y=df_pareto['obj1'], color='blue', label='Pontos Pareto')

    # Plotando o ponto utópico em vermelho
    plt.scatter(x=utopico_point[1], y=utopico_point[0], color='red', marker='x', label='Ponto Utópico', s=100)
    
    # Plotando o ponto anti-utópico em vermelho
    plt.scatter(x=antiutopico_point[1], y=antiutopico_point[0], color='red', marker='o', label='Ponto Anti-Utópico', s=100)
    
    # Plotando o ponto de decisão em verde
    if decision is not None:
        plt.scatter(x=decision[1], y=decision[0], color='green', marker='s', label='Decisão escolhida', s=100)
    
    # Título do gráfico
    plt.title(title)
    
    # Legenda
    plt.legend()
    
    # Mostrar o gráfico
    plt.xlabel('obj2')
    plt.ylabel('obj1')
    plt.show()



def generate_pareto(pareto_df, title='', decision=None):
    # Calculate antiutopic and utopic points
    pareto_df = pareto_df[['obj1', 'obj2']].drop_duplicates().sort_values(by='obj2')
    antiutopico_priority = max(pareto_df['obj1']) + (max(pareto_df['obj1']) - min(pareto_df['obj1'])) * 0.1 
    antiutopico_cost = max(pareto_df['obj2']) + (max(pareto_df['obj2']) - min(pareto_df['obj2'])) * 0.1 
    utopic_cost = min(pareto_df['obj2'])
    utopic_priority = min(pareto_df['obj1'])

    # Calculate hipervolume
    reference_point = np.array([antiutopico_priority, antiutopico_cost])
    hv = HyperVolume(reference_point)
    array_priority = pareto_df['obj1'].to_numpy()
    array_cost = pareto_df['obj2'].to_numpy()
    pareto_front = np.array([[array_priority[elem], array_cost[elem]] for elem in range(len(array_cost))])
    hipervolume = hv.compute(pareto_front)

    # Calculate area covered
    side_priority = antiutopico_priority - utopic_priority
    side_cost = antiutopico_cost - utopic_cost
    max_vol = side_priority * side_cost
    S_metric = hipervolume / max_vol

    # Utopic points as tuples
    utopic = (utopic_priority, utopic_cost)
    antiutopic = (antiutopico_priority, antiutopico_cost)

    # # Calculo de delta metric
    # distancias = np.sqrt((pareto_df['obj1'] - utopic_priority)**2 + (pareto_df['obj2'] - utopic_cost)**2)
    # delta = distancias.mean()

    distances = np.sqrt(np.diff(pareto_df['obj1'])**2 + np.diff(pareto_df['obj2'])**2)
    mean_distance = distances.mean()
    numerador_array = [abs(d - mean_distance) for d in distances]
    denominador = len(distances) * mean_distance
    delta = sum(numerador_array) / denominador

    
    print(f'Hipervolume : {hipervolume}')
    # print(f'S_metric : {S_metric}')
    print(f'Delta_metric: {delta} ')
    print_pareto(pareto_df, title, utopic, antiutopic, decision)