import pandas as pd
import json
from src.optimizer import PandemicAlocationOptimizer

# Define caminhos dos arquivos de dados
data_path = 'data/'

optimizer = PandemicAlocationOptimizer(
    data_path=data_path
)

optimizer.run()