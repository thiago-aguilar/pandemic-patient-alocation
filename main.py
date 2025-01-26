import pandas as pd
import json
from src.optimizer import PortfolioOptimizer

# Define caminhos dos arquivos de dados
data_path = 'data/'

optimizer = PortfolioOptimizer(
    data_path=data_path
)

optimizer.run()