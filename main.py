import pandas as pd
import json
from src.optimizer import PortfolioOptimizer

# Define caminhos dos arquivos de dados
estoque_file = "data/Estoque.xlsx"
obras_file = "data/Obras.xlsx"
custos_transporte_file = "data/CustosTransp.json"

optimizer = PortfolioOptimizer(
    estoque_file,
    obras_file,
    custos_transporte_file
)

optimizer.run()