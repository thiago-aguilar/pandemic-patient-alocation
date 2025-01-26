import pandas as pd
import json
from abc import abstractmethod
from src.load_data import carregar_estoque, carregar_custos_transporte, carregar_obras
import numpy as np

from gurobipy import Model, GRB, quicksum
from tqdm import tqdm
import copy
import os
import shutil

class PortfolioOptimizer:

    def __init__(self, estoque_file, obras_file, custos_transporte_file):
        """
        Argrs:
            estoque_file (str): Arquivo de estoque
            obras_file (str): Arquivo de obras
            custo_transporte_file (str): Arquivo com custos de transporte
        """
        self.estoque_file = estoque_file
        self.obras_file = obras_file
        self.custos_transporte_file = custos_transporte_file
        self.clear_solutions_directory()

        self.obj_1 = None
        self.obj_2 = None
        
    def run(self):
        # Inicializando o modelo
        self.model = Model('PortfolioOptimization')

        # Leitura de preparação de dados
        self.read_inputs()
        self.define_sets()
        self.define_parameters()

        # Criação do modelo
        self.create_decision_variables()
        self.create_constraints()

        # Cria curva pareto        
        self.create_pareto_curve()


    def read_inputs(self):

        print('Lendo inputs')
        self.estoque = carregar_estoque(self.estoque_file)
        self.obras = carregar_obras(self.obras_file)
        self.custos_transporte = carregar_custos_transporte(self.custos_transporte_file)
    

    def define_sets(self):
        
        print('Definindo conjuntos')
        # Inicializa conjuntos
        model_sets = dict()

        # I - Conjunto de Obras
        model_sets['I'] = self.obras["obra"].unique()

        # J - Conjuntos de depósitos
        model_sets['J'] = self.estoque["cod_dep"].unique()
        
        # M - Conjunto de materiais
        model_sets['M'] = self.estoque["cod_mat"].unique()

        # J_i - Subconjunto de depósitos que executam a obra i
        model_sets['J_i'] = (
            self.obras
            .drop_duplicates(subset=['obra', 'cod_dep'])
            .groupby('obra')
            .agg({'cod_dep': 'unique'})
            .to_dict()
            ['cod_dep']
        )

        # I[j] - Subconjunto de obras que o depósito j pode executar
        model_sets['I_j'] = (
            self.obras
            .drop_duplicates(subset=['obra', 'cod_dep'])
            .groupby('cod_dep')
            .agg({'obra': 'unique'})
            .to_dict()
            ['obra']
        )

        # M[i] - Subconjunto de materiais que a obra i requer
        model_sets['M_i'] = (
            self.obras
            .drop_duplicates(subset=['obra', 'cod_mat'])
            .groupby('obra')
            .agg({'cod_mat': 'unique'})
            .to_dict()
            ['cod_mat']
        )

        # possible_transfer - conjunto de tuplas (k,j,m) com todas as transferencias possiveis
        model_sets['possible_transfer'] = set(self.custos_transporte.keys())

        self.model_sets = model_sets

        
    def define_parameters(self):
        
        print('Criando parâmetros')
        # Inicializa parâmetros
        parameters = dict()

        # w[i] - prioridade de uma dada obra i
        parameters['w'] = (
            self.obras
            .drop_duplicates(subset=['obra', 'prioridade'])
            .set_index('obra')
            .to_dict()
            ['prioridade']
        )

        # q[i,m] - quantidade do material m necessario na obra i
        parameters['q'] = (
            self.obras
            .groupby(['obra', 'cod_mat'])
            .agg({'qtd_dem': 'sum'})
            .to_dict()
            ['qtd_dem']
        )

        # Q[j,m] - estoque inicial do material m no deposito j
        parameters['Q'] = (
            self.estoque
            .set_index(['cod_dep', 'cod_mat'])
            .to_dict()
            ['estoque']
        )
        
        # c[k,j,m] - custo de transporte de uma unidade do material m do deposito k para o deposito j
        parameters['c'] = self.custos_transporte

        # N - quantidade total de obras
        parameters['N'] = self.obras['obra'].nunique()

        # D - quantidade total de depositos
        parameters['D'] = self.obras['cod_dep'].nunique()

        self.params = parameters


    def create_decision_variables(self):

        print('Criando variáveis de decisão')
        decision_variables = dict()

        # x[i,j] - Binária: 1 se a obra i é executada no depósito j
        decision_variables['x'] = self.model.addVars(
            [(i, j) for i in self.model_sets['I'] for j in self.model_sets['J_i'][i]],
            vtype=GRB.BINARY,
            name="x"
        )

        # t[k,j,m]: Continua: Quantidade de material m transferida do depósito k para o depósito j
        decision_variables['t'] = self.model.addVars(
            [(k, j, m) for k in self.model_sets['J'] for j in self.model_sets['J'] for m in self.model_sets['M'] if (j != k) and ((k,j,m) in self.model_sets['possible_transfer']) ],
            vtype=GRB.CONTINUOUS,
            name="t",
            lb=0
        )
    
        self.vars = decision_variables


    def create_constraints(self):

        # Restrição 1 - Obra executada no máximo uma vez
        print('Criando restrição 1')
        for i in tqdm(self.model_sets['I']):
            self.model.addConstr(
                quicksum(self.vars['x'][i,j] for j in self.model_sets['J_i'][i]) <= 1,
                name=f"C1_OneExecution_{i}"
            )
        
        # Restrição 2 - Balanço de massa do estoque de materiais
        print('Criando restrição 2')
        for m in tqdm(self.model_sets['M']):
            for j in self.model_sets['J']:  
                
                LHS = quicksum(self.vars['x'][i, j] * self.params['q'][i, m] for i in self.model_sets['I_j'][j] if m in self.model_sets['M_i'][i]) + \
                      quicksum(self.vars['t'][j, k, m] for k in self.model_sets['J'] if (k!=j) and ((j,k,m) in self.model_sets['possible_transfer']))              
                

                tuple_get = (j,m)
                initial_stock = self.params['Q'].get(tuple_get, 0)
        
                RHS = initial_stock + \
                       quicksum(self.vars['t'][k,j,m] for k in self.model_sets['J'] if (k!=j) and ((k,j,m) in self.model_sets['possible_transfer']) )

                self.model.addConstr(
                    # LHS
                    LHS <= RHS,                    
                    name=f"C2_StockBalance_{m}_{j}"
                )

        
    def set_obj_function(self, weight_priority: float, weight_cost: float):
        
        if self.obj_1 is None:
            # obj1 - Maximização de prioridades | Minimização de: - sum(prioridade)
            self.obj_1 = - quicksum(self.vars['x'][i,j] * self.params['w'][i] for i in self.model_sets['I'] for j in self.model_sets['J_i'][i])

        if self.obj_2 is None:
            # obj2 - Minimização de custo
            self.obj_2 = quicksum(self.vars['t'][k, j, m] * self.params['c'][k, j, m] for k in self.model_sets['J'] for j in self.model_sets['J'] for m in self.model_sets['M'] if (j!= k) and ((k,j,m) in self.model_sets['possible_transfer']))


        self.model.setObjective(
            # Obj1 - Maximização de prioridades
            weight_priority * self.obj_1 +
            # Obj2 - Maximização de pesos
            weight_cost * self.obj_2,
            GRB.MINIMIZE
        )

    def clear_solutions_directory(self):
        # Verificar se a pasta 'solutions' existe
        path_output = 'solutions/'
        if os.path.exists(path_output) and os.path.isdir(path_output):
            # Iterar sobre todos os itens dentro da pasta 'solutions'
            for item in os.listdir(path_output):
                item_path = os.path.join(path_output, item)
                
                # Se for um diretório, apagar o conteúdo
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                # Se for um arquivo (não uma pasta), apagar o arquivo
                elif os.path.isfile(item_path):
                    os.remove(item_path)
        else:
            os.mkdir(path_output)


    def generate_results(self):
        

        obj1_value = self.obj_1.getValue()
        obj2_value = self.obj_2.getValue()
        sol_df = pd.DataFrame({
            'Fobj': ['Obj1', 'Obj2'],
            'Result': [obj1_value, obj2_value]
        }, index=[0,1])
        
        rounded_obj1 = abs(np.round(obj1_value, decimals=2))
        rounded_obj2 = np.round(obj2_value, decimals=2)

        sol_name = f'sol_priority_{rounded_obj1}_cost_{rounded_obj2}/'
        path_output = 'solutions/' + sol_name 

        # Caso já exista a pasta, essa solução já foi guardada
        if os.path.exists(path_output):
            return None
        # Caso não exista a pasta, cria e salva a solução
        else:
            os.mkdir(path_output)
        sol_df.to_csv(path_output + 'solution.csv', index=False)
        
        # Get var X result and create a DF with it
        vars_x_on = [key for key, elem in self.vars['x'].items() if np.isclose(elem.X, 1) ]
        array_i = [elem[0] for elem in vars_x_on]
        array_j = [elem[1] for elem in vars_x_on]
        df_output_X = pd.DataFrame({
            'obra': array_i,
            'cod_dep': array_j
        })
        df_output_X['obra_atendida'] = 1
        df_output_X.to_csv(path_output + 'obras_atendidas.csv', index=False)

        # Generate DF grouped to compare with professor's solution
        df_real_output = (
            self.obras[['obra', 'cod_dep', 'prioridade']]
            .drop_duplicates()
            .merge(df_output_X, on=['cod_dep', 'obra'], how='left', validate='1:1')
        )
        df_real_output['obra_atendida'] = df_real_output['obra_atendida'].fillna(0)
        df_real_output['prioridade_atendida'] = df_real_output['obra_atendida'] * df_real_output['prioridade']

        df_grouped = (
            df_real_output
            .groupby(['cod_dep'])
            .agg({
                # NUM_OBRAS_ASSOCIADAS
                'obra': 'nunique', 
                # OBRAS_EXECUTADAS
                'obra_atendida': 'sum',
                # SOMA_PRIORIDADES_EXECUTADAS
                'prioridade_atendida': 'sum',
                # SOMA_PRIORIDADES_ASSOCIADAS
                'prioridade': 'sum'
            })
            .reset_index()
            .rename(
                columns={
                    'obra':'NUM_OBRAS_ASSOCIADAS',
                    'obra_atendida': 'OBRAS_EXECUTADAS',
                    'prioridade_atendida': 'SOMA_PRIORIDADES_EXECUTADAS',
                    'prioridade': 'SOMA_PRIORIDADES_ASSOCIADAS'
                }
            )
        )
        df_grouped.to_csv(path_output + 'resultado_agrupado.csv', index=False)


    def create_pareto_curve(self):
        
        # Optimize only priorities
        self.set_obj_function(weight_priority=1, weight_cost=0)
        self.model.optimize()

        # Get maximum priority
        P_max = abs(self.obj_1.getValue())
        
        # add constraiont to garantee optimal priorities, and minimize cost
        self.model.addConstr(
                self.obj_1 <= - P_max*0.9999,
                name=f"force_max_priority"
            )
        self.set_obj_function(weight_priority=0, weight_cost=1)
        self.model.optimize()
        self.generate_results()

        # Get maximum cost
        C_max = abs(self.obj_2.getValue())

        # Remove created constraint to force max priority
        constraint_added = self.model.getConstrByName('force_max_priority')
        self.model.remove(constraint_added)

        # Optimize priorities with cost equals 0
        self.model.addConstr(
                self.obj_2 == 0,
                name=f"force_zero_cost"
            )
        self.set_obj_function(weight_priority=1, weight_cost=0)
        self.model.optimize() 
        self.generate_results()

        # Store P_min (with zero of cost transfer)
        P_min = abs(self.obj_1.getValue())
        C_min = 0

        # Remove created constraint to force_zero_cost
        constraint_added = self.model.getConstrByName('force_zero_cost')
        self.model.remove(constraint_added)

        # Create weights array for pareto curve
        begin_weight = 1e-6
        sample_amount = 20
        half_sample = int(sample_amount / 2)

        weights_1 = np.logspace(np.log10(begin_weight), np.log10(1), sample_amount, endpoint=False)
        weights_2 = 1 - np.logspace(np.log10(begin_weight), np.log10(1), sample_amount, endpoint=False)
        
        w3_samples = 10
        weights_3 = np.linspace(0.125,0.875, w3_samples, endpoint=False)
        
        weights = np.concatenate([weights_1[half_sample:], weights_2[half_sample:], weights_3])

        # Initialize pareto curves empty
        dict_intitialize_curve = {
            'priority': [],
            'cost': [],
        }
        idx_array = []
        pareto_original = copy.deepcopy(dict_intitialize_curve) 
        pareto_normalized = copy.deepcopy(dict_intitialize_curve)

        # Iterate over weights, optimize, and store pareto curve values
        for idx, w_priority in enumerate(weights):
            w_cost = 1 - w_priority

            # Optimize 
            self.set_obj_function(
                weight_priority = w_priority/(P_max - P_min),
                weight_cost = w_cost/(C_max - C_min)
            )
            self.model.optimize()
            self.generate_results()

            # Get results
            priority_result = self.obj_1.getValue()
            cost_result = self.obj_2.getValue()

            # Store pareto curve value
            pareto_original['priority'].append(priority_result)
            pareto_original['cost'].append(cost_result)
            
            # Store index
            idx_array.append(idx)
            print(f'Optimized Iteration {idx}')

        # Store last data point
        pareto_original['priority'].append(- P_min)
        pareto_original['cost'].append(C_min)

        # # Store first data point
        pareto_original['priority'].append(- P_max)
        pareto_original['cost'].append(C_max)

        idx_array.append(idx+1)
        idx_array.append(idx+2)


        pareto_original_df = pd.DataFrame(pareto_original, index=idx_array)
        pareto_original_df.to_csv('results_pareto/pareto_curve.csv')