import pandas as pd
import json
from abc import abstractmethod
import numpy as np

from gurobipy import Model, GRB, quicksum
from tqdm import tqdm
import copy
import os
import shutil

class PandemicAlocationOptimizer:

    def __init__(self, data_path):
        """
        Argrs:
            estoque_file (str): Arquivo de estoque
            obras_file (str): Arquivo de obras
            custo_transporte_file (str): Arquivo com custos de transporte
        """
        self.data_path = data_path
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
        self.create_pareto_curve_weighted_sum()


    def read_inputs(self):

        self.tipo_paciente_estadia_df = pd.read_csv(self.data_path + 'tipo_paciente_estadia.csv')
        self.tipo_paciente_recurso_df = pd.read_csv(self.data_path + 'paciente_vs_recurso.csv')
        self.hospital_recurso_sypply_df = pd.read_csv(self.data_path + 'hospitais_vs_recursos_qtd.csv')
        self.hospital_area_pop_dist_df = pd.read_csv(self.data_path + 'hospitais_vs_area_pop_dist.csv')
        self.cond_init_df = pd.read_csv(self.data_path + 'cond_init.csv')
        self.area_pop_qtd_pacientes_df = pd.read_csv(self.data_path + 'area_pop_qtd_pacientes.csv')


    def get_set_from_inputs(self, set_sheets, column_name):
        current_set = set()
        for sheet in set_sheets:
            df_input = getattr(self, sheet)
            current_set = current_set.union(df_input[column_name])    
        return current_set

    def define_sets(self):
        
        print('Definindo conjuntos')
        # Inicializa conjuntos
        model_sets = dict()

        # H - Conjunto de hospitais
        hospital_sheets = ['hospital_recurso_sypply_df', 'hospital_area_pop_dist_df', 'cond_init_df']
        hospital_column_name = 'Hospital'
        model_sets['H'] = self.get_set_from_inputs(hospital_sheets, hospital_column_name)

        # A - Áreas de população
        population_sheets = ['area_pop_qtd_pacientes_df', 'hospital_area_pop_dist_df']
        population_column_name = 'Area_pop'
        model_sets['A'] = self.get_set_from_inputs(population_sheets, population_column_name)

        # T - Períodos (dias) no horizonte de planejamento
        period_sheets = ['area_pop_qtd_pacientes_df', 'cond_init_df']
        period_column_name = 'Dia'
        model_sets['T'] = self.get_set_from_inputs(period_sheets, period_column_name)
        self.checa_periodos_validos(model_sets['T'])

        # R - Tipo de recursos
        resourse_sheets = ['hospital_recurso_sypply_df', 'tipo_paciente_recurso_df']
        resource_column_name = 'Recurso'
        model_sets['R'] = self.get_set_from_inputs(resourse_sheets, resource_column_name)

        # P - Tipos de pacientes
        patient_type_sheets = ['area_pop_qtd_pacientes_df', 'cond_init_df', 'tipo_paciente_recurso_df', 'tipo_paciente_estadia_df']
        patient_type_columns_name = 'Tipo_de_paciente'
        model_sets['P'] = self.get_set_from_inputs(patient_type_sheets, patient_type_columns_name)

        # S_r[r] - Tipos de pacientes (os mesmos tipos do set P) que demandam o recurso r
        filtro_recurso = self.tipo_paciente_recurso_df['Usa_recurso'] == 1
        model_sets['S_r'] = (
            self.tipo_paciente_recurso_df
            [filtro_recurso]
            .groupby('Recurso')
            .agg({'Tipo_de_paciente': 'unique'})
            .to_dict()
            ['Tipo_de_paciente']
        )

        self.model_sets = model_sets


    @staticmethod
    def checa_periodos_validos(periodos_set):
        min_day = min(periodos_set)
        max_day = max(periodos_set)
        for day in range(min_day, max_day + 1):
            if day not in (periodos_set):
                raise Exception('O dia {day} deve ser informado nos inputs, pois está entre o dia minimo e maximo informado')
        
    def define_parameters(self):
        
        print('Criando parâmetros')
        # Inicializa parâmetros
        parameters = dict()
        
        # Distance[a,h]  - Distance from populational area a until hospital h. Unit: kilometers
        parameters['Distance'] = (
            self.hospital_area_pop_dist_df.
            set_index(['Area_pop', 'Hospital'])
            .to_dict()
            ['Distancia']
        )

        # Demand[p,a,t] - Demand of patient type p, in populational era a, at the day t. Unit: # of patients
        parameters['Demand'] = (
            self.area_pop_qtd_pacientes_df
            .set_index(['Tipo_de_paciente', 'Area_pop', 'Dia'])
            .to_dict()['Qtd_pacientes']
        )

        # InitPatients[p,h] - Quantity of patient type p, at the hospital h, that entered before the begining of planing horizon. Unit: # of patients
        parameters['InitPatients'] = (
            self.cond_init_df
            .groupby(['Tipo_de_paciente', 'Hospital'])
            .agg({'Qtd_pacientes_liberados': 'sum'})
            .to_dict()['Qtd_pacientes_liberados']
        )

        # ReleasedPatients[p,h,t]- Quantity of patient type p, at the hospital h, before the befining of planing horizon, that are released at day t. Unit: # of patients
        parameters['ReleasedPatients'] = (
            self.cond_init_df
            .set_index(['Tipo_de_paciente', 'Hospital', 'Dia'])
            .to_dict()
            ['Qtd_pacientes_liberados']
        )

        # LenghOfStay[p] - Quantity of days that the patient type p stays at a hospital
        parameters['LenghOfStay'] = (
            self.tipo_paciente_estadia_df
            .set_index(['Tipo_de_paciente'])
            .to_dict()
            ['Estadia']
        )
        # self.area_pop_qtd_pacientes_df.groupby('Dia').agg({'Qtd_pacientes': 'sum'})
        #  self.area_pop_qtd_pacientes_df

        # ResourceCapacity[r,h] - Amount of resource type r, available at the hospital h. Note that the amount is the same along all planing horizon. Unit: # of recources type r
        self.hospital_recurso_sypply_df['Qtd_recurso'] *= 30
        parameters['ResourceCapacity'] = (
            self.hospital_recurso_sypply_df
            .set_index(['Recurso', 'Hospital'])
            .to_dict()
            ['Qtd_recurso']
        )

        # BigM - Big number to model binary variable Y. Set as the sum of all patient demand in the scenario. 
        parameters['BigM'] = self.area_pop_qtd_pacientes_df['Qtd_pacientes'].sum()

        self.params = parameters


    def create_decision_variables(self):

        print('Criando variáveis de decisão')
        decision_variables = dict()

        # X[p,a,h,t] - Continuous : Number of patients type p, area a, that enters in the hospital h, at the day t. Unit: # of patients
        decision_variables['X'] = self.model.addVars(
            [
                (p,a,h,t) 
                for p in self.model_sets['P'] 
                for a in self.model_sets['A'] 
                for h in self.model_sets['H']
                for t in self.model_sets['T']
             ],
            vtype=GRB.CONTINUOUS,
            name="X",
            lb=0
        )

        # N[p,h,t] - Continuous : Total number of patients type p, at the hospital h, day t. Unit: # of patients
        decision_variables['N'] = self.model.addVars(
            [
                (p,h,t) 
                for p in self.model_sets['P'] 
                for h in self.model_sets['H']
                for t in self.model_sets['T']
             ],
            vtype=GRB.CONTINUOUS,
            name="N",
            lb=0
        )

        # D - Continuous : Maximum distance that a patient has to travel to get to its hospital. 
        decision_variables['D'] = self.model.addVar(
            lb=0,
            vtype=GRB.CONTINUOUS,
            name='T'
        )

        # Y[a,h] - Binary : Represents if a patient from area a is assigned to hospital h along all planing horizont.
        decision_variables['Y'] = self.model.addVars(
            [
                (a,h) 
                for a in self.model_sets['A'] 
                for h in self.model_sets['H']
             ],
            vtype=GRB.BINARY,
            name="Y",
            lb=0
        )
    
        self.vars = decision_variables


    def create_constraints(self):

        # Constraint 1 - All patients demand need to be allocated to hospitals
        print('Creating constraint 1 - Patients Demand')
        for p in tqdm(self.model_sets['P']):
            for a in self.model_sets['A']:
                for t in self.model_sets['T']:
                    demand_tuple = (p,a,t)
                    self.model.addConstr(
                        quicksum(self.vars['X'][p,a,h,t] for h in self.model_sets['H']) == self.params['Demand'].get(demand_tuple,0),
                        name=f"C1_PatientAlocation_{p}_{a}_{t}"
                    )
        
        # Constraint 2 - Patient flow at initial day
        print('Creating Constraint 2 - Patient flow at initial day')
        for p in tqdm(self.model_sets['P']):
            for h in self.model_sets['H']:               
                LHS = self.vars['N'][p,h,1]

                tuple_get_released_patients = (p,h,1)
                RHS = self.params['InitPatients'][p,h] \
                    + quicksum(self.vars['X'][p,a,h,1] for a in self.model_sets['A']) \
                    - self.params['ReleasedPatients'].get(tuple_get_released_patients, 0)
                
                self.model.addConstr(
                        LHS == RHS,
                        name=f"C2_PatientFlowInitialDay_{p}_{h}"
                    )
                
        # Constraint 3 - Patient flow for all other days
        print('Creating Constraint 3 - Patient flow for all other days')
        for p in tqdm(self.model_sets['P']):
            for h in self.model_sets['H']:
                for t in self.model_sets['T']:
                    if t==1:
                        continue
                    
                    LHS = self.vars['N'][p,h,t]

                    release_patients_tuple = (p,h,t)
                    RHS = self.vars['N'][p,h,t-1] \
                        + quicksum(self.vars['X'][p,a,h,t] for a in self.model_sets['A']) \
                        - (
                            quicksum(self.vars['X'][p,a,h,t-self.params['LenghOfStay'][p]] for a in self.model_sets['A'] if ((t-self.params['LenghOfStay'][p]) >= 1)) \
                            + self.params['ReleasedPatients'].get(release_patients_tuple, 0)
                        )
                    self.model.addConstr(
                        LHS == RHS,
                        name=f"C3_PatientFlowOtherDays_{p}_{h}_{t}"
                    )

                    
        # Constraint 4 - Maximum capacity of resources at hospitals
        print('Creating Constraint 4 - Maximum capacity of resources at hospitals')
        for r in tqdm(self.model_sets['R']):
            for h in self.model_sets['H']:
                for t in self.model_sets['T']:
                    self.model.addConstr(
                        quicksum(self.vars['N'][p,h,t] for p in self.model_sets['S_r'][r]) <= self.params['ResourceCapacity'][r,h],
                        name=f"C4_ResourceMaxCapacity_{r}_{h}_{t}"
                    )

        # Constraint 5 - Function to activate Y binary variable for a given (a,h) if any patient from a is attended at hospital h
        for a in self.model_sets['A']:
            for h in self.model_sets['H']:
                self.model.addConstr(
                        quicksum(self.vars['X'][p,a,h,t] for p in self.model_sets['P'] for t in self.model_sets['T']) <= self.params['BigM'] *  self.vars['Y'][a,h],
                        name=f"C5_Y_binary_activation_{a}_{h}"
                    )
                
        # Constraint 6 - Calculus of maximum distance
        for a in self.model_sets['A']:
            for h in self.model_sets['H']:
                self.model.addConstr(
                        self.params['Distance'][a,h] * self.vars['Y'][a,h] <= self.vars['D'],
                        name=f"C6_Maximum_distance_{a}_{h}"
                    )
                
                    

        
    def set_obj_function(self, weight_obj_1: float, weight_obj_2: float):
        
        if self.obj_1 is None:
            # Obj1 - Minimizing total distance of all patients 
            self.obj_1 = quicksum(
                self.vars['X'][p,a,h,t] * self.params['Distance'][a,h] 
                for p in self.model_sets['P'] 
                for a in self.model_sets['A'] 
                for h in self.model_sets['H'] 
                for t in self.model_sets['T']
            )

        if self.obj_2 is None:
            # obj2 - Minimizing maximum distance
            self.obj_2 = quicksum([self.vars['D']])


        self.model.setObjective(
            # Obj1 - Maximização de prioridades
            weight_obj_1 * self.obj_1 +
            # Obj2 - Maximização de pesos
            weight_obj_2 * self.obj_2,
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


    def create_pareto_curve_weighted_sum(self):
        
        # Optimize only priorities
        self.set_obj_function(weight_obj_1=1, weight_obj_2=0)
        self.model.optimize()

        obj1_min = self.obj_1.getValue()

        # add constraint to garantee optimal distance, and minimize maximum dist
        self.model.addConstr(
                self.obj_1 <= obj1_min * 1.00001,
                name=f"force_best_obj1"
            )
        self.set_obj_function(weight_obj_1=0, weight_obj_2=1)
        self.model.optimize()
        # Get maximum cost
        obj2_max = self.obj_2.getValue()

        # Remove created constraint to force max priority
        constraint_added = self.model.getConstrByName('force_best_obj1')
        self.model.remove(constraint_added)

        # Optimize only maximum distance 
        self.set_obj_function(weight_obj_1=0, weight_obj_2=1)
        self.model.optimize()
        obj2_min = self.obj_2.getValue()

        # Optimize priorities with cost equals 0
        self.model.addConstr(
                self.obj_2 <= obj2_min * 1.00001,
                name=f"force_best_obj2"
            )
        self.set_obj_function(weight_obj_1=1, weight_obj_2=0)
        self.model.optimize() 

        obj1_max = self.obj_1.getValue()
        
        extreme_point_1 = (obj1_min, obj2_max)
        extreme_point_2 = (obj1_max, obj2_min)
        extreme_points = [extreme_point_1, extreme_point_2]
        delta_obj1 = obj1_max - obj1_min
        delta_obj2 = obj2_max - obj2_min


        # Remove created constraint to force_zero_cost
        constraint_added = self.model.getConstrByName('force_best_obj2')
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
            'obj1': [],
            'obj2': [],
        }
        idx_array = []
        pareto_original = copy.deepcopy(dict_intitialize_curve) 

        # Iterate over weights, optimize, and store pareto curve values
        for idx, w_obj1 in enumerate(weights):
            w_obj2 = 1 - w_obj1

            # Optimize 
            self.set_obj_function(
                weight_obj_1 = w_obj1/(obj1_max),
                weight_obj_2 = w_obj2/(obj2_max)
            )
            self.model.optimize()
            # self.generate_results()

            # Get results
            priority_result = self.obj_1.getValue()
            cost_result = self.obj_2.getValue()

            # Store pareto curve value
            pareto_original['obj1'].append(priority_result)
            pareto_original['obj2'].append(cost_result)
            
            # Store index
            idx_array.append(idx)
            print(f'Optimized Iteration {idx}')

        # Store last data point
        pareto_original['obj1'].append(obj1_min)
        pareto_original['obj2'].append(obj2_max)

        # # Store first data point
        pareto_original['obj1'].append(obj1_max)
        pareto_original['obj2'].append(obj2_min)

        idx_array.append(idx+1)
        idx_array.append(idx+2)

        pareto_original_df = pd.DataFrame(pareto_original, index=idx_array)
        pareto_original_df.to_csv('results_pareto/pareto_curve.csv')
        breakpoint()