import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pickle
from datetime import datetime
import copy
from utils import *

# Select folders
output_folder = '../output'

survival_rate = .25
eps = 1
Nraces = 25
Nsegments_min = 4
Nsegments_max = 10
# Nturns = 1
car_interval_width = 1
MaxTime = 10*60

Nagents = 5000
# include_car_vars = {'engine_efficiency': False, 'max_acceleration': False, 'max_deceleration': False, 'tank': False, 'grip': False, 'tyre_max_dist': False, 'roll_resistance_coef': False, 'air_resistance_coef': False}
# include_state_vars = {'health_tank': False, 'tyres_deterioration': False, 'distance_to_finish': False, 'position_local': [True,True], 'velocity_local': [True,True], 'orientation_local': True, 'omega': True, 'length_future': [True,False,False], 'curvature_future': [True,True,False,False]}
# agents = [AGENT(include_car_vars,include_state_vars, eps=eps) for dummy in range(Nagents)]
with open('../output/20240203_2222_gen100', 'rb') as f:
    print('Loading agents')
    include_car_vars = pickle.load(f)
    include_state_vars = pickle.load(f)
    agents = pickle.load(f)
    print(include_car_vars)
    print(include_state_vars)

# print('Upgrading agents')
# include_car_vars_prev = include_car_vars
# include_state_vars_prev = include_state_vars
# include_car_vars = {'engine_efficiency': True, 'max_acceleration': True, 'max_deceleration': True, 'tank': False, 'grip': True, 'tyre_max_dist': False, 'roll_resistance_coef': True, 'air_resistance_coef': True}
# include_state_vars = {'health_tank': True, 'tyres_deterioration': True, 'distance_to_finish': False, 'position_local': [True,True], 'velocity_local': [True,True], 'orientation_local': True, 'omega': True, 'length_future': [True,False,False], 'curvature_future': [True,True,False,False]}
# for agent in agents:
#     agent.upgrade(include_car_vars_prev, include_state_vars_prev, include_car_vars, include_state_vars)

generation = 0
Nfinishers = 0
# next_change_generation = 0
while True:
    generation += 1

    age = generation - np.asarray([agent.generation for agent in agents])
    Nage1 = np.mean(age==1)
    if Nage1 > (1-survival_rate) and eps < 1:
        eps *= 1.1
    elif Nage1 < (1-survival_rate)/4 and eps > .01:
        eps *= .9

    # if Nfinishers > survival_rate * Nagents: #and generation >= next_change_generation:
        # if not(Nturns is None):
        #     # increase complexity of circuit
        #     if Nturns < Nsegments_max:
        #         Nturns += 1
        #         MaxTime *= (Nturns/(Nturns-1))
        #         print(f'Nturns increased to {Nturns}')
        #     else:
        #         Nturns = None
        #         print('Nturns set to maximum')
        # else:
            ## increase complexity of inputs
            # include_car_vars_prev = copy.deepcopy(include_car_vars)
            # include_state_vars_prev = copy.deepcopy(include_state_vars)
            # if not(include_state_vars['curvature_future'][1]):
            #     include_state_vars['curvature_future'][1] = True
            #     print('curvature_future[1] added')
            # elif not(include_state_vars['health_tank']):
            #     include_state_vars['health_tank'] = True
            #     print('health_tank added')
            # elif not(include_state_vars['tyres_deterioration']):
            #     include_state_vars['tyres_deterioration'] = True
            #     print('tyres_deterioration added')
            # elif not(include_state_vars['distance_to_finish']):
            #     include_state_vars['distance_to_finish']= True
            #     print('distance_to_finish added')
            # for agent in agents:
            #     agent.upgrade(include_car_vars_prev, include_state_vars_prev, include_car_vars, include_state_vars)
        # next_change_generation = generation + 10

        # car_interval_width = min(car_interval_width*1.1, 1)
        # print(f'car_interval_width increased to {car_interval_width}')

    print(f"gen={generation}, Nfinishers={Nfinishers}, Nage1={Nage1}, eps={eps}")

    # Create offspring
    while len(agents) < Nagents:
        new_agents = copy.deepcopy(agents)
        for agent in new_agents:
            agent.mutate(eps, generation=generation)
        agents = agents + new_agents

    # Simulate races
    penalties = np.zeros(len(agents))
    Nfinishers = 0
    for racenum in range(Nraces):
        while True:
            try:
                Nsegments = np.random.randint(Nsegments_min, Nsegments_max)
                # car = CAR(np.repeat(.5,8))
                car = CAR(np.random.uniform(.5-car_interval_width/2, .5+car_interval_width/2, 8))
                circuit = CIRCUIT(circuit_diagonal=np.random.uniform(200,350), N=Nsegments, start_coordinates=np.random.uniform([0,-.5],[-1,.5]))
                # circuit = CIRCUIT(circuit_diagonal=np.random.uniform(200,350), N=Nsegments, Nturns=min(Nturns,Nsegments), start_coordinates=np.random.uniform([0,-.5],[-1,.5]))
                race = RACE(circuit=circuit,car=car,laps=1,agents=agents,include_car_vars=include_car_vars,include_state_vars=include_state_vars,MaxTime=MaxTime)

                save = (generation%25 == 0 and racenum == 0)
                if save:
                    current_datetime = datetime.now()
                    date_string = current_datetime.strftime("%Y%m%d_%H%M")
                    filename = date_string + '_gen' + str(generation)
                    with open(output_folder+'/'+filename, 'wb') as f:
                        pickle.dump(include_car_vars, f)
                        pickle.dump(include_state_vars, f)
                        pickle.dump(agents, f)
                    race.simulate(saveas=output_folder+'/'+filename+'.mp4')
                else:
                    race.simulate()
                    break
            except:
                print('An error occurred. Trying again.')
                continue

        penalties += race.penalty
        Nfinishers += race.Nfinishers
    Nfinishers /= Nraces

    # Keep the best agents
    agents = [agents[idx] for idx in np.argsort(penalties)[:int(survival_rate*Nagents)]]