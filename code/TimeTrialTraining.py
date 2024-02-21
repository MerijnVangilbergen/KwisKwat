import numpy as np
import pickle
from datetime import datetime
import copy
from utils import *

survival_rate = .2
eps = 1
Nraces = 20
Nsegments_min = 4
Nsegments_max = 10

Nagents = 5000

## Start from scrath
# include_car_vars = {'engine_efficiency': True, 'max_acceleration': True, 'max_deceleration': True, 'tank': False, 'grip': True, 'tyre_max_dist': False, 'roll_resistance_coef': True, 'air_resistance_coef': True}
# include_state_vars = {'health_tank': True, 'tyres_deterioration': True, 'distance_to_finish': False, 'position_local': [True,True], 'velocity_local': [True,True], 'orientation_local': True, 'omega': True, 'length_future': [True,False,False], 'curvature_future': [True,True,False,False]}
# agents = [AGENT(include_car_vars,include_state_vars, eps=eps) for dummy in range(Nagents)]

# Start from previously trained agents
with open('../output/20240218_2252_gen300_updated', 'rb') as f:
    print('Loading agents')
    include_state_vars = pickle.load(f)
    agents = pickle.load(f)
    print(include_state_vars)

# # print('Upgrading agents')
# include_state_vars_prev = include_state_vars
# include_state_vars = {'health_tank': True, 'tyres_deterioration': True, 'distance_to_finish': False, 'position_local': [True,True], 'velocity_local': [True,True], 'orientation_local': True, 'omega': True, 'length_future': [True,False,False], 'curvature_future': [True,True,False,False]}
# for agent in agents:
#     agent.upgrade(include_state_vars_prev, include_state_vars)


generation = 300
Nfinishers = 0

while True:
    generation += 1

    age = generation - np.asarray([agent.generation for agent in agents])
    Nage1 = np.mean(age==1)
    if Nage1 > (1-survival_rate) and eps < 1:
        eps *= 1.1
    elif Nage1 < (1-survival_rate)/4 and eps > .01:
        eps *= .9

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
                car = CAR(np.random.uniform(0,1,8), color='red')
                circuit = CIRCUIT(circuit_diagonal=np.random.uniform(200,350), N=Nsegments)
                race = RACE(circuit=circuit,cars=[car],laps=1,agents=agents,include_state_vars=include_state_vars,MaxTime=10*min(generation,60))

                save = (generation%25 == 0 and racenum == 0)
                if save:
                    current_datetime = datetime.now()
                    date_string = current_datetime.strftime("%Y%m%d_%H%M")
                    filename = date_string + '_gen' + str(generation)
                    with open('../output/'+filename, 'wb') as f:
                        pickle.dump(include_state_vars, f)
                        pickle.dump(agents, f)
                    race.simulate(saveas='../output/'+filename+'.mp4')
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