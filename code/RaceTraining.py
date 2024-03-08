import numpy as np
import pickle
from datetime import datetime
import copy
from utils import *

colors = ['red', 'purple', 'blue', 'green', 'yellow', 'orange', 'white', 'pink']

survival_rate = .25
eps = .01
Nraces = 200
Nsegments_min = 4
Nsegments_max = 10
MaxTime = 7*60


# # Start from previously trained agents on the time trial
# with open('../output/20240218_2252_gen300_updated', 'rb') as f:
#     print('Loading agents')
#     include_state_vars = pickle.load(f)
#     agents = pickle.load(f)
#     # print(include_state_vars)
# 
# print('Upgrading agents')
# include_state_vars_prev = dict(include_state_vars)
# for var in ['rel_position', 'rel_velocity']:
#     include_state_vars_prev[var] = [False, False]
#     include_state_vars[var] = [True, True]
# for var in ['rel_orientation', 'rel_omega']:
#     include_state_vars_prev[var] = False
#     include_state_vars[var] = True
# 
# for agent in agents:
#     agent.upgrade(include_state_vars_prev, include_state_vars)


# Start from previously trained agents
with open('../output/20240306_1751_gen70', 'rb') as f:
    print('Loading agents')
    include_state_vars = pickle.load(f)
    agents = pickle.load(f)
    # print(include_state_vars)

Nagents = len(agents) 
assert Nagents % 60 == 0 #Nagents is a multiple of 2,3,4,5,6

generation = 0
while True:
    generation += 1

    age = generation - np.asarray([agent.generation for agent in agents])
    Nage1 = np.mean(age==1)
    if Nage1 > (1-survival_rate) and eps < .01:
        eps *= 1.01
    elif Nage1 < (1-survival_rate)/2:
        eps *= .9

    print(f"gen={generation}, Nage1={Nage1}, eps={eps}, Nagents={Nagents}, Nraces={Nraces}")

    # Create offspring
    agents_prev = copy.deepcopy(agents)
    while len(agents) < Nagents:
        new_agents = copy.deepcopy(agents_prev)
        for agent in new_agents:
            agent.mutate(eps, generation=generation)
        agents = agents + new_agents
    assert len(agents) == Nagents

    # Simulate races
    penalties = np.zeros(len(agents))
    racenum = 0
    while racenum < Nraces:
        while True:
            try:
                circuit = CIRCUIT(circuit_diagonal=np.random.uniform(200,350), 
                                  Nsegments=np.random.randint(Nsegments_min, Nsegments_max))
                Ncars = np.random.randint(1,7)
                cars = [CAR(np.random.uniform(0,1,8), color=colors[dummy]) for dummy in range(Ncars)]
                new_penalties = np.zeros(Nagents)

                permutation = np.random.permutation(Nagents)
                groups = np.array([permutation[n*Ncars:(n+1)*Ncars] for n in range(Nagents//Ncars)]) # shape(Nraces, Ncars)
                    # All agents in group n ride car n
            
                for iteration in range(Ncars):
                    if iteration > 0:
                        # roll groups such that each agent rides each car once
                        groups = np.roll(groups, 1, axis=1)
                        # shuffle agents within each group
                        for row in range(1,groups.shape[0]):
                            groups[row] = np.random.permutation(groups[row])
                    input_order = groups.flatten()
                    agents_reordered = [agents[idx] for idx in input_order]
                    race = RACE(circuit=circuit,cars=cars,laps=1,agents=agents_reordered,include_state_vars=include_state_vars,MaxTime=MaxTime)

                    save = (generation%10 == 0 and racenum == 0 and iteration == 0)
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
                    new_penalties[input_order] += ((race.time_elapsed + race.distance_left)/MaxTime + race.ranking)
                break
            except:
                print('An error occurred. Trying again.')
                continue

        racenum += Ncars
        penalties += new_penalties

    # Keep the best agents
    if generation%10 == 0 and Nagents > 60:
        Nagents_prev = Nagents
        Nagents = Nagents_prev - 60
        Nraces = int(Nraces*Nagents_prev/Nagents)
    agents = [agents[idx] for idx in np.argsort(penalties)[:int(survival_rate*Nagents)]]