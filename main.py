import numpy as np
import pandas as pd
import tkinter as tk
from collections import namedtuple
from PIL import ImageTk, Image
import time
import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

def vector2angle(vector,deg=False):
    complex_number = vector[0] + vector[1] *1j
    return np.angle(complex_number, deg=deg)

class CIRCUIT:
    def __init__(self,size,startlength,N):
        def select_checkpoints():
            # Select uniformly distributed points in the rectangular area
            p = np.vstack([ [(size[0]-startlength)/2, 0], #start finish line
                            [(size[0]+startlength)/2, 0], #end finish line
                            np.column_stack([np.random.uniform(0,size[0],N-2), np.random.uniform(0,size[1],N-2)]) ])
            # Replace points that are too close to each other
            minimal_separation_distance = 3*default_width
            for ii in range(2,len(p)):
                while True:
                    separation = np.linalg.norm(p[ii]-p[0:ii],axis=1)
                    if np.any(separation < minimal_separation_distance):
                        p[ii] = np.column_stack([np.random.uniform(0,size[0]), np.random.uniform(0,size[1])])
                    else:
                        break
            p = np.roll(p,-1,axis=0) # Such that the finish line is the line segment between checkpoints N and 0

            # Choose an order of check points such that there are no intersections
            p, dir = untangle_knot(p)
            length = np.linalg.norm(dir, axis=1)
            tangent = dir / length[:,np.newaxis]
            normal = np.column_stack([-tangent[:,1], tangent[:,0]])

            plt.plot(p[:,0],p[:,1])
            plt.show()

            # At this point, it is guaranteed that the line segments do not mutually intersect, but overlap is possible due to the road width.
            p,tangent,normal,length = resolve_overlap(p,tangent,normal,length)
            plt.plot(p[:,0],p[:,1])
            plt.show()

            # At this point, we have a valid circuit. However, we move some check_points in order to utilise space more efficient. Line segments become longer and turnings become sharper.
            # p,tangent,normal,length = spread_out(p,tangent,normal,length)
            # plt.plot(p[:,0],p[:,1])
            # plt.show()

            return (p,tangent,normal,length)

        def check_points_to_frenet(p,cyclic=False):
            # Input: p is a sequence of 2D points.
            # Output: the tangent, normal and lenght of the line segments connecting the check points p
            # Convention: p[ii] = p[ii-1] + length[ii]*tangent[ii]
            if cyclic:
                tangent = np.diff(np.vstack([p[-1], p]), axis=0)
            else:
                tangent = np.diff(p, axis=0)
            length = np.linalg.norm(tangent, axis=1)
            tangent = tangent / length[:,np.newaxis]
            normal = np.column_stack([-tangent[:,1], tangent[:,0]])
            return (tangent, normal, length)

        def untangle_knot(p):
            dir = np.diff(np.vstack([p[-1],p]), axis=0)
            ii = 1
            jj = N-1
            while ii < N-2:
                # Check for intersection:
                # Find lambda1 and lambda2 such that p[ii]-lambda0*dir[ii] = p[jj]-lambda1*dir[jj]
                # Intersection found if both lambda0 and lambda1 are within the interval [0,1]
                lambda_ = np.linalg.solve(np.column_stack((dir[ii], -dir[jj])), p[ii]-p[jj])
                if np.all(lambda_>0) and np.all(lambda_<1):
                    # An intersection was found. Flip the loop to resolve the intersection.
                    p[ii:jj] = np.flip(p[ii:jj], axis=0)
                    dir[ii] = p[ii] - p[ii-1]
                    dir[jj] = p[jj] - p[jj-1]
                    dir[ii+1:jj] = -np.flip(dir[ii+1:jj], axis=0)
                    # Restart the search for intersections
                    ii = 1
                    jj = N-1
                elif jj-ii <= 2:
                    # No intersection found between segment ii and any other segment (jj>ii). Next, we look for intersections between segment ii+1 and any other segment (jj>ii).
                    ii += 1
                    jj = N-1
                else:
                    # No intersection found between segments ii and jj. Next, we look for an intersection between segments ii and jj-1.
                    jj -= 1
            return p, dir

        def resolve_overlap(p,tangent,normal,length):
            # Input: The points p must be ordered in a way that there are no intersections.
            # Taking into account the road width, some line segments may overlap. Any such occurrences are resolved by moving at least one check_point away from a line segment.
            ii = 0
            jj = 1
            while ii < N:
                #Check whether any check_point jj is too close to line segment ii.
                minimal_distance = default_width * 1.1
                coord = (p[jj]-p[ii]) @ np.column_stack([tangent[ii],normal[ii]])
                if np.abs(coord[1])<minimal_distance and coord[0]<=0 and coord[0]>=-length[ii]:
                    #Check_point jj is too close to line segment ii. We move check_point jj away from line segment ii along the normal of line segment ii.
                    print('Check point ',jj,' was moved from ',p[jj],' to ',p[jj]+np.sign(coord[1]) * (minimal_distance-np.abs(coord[1])) * normal[ii],' (away from line segment ',ii,').')
                    p[jj] += np.sign(coord[1]) * (minimal_distance-np.abs(coord[1])) * normal[ii]
                    tangent[[jj,(jj+1)%N]], normal[[jj,(jj+1)%N]], length[[jj,(jj+1)%N]] = check_points_to_frenet(p[[jj-1,jj,(jj+1)%N]])
                jj = (jj+1)%N
                if jj == (ii-1)%N:
                    ii += 1
                    jj = (ii+1)%N
            return (p,tangent,normal,length)

        def spread_out(p,tangent,normal,length):
            difftangent = np.diff(np.vstack([tangent,tangent[0]]), axis=0)
            difftangent_len = np.linalg.norm(difftangent, axis=1)
            difftangent = difftangent / difftangent_len[:,np.newaxis]

            #We iterate over all checkpoints in the order of increasing sharpness
            for ii in np.argsort(difftangent_len):
                if ii in [0,N-1]:
                    break
                #What is the maximal distance over which we can move p[ii] in the direction of -difftangent[ii] ?
                try:
                    #Find first intersection of p[ii]-lambda*difftangent[ii] with line segments
                    max_mov_dist = first_intersection(p, point=p[ii], dir=-difftangent[ii])
                except Exception:
                    #Find first intersection of p[ii]-lambda*difftangent[ii] with rectangular border
                    max_mov_dist = first_intersection(np.array([[0,0], [size[0],0], [size[0],size[1]], [0,size[1]]]), point=p[ii], dir=-difftangent[ii])
                print(max_mov_dist)
                print('Check point ',ii,' was moved from ',p[ii],' to ',p[ii] - max_mov_dist * difftangent[ii],'.')
                p[ii] -= max_mov_dist * difftangent[ii]
                tangent[[ii,(ii+1)%N]], normal[[ii,(ii+1)%N]], length[[ii,(ii+1)%N]] = check_points_to_frenet(p[[ii-1,ii,(ii+1)%N]])
                difftangent[[ii-1,ii,(ii+1)%N]] = np.diff(tangent[[ii-1,ii,(ii+1)%N,(ii+2)%N]], axis=0)
                plt.plot(p[:,0],p[:,1])
                plt.show()
            return (p,tangent,normal,length)

        def first_intersection(p,point,dir):
            #Input: dir is a unit vector.
            #Output: the distance between point and the first intersection along point+lambda*dir for lambda>0.
            N = np.shape(p)[0]
            normal_dir = np.array([-dir[1],dir[0]])
            #Express the checkpoints in the coordinate system defined by dir and normal_dir
            coords = (p-point) @ np.column_stack((dir,normal_dir))

            # An intersection with the path is present when a sign switch occurs in the normal coordinate
            intersect = np.roll(coords[:,1],-1) * coords[:,1] <= 0
            intersect_idx = np.where(intersect)[0]
            dist_to_intersection = np.array([coords[idx-1,0] - coords[idx-1,1] * (coords[idx,0]-coords[idx-1,0])/(coords[idx,1]-coords[idx-1,1]) for idx in intersect_idx])

            if dist_to_intersection.size == 0:
                raise Exception("No intersections found")
            
            intersect_idx = intersect_idx[dist_to_intersection>0]
            dist_to_intersection = dist_to_intersection[dist_to_intersection>0]

            if dist_to_intersection.size == 0:
                print(p,point,dir)
                raise Exception("No intersections found")
            else:
                argmin_ = np.argmin(dist_to_intersection)
                intersect_idx = intersect_idx[argmin_]
                dist_to_intersection = dist_to_intersection[argmin_]
                return dist_to_intersection
        
        self.size = size
        check_points,tangent,normal,length = select_checkpoints()
        
        difftangent = np.diff(np.vstack([tangent,tangent[0]]), axis=0)
        difftangent_len = np.linalg.norm(difftangent, axis=1)
        difftangent = difftangent / difftangent_len[:,np.newaxis]
        dist2center = np.zeros(N)
        for ii in range(N):
            dist2center_min = default_width / np.sqrt(2*(1-np.abs(np.dot(tangent[ii],tangent[(ii+1)%N]))))
            dist2center_max = np.min(length[[ii,(ii+1)%N]]) / np.sqrt(2*(1+np.abs(np.dot(tangent[ii],tangent[(ii+1)%N]))))
            # while dist2center_min > dist2center_max:
            #     # Move the checkpoint further from its neighbouring checkpoints if the turning is too narrow.
            #     check_points[ii] -= difftangent[ii] * default_width
            #     tangent[[ii,(ii+1)%N]], normal[[ii,(ii+1)%N]], length[[ii,(ii+1)%N]] = check_points_to_frenet(check_points[[ii-1,ii,(ii+1)%N]])
            #     print('Checkpoint ',ii,' was moved in order to handle a sharp turning')
            #     dist2center_min = 
            #     dist2center_max = 
            print(dist2center_min,dist2center_max)
            dist2center[ii] = np.random.uniform(dist2center_min,dist2center_max)
        
        centers = check_points + dist2center[:,np.newaxis] * difftangent
        length = length - np.sum((check_points-centers)*tangent,axis=1) - np.sum(np.roll(centers-check_points,1,axis=0)*tangent,axis=1)
        destination = check_points - np.sum((check_points-centers)*tangent,axis=1)[:,np.newaxis] * tangent

        radius = np.sum((centers-destination) * normal, axis=1)
        curvature = 1/radius
        turning_left = np.sign(radius)
        radius = np.abs(radius)
        angle_change = turning_left * np.arccos( np.sum(tangent * np.roll(tangent,-1,axis=0), axis=1) )
        angle_final = np.array([vector2angle(-turning_left[ii] * normal[(ii+1)%N]) for ii in range(N)])

        self.data = pd.DataFrame(data={'tangent': [tangent[dummy] for dummy in range(N)],
                                       'normal': [normal[dummy] for dummy in range(N)],
                                       'length': [length[dummy] for dummy in range(N)],
                                       'destination': [destination[dummy] for dummy in range(N)],
                                       'center': [centers[dummy] for dummy in range(N)],
                                       'radius': radius,
                                       'curvature': curvature,
                                       'angle_change': angle_change,
                                       'angle_final': angle_final})
        print('circuit data: \n ', self.data)
    def plot(self):
        # We construct a series of 2D points to be plotted
        all_points = np.empty((0,2))
        for ii in range(len(self.data)):
            theta = np.linspace(self.data.at[ii, 'angle_final'] - self.data.at[ii, 'angle_change'], 
                                self.data.at[ii, 'angle_final'])
            turning = self.data.at[ii, 'center'] + self.data.at[ii, 'radius'] * np.column_stack([np.cos(theta), np.sin(theta)])
            all_points = np.vstack([all_points, self.data.at[ii, 'destination'], turning])
        # all_points = np.vstack([all_points,all_points[0]])

        # Plotting the race circuit
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_axes([0,0,1,1])
        ax.set_aspect('equal')
        ax.set_facecolor('green')
        ax.plot(all_points[:,0], all_points[:,1], 'r-', linewidth=fig.dpi*ipm*default_width*1.1)
        ax.plot(all_points[:,0], all_points[:,1], 'w--',linewidth=fig.dpi*ipm*default_width*1.1)
        ax.plot(all_points[:,0], all_points[:,1], color=(0.3, 0.3, 0.3), linestyle='-', linewidth=fig.dpi*ipm*default_width)
        plt.show()

CAR = namedtuple("CAR", "acceleration brake grip tank")

class AGENT:
    def __init__(self):
        self.neural_network = 1
    def getAction(self,state):
        acceleration = np.random.uniform(-1,1)
        steering = np.random.uniform(-1,1)
        return acceleration, steering

class RACE:
    def __init__(self,circuit,agents,cars):
        self.circuit = circuit
        self.agents = agents
        self.cars = cars
        numOfSteps = 4

        # Create a 3D dataframe for storage of results
        A = np.array(['car'+str(driver_idx) for driver_idx in range(len(agents))])
        B = np.array(['position','velocity','health','segment','reached_turning'])
        C = numOfSteps*[len(A)* [np.zeros(2), np.zeros(2), 1, 0, False]]
        self.data = pd.DataFrame(data=C, columns=pd.MultiIndex.from_tuples(zip(np.repeat(A, len(B)), np.tile(B, len(A)))))
        print('race data: \n', self.data)
    def simulate(self):
        for tt in range(self.data.shape[0]-1):
            # Determine the inputs to the neural networks, i.e. what the agent sees.
            agent_state = self.map_to_agent_state(tt)
            # The agent determines which action to take
            actions = [agent.getAction(agent_state) for agent in self.agents]
            # Calculate the next state using Forward Euler (or any other integration scheme)
            self.step_simulator(tt,actions)
            print('race data: \n', self.data)
    def map_to_agent_state(self,tt):
        # Input: the global state at some instance of time (stored in self.data.iloc[tt])
        # Output: the agent state, stored as a numpy array. The elements represent the following:
            # Car stats
            # --- time left in the race
            # position along the tangent wrt the destination (distance to next segment)
            # position along the normal wrt the destination (deviation from the road center)
            # velocity along the tangent wrt the destination
            # velocity along the normal wrt the destination
            # curvature of current segment
            # curvature of next segment
            # length of next segment
            # --- orientation or drift
            # --- difference in total elapsed distance wrt first car ahead
            # --- normal coordinate of first car ahead
            # --- difference in total elapsed distance wrt first car behind
            # --- normal coordinate of first car behind

        car=0 #MAKE THIS AN INPUT ARGUMENT
        car_string = 'car'+str(car)
        segment = self.data.at[tt,(car_string,'segment')]
        if self.data.at[tt,(car_string,'reached_turning')]:
            radial_position = self.data.at[tt,(car_string,'position')] - self.circuit.data.at[segment,'centers']
            angle = vector2angle(radial_position)
            radius = np.linalg(radial_position)
            return np.concatenate(( self.cars[car], #car stats
                                   [self.circuit.data.at[segment,'radius'] * (angle-self.circuit.data.at[segment,'angle_final']), #distance to next segment
                                    (radius-self.circuit.data.at[segment,'radius']) * np.sign(self.circuit.data.at[segment,'curvature']), #deviation from the road center
                                    #velocity
                                    self.circuit.data.at[segment,'curvature'], #curvature of current segment
                                    0 #curvature of next segment
                                   ]
                                    ))
        else:
            return np.concatenate(( self.cars[car], #car stats
                                    (np.vstack([ self.data.at[tt,(car_string,'position')]-self.circuit.data.at[segment,'destination'],
                                                 self.data.at[tt,(car_string,'velocity')] ])
                                     @ np.column_stack([self.circuit.data.at[segment,'tangent'], self.circuit.data.at[segment,'normal']])
                                    ).reshape(-1), #position and velocity
                                    [0, self.circuit.data.at[(segment+1)%self.circuit.data.shape[0],'curvature']] #curvature of current and next segment
                                    ))

    def step_simulator(self,tt,actions):
        # Input: the global state at some instance of time (stored in self.data.iloc[tt])
        # Input: the action of each driver
        # Output: the global state at the next instance of time (stored in self.data.iloc[tt+1])

        # state = self.data.iloc[tt].copy(deep=True)
        # state.loc['car0','velocity'] += np.array([2,0])

        for car in range(len(self.agents)):
            car_string = 'car'+str(car)

            acceleration = actions[car][0] * np.array([1,0])

            self.data.at[tt+1,(car_string,'velocity')] = self.data.at[tt,(car_string,'velocity')] + dt * acceleration
            self.data.at[tt+1,(car_string,'position')] = self.data.at[tt,(car_string,'position')] + dt/2 * (self.data.at[tt,(car_string,'velocity')]+self.data.at[tt+1,(car_string,'velocity')])
    def display(self):
        self.circuit.plot()

        # # Create an image for each car
        # car_images = []
        # for car in range(len(self.agents)):
        #     car_images.append(Image.open("Car.png")) #CHANGE COLORS
        #     photo_image = ImageTk.PhotoImage(car_image.rotate(90))
        #     canvas.create_image(200,200,image=photo_image)
        
        # # Update positions for each time step
        # for tt in range(np.shape(self.data)[0]):
        #     start_time = time.time()
        #     for car in range(len(self.agents)):
        #         car_string = 'car'+str(car)
        #         pos = self.data.at[tt,(car_string,'position')]
        #     end_time = time.time()
        # time.sleep(max(0,dt-(end_time-start_time)))

        # window.mainloop()

default_width = 12 #[meters]
dt = 1

# fig_size_pixels = (1920,1080) #[pixels]
fig_size = (12.8,7.2) #[inches]
circuit_width = 500 #[meters]
ipm = fig_size[0]/circuit_width #inches per meter
circuit_size = (circuit_width, fig_size[1] /ipm) #[meters]

circuit = CIRCUIT(size=circuit_size,startlength=250,N=20)
circuit.plot()

# agent1 = AGENT()
# agent2 = AGENT()
# car1 = CAR(1,1,1,1)
# car2 = CAR(1,1,1,1)
# race = RACE(circuit,[agent1,agent2], [car1,car2])
# race.simulate()
# race.display()