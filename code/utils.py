import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.transforms as transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import time

# Select folders
output_folder = '../output'


def vector2angle(vector):
    # Given a (radial) vector, return the angle with the x-axis.
    # The angle is expressed in the interval [-pi, pi].
    if vector.ndim == 1:
        # The input is np.array([x,y])
        return np.arctan2(vector[1], vector[0])
    else:
        # The input is a Nx2 array. Every row corresponds to one vector. The output is an array of length N with all corresponding angles.
        return np.arctan2(vector[:,1], vector[:,0])

def angle_minus(a,b):
    # Calculate a-b such that the output is in (-pi,pi] (assuming both a and b are withing (-pi,pi]).
    diff = a-b
    diff[diff <= -np.pi] += 2*np.pi
    diff[diff > np.pi] -= 2*np.pi
    return diff

def rotate90left(vector):
    if vector.ndim == 1:
        # Both input and output vector have length 2.
        return np.array([-vector[1], vector[0]])
    else:
        # Both input and output vector shape (Nwheels,2).
        return np.column_stack([-vector[:,1], vector[:,0]])

def get_TN_from_angle(angle):
    # Define TN such that TN[n,:,:] is a 2*2 matrix of which the columns represent the tangent and normal corresponding to angle[n].
    TN = np.zeros((len(angle),2,2))
    cos = np.cos(angle)
    sin = np.sin(angle)
    TN[:,0,0] = cos     #Tx = cos(angle)
    TN[:,1,0] = sin     #Ty = sin(angle)
    TN[:,0,1] = -sin    #Nx = -sin(angle)
    TN[:,1,1] = cos     #Nx = cos(angle)
    return TN

def to_frame(vector,frame_TN):
    # The input vector is expressed in the global frame.
    # The output vector is expressed in an alternative frame with frame_TN as [Tangent,Normal]-pair with respect to the global frame.
    if vector.ndim == 1:
        # frame_TN is a 2D-array of size 2x2. Its columns represent the tangent and the normal of the alternative frame with respect to the global frame.
        # The output represents the same vector as the input, but is expressed in the alternative frame.
        return np.transpose(frame_TN) @ vector
    elif vector.ndim == 2 and frame_TN.ndim == 2:
        # vector has shape (N,2)
        # frameTN has shape (2,2)
        return np.matmul(np.transpose(frame_TN)[np.newaxis,:,:], vector[:,:,np.newaxis]).squeeze(axis=2) # shape (N,2)
    elif vector.ndim == 2 and frame_TN.ndim == 3:
        # vector has shape (N,2)
        # frameTN has shape (N,2,2)
        return np.matmul(np.transpose(frame_TN,[0,2,1]), vector[:,:,np.newaxis]).squeeze(axis=2) # shape (N,2)
    elif vector.ndim == 3 and frame_TN.ndim == 4:
        # vector has shape (Nagents,Nwheels,2)
        # frameTN has shape (Nagents,Nweels,2,2)
        return np.matmul(np.transpose(frame_TN,[0,1,3,2]), vector[:,:,:,np.newaxis]).squeeze(axis=3) # shape (Nagents,Nwheels,2)
    else:
        raise NotImplementedError("The function to_frame was not implemented for {vector.ndim}-dimensional vector and {frame_TN.ndim}-dimensional frame_TN.")

class CIRCUIT:
    def __init__(self,circuit_diagonal,N,Nstartspots=1,start_coordinate=-.8,start_coordinate2=0):
        # start_coordinate2 is ignored if Nstartspots>1.
        assert N > 3 # For N=3, an infinite loop may/will occur in the untangle_knot function.

        # Create figure
        self.fig = plt.figure(figsize=fig_size)
        ax = self.fig.add_axes([0, 0, 1, 1])
        # Remove x and y axes
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False,
                    left=False, labelleft=False)

        # Extract ax_size [inches] and circuit_size [meters]
        ax_size = fig_size * np.array([ax.figure.subplotpars.right - ax.figure.subplotpars.left, 
                                       ax.figure.subplotpars.top - ax.figure.subplotpars.bottom])
        ipm = np.linalg.norm(ax_size) / circuit_diagonal #inches per meter
        self.dpm = self.fig.dpi * ipm * .87 #dots per meter
            # This number is solely used for line_widths in the circuit plot. The factor .87 is a manual correction based on visual results.
        self.size = ax_size / ipm #circuit_size [meters]

        # Close the figure
        plt.close(self.fig)

        # Construct a primitive circuit (random choices) and copy the data
        while True:
            try:
                primitive = self.PRIMITIVE_CIRCUIT(self.size,N)
                break
            except Exception as e:
                # print(e)
                pass

        self.Nsegments = primitive.Nsegments
        self.destination = primitive.destination
        self.TN = primitive.TN
        self.length = primitive.length
        
        # Define turnings (random choices)
        dist2center = np.random.uniform(primitive.dist2center_min, primitive.dist2center_max)
        
        difftangent = np.diff(np.vstack([self.TN[:,:,0],self.TN[0,:,0]]), axis=0)
        difftangent = difftangent / np.linalg.norm(difftangent, axis=1)[:,np.newaxis]
        self.center = self.destination + dist2center[:,np.newaxis] * difftangent # shape (Nsegments,2)
        coord_center = to_frame(vector=self.center-self.destination, frame_TN=self.TN) # shape (Nsegments,2)
        displacement_destination = coord_center[:,0]
        self.radius = coord_center[:,1] # shape Nsegments
        self.destination += displacement_destination[:,np.newaxis] * self.TN[:,:,0]
        self.length += (displacement_destination + np.roll(displacement_destination,1,axis=0))

        # Roll data such that segment 0 is the longest segment.
        roll_idx = np.argmax(self.length)
        self.destination = np.roll(self.destination, -roll_idx, axis=0)
        self.TN = np.roll(self.TN, -roll_idx, axis=0)
        self.length = np.roll(self.length, -roll_idx)
        self.center = np.roll(self.center, -roll_idx, axis=0)
        self.radius = np.roll(self.radius, -roll_idx)

        # Derive other data (no choices)
        self.curvature = 1/self.radius # shape Nsegments
        turning_left = np.sign(self.radius)
        self.radius = np.abs(self.radius) # shape Nsegments
        self.angle_change = turning_left * np.arccos( np.sum(self.TN[:,:,0] * np.roll(self.TN[:,:,0],-1,axis=0), axis=1) ) # shape Nsegments
        self.angle_final = vector2angle(-turning_left[:,np.newaxis] * np.roll(self.TN[:,:,1],-1,axis=0)) # shape Nsegments
        self.length_turning = self.radius * np.abs(self.angle_change)

        self.orientation = vector2angle(self.TN[:,:,0]) # shape Nsegments

        # Select start and finish
        if Nstartspots == 1:
            self.start = self.destination[0,:] + self.TN[0] @ np.array([start_coordinate*self.length[0], start_coordinate2*road_width])
            self.start = self.start[np.newaxis,:] # shape (Nstartspots,2)
        else:
            self.start = self.destination[0,:][np.newaxis,:] + start_coordinate * self.length[0] * self.TN[0,:,0][np.newaxis,:] + np.linspace(-(Nstartspots-1)/(2*Nstartspots)*road_width,(Nstartspots-1)/(2*Nstartspots)*road_width,Nstartspots)[:,np.newaxis] * self.TN[0,:,1][np.newaxis,:] # shape (Nstartspots,2)
        self.finish = -.2 * self.length[0] # This is the tangential component of the local position in segment[0]

        temp = self.length + self.length_turning
        temp = np.cumsum(temp[::-1])[::-1] # backward cumsum
        self.lap_distance = temp[0]
        temp = np.concatenate((temp[1:], [0]))
        self.distance_to_finish = temp + (self.length[0] + self.finish)
        
    def plot(self):
        ax = self.fig.get_axes()[0]

        ax.set_aspect('equal')
        ax.set_xlim(0,self.size[0])
        ax.set_ylim(0,self.size[1])

        # Background
        ax.set_facecolor('green')

        # # Plot the pitstop
        # ax.plot(pitstop_points[:,0], pitstop_points[:,1], 'w-', linewidth=fig.dpi*ipm*road_width*2.1)
        # # ax.plot(pitstop_points[:,0], pitstop_points[:,1], 'r--',linewidth=fig.dpi*ipm*road_width*2.1, dashes=(1,1))
        # ax.plot(pitstop_points[:,0], pitstop_points[:,1], color=(0.3, 0.3, 0.3), linestyle='-', linewidth=fig.dpi*ipm*road_width*2)

        # Plot the track
        all_points = self.destination[0] + self.finish * self.TN[0,:,0]
        for segment in range(self.Nsegments):
            theta = np.linspace(self.angle_final[segment] - self.angle_change[segment], 
                                self.angle_final[segment])
            turning = self.center[segment,:] + self.radius[segment] * np.column_stack([np.cos(theta), np.sin(theta)])
            all_points = np.vstack([all_points, self.destination[segment,:], turning])
        all_points = np.vstack([all_points,all_points[0]])
        ax.plot(all_points[:,0], all_points[:,1], 'w-', linewidth=self.dpm*road_width*1.1)
        ax.plot(all_points[:,0], all_points[:,1], 'r--',linewidth=self.dpm*road_width*1.1, dashes=(1,1))
        ax.plot(all_points[:,0], all_points[:,1], color=(0.3, 0.3, 0.3), linestyle='-', linewidth=self.dpm*road_width)

        # Plot the finish line
        finish_points = self.destination[0,:][:,np.newaxis] + self.TN[0] @ np.column_stack([[self.finish, -road_width/2], [self.finish, road_width/2]])
        ax.plot(finish_points[0,:], finish_points[1,:], 'w-', linewidth=self.dpm*.5) # 0.5 meters
    
    class PRIMITIVE_CIRCUIT:
        def __init__(self,size,N):
            max_execution_time = 10
            start_primitive = time.time()

            self.size = size
            self.Nsegments = N
            self.select_checkpoints(size,minimal_separation=5*road_width)
                # This creates self.destination with shape (Nsegments,2)
            
            dir = np.diff(np.vstack([self.destination[-1], self.destination]), axis=0)
            self.length = np.linalg.norm(dir, axis=1)        # shape Nsegments
            self.TN = np.zeros((self.Nsegments,2,2))         # shape (Nsegments,2,2)
            self.TN[:,:,0] = dir / self.length[:,np.newaxis] # tangent
            self.TN[:,:,1] = rotate90left(self.TN[:,:,0])    # normal
            cos_2theta = np.abs(np.sum(self.TN[:,:,0] * np.roll(self.TN[:,:,0], -1, axis=0), axis=1))
            cos_theta = np.sqrt((1+cos_2theta)/2)
            sin_theta = np.sqrt((1-cos_2theta)/2)
            self.dist2center_min = road_width/2 / sin_theta
            self.dist2center_max = np.min(np.column_stack([self.length, np.roll(self.length,-1)]), axis=1) /2 / cos_theta

            # Choose an order of check points such that there are no intersections
            self.untangle_knot()

            # At this point, it is guaranteed that the line segments do not mutually intersect, but overlap is possible due to the road width.
            any_change = True
            while any_change:
                self.resolve_overlap(minimal_separation=2*road_width)
                any_change = self.untangle_knot()
                if time.time()-start_primitive > max_execution_time:
                    raise Exception("The initialisation of the primitive circuit was terminated because it took longer than", max_execution_time, "seconds.")
            # self.plot()

            while np.any(self.dist2center_min > self.dist2center_max):
                self.resolve_turning_issues()
                any_change = True
                while any_change:
                    self.resolve_overlap(minimal_separation=1.5*road_width)
                    any_change = self.untangle_knot()
                    if time.time()-start_primitive > max_execution_time:
                        raise Exception("The initialisation of the primitive circuit was terminated because it took longer than", max_execution_time, "seconds.")
            # self.plot()
            
            self.translate_to_center()
        
        def select_checkpoints(self,size,minimal_separation):
            # Select uniformly distributed points in the rectangular area
            self.destination = np.vstack([ np.column_stack([np.random.uniform(road_width/2, size[0]-road_width/2, self.Nsegments), 
                                                            np.random.uniform(road_width/2, size[1]-road_width/2, self.Nsegments)]) ])
            # Replace points that are too close to each other
            for ii in range(self.Nsegments):
                attempts = 0
                while True:
                    separation = np.linalg.norm(self.destination[ii]-self.destination[:ii],axis=1)
                    if attempts >= 100:
                        raise Exception("The function select_checkpoints was terminated because it took longer than 100 attempts to find a suitable location for a checkpoint.")
                    if np.any(separation < minimal_separation):
                        self.destination[ii] = [np.random.uniform(0,size[0]), np.random.uniform(0,size[1])]
                        attempts += 1
                    else:
                        break

        def plot(self):
            plt.figure()
            plt.plot(self.destination[:,0],self.destination[:,1])
            plt.show()

        def untangle_knot(self):
            start_untangle = time.time()
            any_change = False
            ii = 0
            jj = self.Nsegments-2
            while ii < self.Nsegments-2:
                if time.time()-start_untangle > 1:
                    raise Exception("The function untangling_knot was terminated because it took longer than one second.")
                # Check for intersection:
                # Find lambda1 and lambda2 such that checkpoint[ii]-lambda0*dir[ii] = checkpoint[jj]-lambda1*dir[jj]
                # Intersection found if both lambda0 and lambda1 are within the interval [0,1]
                lambda_ = np.linalg.solve(np.column_stack((self.TN[ii,:,0], -self.TN[jj,:,0])), self.destination[ii]-self.destination[jj])
                if np.all(lambda_>0) and np.all(lambda_<self.length[[ii,jj]]):
                    # An intersection was found. Flip the loop to resolve the intersection.
                    any_change = True

                    self.destination[ii:jj] = np.flip(self.destination[ii:jj], axis=0)
                    self.length[ii+1:jj] = np.flip(self.length[ii+1:jj])
                    self.TN[ii+1:jj] = -np.flip(self.TN[ii+1:jj], axis=0)
                    self.dist2center_min[ii+1:jj-1] = np.flip(self.dist2center_min[ii+1:jj-1])
                    self.dist2center_max[ii+1:jj-1] = np.flip(self.dist2center_max[ii+1:jj-1])

                    self.update_checkpoint(ii)
                    self.update_checkpoint(jj-1)

                    # Restart the search for intersections
                    ii = 0
                    jj = self.Nsegments-2
                elif jj-ii <= 2:
                    # No intersection found between segment ii and any other segment (jj>ii). Next, we look for intersections between segment ii+1 and any other segment (jj>ii).
                    ii += 1
                    jj = self.Nsegments-1
                else:
                    # No intersection found between segments ii and jj. Next, we look for an intersection between segments ii and jj-1.
                    jj -= 1
            return any_change
            

        def resolve_overlap(self,minimal_separation):
            # Input: The destination must be ordered in a way that there are no intersections.
            # Taking into account the road width, some line segments may overlap. Any such occurrences are resolved by moving at least one check_point away from a line segment.
            any_change = True
            count = 0
            while any_change:
                any_change = False
                ii = 0
                jj = 1
                while ii < self.Nsegments:
                    #Check whether any check_point jj is too close to line segment ii.
                    coord = to_frame(vector=self.destination[jj]-self.destination[ii], frame_TN=self.TN[ii])
                    if np.abs(coord[1])<minimal_separation and coord[0]<=0 and coord[0]>=-self.length[ii]:
                        #Check_point jj is too close to line segment ii. We perform the following changes:
                        # 1. We move check_point jj away from line segment ii along the normal of line segment ii.
                        # 2. We move line segment ii away from check_point jj along the normal of line segment ii.
                        last_jj = jj
                        any_change = True
                        r = np.random.uniform(0, 1)
                        dir = np.sign(coord[1]) * self.TN[ii,:,1]
                        self.move_along_dir(idx=jj  , dir= dir, dist=1.1*(1-r)*(minimal_separation-np.abs(coord[1])), minimal_separation=minimal_separation)
                        self.move_along_dir(idx=ii  , dir=-dir, dist=1.1*   r *(minimal_separation-np.abs(coord[1])), minimal_separation=minimal_separation)
                        self.move_along_dir(idx=ii-1, dir=-dir, dist=1.1*   r *(minimal_separation-np.abs(coord[1])), minimal_separation=minimal_separation)
                        
                    jj = (jj+1)%self.Nsegments
                    if jj == (ii-1)%self.Nsegments:
                        ii += 1
                        jj = (ii+1)%self.Nsegments
                if any_change:
                    count += 1
                    if count >= 100:
                        self.remove_checkpoint(idx=last_jj)
        
        def resolve_turning_issues(self):
            prev_idx = -1
            count = 0
            while np.any(self.dist2center_min > self.dist2center_max):
                idx = np.argmax(self.dist2center_min - self.dist2center_max)
                # print('Resolving checkpoint', idx)

                # 1. Try -difftangent
                difftangent = self.TN[idx,:,0]-self.TN[(idx+1)%self.Nsegments,:,0]
                difftangent = difftangent/np.linalg.norm(difftangent)
                self.move_along_dir(idx=idx, dir=-difftangent, minimal_separation=2*road_width)
                if self.dist2center_min[idx] <= self.dist2center_max[idx]:
                    continue
                
                # 2. Try difftangent
                self.move_along_dir(idx=idx, dir=difftangent, minimal_separation=2*road_width)
                if self.dist2center_min[idx] <= self.dist2center_max[idx]:
                    continue

                # 3. Try -difftangent again
                difftangent = self.TN[idx,:,0]-self.TN[(idx+1)%self.Nsegments,:,0]
                difftangent = difftangent/np.linalg.norm(difftangent)
                self.move_along_dir(idx=idx, dir=-difftangent, minimal_separation=2*road_width)
                if self.dist2center_min[idx] <= self.dist2center_max[idx]:
                    continue

                # 4. Try neighbouring points
                if self.length[idx] <= self.length[(idx+1)%self.Nsegments]:
                    self.move_along_dir(idx=idx-1, dir=-self.TN[idx,:,0], minimal_separation=2*road_width)
                if self.length[idx] >= self.length[(idx+1)%self.Nsegments]:
                    self.move_along_dir(idx=(idx+1)%self.Nsegments, dir=self.TN[(idx+1)%self.Nsegments,:,0], minimal_separation=2*road_width)
                if self.length[idx] < self.length[(idx+1)%self.Nsegments]:
                    self.move_along_dir(idx=idx-1, dir=-self.TN[idx,:,0], minimal_separation=2*road_width)
                
                if self.dist2center_min[idx] <= self.dist2center_max[idx]:
                    continue

                # 5. Try moving the turning point away from the neirest neighbour
                if self.length[idx] < self.length[(idx+1)%self.Nsegments]:
                    self.move_along_dir(idx=idx, dir=self.TN[idx,:,0], minimal_separation=1.5*road_width)
                elif self.length[idx] > self.length[(idx+1)%self.Nsegments]:
                    self.move_along_dir(idx=idx, dir=-self.TN[(idx+1)%self.Nsegments,:,0], minimal_separation=1.5*road_width)

                if self.dist2center_min[idx] <= self.dist2center_max[idx]:
                    continue

                # 6. Try moving the turning point towards its farest neighbour
                if self.length[idx] < self.length[(idx+1)%self.Nsegments]:
                    self.move_along_dir(idx=idx, dir=self.TN[(idx+1)%self.Nsegments,:,0], dist=self.length[(idx+1)%self.Nsegments]/2)
                elif self.length[idx] > self.length[(idx+1)%self.Nsegments]:
                    self.move_along_dir(idx=idx, dir=-self.TN[idx,:,0], dist=self.length[idx]/2)

                if self.dist2center_min[idx] <= self.dist2center_max[idx]:
                    continue

                # print('Issue not resolved')
                if idx == prev_idx:
                    count += 1
                    if count >= 3:
                        self.remove_checkpoint(idx)
                        self.untangle_knot()
                else:
                    prev_idx = idx
                    count = 0
                
        def update_checkpoint(self,idx,new_checkpoint=np.array([])):
            if new_checkpoint.size > 0:
                # print('Check point', idx,'was moved from', self.destination[idx], 'to', new_checkpoint)
                # Update destination
                self.destination[idx] = new_checkpoint

            neighbouring_points = [idx-1, idx, (idx+1)%self.Nsegments]
            neighbouring_segments = [idx, (idx+1)%self.Nsegments]
            neighbouring_segments_double = [idx-1, idx, (idx+1)%self.Nsegments, (idx+2)%self.Nsegments]
            
            # Update TN and length
            dir = np.diff(self.destination[neighbouring_points], axis=0)
            length = np.linalg.norm(dir, axis=1)
            tangent = dir / length[:,np.newaxis]
            self.length[neighbouring_segments] = length
            self.TN[neighbouring_segments,:,0] = tangent
            self.TN[neighbouring_segments,:,1] = rotate90left(tangent)

            # Update dist2center_min and dist2center_max
            cos_2theta = np.abs(np.sum(self.TN[neighbouring_segments_double[:-1],:,0] * self.TN[neighbouring_segments_double[1:],:,0], axis=1))
            cos_theta = np.sqrt((1+cos_2theta)/2)
            sin_theta = np.sqrt((1-cos_2theta)/2)
            self.dist2center_min[neighbouring_points] = road_width/2 / sin_theta
            self.dist2center_max[neighbouring_points] = np.min(np.column_stack([self.length[neighbouring_segments_double[:-1]], self.length[neighbouring_segments_double[1:]]]), axis=1) /2 / cos_theta

        def remove_checkpoint(self,idx):
            self.destination = np.delete(self.destination, idx, axis=0)
            self.TN = np.delete(self.TN, idx, axis=0)
            self.length = np.delete(self.length, idx)
            self.dist2center_min = np.delete(self.dist2center_min, idx)
            self.dist2center_max = np.delete(self.dist2center_max, idx)
            self.Nsegments -= 1
            self.update_checkpoint(idx%self.Nsegments)
            # print('Checkpoint',idx,'was deleted from the circuit.')

        def move_along_dir(self,idx,dir,minimal_separation=0,dist=np.inf):
            original = self.destination[idx]

            # What is the maximal distance over which we can move checkpoint idx in the direction dir?
            max_mov_dist = np.maximum(self.first_intersection(point=self.destination[idx], dir=dir, drop=[idx,(idx+1)%self.Nsegments]) - minimal_separation, 0)
            dist = np.minimum(dist, max_mov_dist)
            
            self.update_checkpoint(idx, self.destination[idx] + dist * dir)

            count = 0
            while not (self.first_intersection(point=self.destination[idx], dir=-self.TN[idx,:,0]                  , drop=[idx-1,idx,(idx+1)%self.Nsegments]) > self.length[idx]     and 
                       self.first_intersection(point=self.destination[idx], dir=self.TN[(idx+1)%self.Nsegments,:,0], drop=[idx,(idx+1)%self.Nsegments,(idx+2)%self.Nsegments]) > self.length[(idx+1)%self.Nsegments]):
                self.update_checkpoint(idx, self.destination[idx] - dist/5 * dir)
                count += 1
                if count >= 5:
                    self.destination[idx] = original
                    break
         
        def first_intersection(self,point,dir,checkpoints=[],drop=[]):
            #Input: dir is a unit vector.
            #Output: the distance between point and the first intersection along point+lambda*dir for lambda>0.
            if len(checkpoints) == 0:
                checkpoints = self.destination

            dir_TN = np.column_stack([dir, rotate90left(dir)])

            #Express the checkpoints in the coordinate system defined by dir_TN
            coords = to_frame(vector=checkpoints-point, frame_TN=dir_TN)

            # An intersection with the circuit is present when a sign switch occurs in the normal coordinate
            intersect = np.roll(coords[:,1],1) * coords[:,1] <= 0
            intersect[drop] = False

            idx = np.where(intersect)[0]
            dist_to_intersection = (coords[idx-1,0] * coords[idx,1] - coords[idx,0] * coords[idx-1,1]) / (coords[idx,1] - coords[idx-1,1])
            
            dist_to_intersection = dist_to_intersection[dist_to_intersection>0]

            if dist_to_intersection.size == 0:
                # No intersection with the circuit found. Looking for intersection with rectangular border
                return self.first_intersection(point=point, dir=dir, checkpoints=np.vstack([[0,0], [self.size[0],0], self.size, [0,self.size[1]]]))
            else:
                return np.min(dist_to_intersection)
        
        def translate_to_center(self):
            self.destination -= (np.min(self.destination, axis=0) + np.max(self.destination, axis=0) - self.size)[np.newaxis,:]

class CAR:
    def __init__(self,quality,color='red'):
        def get_value(coordinate,min,max):
            return min + coordinate*(max-min)
        self.img = mpimg.imread(f'../Car_{color}.png')
        self.engine_efficiency     = get_value(quality[0], .2, .55) # scalar
        self.max_acceleration      = get_value(quality[1], 2.25, 14.2) # m/s²
        self.max_deceleration      = get_value(quality[2], 5., 45.) # m/s²
        self.tank                  = get_value(quality[3], 1e5 / tank_unit, 1e7 / tank_unit) # max 110 liter & 12.889 Wh/liter -> max 3.96e8 Joule
        self.grip                  = get_value(quality[4], .9*g, 1.5*g) # m/s² , friction_coeficient * g
        self.tyre_max_dist         = get_value(quality[5], 1e5, 4e5) # km
        self.roll_resistance_coef  = get_value(quality[6], .03*g, .007*g)
        drag_coefficient           = get_value(quality[7], 1.1, .7) #usually ranges between 0.7 to 1.1
        self.air_resistance_coef = .5 * 1.225 * 1.75 * drag_coefficient
            # air_density = 1.225 kg/m³
            # frontal_area = 1.75 m²
            # frontal_area = 1.75 m² around 1.5 to 2.0 square meters

class AGENT:
    def __init__(self,include_car_vars,include_state_vars,eps=0,generation=0):
        Ninputs = (   len([attr for attr in include_car_vars.keys() if include_car_vars[attr]==True]) # number of 1-dimensional car variables
                    + np.sum([np.sum(include_car_vars[attr]) for attr in include_car_vars.keys() if isinstance(include_car_vars[attr], list)]) # number of dimensions from higher dimensional car variables
                    + len([attr for attr in include_state_vars.keys() if include_state_vars[attr]==True]) # number of 1-dimensional state variables
                    + np.sum([np.sum(include_state_vars[attr]) for attr in include_state_vars.keys() if isinstance(include_state_vars[attr], list)]) ) # number of dimensions from higher dimensional state variables
        Nlayers = 4
        Nneurons = np.round(np.linspace(Ninputs+1, 2+1, Nlayers, dtype=int)) # bias term included
        self.neural_network = []
        for ii in range(len(Nneurons)-1):
            self.neural_network.append(np.zeros((Nneurons[ii+1]-1,Nneurons[ii])))
        self.generation = generation
        self.mutate(eps=eps, generation=generation)
    
    def mutate(self,eps,generation):
        self.neural_network = [layer + np.random.uniform(-eps,eps,np.shape(layer))
                               for layer in self.neural_network]
        self.generation = generation
    
    # def add_layer(self):
    #     Ninputs = len(self.neural_network[0]) - 1
    #     Nlayers = len(self.neural_network) + 1
    #     Nneurons = np.round(np.linspace(Ninputs+1, 2+1, Nlayers, dtype=int)) # bias term included
    
    def upgrade(self, include_car_vars_prev, include_state_vars_prev, include_car_vars, include_state_vars):
        # Upgrade the neural network to include new variables.
        previous_neural_network = self.neural_network

        include_booleans_prev = np.array([True], dtype=bool) # bias term included
        for incl in list(include_car_vars_prev.values()) + list(include_state_vars_prev.values()):
            include_booleans_prev = np.append(include_booleans_prev, incl)

        include_booleans = np.array([True], dtype=bool) # bias term included
        for incl in list(include_car_vars.values()) + list(include_state_vars.values()):
            include_booleans = np.append(include_booleans, incl)
        
        Ninputs = np.sum(include_booleans) # bias term included
        Nlayers = len(previous_neural_network) + 1
        Nneurons = np.round(np.linspace(Ninputs, 2+1, Nlayers, dtype=int)) # bias term included
        self.neural_network = []
        for ii in range(len(Nneurons)-1):
            new_layer = np.zeros((Nneurons[ii+1]-1,Nneurons[ii]))
            if ii == 0:
                new_layer[:np.shape(previous_neural_network[ii])[0],include_booleans_prev[include_booleans]] = previous_neural_network[ii]
            else:
                new_layer[:np.shape(previous_neural_network[ii])[0],:np.shape(previous_neural_network[ii])[1]] = previous_neural_network[ii]
            self.neural_network.append(new_layer)

class RACE:
    def __init__(self,circuit,car,laps,agents,include_car_vars,include_state_vars,MaxTime,interaction=True):
        self.circuit = circuit
        self.car = car
        self.laps = laps
        self.MaxTime = MaxTime
        self.Nagents = len(agents)
        self.include_car_vars = include_car_vars
        self.include_state_vars = include_state_vars
        self.interaction = interaction
        self.state = RACE.STATE(Nagents=self.Nagents, circuit=self.circuit, Nlaps=self.laps, car=self.car)
        self.neural_networks = [np.rollaxis(np.dstack([agent.neural_network[layer] for agent in agents]), -1)
                                for layer in range(len(agents[0].neural_network))] # shape (Nlayers,Nagents,Ncols,Nrows)
        self.remaining_drivers = np.arange(self.Nagents, dtype=int)
        self.Nfinishers = 0
        self.penalty = np.zeros(self.Nagents)
        
    def simulate(self,saveas=''):
        def draw_cars_and_save_frame():
            # Draw the cars in the correct position and orientation
            car_images = []
            zorder = max(artist.get_zorder() for artist in ax.get_children())
            for aa in range(min(self.Nagents,15)):
                zorder += 1
                p = self.state.position[[aa],:].T # This is a 2*1 column array
                TN = self.state.car_TN[aa]        # This is a 2*2 array
                transformation = transforms.Affine2D(matrix=np.block([[TN,p], [0,0,1]]))
                car_images.append(ax.imshow(self.car.img,
                                            extent = [-car_size[0]/2, car_size[0]/2, -car_size[1]/2, car_size[1]/2],
                                            transform = transformation + ax.transData,
                                            zorder = zorder))
        
            # Draw the plot onto a canvas
            canvas = FigureCanvas(self.circuit.fig)
            canvas.draw()

            # Remove car_images for next iteration
            for image in car_images:
                image.remove()

            # Get the RGBA buffer from the canvas and convert to BGR
            frame = np.array(canvas.renderer._renderer)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Write the frame to the video
            out.write(frame)
        
        save = saveas!=''
        if save:
            self.circuit.plot()
            ax = self.circuit.fig.get_axes()[0]

            # Initialise video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(saveas, fourcc, fps=video_speedup/dt, frameSize=frameSize)
            
            draw_cars_and_save_frame()

        MaxSteps = int(self.MaxTime / dt)
        tt = 0
        stop = False
        while not(stop):
            # Determine the inputs to the neural networks, i.e. what the agent sees. (one vector for every agent)
            state = np.column_stack([np.tile([getattr(self.car, attr)[self.include_car_vars[attr]].squeeze() for attr in self.include_car_vars.keys() if not(self.include_car_vars[attr]==False)], (self.Nagents,1))] +
                                    [getattr(self.state, attr)[:,self.include_state_vars[attr]] for attr in self.include_state_vars.keys() if not(self.include_state_vars[attr]==False)])

            # The agents determine which action to take
            actions = self.get_actions(state) # shape (Nagents,Nactions)

            # Simulate to get the new state
            self.state.simulate(actions,dt,self.circuit,self.car,self.laps,self.interaction)

            if save:
                draw_cars_and_save_frame()
            
            tt += 1
            finished = self.state.distance_to_finish <= 0
            if np.any(finished):
                self.Nfinishers += np.sum(finished)
                self.penalty[self.remaining_drivers[np.where(finished)[0]]] = tt*dt
                self.remove_agents(finished)
            
            if tt < MaxSteps and self.Nagents > self.Nfinishers:
                abandoned = np.logical_or(np.abs(self.state.position_local[:,1]) > 2*road_width, 
                                          np.logical_and(self.state.health_tank <= 0, np.linalg.norm(self.state.velocity,axis=1)<1e-2))
            else:
                abandoned = np.ones(self.Nagents, dtype=bool)
                stop = True
            if np.any(abandoned):
                self.penalty[self.remaining_drivers[np.where(abandoned)[0]]] = MaxSteps*dt + self.state.distance_to_finish[abandoned]
                self.remove_agents(abandoned)

        if save:
            out.release()
    
    def get_actions(self,input):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x)) # RuntimeWarning may arise due to overflow. This can be neglected because numpy replaces overflow by inf, which leads to the correct result.
        
        # Input has shape (Nagents, Ninputs)
        # The layers have shape (Nagents, Noutputs, Ninputs)
        bias = np.repeat(1,self.Nagents)
        output = input
        for layer in self.neural_networks:
            input = np.column_stack((bias, output))
            output = sigmoid(np.matmul(layer, input[:,:,np.newaxis]).squeeze(axis=2))
        output =  2*output - 1 # All in the interval [-1,1]
        return output
    
    def remove_agents(self,mask):
        self.Nagents -= np.sum(mask)
        self.remaining_drivers = self.remaining_drivers[~mask]
        self.neural_networks = [layer[~mask] for layer in self.neural_networks] # shape (Nlayers,Nagents,Ncols,Nrows)
        self.state.remove_agents(mask)
    
    class STATE:
        def __init__(self,Nagents,circuit,Nlaps,car):
            if np.shape(circuit.start)[0] == 1:
                self.position = np.repeat(circuit.start, Nagents, axis=0)     # shape (Nagents,2)
            else:
                self.position = circuit.start                                 # shape (Nagents,2)
            self.velocity = np.zeros((Nagents, 2))                            # shape (Nagents,2)
            self.orientation = np.repeat(circuit.orientation[0], Nagents)     # shape Nagents
            self.car_TN = get_TN_from_angle(self.orientation)                 # shape (Nagents,2,2)
            self.omega = np.zeros(Nagents)                                    # shape Nagents
            self.health_tank = np.repeat(car.tank, Nagents)                   # shape Nagents
            self.tyres_deterioration = np.zeros(Nagents)                      # shape Nagents

            # Counters
            self.laps = np.zeros(Nagents, dtype=int)                          # shape Nagents
            self.segment = np.repeat(0, Nagents)                              # shape Nagents
            self.reached_turning = np.repeat(False,Nagents)                   # shape Nagents

            # Local state
            self.position_local = np.zeros((Nagents,2))                       # shape (Nagents,2)
            self.velocity_local = np.zeros((Nagents,2))                       # shape (Nagents,2)
            self.orientation_local = np.zeros(Nagents)                        # shape Nagents
            self.TN_local = np.zeros((Nagents,2,2))                           # shape (Nagents,2,2)
            self.update_local(circuit)

            # Agent states
            self.distance_to_finish = np.maximum(0,-self.position_local[:,0]) + circuit.distance_to_finish[self.segment] + (Nlaps-self.laps-1) * circuit.lap_distance # shape Nagents
            self.distance_to_finish[~self.reached_turning] += circuit.length_turning[self.segment[~self.reached_turning]]
            self.length_future = np.zeros((Nagents, 3))                       # shape (Nagents,3)
            self.curvature_future = np.zeros((Nagents, 4))                    # shape (Nagents,4)

            next_segment = (self.segment+1) %circuit.Nsegments
            self.length_future[~self.reached_turning, 0] = circuit.length_turning[self.segment[~self.reached_turning]]
            self.length_future[~self.reached_turning, 1] = circuit.length[next_segment[~self.reached_turning]]
            self.length_future[~self.reached_turning, 2] = circuit.length_turning[next_segment[~self.reached_turning]]
            self.length_future[self.reached_turning, 0] = circuit.length[next_segment[self.reached_turning]]
            self.length_future[self.reached_turning, 1] = circuit.length_turning[next_segment[self.reached_turning]]
            self.length_future[self.reached_turning, 0] = circuit.length[(next_segment[self.reached_turning]+1) %circuit.Nsegments]
            self.curvature_future[self.reached_turning, 0] = circuit.curvature[self.segment[self.reached_turning]]
            self.curvature_future[~self.reached_turning, 1] = circuit.curvature[self.segment[~self.reached_turning]]
            self.curvature_future[self.reached_turning, 2] = circuit.curvature[next_segment[self.reached_turning]]
            self.curvature_future[~self.reached_turning, 3] = circuit.curvature[next_segment[~self.reached_turning]]
        
        def update_local(self,circuit):
            # Express position and velocity in local coordinates
            self.TN_local[np.logical_not(self.reached_turning)] = circuit.TN[self.segment[np.logical_not(self.reached_turning)]]
            disp_from_center = self.position[self.reached_turning] - circuit.center[self.segment[self.reached_turning]]
            dist_from_center = np.linalg.norm(disp_from_center, axis=1)
            self.TN_local[self.reached_turning,:,1] = disp_from_center / (-np.sign(circuit.curvature[self.segment[self.reached_turning]]) * dist_from_center)[:,np.newaxis]
            self.TN_local[self.reached_turning,0,0] = self.TN_local[self.reached_turning,1,1] #Tx = Ny
            self.TN_local[self.reached_turning,1,0] = -self.TN_local[self.reached_turning,0,1] #Ty = -Nx

            self.position_local[np.logical_not(self.reached_turning)] = np.matmul((self.position[np.logical_not(self.reached_turning)]-circuit.destination[self.segment[np.logical_not(self.reached_turning)],:])[:, np.newaxis, :], self.TN_local[np.logical_not(self.reached_turning)]).squeeze(axis=1)
            self.position_local[self.reached_turning,1] = -np.sign(circuit.curvature[self.segment[self.reached_turning]]) * (dist_from_center - circuit.radius[self.segment[self.reached_turning]])
            self.position_local[self.reached_turning,0] = circuit.radius[self.segment[self.reached_turning]] * angle_minus(vector2angle(disp_from_center), circuit.angle_final[self.segment[self.reached_turning]]) * np.sign(circuit.angle_change[self.segment[self.reached_turning]])

            self.velocity_local = np.matmul(self.velocity[:, np.newaxis, :], self.TN_local).squeeze(axis=1)

            self.orientation_local = angle_minus(self.orientation, vector2angle(self.TN_local[:,:,0]))

        def update_counters(self,circuit):
            reached_next_checkpoint = np.logical_and(self.position_local[:,0] >= 0, np.abs(self.position_local[:,1]) <= road_width/2)
            if np.any(reached_next_checkpoint):
                any_change = True
                self.segment[reached_next_checkpoint & self.reached_turning] += 1
                self.reached_turning[reached_next_checkpoint] = np.logical_not(self.reached_turning[reached_next_checkpoint])

                finished_lap = self.segment == circuit.Nsegments
                self.segment[finished_lap] = 0
                self.laps[finished_lap] += 1

                # Update lengths and curvatures of the future path
                self.length_future[reached_next_checkpoint & ~self.reached_turning] = np.column_stack([self.length_future[reached_next_checkpoint & ~self.reached_turning, 1:], 
                                                                                                       circuit.length_turning[(self.segment[reached_next_checkpoint & ~self.reached_turning]+1) %circuit.Nsegments]])
                self.length_future[reached_next_checkpoint & self.reached_turning] = np.column_stack([self.length_future[reached_next_checkpoint & self.reached_turning, 1:], 
                                                                                                      circuit.length[(self.segment[reached_next_checkpoint & self.reached_turning]+2) %circuit.Nsegments]])
                self.curvature_future[reached_next_checkpoint & ~self.reached_turning] = np.column_stack([self.curvature_future[reached_next_checkpoint & ~self.reached_turning, 1:], 
                                                                                                          circuit.curvature[(self.segment[reached_next_checkpoint & ~self.reached_turning]+1) %circuit.Nsegments]])
                self.curvature_future[reached_next_checkpoint & self.reached_turning] = np.column_stack([self.curvature_future[reached_next_checkpoint & self.reached_turning, 1:], 
                                                                                                         np.zeros(np.sum(reached_next_checkpoint & self.reached_turning))])
            else:
                any_change = False
            return any_change

        def simulate(self,actions,dt,circuit,car,Nlaps,interaction):
            ### Derive the forces exerted by the road ###
            # Determine whether wheels are on road or grass
            car_Normal_local = np.matmul(np.transpose(self.TN_local,(0,2,1)), self.car_TN[:,:,1,np.newaxis]).squeeze(axis=2) # shape (Nagents,2)
                # car_TN_local = TN_local.transpose() @ car_TN
                # car_Normal_local = TN_local.transpose() @ car_Normal
            wheels_Npos_local = self.position_local[:,np.newaxis,1] + np.tensordot(car_Normal_local[:,np.newaxis,:], wheel_rvectors[np.newaxis,:,:], axes=(2,2)).squeeze(axis=(1,2)) #shape (Nagents,Nwheels)
                # wheels_Npos_local = position_local[1] + np.dot(car_Normal_local, wheel_rvector)
            
            mask_offroad = np.abs(wheels_Npos_local) > road_width/2 #shape (Nagents,Nwheels)
            grip = np.repeat(car.grip / (1+self.tyres_deterioration[:,np.newaxis]**2), 4, axis=1) #shape (Nagents,Nwheels)
            grip[mask_offroad] *= offroad_grip_multiplyer
            roll_resistance_coef = np.where(mask_offroad, car.roll_resistance_coef * offroad_roll_resistance_multiplyer, car.roll_resistance_coef)

            # Determine the desired tangential acceleration and the fuel usage
            action_acceleration = actions[:,0]
            action_acceleration[self.health_tank<=0] = 0 # No more acceleration when tank is empty
            positive_acceleration = action_acceleration > 0
            acceleration_linear = np.where(positive_acceleration, action_acceleration * car.max_acceleration, 
                                                                  action_acceleration * car.max_deceleration)
            efficiency = np.maximum(1e-3, np.where(positive_acceleration, car.engine_efficiency * (1-np.exp(action_acceleration-1)) / (1-np.exp(-1)), np.inf))
                # efficiency = base_efficiency * (1 - exp(acceleration / max_acceleration))
            self.health_tank -= acceleration_linear * np.linalg.norm(self.velocity,axis=1) / efficiency * dt / tank_unit
                # Energy consumption = int(F*v)*dt
            
            # We work in the car frame, meaning that (1,0) corresponds to the car tangent and (0,1) corresponds to the car normal.
            sintheta = actions[:,1]  # shape Nagents #theta is the angle between the car tangent and the front wheel tangent
            costheta = np.sqrt(1-sintheta**2)
            front_wheel_TN = np.zeros((len(sintheta),2,2,2)) # shape (Nagents,Nfrontwheels,2,2)
            front_wheel_TN[:,:,0,0] = costheta[:,np.newaxis]
            front_wheel_TN[:,:,0,1] = -sintheta[:,np.newaxis]
            front_wheel_TN[:,:,1,0] = sintheta[:,np.newaxis]
            front_wheel_TN[:,:,1,1] = costheta[:,np.newaxis]

            velocity_car_frame = to_frame(self.velocity, frame_TN=self.car_TN) # shape (Nagents,2)
            wheel_velocity = velocity_car_frame[:,np.newaxis,:] + self.omega[:,np.newaxis,np.newaxis] * rotate90left(wheel_rvectors)[np.newaxis,:,:] # shape (Nagents,Nwheels,2)
                # Expressed in the car frame

            # We work in the wheel frame, meaning that (1,0) corresponds to the wheel tangent and (0,1) corresponds to the wheel normal.
            wheel_velocity[:,[0,1],:] = to_frame(wheel_velocity[:,[0,1],:], frame_TN=front_wheel_TN)
                # Conversion from car frame to wheel frame
            acceleration_surface = np.zeros((np.shape(wheel_velocity))) # shape (Nagents,Nwheels,2)
            acceleration_surface[:,:,0] = acceleration_linear[:,np.newaxis] - np.sign(wheel_velocity[:,:,0]) * roll_resistance_coef
                # maximal acceleration along wheel tangent = acceleration_linear - sign(wheel speed along wheel tangent) * roll_resistance_coef
            acceleration_surface[:,:,1] = -wheel_velocity[:,:,1] /dt
                # maximal acceleration along wheel normal = centripital friction required to prevent centrifugal drift
                # speed - dt*acceleration = 0, so maximal acceleration along wheel normal = -(wheel speed along wheel normal) /dt
            acc_norm = np.linalg.norm(acceleration_surface,axis=-1) # shape (Nagents,Nwheels)
            drift = acc_norm > grip
            acceleration_surface[drift, :] *= (grip[drift, np.newaxis]*drift_grip_multiplyer / acc_norm[drift, np.newaxis])
            self.tyres_deterioration -= dt / car.tyre_max_dist * np.mean(wheel_velocity[:,:,0] * (1+drift*drift_deterioration_multiplyer), axis=1)

            # Prevent driving backwards
            acceleration_surface[:,:,0] = np.maximum(acceleration_surface[:,:,0], -wheel_velocity[:,:,0]/dt)
            
            # We work in the car frame, meaning that (1,0) corresponds to the car tangent and (0,1) corresponds to the car normal.
            acceleration_surface[:,[0,1],:] = np.matmul(front_wheel_TN, acceleration_surface[:,[0,1],:,np.newaxis]).squeeze(axis=3) # shape (Nagents,Nwheels,2)
                # Conversion from wheel frame to car frame

            # Equivalent forces
            acceleration_surface_eq = np.mean(acceleration_surface, axis=1) # shape (Nagents,2)
                # translational force
            torque_per_mass = wheel_rvectors[np.newaxis,:,0] * acceleration_surface[:,:,1] - wheel_rvectors[np.newaxis,:,1] * acceleration_surface[:,:,0]  # shape (Nagents,Nwheels)
                # torque = r x F, so torque_z = r_x * F_y - r_y * F_x
            torque_per_mass = np.mean(torque_per_mass, axis=1) # shape (Nagents,2)
                # rotational force
                # We use mean and not sum. That is because we are working with acceleration and not force. If we would be working with force, we should've used mass/4 for each wheel.

            ### Add air resistance ###
            acceleration_surface = np.matmul(self.car_TN, acceleration_surface_eq[:,:,np.newaxis]).squeeze(axis=2) # shape (Nagents,2)
                # Conversion to global frame
            acceleration = acceleration_surface - car.air_resistance_coef * np.linalg.norm(self.velocity,axis=1)[:,np.newaxis] * self.velocity # shape (Nagents,2)

            if interaction:
                copy_position = np.array(self.position)
                copy_velocity = np.array(self.velocity)
                copy_orientation = np.array(self.orientation)
                copy_omega = np.array(self.omega)
                copy_car_TN = np.array(self.car_TN)

            ### Apply integration scheme ###
            self.position += dt/2 * self.velocity
            self.velocity += dt * acceleration
            self.position += dt/2 * self.velocity

            self.orientation += dt/2 * self.omega
            self.omega += dt / moment_of_inertia_per_mass * torque_per_mass
            self.car_TN = get_TN_from_angle(self.orientation)
            self.orientation += dt/2 * self.omega

            ### Check for colisions ###
            if interaction:
                in_range = np.linalg.norm(self.position[:,np.newaxis,:] - self.position[np.newaxis,:,:], axis=2) < np.linalg.norm(car_size)
                IDX1, IDX2 = np.where(np.triu(in_range, k=1))
                for idx1, idx2 in zip(IDX1, IDX2):
                    #get the positions of the wheels of both cars
                    wheels_pos = self.position[[idx1,idx2],np.newaxis,:] + np.matmul(self.car_TN[[idx1,idx2],np.newaxis,:,:], wheel_rvectors[np.newaxis,:,:,np.newaxis]).squeeze() # shape (2,Nwheels,2)
                    #check whether colision occurs
                    wheels_car1_in_frame_car2 = to_frame(wheels_pos[0,:,:].squeeze()-self.position[idx2],self.car_TN[idx2]) # The coordinates of the wheels of car 1 in the frame of car 2
                    wheels_car2_in_frame_car1 = to_frame(wheels_pos[1,:,:].squeeze()-self.position[idx1],self.car_TN[idx1]) # The coordinates of the wheels of car 2 in the frame of car 1
                    intersect1 = np.all(np.abs(wheels_car1_in_frame_car2) - car_size/2 <= 0, axis=1)
                    intersect2 = np.all(np.abs(wheels_car2_in_frame_car1) - car_size/2 <= 0, axis=1)
                    if ~np.any(intersect1) and ~np.any(intersect2):
                        #no colision occurs
                        continue
                    #find the point of colision
                    r_hit = np.mean(np.concatenate((wheels_pos[0,intersect1,:], wheels_pos[1,intersect2,:]), axis=0), axis=0).squeeze()
                    velocity_r_hit1 = self.velocity[idx1,:] + self.omega[idx1] * np.asarray([[0,-1],[1,0]]) @ (r_hit-self.position[idx1])
                    velocity_r_hit2 = self.velocity[idx2,:] + self.omega[idx2] * np.asarray([[0,-1],[1,0]]) @ (r_hit-self.position[idx2])
                    rvector1 = to_frame(r_hit-self.position[idx1], self.car_TN[idx1])
                    rvector2 = to_frame(r_hit-self.position[idx2], self.car_TN[idx2])

                    # Add force to both cars
                    force = -(velocity_r_hit1-velocity_r_hit2) / dt # exerted on car1
                    acceleration[idx1] += force
                    torque_per_mass[idx1] += (rvector1[0] * force[1] - rvector1[1] * force[0])
                    acceleration[idx2] -= force
                    torque_per_mass[idx2] -= (rvector2[0] * force[1] - rvector2[1] * force[0])

                    #undo integration scheme
                    self.position = copy_position
                    self.velocity = copy_velocity
                    self.orientation = copy_orientation
                    self.omega = copy_omega
                    self.car_TN = copy_car_TN

                    #re-apply integration scheme
                    self.position += dt/2 * self.velocity
                    self.velocity += dt * acceleration
                    self.position += dt/2 * self.velocity

                    self.orientation += dt/2 * self.omega
                    self.omega += dt / moment_of_inertia_per_mass * torque_per_mass
                    self.car_TN = get_TN_from_angle(self.orientation)
                    self.orientation += dt/2 * self.omega

            ### Update local positions and counters ###
            any_change = True
            while any_change:
                self.update_local(circuit)
                any_change = self.update_counters(circuit)
            self.distance_to_finish = np.maximum(0,-self.position_local[:,0]) + circuit.distance_to_finish[self.segment] + (Nlaps-self.laps-1) * circuit.lap_distance
            self.distance_to_finish[~self.reached_turning] += circuit.length_turning[self.segment[~self.reached_turning]]
        
        def remove_agents(self,mask):
            for attr in self.__dict__:
                setattr(self, attr, getattr(self, attr)[~mask])

# Size parameters
road_width = 10 #[meters]
car_size = np.array([5,2]) #[meters]
wheel_rvectors = np.array([ [ car_size[0], car_size[1]], #front left wheel
                            [ car_size[0],-car_size[1]], #front right wheel
                            [-car_size[0], car_size[1]], #back left wheel
                            [-car_size[0],-car_size[1]]  #back right wheel
                            ]) /2                        #shape (4,2)

# Physical parameters
dt = 1/4 #[s]
g = 9.81 #[m/s²]
drift_grip_multiplyer = .85
offroad_grip_multiplyer = .55
offroad_roll_resistance_multiplyer = 50
moment_of_inertia_per_mass = np.sum(car_size**2) /12 + np.sum(car_size) /8
drift_deterioration_multiplyer = 20
tank_unit = 1e6 # mega-Joule

# Display parameters
video_speedup = 4
frameSize = (2000,1130) #[pixels]
fig_size = np.array([20, 11.3]) #[inches]