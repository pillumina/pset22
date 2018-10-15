import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class BigBang:

    def __init__(self, W, L, pe):
        '''
        Creates gridded world/environment in which agent will operate.

        Inputs:
            W - INT indicating number of rows in grid (width)
            L - INT indicating number of columns in grid (length)

        NOTE: policy is held in two 1-D arrays (one for orientation, one for location).
        All other info is in matrices.
        '''
        assert 0 <= pe <= 1, 'Probability of rotation error (pe) must be: 0 <= pe <= 1'
        self.width = W  # x (column) dimension
        self.length = L  # y (row) dimension
        self.pe = pe
        self.ID_goal = []  # Value is set by self.InitializePolicy(ID_goal)
        self.PolicyValue = np.zeros(W * L)

        # rewardspace will not include orientation. It only holds a reward for physical location.
        self.rewardspace = np.zeros(W * L)

        # Initialize gradient, which will be a vector field pointing to the goal
        self.gradient = np.zeros(self.width * self.length)

        time = [i for i in range(12)]
        radians = [(2 * np.pi - (t / 12) * 2 * np.pi) % (2 * np.pi) for t in time]
        radians = np.roll(radians, 3)
        self.time2radians = {k: v for k, v in zip(time, radians)}

        # Create set of possible actions. Note: actions 0 and 2 are only
        # permissible if the robot is on the perimeter. All others may occur on
        # the interior and exterior. However, agent is not allowed to move off
        # the grid.
        # Key:
        #   position 0 (translation) - stay(0),up(0),right(3),down(6),left(9)
        #   position 1 (rotation) - Counter-Clockwise(-1),stay(0),Clockwise(1)
        self.actionspace = np.array([[-10, -1],
                                     [-10, 0],
                                     [-10, 1],
                                     [0, -1],
                                     [0, 0],
                                     [0, 1],
                                     [3, -1],
                                     [3, 0],
                                     [3, 1],
                                     [6, -1],
                                     [6, 0],
                                     [6, 1],
                                     [9, -1],
                                     [9, 0],
                                     [9, 1]])

        self.validdirections = list(set(self.actionspace[:, 0]))

        # Initialize an array of matrices, one for each possible translation command
        N_directions = len(self.validdirections)  # Number of travel directions(stay,up,right,down,right)
        self.adjacencytensor = np.zeros([W * L, W * L, N_directions], dtype=int)

        # Create offset arrays(stay,up,left,down,right)
        directions = [np.array([0, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])]
        N_states = self.length * self.width
        for ID in range(N_states):

            coords0 = self.ID2Coords(ID)
            coords0 = np.array(coords0)

            # Determine neighbor state based on translation direction
            idx = 0
            for offset in directions:
                coords_neighbor = coords0 + offset  # Get location of neighbor

                # Try indexing the rewardspace using the neighbor's coordinates.
                # If successful, change a 0 to a 1 in the adjacency matrix.
                ID_neighbor = self.Coords2ID(coords_neighbor)
                if ID_neighbor != np.inf:
                    self.adjacencytensor[ID, ID_neighbor, idx] = 1
                else:
                    pass

                idx += 1

        # Check that adjacency matrix is symmetric since graph is undirected
        assert np.sum(self.Symmetricity(True)) == 0

        print('Adjacency matrix successfully populated!')

        # Populate state IDs on grid
        self.gridspace = np.zeros([W, L])

        # Create grid with each position containing its ID
        count = 0
        for row, col in zip(range(self.width), range(self.length)):
            self.gridspace[row, col] = count
            count += 1

        self.policy = np.zeros([12, W * L, 2])  # Initial policy: face 0 and stand still

    def Coords2ID(self, coords):
        '''
        Converts a list of nested lists, each containing grid row & col coordinates,
        to a list state IDs. The result is two nested lists, one with state IDs
        and the other list containing the reward for the state ID in the
        corresponding position in the first list (state IDs). The output is
        compatible as an input to GenStateSpace

        Inputs:
            coords - LIST of length 2 lists each containing the row & col coordinates
                    of each state that will be assigned a reward
            rewards - LIST (optional) of reward values for each rol-col combination

        Outputs:
            output - LIST of two lists. First one containing each state ID, second
                    one containing the reward for each state in the first list.
                    If no rewards are provided, second list is empty.
        '''
        # Determine the state ID
        assert len(coords) == 2

        if 0 <= coords[0] < self.length and 0 <= coords[1] < self.width:
            row = coords[0]
            col = coords[1]
            ID = self.length * row + col % self.width

        else:
            ID = np.inf

        return ID

    def ID2Coords(self, ID):
        '''
        Converts list of state IDs to lists coordinates for each state ID.

        Inputs:
           ID - INT with grid ID

        Outputs:
           [row,col] - LIST  containing the row & col coordinates of the ID
        '''
        row = ID // self.length
        col = ID % self.length
        assert (row < self.width and col < self.length), "Invalid state ID"

        return [row, col]

    def AssignRewards(self, state_reward):
        '''
        Assigns reward values to states in grid.

        Inputs:
            state_reward - LIST (nested). First list is IDs to which rewards will
                           be assigned. Second is corresponding reward vaules

        Outputs:
            None (modifies self.rewardspace)
        '''
        states = state_reward[0]
        rewards = state_reward[1]
        for idx, reward in zip(states, rewards):
            self.rewardspace[idx] = reward

    def ReadPolicy(self, state):
        '''
        Return the optimal policy based on the current state.
        '''
        theta = state[0]
        ID = state[1]
        action = self.policy[theta, ID, :]
        return action

    def GetReward(self, ID):
        return self.rewardspace[ID]

    def IsInterior(self, ID):
        '''
        Determines whether a state lies on the perimeter of the grid.

        Inputs:
            ID - INT of state ID to check

        Outputs:
            result - BOOL indicating whether state is on perimeter
        '''
        N_neighbors = np.sum(self.adjacencytensor[ID, :])

        if N_neighbors == 4:
            result = True
        else:
            result = False
        return result

    def IsPerimeter(self, ID):
        '''
        Determines whether a state lies on the perimeter of the grid.

        Inputs:
            ID - INT of state ID to check

        Outputs:
            result - BOOL indicating whether state is in a corner
        '''
        IDs_perimeter = []
        IDs_perimeter.extend([i for i in range(self.length)])
        a = self.length * (self.length - 1) + self.length
        b = self.length
        IDs_perimeter.extend([i for i in range(0, a, b)])
        IDs_perimeter.extend([i + (self.length - 1) for i in range(0, a, b)])
        IDs_perimeter.extend([i for i in range(self.length * (self.length - 1), self.width * self.length)])

        if ID in IDs_perimeter:
            result = True
        else:
            result = False
        return result

    def Adjacency(self, ID0, ID1, direction):
        '''
        Determines whether two grid squares are adjacent.

        Inputs:
            ID0 - INT of current state ID (s)
            ID1 - INT of desired state ID (s')
            direction - INT indicating direction of travel

        Outputs:
            result - BOOL indicating true or false for adjacency
        '''
        if not (direction in self.validdirections):
            raise ValueError("Invalid action provided as input! Must be in the set {-10,0,3,6,9}")

        # Determine whether states are adjacent
        idx = self.validdirections.index(direction)
        matrixentry = self.adjacencytensor[ID0, ID1, idx]
        result = bool(matrixentry)

        return result

    def Symmetricity(self, hiddenmode=False):
        '''
        Calculates the difference between the adjacency matrix and its transpose.
        Can choose whether to display a figure or return the difference.
        Inputs:
            hiddenmode - BOOL; if False, a figure will be displayed showing the difference image.
                               if True, no figure will be displayed and matrix will be returned.
        Outputs:
            difference - ARRAY; if hiddenmode == True
        '''
        if not (hiddenmode):
            assert np.sum(self.adjacencytensor) != 0, "Adjacency tensor has not been populated yet."

        image = np.sum(self.adjacencytensor, axis=2)
        difference = image - np.transpose(image)

        if not (hiddenmode):
            mpl.pyplot.subplot(1, 2, 1)
            mpl.pyplot.title('Adjacency Matrix')
            mpl.pyplot.imshow(image)

            mpl.pyplot.subplot(1, 2, 2)
            mpl.pyplot.title('Original Minus Transpose')
            mpl.pyplot.imshow(difference)
        else:
            return difference

    def GetNextState(self, state0, action):
        '''
        Given the current state [theta,ID] and a translation action, determine
        the state that will follow. This does not incorporate rotation error.

        Input:
            state0 - LIST (state vector)
            action - LIST (action vector)
        '''
        theta0 = state0[0]
        ID0 = state0[1]

        a_rot = action[1]

        a_trans = action[0]
        orientation = self.RoundTheta(theta0)
        # Either flip heading, stay in place, or maintain heading based on command
        if a_trans == 1:
            heading = orientation
        elif a_trans == 0:
            heading = -10
        elif a_trans == -1:
            heading = (orientation - 6) % 12
        else:
            errormsg = ' '.join(['Invalid drive command received:', str(a_trans)])
            raise ValueError(errormsg)

        if heading == -10:
            adjacent = self.adjacencytensor[ID0, :, 0]
        elif heading == 0:
            adjacent = self.adjacencytensor[ID0, :, 1]
        elif heading == 3:
            adjacent = self.adjacencytensor[ID0, :, 2]
        elif heading == 6:
            adjacent = self.adjacencytensor[ID0, :, 3]
        elif heading == 9:
            adjacent = self.adjacencytensor[ID0, :, 4]

        # If on perimeter and no adjacent states, robot stays still but can rotate.
        if np.sum(adjacent) == 0 and self.IsPerimeter(ID0):
            ID1 = ID0
            theta1 = (theta0 + a_rot) % 12

        # If not on perimeter and no adjacent states, something is wrong.
        elif np.sum(adjacent) == 0 and not (self.IsPerimeter(ID0)):
            errormsg = ' '.join(
                ['No adjacent cell found, but agent is not on the perimeter. Something is wrong! ID:', str(ID0),
                 ', direction', str(heading)])
            raise ValueError(errormsg)

        # If adjacent state found, robot translates and rotates.
        elif np.sum(adjacent) == 1:
            ID1 = np.where(adjacent == 1)[0][0]
            theta1 = (theta0 + a_rot) % 12

        # If no cases above are satisfied, something is wrong.
        else:
            errormsg = ' '.join(
                ['Invalid number of adjacent states found for cell', str(ID0), 'direction', str(heading)])
            raise ValueError(errormsg)

        state1 = [theta1, ID1]
        return state1

    def Probability(self, state0, action, state1, debug=False):

        theta0 = state0[0]
        ID0 = state0[1]

        P_path = [self.pe, 1 - 2 * self.pe, self.pe]
        rotations = [-1, 0, 1]
        theta_error = [(theta0 + d) % 12 for d in rotations]
        theta_tilde = [self.RoundTheta(theta) for theta in theta_error]  # Call this heading later
        # a_trans is squared as a quick fix. Algorithm was getting confused on directionality.
        ID_tilde = [self.GetNextState([t, ID0], action)[1] for t in theta_tilde]

        # Get a list of all possible states reachable from current state using
        # the chosen action.
        allstates = []
        for ID in ID_tilde:
            allstates.append([])
            for rot in rotations:
                a = (theta0 + rot) % 12
                b = ID
                allstates[-1].append([a, b])

        # Calculate total probability
        TotalProbability = 0
        for i in range(len(allstates)):
            group = allstates[i]
            if state1 in group:
                TotalProbability += P_path[i]

        if debug:
            return TotalProbability, allstates
        else:
            return TotalProbability

    def RoundTheta(self, theta):
        '''
        Round current orientation to the nearest cardinal direction
        '''
        if theta in [11, 0, 1]:
            orientation = 0
        elif theta in [2, 3, 4]:
            orientation = 3
        elif theta in [5, 6, 7]:
            orientation = 6
        elif theta in [8, 9, 10]:
            orientation = 9

        else:
            errormsg = ' '.join(['Invalid theta received as input:', str(theta)])
            raise ValueError(errormsg)

        return orientation

    def InitializePolicy(self, ID_goal):
        '''
        Creates initial policy for agent based on gradients and geometry.
        Inputs:
            ID_goal - INT which is the ID of the grid cell with the goal

        Outputs:
            None (modified self.policy)
        '''
        self.ID_goal = ID_goal
        self.InitializeGradient(ID_goal)  # Calculate gradient

        # Iterate through all orientations and grid cells
        for theta0 in range(12):
            phi = self.time2radians[theta0]  # Orientation
            u = np.array([np.cos(phi), np.sin(phi)])  # Orientation unit vector
            for ID0 in range(self.width * self.length):

                # Set the current state and then weigh the options of going forwards of backwards.
                state0 = [theta0, ID0]

                ##################
                # FORWARDS
                action0_f = [1, 0]  # Move fwd, no rotation
                state1_f = self.GetNextState(state0, action0_f)
                ID1_f = state1_f[1]
                delta_f = abs(theta0 - self.gradient[ID1_f])

                # BACKWARDS
                action0_b = [-1, 0]  # Move bwd, no rotation
                state1_b = self.GetNextState(state0, action0_b)
                ID1_b = state1_b[1]
                delta_b = abs(theta0 - self.gradient[ID1_b])
                ##################

                # Deterine which action resulted in the smallest turning angle.
                # If turning angles are same, choose the one that gets us closer to the goal.
                # Then, figure out which direction the turn is (CW vs CCW)
                if abs(delta_f - delta_b) < 5:
                    coords_goal = self.ID2Coords(ID_goal)

                    coords1_f = self.ID2Coords(state1_f[1])
                    displ1_f = [coords1_f[1] - coords_goal[1], coords1_f[0] - coords_goal[0]]
                    dist1_f = np.linalg.norm(displ1_f)

                    coords1_b = self.ID2Coords(state1_b[1])
                    displ1_b = [coords1_b[1] - coords_goal[1], coords1_b[0] - coords_goal[0]]
                    dist1_b = np.linalg.norm(displ1_b)

                    arg = np.argmin([dist1_b, dist1_f])
                    a_trans_star = arg * 2 - 1  # Mapping from [0,1] -> [-1,1]

                else:
                    arg = np.argmin([delta_b, delta_f])
                    a_trans_star = arg * 2 - 1  # Mapping from [0,1] -> [-1,1]

                # If fwd provides an easier turn:
                if a_trans_star == 1:
                    time_gradient = self.gradient[ID1_f]

                # If bwd provides an easier turn:
                elif a_trans_star == -1:
                    time_gradient = self.gradient[ID1_b]

                psi = self.time2radians[time_gradient]  # Angle of gradient vector
                v = np.array([np.cos(psi), np.sin(psi)])

                # Get direction of optimal rotation [-1,1] -> [CCW,CW]
                rot_sign = np.cross(v, u)
                if rot_sign >= 0:
                    a_rot_star = 1
                else:
                    a_rot_star = -1

                self.policy[theta0, ID0, 0] = a_trans_star
                self.policy[theta0, ID0, 1] = a_rot_star

    def InitializeGradient(self, ID_goal):
        '''
        Generate a vector "gradient" field pointing towards the goal from all other states.
        This function is called within the policy initialization function.

        Input:
            The grid ID of the goal.

        Output:
            None (modifies self.gradient)
        '''
        IDs = np.zeros([np.size(self.gridspace, 0), np.size(self.gridspace, 1)])

        count = 0
        for row in range(self.length):
            for col in range(self.width):
                IDs[row, col] = count
                count += 1

        # Create cartesian coordinate system for trigonometry calculations
        cartesian = np.zeros([np.size(self.gridspace, 0), np.size(self.gridspace, 1), 2])

        # X values
        for col in range(self.width):
            cartesian[:, col, 0] = col

        # Y values
        for row in range(self.length):
            cartesian[row, :, 1] = -row

        # Shift cartesian space so goal is at origin
        coords_goal = self.ID2Coords(ID_goal)
        cartesian[:, :, 0] -= cartesian[coords_goal[0], coords_goal[1], 0]
        cartesian[:, :, 1] -= cartesian[coords_goal[0], coords_goal[1], 1]

        # Iterate through all possible states (orientation, location) to generate policy
        count = 0
        for y in cartesian[:, 0, 1]:

            for x in cartesian[0, :, 0]:

                if x == 0 and y == 0:
                    psi = np.pi / 2

                else:
                    v = np.array([x, y])
                    v = np.transpose(v)
                    v = -v / np.linalg.norm(v, 2)

                    psi = np.arctan2(v[1], v[0])

                orientation = round((2 * np.pi - psi + np.pi / 2) / (2 * np.pi / 12)) % 12

                self.gradient[count] = orientation

                count += 1


# Uncomment for deugging to see the policy in the physical grid format
#            output = np.zeros([W,L,12])
#
#            for i in range(12):
#                output[:,:,i] = np.vstack([policy[i,a:(a+L),0] for a in range(0,W*L,L)])

class Agent(BigBang):
    def __init__(self, world, state=[0, 0], patience=18):
        self.state = state
        self.statehist = [self.state]
        self.errorhist = []
        self.actionhist = []
        self.patience = patience
        self.world = world
        self.foundcake = False
        self.stuck = False
        initmsg = ' '.join(['Agent initialized with state', str(self.state)])
        print(initmsg)

    def UpdateHist(self, state=[], error=[]):
        '''
        '''
        if error in [-1, 0, 1] and state == []:
            self.errorhist.append(error)
            assert (len(self.statehist) == len(
                self.errorhist)), 'When updating the error history, it must be as long as the state history.'


        elif len(state) == 2 and error == []:
            self.statehist.append(state)

        elif len(state) == 2 and error in [-1, 0, 1]:
            errormsg = 'Cannot update state history and error history simultaneously since they occur in sequence.'
            raise ValueError(errormsg)

        else:
            errormsg = ' '.join(['Invalid state and/or error update! State:', str(state), 'Error:', str(error)])
            raise ValueError(errormsg)

    def CheckStatus(self):
        '''
        '''
        if self.world.GetReward(self.state[1]) == np.max(self.world.rewardspace):
            self.foundcake = True

            print('Agent found the cake!')

        if len(self.statehist) >= self.patience:
            result = True
            N = min(self.patience, len(self.statehist))
            for n in range(N - 1, 0, -1):
                result = result and (self.statehist[n] == self.statehist[n - 1])

            if result:
                self.stuck = True
                print('Agent got stuck and gave up!')

    def Translate(self, action):
        '''
        Takes an action as an input and updates the agent's state

        Inputs:
            action - LIST [{-1,0,1},{-1,0,1}] (state vector)
        '''
        theta0 = self.state[0]
        ID0 = self.state[1]

        heading = self.RoundTheta(theta0)

        ID1 = self.world.GetNextState([heading, ID0], action)[1]

        self.state = [theta0, ID1]

        # No state update here because

    def Rotate(self, action):
        '''
        '''
        delta = action[1]
        self.state[0] = int((self.state[0] + delta) % 12)

        self.UpdateHist(state=self.state, error=[])

    #    def RotateToHighestReward(self):
    #        if self.IsPerimeter(self.state):
    #            ID0 = self.state[1]
    #            adjacency_neighbors = np.sum(self.adjacencytensor[ID0,:,1:],2)
    #            IDs_neighbors = [ID for ID,adj in enumerate(adjacency_neighbors) if adj == 1]
    #            Rewards_neighbors = [self.GetReward(ID) for ID in IDs_neighbors]
    #        else:
    #            raise ValueError('Pure rotation command encountered in a non-perimeter state.')

    def RotationError(self):
        '''
        Determines whether a rotation error is incurred and then updates the agent's error history.

        Inputs:
            None

        Outputs:
            None (modified agent's state directly)
        '''
        error_eff = 0
        if np.random.rand() <= self.pe:
            self.Rotate(1)
            error_eff += 1

        if np.random.rand() <= self.pe:
            self.Rotate(-1)
            error_eff -= 1

        self.UpdateHist(state=[], error=error_eff)

    def Act(self, action):
        '''
        '''
        # Agent's state is modified directly by each of these steps
        self.RotationError  # Possibly incur rotation error
        self.Translate(action)  # Move in commanded direction
        self.Rotate(action)  # Post-translation rotation

    def Navigate(self, iters_max=40):
        '''
        '''
        # Get initial proposed action
        action_proposed = self.world.ReadPolicy(self.state)

        # While agent has not found cake, is not stuck, and has not given up,
        # navigate the state space.
        count = 0
        while not (self.foundcake) and not (self.stuck) and count <= iters_max:
            state_proposed = self.world.GetNextState(self.state, action_proposed)

            # If current and proposed states are adjacent via proposed action, act.
            if self.world.Probability(self.state, action_proposed, state_proposed) > 0:
                self.Act(action_proposed)

            self.CheckStatus()  # Check to see if robot is stuck.
            action_proposed = self.world.ReadPolicy(self.state)

            count += 1

    def PolicyEvaluator(self):
        pass


def drawArrow(Werld, state, ax):
    '''
    Draw an arrow in a given grid cell and orientation
    '''

    def R(phi):
        '''
        2-dimensional CCW rotation by phi degrees

        Input:
            phi - FLOAT indicating rotation angle in radians

        Output:
            vector_rotated - NUMPY ARRAY containing the 2 components of the rotated vector
        '''
        if phi == Werld.time2radians[0]:
            phi -= 0.1
        phi += np.pi

        vector_rotated = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        return vector_rotated

    # Extract state information, get coordinates, and get the orientation in terms of clock time
    time = state[0]
    ID = state[1]
    coords = np.array([ID % 6, ID // 6])
    phi = Werld.time2radians[time]

    # Create reference vector (a rightward arrow at 3:00 is used since that's where
    # rotation on the unit circle starts. All other orientations will be attained
    # by rotating this vector CCW).
    XY = np.array([coords[0] - 0.375, coords[1]])  # Vector head
    XYTEXT = np.array([coords[0] + 0.375, coords[1] - 0.05])  # Vector tail

    # Rotate the vector and convert it to a tuple
    xy = (XY - coords) @ R(phi) + coords
    xy = tuple(xy)

    xytext = (XYTEXT - coords) @ R(phi) + coords
    xytext = tuple(xytext)

    # Draw the arrow
    ax.annotate(' ',
                xy=xy,
                xycoords='data',
                xytext=xytext,
                arrowprops=dict(facecolor='black'))


def plotPolicy(Policy_Aprx, StateValue, Werld, title=[]):
    '''
    Plot a policy on the grid space
    '''

    # Initialize state value matrix
    StateValue_matrix = np.zeros([6, 6])
    N_states = len(Policy_Aprx)

    # If state values weren't provided, set all the values to zero for the plot.
    if StateValue != []:
        for s in range(N_states):
            StateValue_matrix[s % 6, s // 6] = StateValue[s]

    # Prepare the figure
    fig = mpl.pyplot.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    mpl.pyplot.imshow(StateValue_matrix, cmap='Blues')
    mpl.pyplot.colorbar()

    # Draw arrows for each state
    for ID in np.arange(N_states):
        theta = Policy_Aprx[ID]
        drawArrow(Werld, [theta, ID], ax)

    # If a title was provided, add it to the plot.
    if title != []:
        mpl.pyplot.title(title)

    mpl.pyplot.show()


def plotStateHistory(Werld, statehist, StateValue=[]):
    '''
    Plot the agent's state history (path and orientation) on the grid

    Inputs:
        Werld - BIGBANG object
        statehist - LIST (nested) with length 2 lists containing state history
        StateValue - Value to be used as temperature in the plot. Typically the state value or reward

    Ouputs:
        None (displays agent's path)
    '''
    StateValue_matrix = np.zeros([6, 6])
    N_states = Werld.length * Werld.width

    if StateValue != []:
        for s in range(N_states):
            StateValue_matrix[s % 6, s // 6] = StateValue[s]

    fig = mpl.pyplot.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    mpl.pyplot.imshow(StateValue_matrix, cmap='Blues')
    mpl.pyplot.colorbar()

    state0 = statehist[0]
    drawArrow(Werld, state0, ax)
    for state1 in statehist[1:]:
        ID0 = state0[1]
        ID1 = state1[1]

        coords0 = [ID0 % 6, ID0 // 6]
        coords1 = [ID1 % 6, ID1 // 6]

        mpl.pyplot.plot([coords0[0], coords1[0]], [coords0[1], coords1[1]], 'r--', lw=2)

        drawArrow(Werld, state1, ax)

        state0 = state1

    coords_goal = Werld.ID2Coords(Werld.ID_goal)
    mpl.pyplot.scatter(coords_goal[1], coords_goal[0], marker='*', s=2000, c='g')


World = BigBang(6, 6, 0)

IDs_red = [0, 1, 2, 3, 4, 5, 6, 11, 12, 17, 18, 23, 24, 29, 30, 31, 32, 33, 34, 35]
rewards_red = [-100 for i in range(len(IDs_red))]

IDs_yellow = [8, 10, 14, 16, 20, 22]
rewards_yellow = [-1 for i in range(len(IDs_yellow))]

IDs_green = [9]
rewards_green = [1]

World.AssignRewards([IDs_red, rewards_red])
World.AssignRewards([IDs_yellow, rewards_yellow])
World.AssignRewards([IDs_green, rewards_green])

# World.GetNextState([6,0],[-1,1])

World.InitializePolicy(9)

agent0 = Agent(World, state=[6, 7], patience=18)
agent0.Navigate()

plotStateHistory(World, agent0.statehist[0:20])