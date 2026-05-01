import numpy as np
from .ArtificialPotentialField import ArtificialPotentialField
from .Abstraction import fill_obstacles, abstract

class ModelPredicitveController:
    def __init__(self, horizon, samples, cov, min_u, max_u, n=3, APF=None, path=None, dist_w=20, obs_w=20, rem_w=1000, path_w=2000, returned_horizon=1):
        self.H = horizon
        self.K = samples
        self.dimension = n
        self.cov = cov
        self.max_u = max_u
        self.min_u = min_u
        self.dist_weight = dist_w # 0
        self.obs_weight = obs_w # 20
        self.rem_weight = rem_w # 10_000
        self.delta_weight = 100000
        self.path_weight = path_w # 200
        self.prev_u = np.array([0, 0, 0])
        if APF is not None:
            self.APF = APF
        else:
            self.APF = None
        self.path = path
        self.path_index = 0
        self.returned_horizon = int(np.clip(returned_horizon, 1, horizon))

    def control(self, x0, targets_arr, obstacles):
        """
        This is where control is implemented.
        Parameters:
            - x0: Initial cartesian position at the current time step
        """

        noise = np.random.multivariate_normal(np.zeros(self.dimension), self.cov, (self.K, self.H))
        costs = np.zeros(self.K)

        for k in range(self.K):
            targets = np.copy(targets_arr)
            pos = np.copy(x0)
            cost = 0.0
            for h in range(self.H):
                u = noise[k, h]
                if self.APF is not None:
                    force = self.APF.resultant_force(pos, targets, obstacles)
                    # print(force, pos)
                    control = np.clip(u + force, self.min_u, self.max_u)
                    pos = pos + control
                else:
                    u = np.clip(u, self.min_u, self.max_u)
                    pos = pos + u
                c, targets = self.cost_function(pos, targets, obstacles, u)
                cost += c
            costs[k] = cost

        beta = np.min(costs)
        # weights = np.exp(-(costs - beta))
        # weights /= np.sum(weights) + 1e-12
        # delta_U = np.sum(weights[:, None, None] * noise, axis=0)
        # u0 = delta_U[0]
        index = np.argmin(costs)
        if self.APF:
            pos = np.copy(x0)
            force = self.APF.resultant_force(pos, targets, obstacles)
            u0 = noise[index, 0:self.returned_horizon, :] + force
            # u0 += force
        else:
            u0 = noise[index, 0:self.returned_horizon, :]
        # if beta == np.inf:
        #     print(force, u0, pos)

        return np.clip(u0, self.min_u, self.max_u), beta
    
    def cost_function(self, new_pos, targets, obstacles, u):
        if self.path is not None:
            path_distances = np.linalg.norm(self.path - new_pos, axis=1)
            if path_distances[self.path_index] <= 0.2:
                self.path_index += 1
                self.path_index = self.path_index % len(path_distances) # Avoid indexing error
            self.path_index = np.argmin(path_distances)
            path_cost = path_distances[self.path_index]
            if self.path_index != len(path_distances) - 1:
                p = path_distances[self.path_index + 1:self.path_index + 5]
                p = p * np.linspace(1, len(p), len(p)) * 2
                path_cost += np.sum(p)
        else:
            path_cost = 0

        length = len(targets)

        if self.dimension == 3:
            targets = adjust_targets(new_pos, targets)
        else:
            targets = adjust_targets_2d(new_pos, targets)
        distances = np.linalg.norm(targets - new_pos, axis=1)

        if obstacles is not None:
            obstacle_distance = np.linalg.norm(obstacles - new_pos, axis=1)
        if self.dimension == 3:
            #t = (obstacles - new_pos) ** 2
            #obstacle_distance = np.sum(t, axis=1)
            temp = np.linalg.norm(obstacles[:, :2] - new_pos[:2], axis=1)
            #temp = np.sum(t[:, :2], axis=1)
            if new_pos[2] <= -0.1: #np.any(new_pos[2] < (obstacles[temp < 0.01])[:, 2]):
                return np.inf, targets
        
        if obstacles is not None:
            obs_cost = np.sum(0.04 - obstacle_distance[obstacle_distance < 0.04]) * self.obs_weight
        else:
            obs_cost = 0
        dist_cost = np.sum(distances) * self.dist_weight
        if len(targets) != 0:
            dist_cost /= len(targets)
        rem_cost = (2 * len(targets) - length) * self.rem_weight

        #effort_cost = self.delta_weight * angle_between_vectors(u, self.prev_u)
        cost = rem_cost + obs_cost + dist_cost + path_cost * self.path_weight # + effort_cost
        return cost, targets

def angle_between_vectors(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'.
    The angle will be in the range [0, pi].
    """
    # Compute the dot product
    dot_product = np.dot(v1, v2)
    
    # Compute the magnitudes (norms)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Compute the cosine of the angle
    # Use np.clip to clamp the value within [-1.0, 1.0] to avoid
    # potential floating-point errors (e.g., 1.0000000000000002)
    cosine_angle = np.clip(dot_product / (norm_v1 * norm_v2 + 1e-12), -1.0, 1.0)
    
    # Compute the angle in radians using arccos
    angle_rad = np.arccos(cosine_angle)
    
    return angle_rad

def adjust_targets(new_pos, targets):
        distances = np.linalg.norm(targets[:, :2] - new_pos[:2], axis=1)
        heights = targets[:, 2] - new_pos[2]
        #print(heights)
        condition = ((distances < 0.02) * ((heights < -0.05) + (heights > 0.05))) + (distances > 0.02)
        targets = targets[condition]
        #print(targets)
        return targets

def adjust_targets_2d(new_pos, targets):
        distances = np.linalg.norm(targets - new_pos, axis=1)
        #print(heights)
        condition = (distances > 0.04)
        targets = targets[condition]
        #print(targets)
        return targets

def eval(positions):
    # To implement this, the code will subtract 2 cummulative positions to find the difference between them and then take the absolute value.
    # The sum of all the absolute cummulative differences will give the total distance moved.
    # To do the subtraction, the code will right shift all the elements in the position array and perform the subtraction with the original array.

    right_shifted_array = positions[:-1]
    right_shifted_array = right_shifted_array.tolist()
    right_shifted_array.insert(0, [0.0, 0.0, 0.0]) # Keeps the array sizes the same
    right_shifted_array = np.array(right_shifted_array)
    positions = np.array(positions)
    differences = np.linalg.norm(positions - right_shifted_array, axis=1)
    return np.sum(differences)

def run_multiple_tests(n=10):
    import matplotlib.pyplot as plt
    avg_pct = []
    rem = []
    dist = []
    coll = []

    for i in range(n):
        targets_arr = []
        z = -0.05
        for i in range(1, 11):
            x = i / 20 + 0.1
            for j in range(1, 11):
                y = j / 20 + 0.1
                targets_arr.append([x, y, z])

        targets_arr = np.array(targets_arr)
        targets_arr = targets_arr + np.array([0.3, 0.3, 0.05])
        obstacles = np.linspace(0, 2, 307200)
        obstacles = np.array([obstacles, obstacles, np.ones(307200) * -0.05]).T
        #obstacles = targets_arr - np.array([0.0, 0.0, 0.05])
        labels, centres = abstract(obstacles, size=200)

        pos = np.array([0, 0, 0])
        positions = []
        pcts = []
        collisions = 0
        print("Start")

        for t in range(180):
            obs = fill_obstacles(pos, labels, centres, obstacles, n=5000)
            obs = np.array(obs)
            u, cost = controller.control(pos, targets_arr, obs)

            force = controller.APF.resultant_force(pos, targets_arr, obs)
            mppi = (u - force)
            pct = np.linalg.norm(mppi) / (np.linalg.norm(force) + np.linalg.norm(mppi))
            pcts.append(pct)
            
            pos = u + pos
            if pos[2] < -0.05:
                collisions += 1
            targets_arr = adjust_targets(pos, targets_arr)
            positions.append(pos)
            if len(targets_arr) == 0:
                break
            print(f"{t} steps")

        coll.append(collisions)
        distance = eval(np.array(positions))
        dist.append(distance)
        remaining = len(targets_arr)
        rem.append(remaining)
        avg_pct.append(np.mean(pcts))
        print("Average Percentage:", np.mean(pcts))
        print("Remaining:", remaining)
        print("Distance Travelled:", distance)
        print("Collisions:", collisions)

    print(f"\n\nAverage Distance Travelled across all tests: {np.mean(dist)}")
    print(f"Average Collisions across all tests: {np.mean(coll)}")
    print(f"Average Targets Remaining across all tests: {np.mean(rem)}")
    print(f"Average Percentage of MPPI control: {np.mean(avg_pct)}")

    x = np.linspace(0, len(positions) - 1, len(positions))
    positions = np.array(positions)

    plt.subplot(1, 2, 1)
    plt.scatter(targets_arr[:, 0], targets_arr[:, 1], c='red')
    plt.plot(positions[:, 0], positions[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 2, 2)
    plt.plot(x, positions[:, 2])
    plt.xlabel('timestep')
    plt.ylabel('z')

    plt.show()

controller = ModelPredicitveController (
    5,
    800,
    np.identity(3) * 0.0015,
    -0.03,
    0.03,
    APF=True
)

if __name__ == "__main__":
    run_multiple_tests(n=1)

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # targets_arr = []

    # z = -0.05
    # for i in range(1, 11):
    #     x = i / 20 + 0.1
    #     for j in range(1, 11):
    #         y = j / 20 + 0.1
    #         targets_arr.append([x, y, z])

    # targets_arr = np.array(targets_arr)
    # targets = np.copy(targets_arr)
    # obstacles = targets_arr - np.array([0.0, 0.0, 0.05])

    # pos = np.array([0, 0, 0])
    # positions = []
    # costs = []
    # pcts = []

    # model = ArtificialPotentialField(
    #     targets=None,
    #     obstacles=None,
    #     att_gain=0.08,
    #     rep_gain=0.15,
    #     max_distance=0.07
    # )
    # for t in range(150):
    #     if t % 10 == 0:
    #         print(t, "steps")
    #     u, cost = controller.control(pos, targets_arr, obstacles)
    #     #u = model.resultant_force(pos, targets_arr, obstacles)

    #     force = model.resultant_force(pos, targets_arr, obstacles)
    #     mppi = (u - force)
    #     pct = np.linalg.norm(mppi) / (np.linalg.norm(force) + np.linalg.norm(mppi))
    #     pcts.append(pct)
        
    #     pos = u + pos
    #     #cost = 0
    #     if pos[2] < -0.1:
    #         print(u, pos)
    #     targets_arr = adjust_targets(pos, targets_arr)
    #     positions.append(pos)
    #     costs.append(cost)
    #     if len(targets_arr) == 0:
    #         break
    # positions = np.array(positions)

    # print(len(targets_arr))
    # print("Percentages:\n", pcts)
    # print("Average Percentage:", np.mean(pcts))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2])
    # print(positions)
    # print(f"Total Distance = {eval(positions)} m")
    # plt.show()

    # x = np.linspace(0, len(costs) - 1, len(costs))
    # plt.plot(x, costs)
    # plt.xlabel('timestep')
    # plt.ylabel('cost')
    # plt.show()

    # plt.subplot(1, 2, 1)
    # plt.scatter(targets[:, 0], targets[:, 1], c='red')
    # plt.plot(positions[:, 0], positions[:, 1])
    # plt.xlabel('x')
    # plt.ylabel('y')

    # plt.subplot(1, 2, 2)
    # plt.plot(x, positions[:, 2])
    # plt.xlabel('timestep')
    # plt.ylabel('z')

    # plt.show()