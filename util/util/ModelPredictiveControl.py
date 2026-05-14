import numpy as np

from .ArtificialPotentialField import ArtificialPotentialField
from .Abstraction import fill_obstacles, abstract


class ModelPredictiveController:
    """
    Sampling-based Model Predictive Path Integral (MPPI) controller.

    This controller generates multiple random control trajectories over a
    finite horizon and evaluates them using a custom cost function. The
    trajectory with the minimum cost is selected as the control action.

    Optionally, an Artificial Potential Field (APF) can be combined with
    the sampled controls to bias movement toward targets and away from
    obstacles.

    Attributes:
        H (int):
            Prediction horizon length.

        K (int):
            Number of sampled trajectories.

        dimension (int):
            State/control dimensionality (2D or 3D).

        cov (np.ndarray):
            Covariance matrix used for sampling control noise.

        max_u (float or np.ndarray):
            Maximum allowed control value(s).

        min_u (float or np.ndarray):
            Minimum allowed control value(s).

        dist_weight (float):
            Weight applied to target distance cost.

        obs_weight (float):
            Weight applied to obstacle proximity cost.

        rem_weight (float):
            Weight applied to remaining-target penalty.

        path_weight (float):
            Weight applied to path-following cost.

        prev_u (np.ndarray):
            Previous control input (currently unused).

        APF (ArtificialPotentialField or None):
            Artificial Potential Field model used for guidance.

        path (np.ndarray or None):
            Optional reference path to follow.

        path_index (int):
            Current closest index along the reference path.

        returned_horizon (int):
            Number of control steps returned to the caller.
    """

    def __init__(
        self,
        horizon,
        samples,
        cov,
        min_u,
        max_u,
        n=3,
        APF=None,
        path=None,
        dist_w=20,
        obs_w=20,
        rem_w=1000,
        path_w=2000,
        returned_horizon=1
    ):
        """
        Initialize the MPPI controller.

        Args:
            horizon (int):
                Prediction horizon length.

            samples (int):
                Number of sampled trajectories.

            cov (np.ndarray):
                Covariance matrix used for Gaussian control sampling.

            min_u (float or np.ndarray):
                Minimum control limits.

            max_u (float or np.ndarray):
                Maximum control limits.

            n (int, optional):
                Number of dimensions (2D or 3D). Defaults to 3.

            APF (ArtificialPotentialField, optional):
                Artificial Potential Field object. Defaults to None.

            path (np.ndarray, optional):
                Optional path to follow. Defaults to None.

            dist_w (float, optional):
                Weight for target distance cost.

            obs_w (float, optional):
                Weight for obstacle avoidance cost.

            rem_w (float, optional):
                Weight for remaining-target penalty.

            path_w (float, optional):
                Weight for path-following cost.

            returned_horizon (int, optional):
                Number of future controls returned by the controller.
        """
        self.H = horizon
        self.K = samples
        self.dimension = n

        self.cov = cov
        self.max_u = max_u
        self.min_u = min_u

        # Cost function weights
        self.dist_weight = dist_w
        self.obs_weight = obs_w
        self.rem_weight = rem_w
        self.delta_weight = 100000
        self.path_weight = path_w

        # Previous control input (unused but reserved for smoothness penalties)
        self.prev_u = np.array([0, 0, 0])

        # Optional Artificial Potential Field
        self.APF = APF if APF is not None else None

        # Optional reference path
        self.path = path
        self.path_index = 0

        # Ensure returned horizon is within valid bounds
        self.returned_horizon = int(
            np.clip(returned_horizon, 1, horizon)
        )

    def control(self, x0, targets_arr, obstacles):
        """
        Compute the optimal control action using trajectory sampling.

        The controller:
        1. Samples random control trajectories.
        2. Simulates future motion over the prediction horizon.
        3. Evaluates each trajectory using the cost function.
        4. Returns the lowest-cost trajectory segment.

        Args:
            x0 (np.ndarray):
                Current position/state of the agent.

            targets_arr (np.ndarray):
                Array of target positions.

            obstacles (np.ndarray or None):
                Array of obstacle positions.

        Returns:
            tuple:
                (
                    optimal_control_sequence,
                    minimum_cost
                )
        """

        # Sample Gaussian control noise for all trajectories
        noise = np.random.multivariate_normal(
            np.zeros(self.dimension),
            self.cov,
            (self.K, self.H)
        )

        # Store total trajectory costs
        costs = np.zeros(self.K)

        # Evaluate every sampled trajectory
        for k in range(self.K):

            targets = np.copy(targets_arr)
            pos = np.copy(x0)

            cost = 0.0

            # Roll out trajectory over horizon
            for h in range(self.H):

                # Sampled control perturbation
                u = noise[k, h]

                if self.APF is not None:

                    # Compute APF guidance force
                    force = self.APF.resultant_force(
                        pos,
                        targets,
                        obstacles
                    )

                    # Combine sampled control with APF guidance
                    control = np.clip(
                        u + force,
                        self.min_u,
                        self.max_u
                    )

                    # Update simulated position
                    pos = pos + control

                else:
                    # Pure MPPI control without APF guidance
                    u = np.clip(u, self.min_u, self.max_u)
                    pos = pos + u

                # Evaluate trajectory state cost
                c, targets = self.cost_function(
                    pos,
                    targets,
                    obstacles,
                    u
                )

                cost += c

            costs[k] = cost

        # Minimum trajectory cost
        beta = np.min(costs)

        # Select best trajectory index
        index = np.argmin(costs)

        # Return first segment of optimal trajectory
        if self.APF:

            pos = np.copy(x0)

            force = self.APF.resultant_force(
                pos,
                targets,
                obstacles
            )

            u0 = (
                noise[index, 0:self.returned_horizon, :]
                + force
            )

        else:
            u0 = noise[index, 0:self.returned_horizon, :]

        return np.clip(u0, self.min_u, self.max_u), beta

    def cost_function(self, new_pos, targets, obstacles, u):
        """
        Compute the cost associated with a simulated trajectory state.

        The cost consists of:
            - Distance to remaining targets
            - Obstacle proximity penalty
            - Remaining-target penalty
            - Optional path-following penalty

        Args:
            new_pos (np.ndarray):
                Simulated future position.

            targets (np.ndarray):
                Remaining target positions.

            obstacles (np.ndarray or None):
                Obstacle positions.

            u (np.ndarray):
                Current control input.

        Returns:
            tuple:
                (
                    total_cost,
                    updated_targets
                )
        """

        # ---------------------------------------------------------
        # Path-following cost
        # ---------------------------------------------------------
        if self.path is not None:

            path_distances = np.linalg.norm(
                self.path - new_pos,
                axis=1
            )

            # Advance path index when sufficiently close
            if path_distances[self.path_index] <= 0.2:

                self.path_index += 1

                # Prevent index overflow
                self.path_index = (
                    self.path_index % len(path_distances)
                )

            # Find closest point on path
            self.path_index = np.argmin(path_distances)

            path_cost = path_distances[self.path_index]

            # Add lookahead path penalty
            if self.path_index != len(path_distances) - 1:

                p = path_distances[
                    self.path_index + 1:
                    self.path_index + 5
                ]

                p = p * np.linspace(1, len(p), len(p)) * 2

                path_cost += np.sum(p)

        else:
            path_cost = 0

        # Number of targets before adjustment
        length = len(targets)

        # Remove targets already reached
        if self.dimension == 3:
            targets = adjust_targets(new_pos, targets)
        else:
            targets = adjust_targets_2d(new_pos, targets)

        # Distance to remaining targets
        distances = np.linalg.norm(
            targets - new_pos,
            axis=1
        )

        # Compute obstacle distances
        if obstacles is not None:
            obstacle_distance = np.linalg.norm(
                obstacles - new_pos,
                axis=1
            )

        # ---------------------------------------------------------
        # 3D collision handling
        # ---------------------------------------------------------
        if self.dimension == 3:

            temp = np.linalg.norm(
                obstacles[:, :2] - new_pos[:2],
                axis=1
            )

            # Penalize positions below safety threshold
            if new_pos[2] <= -0.1:
                return np.inf, targets

        # ---------------------------------------------------------
        # Obstacle cost
        # ---------------------------------------------------------
        if obstacles is not None:

            obs_cost = np.sum(
                0.04 - obstacle_distance[
                    obstacle_distance < 0.04
                ]
            ) * self.obs_weight

        else:
            obs_cost = 0

        # ---------------------------------------------------------
        # Distance-to-target cost
        # ---------------------------------------------------------
        dist_cost = (
            np.sum(distances) * self.dist_weight
        )

        if len(targets) != 0:
            dist_cost /= len(targets)

        # ---------------------------------------------------------
        # Reward target completion
        # ---------------------------------------------------------
        rem_cost = (
            (2 * len(targets) - length)
            * self.rem_weight
        )

        # Total cost
        cost = (
            rem_cost
            + obs_cost
            + dist_cost
            + path_cost * self.path_weight
        )

        return cost, targets


def angle_between_vectors(v1, v2):
    """
    Compute the angle between two vectors.

    Args:
        v1 (np.ndarray):
            First vector.

        v2 (np.ndarray):
            Second vector.

    Returns:
        float:
            Angle in radians within [0, pi].
    """

    # Dot product between vectors
    dot_product = np.dot(v1, v2)

    # Vector magnitudes
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Clamp cosine value for numerical stability
    cosine_angle = np.clip(
        dot_product / (norm_v1 * norm_v2 + 1e-12),
        -1.0,
        1.0
    )

    # Convert cosine to angle
    angle_rad = np.arccos(cosine_angle)

    return angle_rad


def adjust_targets(new_pos, targets):
    """
    Remove targets that have already been reached in 3D space.

    A target is considered reached if:
        - XY distance is sufficiently small
        - Height difference is within tolerance

    Args:
        new_pos (np.ndarray):
            Current agent position.

        targets (np.ndarray):
            Remaining target positions.

    Returns:
        np.ndarray:
            Filtered target array.
    """

    distances = np.linalg.norm(
        targets[:, :2] - new_pos[:2],
        axis=1
    )

    heights = targets[:, 2] - new_pos[2]

    condition = (
        ((distances < 0.02)
         * ((heights < -0.05) + (heights > 0.05)))
        + (distances > 0.02)
    )

    targets = targets[condition]

    return targets


def adjust_targets_2d(new_pos, targets):
    """
    Remove targets that have already been reached in 2D space.

    Args:
        new_pos (np.ndarray):
            Current agent position.

        targets (np.ndarray):
            Remaining target positions.

    Returns:
        np.ndarray:
            Filtered target array.
    """

    distances = np.linalg.norm(
        targets - new_pos,
        axis=1
    )

    condition = (distances > 0.04)

    targets = targets[condition]

    return targets


def eval(positions):
    """
    Compute the total distance travelled by the agent.

    The function calculates pairwise distances between consecutive
    positions and sums them.

    Args:
        positions (np.ndarray):
            Sequence of positions over time.

    Returns:
        float:
            Total travelled distance.
    """

    # Shift positions by one timestep
    right_shifted_array = positions[:-1]

    right_shifted_array = right_shifted_array.tolist()

    # Pad with origin to maintain shape
    right_shifted_array.insert(0, [0.0, 0.0, 0.0])

    right_shifted_array = np.array(right_shifted_array)

    positions = np.array(positions)

    # Compute frame-to-frame displacement
    differences = np.linalg.norm(
        positions - right_shifted_array,
        axis=1
    )

    return np.sum(differences)


def run_multiple_tests(n=10):
    """
    Run multiple controller simulations and evaluate performance.

    Metrics collected:
        - Average MPPI contribution percentage
        - Remaining targets
        - Distance travelled
        - Collision count

    Args:
        n (int, optional):
            Number of simulations to run.
    """

    import matplotlib.pyplot as plt

    avg_pct = []
    rem = []
    dist = []
    coll = []

    for i in range(n):

        # ---------------------------------------------------------
        # Generate target grid
        # ---------------------------------------------------------
        targets_arr = []

        z = -0.05

        for i in range(1, 11):

            x = i / 20 + 0.1

            for j in range(1, 11):

                y = j / 20 + 0.1

                targets_arr.append([x, y, z])

        targets_arr = np.array(targets_arr)

        targets_arr = targets_arr + np.array([0.3, 0.3, 0.05])

        # ---------------------------------------------------------
        # Generate obstacle field
        # ---------------------------------------------------------
        obstacles = np.linspace(0, 2, 307200)

        obstacles = np.array([
            obstacles,
            obstacles,
            np.ones(307200) * -0.05
        ]).T

        # Build obstacle abstraction
        labels, centres = abstract(obstacles, size=200)

        pos = np.array([0, 0, 0])

        positions = []
        pcts = []

        collisions = 0

        print("Start")

        # ---------------------------------------------------------
        # Main simulation loop
        # ---------------------------------------------------------
        for t in range(180):

            # Retrieve local obstacle subset
            obs = fill_obstacles(
                pos,
                labels,
                centres,
                obstacles,
                n=5000
            )

            obs = np.array(obs)

            # Compute control action
            u, cost = controller.control(
                pos,
                targets_arr,
                obs
            )

            # Compute APF contribution percentage
            force = controller.APF.resultant_force(
                pos,
                targets_arr,
                obs
            )

            mppi = (u - force)

            pct = np.linalg.norm(mppi) / (
                np.linalg.norm(force)
                + np.linalg.norm(mppi)
            )

            pcts.append(pct)

            # Update position
            pos = u + pos

            # Collision detection
            if pos[2] < -0.05:
                collisions += 1

            # Remove reached targets
            targets_arr = adjust_targets(pos, targets_arr)

            positions.append(pos)

            # Stop if all targets reached
            if len(targets_arr) == 0:
                break

            print(f"{t} steps")

        # ---------------------------------------------------------
        # Store simulation statistics
        # ---------------------------------------------------------
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

    # -------------------------------------------------------------
    # Print overall statistics
    # -------------------------------------------------------------
    print(
        f"\n\nAverage Distance Travelled across all tests: "
        f"{np.mean(dist)}"
    )

    print(
        f"Average Collisions across all tests: "
        f"{np.mean(coll)}"
    )

    print(
        f"Average Targets Remaining across all tests: "
        f"{np.mean(rem)}"
    )

    print(
        f"Average Percentage of MPPI control: "
        f"{np.mean(avg_pct)}"
    )

    # -------------------------------------------------------------
    # Plot trajectory results
    # -------------------------------------------------------------
    x = np.linspace(
        0,
        len(positions) - 1,
        len(positions)
    )

    positions = np.array(positions)

    plt.subplot(1, 2, 1)

    plt.scatter(
        targets_arr[:, 0],
        targets_arr[:, 1],
        c='red'
    )

    plt.plot(
        positions[:, 0],
        positions[:, 1]
    )

    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 2, 2)

    plt.plot(x, positions[:, 2])

    plt.xlabel('timestep')
    plt.ylabel('z')

    plt.show()


# -----------------------------------------------------------------
# Example controller configuration
# -----------------------------------------------------------------
controller = ModelPredictiveController(
    horizon=5,
    samples=800,
    cov=np.identity(3) * 0.0015,
    min_u=-0.03,
    max_u=0.03,
    APF=True
)


if __name__ == "__main__":
    run_multiple_tests(n=1)
