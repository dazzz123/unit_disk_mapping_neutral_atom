from dimod import BinaryQuadraticModel, quicksum, Binary
import numpy as np
import json

def QUBO_to_qubo(QUBO, variable_names):
    """Convert a dimod.BinaryQuadraticModel to a symmetric NumPy matrix."""
    size = len(variable_names)
    Q = np.zeros((size, size))
    qubo_dict = QUBO.to_qubo()[0]
    var_to_index = {var: idx for idx, var in enumerate(variable_names)}
    
    for (var1, var2), coeff in qubo_dict.items():
        i = var_to_index[var1]
        j = var_to_index[var2]
        Q[i][j] = coeff
        if i != j:
            Q[j][i] = coeff  # Ensure symmetry
    return Q

def ucp_to_qubo_matrix(ucp_dict):
    """Convert a Unit Commitment Problem to a QUBO matrix."""
    # Input validation
    required_keys = ['num_units', 'num_time_periods', 'demand', 'p_min', 'p_max', 'variable_cost', 'startup_cost', 'N']
    for key in required_keys:
        if key not in ucp_dict:
            raise ValueError(f"Missing key '{key}' in ucp_dict")
    
    num_units = ucp_dict['num_units']
    num_time_periods = ucp_dict['num_time_periods']
    demand = ucp_dict['demand']
    p_min = ucp_dict['p_min']
    p_max = ucp_dict['p_max']
    variable_cost = ucp_dict['variable_cost']
    startup_cost = ucp_dict['startup_cost']
    N = ucp_dict['N']

    if len(demand) != num_time_periods or len(p_min) != num_units or len(p_max) != num_units:
        raise ValueError("Input lengths mismatch")
    if len(variable_cost) != num_units or len(startup_cost) != num_units:
        raise ValueError("Length of 'variable_cost' and 'startup_cost' must equal num_units")

    penalty = 1000
    penalty1 = 100

    units = list(range(1, num_units + 1))
    time_periods = list(range(1, num_time_periods + 1))

    indices_for_u = [(i, t) for i in units for t in time_periods]
    extra_bv_indices = [(i, t, k) for i in units for t in time_periods for k in range(N + 1)]

    u = {(i, t): Binary(f"u_{i}_{t}") for i, t in indices_for_u}
    z = {(i, t, k): Binary(f"z_{i}_{t}_{k}") for i, t, k in extra_bv_indices}
    variable_names = [f"u_{i}_{t}" for i, t in indices_for_u] + [f"z_{i}_{t}_{k}" for i, t, k in extra_bv_indices]

    no_of_bin_variables = len(u) + len(z)

    h = [(p_max[i - 1] - p_min[i - 1]) / N for i in units]

    p = {}
    for i in units:
        for t in time_periods:
            p[(i, t)] = quicksum((p_min[i - 1] + h[i - 1] * k) * z[i, t, k] for k in range(N + 1))

    main_objective = quicksum(
        (variable_cost[i - 1] * p[i, t]) + (startup_cost[i - 1] * u[i, t])
        for i in units for t in time_periods
    )

    const1_as_penalty = [
        (quicksum(p[i, t] for i in units) - demand[t - 1]) ** 2
        for t in time_periods
    ]

    final_penalty_term = quicksum(penalty * const1_as_penalty[i] for i in range(len(const1_as_penalty)))

    penalty_term2 = quicksum(
        (quicksum(z[i, t, k] for k in range(N + 1)) - u[i, t]) ** 2
        for i in units for t in time_periods
    )

    QUBO = main_objective + final_penalty_term + penalty1 * penalty_term2
    Q = QUBO_to_qubo(QUBO, variable_names)

    return Q, variable_names

if __name__ == "__main__":
    ucp_data = {
        'num_units': 2,
        'num_time_periods': 5,
        'demand': [1, 2, 1, 2, 1],
        'p_min': [2, 1],
        'p_max': [3, 2],
        'variable_cost': [30, 45],
        'startup_cost': [50, 25],
        'N': 2
    }

    Q_matrix, variable_names = ucp_to_qubo_matrix(ucp_data)

    print("Symmetric QUBO matrix:")
    print(Q_matrix)
    print("\nMatrix dimension:", Q_matrix.shape)
    print("\nVariable ordering:")
    print(variable_names)

    size = len(variable_names)
    qubo_dict = {(i, j): Q_matrix[i][j] for i in range(size) for j in range(size) if Q_matrix[i][j] != 0}
    with open("qubo_matrix_only.json", 'w') as f:
        json.dump({f"{i},{j}": val for (i, j), val in qubo_dict.items()}, f, indent=4)
    print("QUBO matrix saved to qubo_matrix_only.json")
