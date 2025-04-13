subprocess.run(["julia", "mapping.jl"])

with open("julia_output.json", "r") as f:
    julia_data = json.load(f)

# Use Braket SDK Cost Tracking to estimate the cost to run this example
from braket.tracking import Tracker

tracker = Tracker().start()

import numpy as np
#from ahs_utils import show_drive_and_local_detuning, show_final_avg_density, show_register

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.driving_field import DrivingField
from braket.ahs.local_detuning import LocalDetuning
from braket.devices import LocalSimulator

simulator = LocalSimulator("braket_ahs")


omega_max = 15800000.0  # rad / seconds
omega_slew_rate_max = 250000000000000.0  # rad / seconds / seconds
tmin = 5e-08  # seconds

local_detuning_max = (2 * np.pi * 7.5) * 1e6  # rad / seconds
local_detuning_slew_rate_max = (2 * np.pi * 7.5) * 1e12  # rad / seconds / seconds

t_ramp_omega = omega_max / omega_slew_rate_max  # seconds
t_ramp_local_detuning = local_detuning_max / local_detuning_slew_rate_max  # seconds

detuning_max = 125000000.0  # rad / seconds


import matplotlib.pyplot as plt
import re
import numpy as np



# Process matches
coordinates = [(int(x), int(y)) for x, y, _ in julia_data['loc']]
weights = [float(weight) for _, _, weight in julia_data['weights']]

# Output results
print("Coordinates:", coordinates)
print("Weights:", weights)

# Get min and max coordinates
x_coords, y_coords = zip(*coordinates)
min_x, max_x = min(x_coords), max(x_coords)
min_y, max_y = min(y_coords), max(y_coords)

# Define spacing parameter
spacing = 4

# Calculate line positions
x_lines = np.arange(np.floor(min_x/spacing)*spacing + 0.5, np.ceil(max_x/spacing)*spacing + 0.5, spacing)
y_lines = np.arange(np.floor(min_y/spacing)*spacing + 0.5, np.ceil(max_y/spacing)*spacing + 0.5, spacing)

# Create boxes and separate coordinates and weights
boxes = {}
for (x, y), w in zip(coordinates, weights):
    box_x = int(np.floor((x - 0.5) / spacing))
    box_y = int(np.floor((y - 0.5) / spacing))
    box_key = (box_x, box_y)
    if box_key not in boxes:
        boxes[box_key] = {'coordinates': [], 'weights': []}
    boxes[box_key]['coordinates'].append((x, y))
    boxes[box_key]['weights'].append(w)

# Store MWIS solutions for each box
box_mwis_solutions = {}

# Process each box
for box_key, data in boxes.items():
    x_range = (box_key[0]*spacing + 0.5, (box_key[0]+1)*spacing + 0.5)
    y_range = (box_key[1]*spacing + 0.5, (box_key[1]+1)*spacing + 0.5)
    
    if not data['coordinates']:
        print(f"\nSkipping box at x={x_range}, y={y_range} (no coordinates)")
        continue
    
    print(f"\nProcessing box at x={x_range}, y={y_range}:")
    print(f"Coordinates: {data['coordinates']}")
    print(f"Weights: {data['weights']}")
    
    register = AtomArrangement()
    for x, y in data['coordinates']:
        register.add([float(x), float(y)])
    
    min_weight = min(data['weights'])
    max_weight = max(data['weights'])
    if max_weight == min_weight:
        normalized_weights = [0.5 for _ in data['weights']]
    else:
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in data['weights']]
    print("Normalized Weights:", normalized_weights)
    
    t_max = 6.0e-6
 
    
    time_points = [0, t_ramp_omega, t_max - t_ramp_local_detuning - t_ramp_omega, t_max - t_ramp_local_detuning, t_max]
    amplitude_values = [0, omega_max, omega_max, 0, 0]
    phase_values = [0 for _ in range(len(time_points))]
    detuning_0 = -detuning_max / 12
    detuning_slew_rate = local_detuning_max / (t_max - t_ramp_local_detuning)
    detuning_values = [
        detuning_0,
        detuning_0 + detuning_slew_rate * t_ramp_omega,
        detuning_0 + detuning_slew_rate * (t_max - t_ramp_local_detuning - t_ramp_omega),
        detuning_0 + detuning_slew_rate * (t_max - t_ramp_local_detuning),
        detuning_0 + detuning_slew_rate * (t_max - t_ramp_local_detuning),
    ]
    drive = DrivingField.from_lists(time_points, amplitude_values, detuning_values, phase_values)
    
    time_local_detuning = [0, t_max - t_ramp_local_detuning, t_max]
    value_local_detuning = [0, local_detuning_max, 0]
    pattern = [np.round(w, 2) for w in normalized_weights]
    local_detuning = LocalDetuning.from_lists(time_local_detuning, value_local_detuning, pattern)
    
    program = AnalogHamiltonianSimulation(hamiltonian=drive + local_detuning, register=register)
  
    result = simulator.run(program,shots=1000, steps=30, blockade_radius=1.5).result()
    
    def get_counters_from_result(result):
        post_sequences = [list(measurement.post_sequence) for measurement in result.measurements]
        post_sequences = ["".join(["r" if site == 0 else "g" for site in post_sequence]) for post_sequence in post_sequences]
        counters = {}
        for post_sequence in post_sequences:
            counters[post_sequence] = counters.get(post_sequence, 0) + 1
        return counters
    
    counters = get_counters_from_result(result)
    sorted_counters = sorted(counters.items(), key=lambda x: -x[1])
    config_largest_prob = sorted_counters[0][0]
    largest_prob = sorted_counters[0][1] / sum([item[1] for item in sorted_counters])
    
    print(f"The configuration with largest probability reads {config_largest_prob} with probability {largest_prob}")
    
    binary_mwis = [1 if char == 'r' else 0 for char in config_largest_prob]
    print("Binary MWIS Array:", binary_mwis)
    
    box_mwis_solutions[box_key] = {
        'coordinates': data['coordinates'],
        'weights': data['weights'],
        'mwis_binary': binary_mwis
    }

# Combine MWIS solutions and find global MWIS
all_selected_nodes = []
for box_key, solution in box_mwis_solutions.items():
    for i, (coord, weight) in enumerate(zip(solution['coordinates'], solution['weights'])):
        if solution['mwis_binary'][i] == 1:
            all_selected_nodes.append((coord, weight))

# Greedy MWIS algorithm for the full graph
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

blockade_radius = 1.5
global_mwis = []
all_selected_nodes.sort(key=lambda x: x[1], reverse=True)

for node, weight in all_selected_nodes:
    is_independent = True
    for selected_node, _ in global_mwis:
        if distance(node, selected_node) <= blockade_radius:
            is_independent = False
            break
    if is_independent:
        global_mwis.append((node, weight))

# Create binary solution for all 121 nodes
global_mwis_binary = [0] * len(coordinates)  # Initialize with zeros
global_mwis_coords = [node for node, _ in global_mwis]

for i, coord in enumerate(coordinates):
    if coord in global_mwis_coords:
        global_mwis_binary[i] = 1

# Print global MWIS results
print("\nGlobal MWIS:")
print("Selected Nodes and Weights:", global_mwis)
total_weight = sum(weight for _, weight in global_mwis)
print(f"Total Weight of Global MWIS: {total_weight}")
print(f"Global MWIS Binary Solution (0s and 1s, length={len(global_mwis_binary)}):")
print(global_mwis_binary)

with open("python_output.json", "w") as f:
    json.dump(global_mwis_binary, f)

# Visualize with MWIS highlighted
plt.scatter(x_coords, y_coords, s=100, c=weights, cmap='viridis', alpha=0.5, label='All Nodes')
mwis_x, mwis_y = zip(*[node for node, _ in global_mwis])
plt.scatter(mwis_x, mwis_y, s=150, c='red', marker='*', label='Global MWIS')
for x in x_lines:
    plt.axvline(x=x, color='r', linestyle='--', alpha=0.5)
for y in y_lines:
    plt.axhline(y=y, color='r', linestyle='--', alpha=0.5)
padding = 1
plt.xlim(min_x - padding, max_x + padding)
plt.ylim(min_y - padding, max_y + padding)
plt.grid(True, alpha=0.3)
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Grid Graph with Dividing Lines and Global MWIS (spacing=4)')
plt.colorbar(label='Weights')
plt.legend()
plt.show()

subprocess.run(["julia", "julia_part2.jl"])
