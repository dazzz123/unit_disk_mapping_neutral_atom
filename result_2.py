

# Run Julia script to generate input data
subprocess.run(["/Applications/Julia-1.11.app/Contents/Resources/julia/bin/julia", "mapping.jl"])

# Load Julia output
with open("julia_output.json", "r") as f:
    julia_data = json.load(f)

# Start cost tracking
tracker = Tracker().start()

# Initialize simulator
simulator = LocalSimulator("braket_ahs")

# Define parameters
omega_max = 15800000.0  # rad / seconds
omega_slew_rate_max = 250000000000000.0  # rad / seconds / seconds
tmin = 5e-08  # seconds
local_detuning_max = (2 * np.pi * 7.5) * 1e6  # rad / seconds
local_detuning_slew_rate_max = (2 * np.pi * 7.5) * 1e12  # rad / seconds / seconds
t_ramp_omega = omega_max / omega_slew_rate_max  # seconds
t_ramp_local_detuning = local_detuning_max / local_detuning_slew_rate_max  # seconds
detuning_max = 125000000.0  # rad / seconds

# Process matches
coordinates = [tuple(entry["loc"]) for entry in julia_data]
weights = [entry["weight"] for entry in julia_data]

# Output results
print("Coordinates:", coordinates)
print("Weights:", weights)

# Get min and max coordinates
x_coords, y_coords = zip(*coordinates)
min_x, max_x = min(x_coords), max(x_coords)
min_y, max_y = min(y_coords), max(y_coords)

# Define tile parameters
tile_size = 4.0  # Adjust based on node density (e.g., 3.0 if >20 nodes/tile)
overlap = 1.5  # Equal to blockade radius
blockade_radius = 1.5

# Create tiles with overlap
tile_step = tile_size - overlap
x_tiles = np.arange(min_x, max_x + tile_size, tile_step)
y_tiles = np.arange(min_y, max_y + tile_size, tile_step)
tiles = {}

for i, x_start in enumerate(x_tiles):
    for j, y_start in enumerate(y_tiles):
        tile_key = (i, j)
        x_min, x_max = x_start, x_start + tile_size + overlap
        y_min, y_max = y_start, y_start + tile_size + overlap
        tile_nodes = []
        tile_weights = []
        for idx, (x, y) in enumerate(coordinates):
            if x_min <= x <= x_max and y_min <= y <= y_max:
                tile_nodes.append((x, y))
                tile_weights.append(weights[idx])
        if tile_nodes and len(tile_nodes) <= 20:  # Ensure <=20 nodes
            tiles[tile_key] = {'coordinates': tile_nodes, 'weights': tile_weights}
        elif tile_nodes:
            print(f"Warning: Tile {tile_key} has {len(tile_nodes)} nodes, skipping or adjust tile_size")

# Store MWIS solutions for each tile
tile_mwis_solutions = {}

# Process each tile (reuse box calculation logic)
for tile_key, data in tiles.items():
    x_range = (tile_key[0]*tile_step, tile_key[0]*tile_step + tile_size)
    y_range = (tile_key[1]*tile_step, tile_key[1]*tile_step + tile_size)
    
    if not data['coordinates']:
        print(f"\nSkipping tile at x={x_range}, y={y_range} (no coordinates)")
        continue
    
    print(f"\nProcessing tile at x={x_range}, y={y_range}:")
    print(f"Coordinates: {data['coordinates']}")
    print(f"Weights: {data['weights']}")
    a = 1.0
    register = AtomArrangement()
    for x, y in data['coordinates']:
        register.add([float(x)*a, float(y)*a])
    
    min_weight = min(data['weights'])
    max_weight = max(data['weights'])
    if max_weight == min_weight:
        normalized_weights = [0.5 for _ in data['weights']]
    else:
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in data['weights']]
    print("Normalized Weights:", normalized_weights)
    
    t_max = 10.0e-6
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
  
    result = simulator.run(program, shots=5000, steps=50, blockade_radius=1.5).result()
    
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
    
    tile_mwis_solutions[tile_key] = {
        'coordinates': data['coordinates'],
        'weights': data['weights'],
        'mwis_binary': binary_mwis
    }

# Combine MWIS solutions with conflict-aware greedy merging
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Collect all selected nodes
all_selected_nodes = []
for tile_key, solution in tile_mwis_solutions.items():
    for i, (coord, weight) in enumerate(zip(solution['coordinates'], solution['weights'])):
        binary = solution['mwis_binary'][i]
        if binary == 1:
            all_selected_nodes.append((coord, weight, tile_key, i))

# Multiple runs to find best global MWIS
num_runs = 10
best_global_mwis = []
best_global_mwis_binary = [0] * len(coordinates)
best_total_weight = 0

for run in range(num_runs):
    # Sort nodes by weight (descending)
    np.random.shuffle(all_selected_nodes)  # Randomize for tie-breaking
    sorted_nodes = sorted(all_selected_nodes, key=lambda x: x[1], reverse=True)
    
    # Greedy selection
    global_mwis = []
    global_mwis_binary = [0] * len(coordinates)
    selected_coords = set()
    
    for coord, weight, tile_key, idx in sorted_nodes:
        if coord in selected_coords:
            continue
        # Check independence
        is_independent = True
        for selected_coord, _ in global_mwis:
            if distance(coord, selected_coord) <= blockade_radius:
                is_independent = False
                break
        if is_independent:
            global_mwis.append((coord, weight))
            global_idx = coordinates.index(coord)
            global_mwis_binary[global_idx] = 1
            selected_coords.add(coord)
    
    # Update best solution
    total_weight = sum(weight for _, weight in global_mwis)
    if total_weight > best_total_weight:
        best_global_mwis = global_mwis
        best_global_mwis_binary = global_mwis_binary
        best_total_weight = total_weight

# Use best solution
global_mwis = best_global_mwis
global_mwis_binary = best_global_mwis_binary
total_weight = best_total_weight

# Print global MWIS results
print("\nGlobal MWIS with Conflict-Aware Greedy Merging:")
print("Selected Nodes and Weights:", global_mwis)
print(f"Total Weight of Global MWIS: {total_weight}")
print(f"Global MWIS Binary Solution (0s and 1s, length={len(global_mwis_binary)}):")
print(global_mwis_binary)

# Save global MWIS binary solution
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
plt.savefig('mwis_plot.png')  # Save plot instead of showing
# plt.show()  # Commented out as per guidelines

# Run Julia answer script
subprocess.run(["/Applications/Julia-1.11.app/Contents/Resources/julia/bin/julia", "answer.jl"])
