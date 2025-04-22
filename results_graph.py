import matplotlib.pyplot as plt
import numpy as np

# Data
qubos = ['Q10',  'Q20',  'Q30', 'Q40', 'Q50' ]
runs = [
    [-0.313, -0.313, -0.313, -0.246, -0.313],  # Q10
    [-0.325, -0.315, -0.342, -0.315, -0.246],      # Q20
    [-0.413, -0.375, -0.329, -0.319, -0.379],
         [-0.068,-0.086,-0.09,-0.09,-0.086],
              [-0.13,-0.096,-0.15,-0.16,-0.19]     # Q30
]
real_answers = [-0.52, -0.63, -0.53, -0.19,-0.15]

# Calculate means and variances
means = [np.mean(run) for run in runs]
variances = [np.var(run) for run in runs]

# Plotting
plt.figure(figsize=(10, 10))

# Plot mean energies with error bars (variance)
plt.errorbar(qubos, means, yerr=np.sqrt(variances), fmt='o-', label='Algorithm Mean Energy', capsize=5, color='blue')

# Plot real answers
plt.plot(qubos, real_answers, 's-', label='Real Energy', color='red')

# Customize plot
plt.xlabel('QUBO Problems')
plt.ylabel('Energy')
plt.title('Comparison of Algorithm Energies vs Real Solutions')
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig('qubo_energy_comparison.png')
