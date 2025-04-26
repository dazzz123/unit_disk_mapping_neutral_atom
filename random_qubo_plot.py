import matplotlib.pyplot as plt
import numpy as np

# Data
qubos = ['R10', 'R15', 'R20', 'R25', 'R40']
runs = [
    [-5.5, -7, -5.5, -5.5, -5.5],  # Q10
    [-10.5, -9, -10.5, -7, -7],      # Q15
    [-10, -12, -12, -11.5, -10],     # Q20
    [-14.5, -15, -20, -17, -16.5, -14, -19.5],# Q25
     [ -9,-12,-20,-16,-15]#Q40
]
real_answers = [-9, -12, -15, -21,-17]

# Calculate means and variances
means = [np.mean(run) for run in runs]
variances = [np.var(run) for run in runs]

# Plotting
plt.figure(figsize=(10, 6))

# Plot mean energies with error bars (variance)
plt.errorbar(qubos, means, yerr=np.sqrt(variances), fmt='o-', label='GP-NAQC', capsize=5, color='orange')

# Plot real answers
plt.plot(qubos, real_answers, 's-', label='Simulated Annealing', color='green')

# Customize plot
plt.xlabel('QUBO Problems')
plt.ylabel('Energy')
plt.title('Comparison of Algorithm Energies vs Real Solutions')
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig('qubo_energy_comparison.png')
