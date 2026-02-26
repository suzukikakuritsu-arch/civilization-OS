import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sphere_dynamics import SphereDynamics


DIM = 500
ITER = 1000
SEEDS = 30

model = SphereDynamics(dim=DIM)

all_curves = []

for seed in range(SEEDS):
    curve = model.run(iterations=ITER, seed=seed)
    all_curves.append(curve)

all_curves = np.array(all_curves)

mean_curve = np.mean(all_curves, axis=0)
std_curve = np.std(all_curves, axis=0)

# ---- 統計 ----
final_values = np.mean(all_curves[:, -100:], axis=1)

mean_final = np.mean(final_values)
ci = stats.t.interval(
    0.95,
    len(final_values) - 1,
    loc=mean_final,
    scale=stats.sem(final_values)
)

print("===== Experiment Result =====")
print(f"Mean Final Stability: {mean_final:.6f}")
print(f"95% Confidence Interval: {ci}")

# ---- 可視化 ----
plt.figure()
plt.plot(mean_curve)
plt.title("Mean Stability Over Time")
plt.xlabel("Iteration")
plt.ylabel("Stability")
plt.show()

plt.figure()
plt.hist(final_values, bins=10)
plt.title("Distribution of Final Stability")
plt.show()
