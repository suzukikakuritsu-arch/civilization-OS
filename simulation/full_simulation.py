import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# -------------------------
# Metrics
# -------------------------

def gini(x):
    diff_sum = np.sum(np.abs(x[:, None] - x[None, :]))
    return diff_sum / (2 * len(x)**2 * np.mean(x))

# -------------------------
# Civilization OS
# -------------------------

class CivilizationOS:
    def __init__(
        self,
        population=300,
        growth_rate=1.02,
        redistribution_rate=0.1,
        resource_capacity=10000,
        regeneration_rate=0.01,
        seed=42
    ):
        np.random.seed(seed)

        self.population = population
        self.growth_rate = growth_rate
        self.redistribution_rate = redistribution_rate

        self.wealth = np.random.exponential(scale=10, size=population)
        self.trust = np.random.uniform(0.4, 0.6, size=population)

        self.resources = resource_capacity
        self.resource_capacity = resource_capacity
        self.regeneration_rate = regeneration_rate

    # --- Economy ---
    def economic_step(self):
        growth = np.random.normal(self.growth_rate, 0.01, self.population)
        self.wealth *= growth

        avg = np.mean(self.wealth)
        redistribution = self.redistribution_rate * (avg - self.wealth)
        self.wealth += redistribution

    # --- Resources ---
    def resource_step(self):
        consumption = np.sum(self.wealth) * 0.001
        self.resources -= consumption

        regeneration = self.resource_capacity * self.regeneration_rate
        self.resources += regeneration

        self.resources = max(0, min(self.resources, self.resource_capacity))

    # --- Trust ---
    def trust_step(self):
        relative = self.wealth / np.mean(self.wealth)
        self.trust = 0.5 * self.trust + 0.5 * (1 / (1 + np.exp(-relative + 1)))

    # --- Step ---
    def step(self):
        self.economic_step()
        self.resource_step()
        self.trust_step()

    # --- Metrics ---
    def metrics(self):
        return (
            np.mean(self.wealth),
            gini(self.wealth),
            self.resources / self.resource_capacity,
            np.mean(self.trust)
        )

# -------------------------
# Single Simulation Run
# -------------------------

def run_simulation(params, steps=300):
    civ = CivilizationOS(**params)

    history = {
        "wealth": [],
        "gini": [],
        "resources": [],
        "trust": []
    }

    for _ in range(steps):
        civ.step()
        m = civ.metrics()
        history["wealth"].append(m[0])
        history["gini"].append(m[1])
        history["resources"].append(m[2])
        history["trust"].append(m[3])

    return history

# -------------------------
# Stability Detection
# -------------------------

def is_stable(history):
    if history["resources"][-1] <= 0.05:
        return False
    if history["gini"][-1] > 0.6:
        return False
    return True

# -------------------------
# Parameter Sweep
# -------------------------

def parameter_search():
    growth_rates = [1.01, 1.02, 1.03]
    redistribution_rates = [0.05, 0.1, 0.2]
    regeneration_rates = [0.005, 0.01, 0.02]

    stable_configs = []

    for g, r, regen in product(growth_rates, redistribution_rates, regeneration_rates):
        params = {
            "growth_rate": g,
            "redistribution_rate": r,
            "regeneration_rate": regen
        }

        history = run_simulation(params)

        if is_stable(history):
            stable_configs.append(params)

    return stable_configs

# -------------------------
# Visualization
# -------------------------

def plot_history(history):
    plt.figure()
    plt.plot(history["wealth"])
    plt.title("Mean Wealth Over Time")
    plt.xlabel("Step")
    plt.ylabel("Wealth")
    plt.show()

    plt.figure()
    plt.plot(history["gini"])
    plt.title("Gini Coefficient")
    plt.xlabel("Step")
    plt.ylabel("Gini")
    plt.show()

    plt.figure()
    plt.plot(history["resources"])
    plt.title("Resource Ratio")
    plt.xlabel("Step")
    plt.ylabel("Resources")
    plt.show()

    plt.figure()
    plt.plot(history["trust"])
    plt.title("Mean Trust")
    plt.xlabel("Step")
    plt.ylabel("Trust")
    plt.show()

# -------------------------
# Main Execution
# -------------------------

if __name__ == "__main__":

    # 1. 単体実行
    params = {
        "growth_rate": 1.02,
        "redistribution_rate": 0.1,
        "regeneration_rate": 0.01
    }

    history = run_simulation(params)
    plot_history(history)

    # 2. 安定領域探索
    stable = parameter_search()

    print("Stable Configurations Found:")
    for s in stable:
        print(s)
