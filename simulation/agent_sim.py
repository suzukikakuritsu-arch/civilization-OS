import numpy as np

# -------------------------
# Metrics
# -------------------------

def gini(x):
    diff_sum = np.sum(np.abs(x[:, None] - x[None, :]))
    return diff_sum / (2 * len(x)**2 * np.mean(x))

def resource_stability(resources):
    return resources / np.max(resources)

# -------------------------
# Civilization OS
# -------------------------

class CivilizationOS:
    def __init__(
        self,
        population=500,
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

    # -------------------------
    # Economy Update
    # -------------------------

    def economic_step(self):
        growth = np.random.normal(self.growth_rate, 0.01, self.population)
        self.wealth *= growth

        avg = np.mean(self.wealth)
        redistribution = self.redistribution_rate * (avg - self.wealth)
        self.wealth += redistribution

    # -------------------------
    # Resource Dynamics
    # -------------------------

    def resource_step(self):
        consumption = np.sum(self.wealth) * 0.001
        self.resources -= consumption

        regeneration = self.resource_capacity * self.regeneration_rate
        self.resources += regeneration

        self.resources = max(0, min(self.resources, self.resource_capacity))

    # -------------------------
    # Trust Update (Decentralized)
    # -------------------------

    def trust_step(self):
        relative_wealth = self.wealth / np.mean(self.wealth)
        self.trust = 0.5 * self.trust + 0.5 * (1 / (1 + np.exp(-relative_wealth + 1)))

    # -------------------------
    # Full Step
    # -------------------------

    def step(self):
        self.economic_step()
        self.resource_step()
        self.trust_step()

    # -------------------------
    # Metrics
    # -------------------------

    def metrics(self):
        return {
            "mean_wealth": float(np.mean(self.wealth)),
            "gini": float(gini(self.wealth)),
            "resource_ratio": float(self.resources / self.resource_capacity),
            "mean_trust": float(np.mean(self.trust))
        }


# -------------------------
# Run Simulation
# -------------------------

if __name__ == "__main__":
    civ = CivilizationOS()

    history = []

    for t in range(300):
        civ.step()
        history.append(civ.metrics())

    for key in history[-1]:
        print(f"{key}: {history[-1][key]:.4f}")
