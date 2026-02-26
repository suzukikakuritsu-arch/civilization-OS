import numpy as np


class SphereDynamics:
    """
    Multiplicative Gaussian dynamics on the unit sphere.
    w_{t+1} = normalize(w_t * exp(-phi * epsilon_t))
    """

    def __init__(self, dim=1000, phi=1.61803398875, sigma=0.05):
        self.dim = dim
        self.phi = phi
        self.sigma = sigma

    def normalize(self, w):
        norm = np.linalg.norm(w)
        return w / norm if norm != 0 else w

    def step(self, w):
        noise = np.random.normal(0, self.sigma, self.dim)
        w_new = w * np.exp(-self.phi * noise)
        return self.normalize(w_new)

    def stability_metric(self, w):
        return 1.0 / (1.0 + np.var(w))

    def run(self, iterations=1000, seed=None):
        if seed is not None:
            np.random.seed(seed)

        w = self.normalize(np.random.randn(self.dim))

        stabilities = []

        for _ in range(iterations):
            w = self.step(w)
            stabilities.append(self.stability_metric(w))

        return np.array(stabilities)
