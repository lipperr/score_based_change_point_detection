import numpy as np
from algorithms.ew import ExpWeightedForecaster

class FLH(ExpWeightedForecaster):
    def __init__(self, alpha=1, ew_params={}, threshold=np.inf):
        self.alpha = alpha
        self.threshold = threshold

        super().__init__(**ew_params)

    def compute_test_stat(self, samples):
        # self.alpha = self.eta
        super().restart()
        self.weights = []
        self.FLH_cumloss = np.zeros(1)
        self.FLH_predictions = []

        samples = np.array(samples).reshape(len(samples), self.basis.xdim, 1)
        n_samples = len(samples)
        stopping_time = -1
        self.weights = np.append(self.weights, [1])

        for t in range(n_samples):
            x = samples[t]
            expert_losses = np.empty(0, dtype=float)
            expert_predictions = []
            EW_loss_0, EW_pred_0 = super().ewstep(x)
            expert_predictions.append(EW_pred_0)
            expert_losses = np.append(expert_losses, EW_loss_0.item())

            for expert_idx in range(1, t+1):
                expert_predictions.append(super().predict_EW(expert_idx, t))
                expert_losses = np.append(expert_losses, super().compute_loss(expert_predictions[-1]))
            expert_predictions = np.array(expert_predictions)

            FLH_pred = np.sum(self.weights[:, None, None] * expert_predictions, axis=0)
            assert FLH_pred.shape == EW_pred_0.shape
            self.FLH_predictions.append(FLH_pred)

            FLH_loss = super().compute_loss(self.FLH_predictions[-1])
            self.FLH_cumloss = np.append(self.FLH_cumloss, self.FLH_cumloss[-1] + FLH_loss)

            if self.EW_cumloss[-1] - self.FLH_cumloss[-1] > self.threshold:
                stopping_time = t
                break
            
            self.weights = (1+t)/(1+t+1) * self.weights * np.exp(-self.alpha*expert_losses) / np.sum(self.weights * np.exp(-self.alpha*expert_losses))
            self.weights = np.append(self.weights, 1/(1+t+1))

        self.test_statistic = np.array(self.EW_cumloss[1:]) - np.array(self.FLH_cumloss[1:]).flatten()
        return self.test_statistic, stopping_time



