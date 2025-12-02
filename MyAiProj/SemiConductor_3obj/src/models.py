import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement
from botorch.models.transforms import Standardize
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning

from src.utils import convert_to_list


class FormulaOptimizer:
    def __init__(self, parameters, output_dir: str, fig_dir: str, device: torch.device = None, seed=42):
        """
        :param parameters: dict, {"name": (min, max, step)}
        :param output_dir: the location of the output files
        """
        self.parameters = parameters
        self.param_names = list(parameters.keys())
        self.output_dir = output_dir
        self.fig_dir = fig_dir
        self.experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.device = device

        # The bound of parameters
        self.bounds = torch.tensor([
            [param[0] for param in parameters.values()],
            [param[1] for param in parameters.values()]
        ], dtype=torch.float64, device=self.device)

        self.steps = torch.tensor([param[2] for param in parameters.values()], device=self.device)

        # Store experiment data
        self.X = torch.empty((0, len(parameters)), dtype=torch.float64, device=self.device)
        # Three objectives: 2 continuous and 1 discrete
        self.Y = torch.empty((0, 3), dtype=torch.float64, device=self.device)
        self.history = pd.DataFrame(columns=self.param_names + ["Coating efficiency", "Uniformity",
                                                                "Discrete objective", "Timestamp"])

        # Maximize all objectives
        self.ref_point = torch.tensor([-0.1, -0.1, -0.1], dtype=torch.float64, device=self.device)

        self.num_restarts = 20
        self.raw_samples = 64
        self.batch_size = 10
        self.n_init = 5

        # record list
        self.iteration_history = []  # store full record at every iteration
        self.hypervolume_history = []  # store hyper volume

        self.seed = seed
        self._set_seed(seed)

    @staticmethod
    def _set_seed(seed):
        """
        Set random seed for reproducibility
        :param seed: (int): Random seed
        :return:
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def generate_initial_samples(self, n_init, seed):
        """Generate initial samples for optimization"""
        sobel_samples = draw_sobol_samples(bounds=self.bounds, n=n_init, q=1, seed=self.seed).squeeze(1)

        # Discretization
        for i in range(len(self.parameters)):
            sobel_samples[:, i] = torch.round(sobel_samples[:, i] / self.steps[i]) * self.steps[i]
            sobel_samples[:, i] = torch.clamp(sobel_samples[:, i], self.bounds[0, i], self.bounds[1, i])

        return sobel_samples.to(self.device)

    def initialize_model(self):
        train_x = normalize(self.X, self.bounds)

        # Every objective corresponds to SingleTaskGP
        gp1 = SingleTaskGP(
            train_x,
            self.Y[:, 0:1],
            outcome_transform=Standardize(m=1),
        )

        gp2 = SingleTaskGP(
            train_x,
            self.Y[:, 1:2],
            outcome_transform=Standardize(m=1),
        )

        gp3 = SingleTaskGP(
            train_x,
            self.Y[:, 2:3],
            outcome_transform=Standardize(m=1),
        )

        model = ModelListGP(gp1, gp2, gp3)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def simulate_multi_objective_experiment(self, x):
        if len(x.shape) != 2:
            raise ValueError(f"Candidates must be 2D tensor, got shape {x.shape}")

        obj1 = (x[:, 0] * 0.05 + x[:, 1] * 0.07 + x[:, 3] * 0.02 + 0.2 * torch.sin(x[:, 0] + x[:, 1])).unsqueeze(1)
        obj2 = (x[:, 2] * 0.15 + x[:, 4] * 0.2 + x[:, 3] * 0.03 + 0.15 * torch.cos(x[:, 2] + x[:, 4])).unsqueeze(1)
        # Scale to avoid excessive clipping, keep in [0, 1]
        obj1 = (obj1 - obj1.min()) / (obj1.max() - obj1.min() + 1e-6)  # Normalize to [0, 1]
        obj2 = (obj2 - obj2.min()) / (obj2.max() - obj2.min() + 1e-6)

        # Binary objective (strictly 0 or 1)
        binary_obj = ((x[:, 0] > 0.5) & (x[:, 1] > 0.3)).float().unsqueeze(1)
        result = torch.cat([obj1, obj2, binary_obj], dim=-1).to(self.device)
        return result

    def get_human_input(self, candidates):
        """
        Input experimental results for processing batch candidate points
        :param candidates: Tensor (batch_size, 5)
        :return: Tensor (batch_size, 2)
        """
        if len(candidates.shape) != 2:
            raise ValueError(f"Candidates must be 2D tensor, got shape {candidates.shape}")

        batch_results = []
        candidates_np = candidates.cpu().numpy()

        for i in range(candidates.shape[0]):
            print(f"\n=== Experiment {i+1}/{candidates.shape[0]} ===")
            print("Parameters:")
            for name, value in zip(self.param_names, candidates_np[i]):
                print(f"{name}: {value:.4f}")

            while True:
                try:
                    user_input = input("Enter three results (0-1 for first two, 0 or 1 for third, comma-seperated): ").strip()
                    result1, result2, result3 = map(float, user_input.split(','))

                    if not (0 <= result1 <= 1 and 0 <= result2 <= 1 and result3 in [0, 1]):
                        raise ValueError("First two values must be between 0 and 1, third must be 0 or 1")

                    batch_results.append([result1, result2, result3])
                    break
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Invalid input: {e}. Please try again.")

        res = torch.tensor(batch_results, dtype=torch.float64, device=self.device)
        return res

    def _compute_hypervolume(self):
        bd = DominatedPartitioning(ref_point=self.ref_point, Y=self.Y)
        return bd.compute_hypervolume().item()

    def _save_iteration_data(self, record):
        filename = f"{self.output_dir}/optimization_history_{self.experiment_id}.json"
        with open(filename, 'a') as f:
            json.dump(record, f)
            f.write("\n")

    def save_experiment_data(self, x, y):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        x_cpu = x.cpu().numpy() if x.ndim == 2 else x.unsqueeze(0).cpu.numpy()
        y_cpu = y.cpu().numpy() if y.ndim == 2 else y.unsqueeze(0).cpu().numpy()

        new_rows = []
        for i in range(x_cpu.shape[0]):
            data = {name: val for name, val in zip(self.param_names, x_cpu[i])}
            data.update({
                "Coating efficiency": y_cpu[i, 0],
                "Uniformity": y_cpu[i, 1],
                "Binary Objective": y_cpu[i, 2],
                "Timestamp": timestamp
            })
            new_rows.append(data)
        new_data = pd.DataFrame(new_rows, columns=self.history.columns)
        if self.history.empty:
            self.history = new_data
        else:
            self.history = pd.concat([self.history, new_data], ignore_index=True)
        filename = f"{self.output_dir}/experiment_{self.experiment_id}.csv"
        self.history.to_csv(filename, index=False)

    def get_pareto_front(self):
        """Calculate the Pareto front from current observations."""
        pareto_mask = torch.ones(self.Y.shape[0], dtype=torch.bool)
        for i in range(self.Y.shape[0]):
            for j in range(self.Y.shape[0]):
                if i != j and torch.all(self.Y[j] >= self.Y[i]) and torch.any(self.Y[j] > self.Y[i]):
                    pareto_mask[i] = False
                    break
        return self.X[pareto_mask], self.Y[pareto_mask]

    def _record_iteration(self, iteration, candidates, acquisition_values=None):
        pareto_x , pareto_y = self.get_pareto_front()
        record = {
            "iteration": iteration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'X': self.X.cpu().numpy().tolist(),
            'Y': self.Y.cpu().numpy().tolist(),
            "candidates": candidates.cpu().numpy().tolist(),
            "hypervolume": self._compute_hypervolume(),
            "pareto_front": {
                "X": pareto_x.cpu().numpy().tolist(),
                "Y": pareto_y.cpu().numpy().tolist(),
            },
            "acquisition_values": convert_to_list(acquisition_values),
        }

        self.iteration_history.append(record)
        self.hypervolume_history.append(record["hypervolume"])

        self._save_iteration_data(record)

    def optimize(self, n_iter=5, simulation_flag=True):
        self._set_seed(self.seed)
        X_init = self.generate_initial_samples(n_init=self.n_init, seed=self.seed)
        assert len(X_init.shape) == 2, f"X_init shape: {X_init.shape}, not a 2d tensor!"
        print("=== Initialize experiments===")

        for candidate in X_init:
            candidate = candidate.unsqueeze(0)  # Add batch dimension

            if simulation_flag:
                y = self.simulate_multi_objective_experiment(candidate)
            else:
                y = self.get_human_input(candidate)
            self.X = torch.cat([self.X, candidate])
            self.Y = torch.cat([self.Y, y])
            self.save_experiment_data(candidate, y)

        self._record_iteration(
            iteration=0,
            candidates=X_init,
            acquisition_values=None,
        )

        print("=== Optimization phase ===")
        standard_bounds = torch.zeros_like(self.bounds, device=self.device)
        standard_bounds[1, :] = 1.0

        for i in range(1, n_iter + 1):
            print(f"\nIteration {i}/{n_iter}")

            mll, model = self.initialize_model()
            fit_gpytorch_mll(mll)

            # Calculate acquisition function
            train_x = normalize(self.X, self.bounds)
            with torch.no_grad():
                pred = model.posterior(train_x).mean

            partitioning = NondominatedPartitioning(ref_point=self.ref_point, Y=pred)
            acq_func = qLogExpectedHypervolumeImprovement(model=model, ref_point=self.ref_point, partitioning=partitioning)

            candidates, acq_values = optimize_acqf(
                acq_function=acq_func,
                bounds=standard_bounds,
                q=self.batch_size,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                options={"batch_limit": 5, "maxiter": 200, "seed": self.seed},
                sequential=True
            )

            # Discretize and clamp candidates
            candidates = unnormalize(candidates, self.bounds)
            for j in range(len(self.parameters)):
                candidates[:, j] = torch.round(candidates[:, j] / self.steps[j]) * self.steps[j]
                candidates[:, j] = torch.clamp(candidates[:, j], self.bounds[0, j], self.bounds[1, j])

            if simulation_flag:
                y_new = self.simulate_multi_objective_experiment(candidates)
            else:
                y_new = self.get_human_input(candidates)

            # Update data
            self.X = torch.cat([self.X, candidates])
            self.Y = torch.cat([self.Y, y_new])
            self.save_experiment_data(candidates, y_new)

            self._record_iteration(
                iteration=i,
                candidates=candidates,
                acquisition_values=acq_values,
            )

            # Compute Hyper volume
            hv = self._compute_hypervolume()
            print(f"Current hyper volume: {hv:.4f}")

    def plot_pareto_front(self):
        """Plot pareto front including all three objectives"""
        # obtain the non-dominated solution
        objectives = self.Y.cpu().numpy()
        pareto_mask = torch.ones(objectives.shape[0], dtype=torch.bool)

        for i in range(objectives.shape[0]):
            for j in range(objectives.shape[0]):
                if i != j and torch.all(self.Y[j] >= self.Y[i]) and torch.any(self.Y[j] > self.Y[i]):
                    pareto_mask[i] = False
                    break

        pareto_front = self.Y[pareto_mask].cpu().numpy()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(self.Y[:, 0], self.Y[:, 1], c='gray', alpha=0.3, label='All experiments')
        ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                   c="gray", alpha=0.4, label="All experiments")

        if len(pareto_front) > 0:
            # sorted_idx = torch.argsort(pareto_front[:, 0])
            # plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='red', label='pareto front')
            # plt.plot(pareto_front[sorted_idx, 0], pareto_front[sorted_idx, 1], 'r--')
            ax.scatter(
                pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
                c='red', label="Pareto front"
            )

        ax.set_xlabel('Coating efficiency')
        ax.set_ylabel('Uniformity')
        ax.set_zlabel('Binary objective')
        ax.set_zticks([0, 1])
        ax.set_title('Pareto front of formula optimization')
        ax.legend()
        ax.grid(True)
        fig_name = f"{self.fig_dir}/experiment_{self.experiment_id}.png"
        plt.savefig(fig_name)
        plt.close(fig)
        # plt.show()
