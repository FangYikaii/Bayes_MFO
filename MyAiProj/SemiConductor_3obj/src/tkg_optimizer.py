import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement
from botorch.models.transforms import Standardize
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from src.utils import convert_to_list


class TraceAwareKGOptimizer:
    """Trace-Aware Knowledge Gradient (taKG) Multi-fidelity Bayesian Optimization"""
    
    def __init__(self, output_dir, fig_dir, seed=42, device=None):
        """
        Initialize the Trace-Aware KG Optimizer
        
        Args:
            output_dir: Directory to save output files
            fig_dir: Directory to save figures
            seed: Random seed for reproducibility
            device: Torch device to use
        """
        self.output_dir = output_dir
        self.fig_dir = fig_dir
        self.experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.device = device if device is not None else torch.device('cpu')
        
        # Define parameter spaces based on user requirements
        self._define_parameter_spaces()
        
        # Store experiment data
        self.X = torch.empty((0, len(self.param_names)), dtype=torch.float64, device=self.device)
        self.Y = torch.empty((0, 3), dtype=torch.float64, device=self.device)  # Three objectives: uniformity, coverage, adhesion
        self.history = pd.DataFrame(columns=self.param_names + ["Uniformity", "Coverage", "Adhesion", "Timestamp"])
        
        # Multi-objective optimization settings
        self.ref_point = torch.tensor([-0.1, -0.1, -0.1], dtype=torch.float64, device=self.device)
        
        # Optimization hyperparameters
        self.num_restarts = 5  # Reduced from 20 to speed up computation
        self.raw_samples = 16  # Reduced from 64 to speed up computation
        self.batch_size = 3  # Reduced from 10 to speed up computation
        self.n_init = 5
        
        # Record lists
        self.iteration_history = []
        self.hypervolume_history = []
        
        # Optimization phases (1: simple systems, 2: complex systems)
        self.phase = 1
        self.phase_1_iterations = 3  # Number of iterations in phase 1
        
        # Independent iteration counter
        self.current_iteration = 0
        
        self.seed = seed
        self._set_seed(seed)
        
        print(f"【INFO】Initialized TraceAwareKGOptimizer with device: {self.device}")
        print(f"【INFO】Parameter space: {self.param_names}")
        print(f"【INFO】Bounds: {self.bounds}")
    
    def _define_parameter_spaces(self):
        """Define parameter spaces based on user requirements"""
        # Organic formula parameters
        self.organic_params = {
            'formula': (1, 30, 1),          # 1-30, step 1
            'concentration': (0.1, 5, 0.1),  # 0.1-5%, step 0.1
            'temperature': (25, 40, 5),      # 25-40°C, step 5
            'soak_time': (1, 30, 1),         # 1-30min, step 1
            'ph': (2.0, 14.0, 0.5),          # 2.0-14.0, step 0.5 (with safety constraints)
            'curing_time': (10, 30, 5)       # 10-30min, step 5
        }
        
        # Metal oxide parameters
        self.oxide_params = {
            'metal_a_type': (1, 20, 1),      # 1-20, step 1
            'metal_a_concentration': (10, 50, 10),  # 10-50%, step 10
            'metal_b_type': (0, 20, 1),      # 0-20, step 1 (0 means no metal B)
            'molar_ratio_b_a': (1, 10, 1)    # 1-10%, step 1
        }
        
        # Experiment condition parameter
        self.condition_params = {
            'experiment_condition': (1, 3, 1)  # 1-3, step 1 (1: organic only, 2: oxide only, 3: both)
        }
        
        # Combine all parameters
        self.parameters = {**self.organic_params, **self.oxide_params, **self.condition_params}
        self.param_names = list(self.parameters.keys())
        
        # Parameter bounds
        self.bounds = torch.tensor([
            [param[0] for param in self.parameters.values()],
            [param[1] for param in self.parameters.values()]
        ], dtype=torch.float64, device=self.device)
        
        # Parameter steps for discretization
        self.steps = torch.tensor([param[2] for param in self.parameters.values()], device=self.device)
        
        # Safety constraints for pH based on formula ID
        # Create a direct mapping from formula ID to pH range for better performance and readability
        self.pH_safety_constraints = {}
        
        # Epoxy silanes (ID 1-4): pH=4.0-6.0
        for formula_id in range(1, 5):
            self.pH_safety_constraints[formula_id] = (4.0, 6.0)
        
        # Active hydrogen silanes (ID 5-7)
        self.pH_safety_constraints[5] = (7.0, 10.5)  # AEAPTMS
        self.pH_safety_constraints[6] = (7.0, 10.5)  # APTES
        self.pH_safety_constraints[7] = (7.0, 10.5)  # APTMS
        
        # Linear self-assembled molecules (ID 8-13)
        self.pH_safety_constraints[8] = (3.5, 6.5)   # MPTES
        self.pH_safety_constraints[9] = (3.5, 6.5)   # 3-巯丙基三甲氧基硅烷
        self.pH_safety_constraints[10] = (4.5, 6.0)  # 全氟辛基三乙氧基硅烷
        self.pH_safety_constraints[11] = (2.0, 5.0)  # 四乙氧基硅烷(TEOS)
        self.pH_safety_constraints[12] = (4.0, 5.5)  # 甲基三氯硅烷
        self.pH_safety_constraints[13] = (4.0, 6.0)  # 乙烯基三甲氧基硅烷
        
        # Other silane compounds (ID 14-17)
        self.pH_safety_constraints[14] = (3.5, 4.5)  # γ-甲基丙烯酰氧基丙基三甲氧基硅烷
        self.pH_safety_constraints[15] = (6.0, 8.0)  # 3-叔丙基三甲氧基硅烷
        self.pH_safety_constraints[16] = (7.0, 10.0) # 哌嗪基丙基甲基二甲氧基硅烷
        self.pH_safety_constraints[17] = (5.0, 7.0)  # 乙酰氧基丙基三甲氧基硅烷
        
        # Other organic compounds (ID 18-21)
        self.pH_safety_constraints[18] = (2.0, 7.0)  # 二乙基磷酰乙基三乙氧基硅烷
        self.pH_safety_constraints[19] = (3.0, 7.0)  # 月桂酸
        self.pH_safety_constraints[20] = (7.0, 11.0) # 聚乙烯亚胺(PEI)
        self.pH_safety_constraints[21] = (3.0, 7.0)  # 聚丙烯酸(PAA)
        
        # Additional compounds (ID 22-30)
        self.pH_safety_constraints[22] = (4.0, 6.0)  # 双-[3-(三乙氧基硅基)丙基]-四硫化物(BTSPS)
        self.pH_safety_constraints[23] = (8.0, 10.0) # 双(三甲氧基硅基)丙酸
        self.pH_safety_constraints[24] = (4.0, 6.0)  # 1,2-双(三甲氧基硅基)乙烷
        self.pH_safety_constraints[25] = (4.0, 6.0)  # 1,3-双(三甲氧基硅基)丙烷
        self.pH_safety_constraints[26] = (9.0, 11.0) # 1,3-双(3-氨基丙基)-1,1,3,3-四甲基二硅氧烷
        self.pH_safety_constraints[27] = (8.0, 10.0) # 双端环氧丙基甲基硅氧烷(n≈10)
        self.pH_safety_constraints[28] = (6.0, 8.0)  # 双(三甲氧基硅基丙基)硫化物
        self.pH_safety_constraints[29] = (4.0, 6.0)  # 三(三甲氧基甲硅烷基丙基)异氰脲酸酯
        self.pH_safety_constraints[30] = (7.0, 9.0)  # 2,4,6,8-四甲基-2,4,6,8-四(丙基缩水甘油醚)环四硅氧烷
    
    @staticmethod
    def _set_seed(seed):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _apply_safety_constraints(self, candidates):
        """Apply safety constraints to candidates"""
        for i in range(candidates.shape[0]):
            formula_id = int(round(candidates[i, 0].item()))
            
            # Apply pH safety constraints based on formula ID
            if formula_id in self.pH_safety_constraints:
                min_ph, max_ph = self.pH_safety_constraints[formula_id]
                candidates[i, 4] = torch.clamp(candidates[i, 4], min_ph, max_ph)
            
            # Apply metal oxide constraints: metal A and B cannot be the same type
            metal_a_type = int(round(candidates[i, 6].item()))
            metal_b_type = int(round(candidates[i, 8].item()))
            if metal_b_type != 0 and metal_a_type == metal_b_type:
                # If same, adjust metal B type to a different value
                candidates[i, 8] = torch.clamp(
                    candidates[i, 8] + 1 if candidates[i, 8] < self.bounds[1, 8] else candidates[i, 8] - 1,
                    self.bounds[0, 8], 
                    self.bounds[1, 8]
                )
        
        return candidates
    
    def _apply_phase_constraints(self, candidates):
        """Apply phase-specific constraints"""
        if self.phase == 1:
            # Phase 1: only simple systems (condition 1 or 2, not 3)
            for i in range(candidates.shape[0]):
                # Randomly assign to condition 1 or 2
                condition = 1 if torch.rand(1) > 0.5 else 2
                candidates[i, -1] = condition
        
        return candidates
    
    def generate_initial_samples(self, n_init=None):
        """Generate initial samples"""
        n = n_init if n_init is not None else self.n_init
        sobel_samples = draw_sobol_samples(bounds=self.bounds, n=n, q=1, seed=self.seed).squeeze(1).to(self.device)
        
        # Discretization and constraints
        for i in range(len(self.parameters)):
            sobel_samples[:, i] = torch.round(sobel_samples[:, i] / self.steps[i]) * self.steps[i]
            sobel_samples[:, i] = torch.clamp(sobel_samples[:, i], self.bounds[0, i], self.bounds[1, i])
        
        # Apply safety and phase constraints
        sobel_samples = self._apply_safety_constraints(sobel_samples)
        sobel_samples = self._apply_phase_constraints(sobel_samples)
        
        return sobel_samples
    
    def initialize_model(self):
        """Initialize GP models for each objective"""
        train_x = normalize(self.X, self.bounds)
        
        # Create separate GP for each objective
        gp1 = SingleTaskGP(
            train_x,
            self.Y[:, 0:1],
            outcome_transform=Standardize(m=1),
        ).to(self.device)
        
        gp2 = SingleTaskGP(
            train_x,
            self.Y[:, 1:2],
            outcome_transform=Standardize(m=1),
        ).to(self.device)
        
        gp3 = SingleTaskGP(
            train_x,
            self.Y[:, 2:3],
            outcome_transform=Standardize(m=1),
        ).to(self.device)
        
        model = ModelListGP(gp1, gp2, gp3).to(self.device)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model
    
    def _compute_trace_aware_knowledge_gradient(self, model, candidates):
        """Compute Trace-Aware Knowledge Gradient acquisition function"""
        # This is a simplified implementation of taKG
        # In practice, taKG would consider the trace of model performance across fidelities
        
        # For now, we'll use qLogExpectedHypervolumeImprovement with trace-aware modifications
        train_x = normalize(self.X, self.bounds)
        with torch.no_grad():
            pred = model.posterior(train_x).mean
        
        partitioning = NondominatedPartitioning(ref_point=self.ref_point, Y=pred)
        acq_func = qLogExpectedHypervolumeImprovement(
            model=model, 
            ref_point=self.ref_point, 
            partitioning=partitioning
        )
        
        return acq_func
    
    def simulate_experiment(self, x):
        """Simulate experiment results"""
        if len(x.shape) != 2:
            raise ValueError(f"Candidates must be 2D tensor, got shape {x.shape}")
        
        # Extract parameters
        formula = x[:, 0]
        concentration = x[:, 1]
        temperature = x[:, 2]
        soak_time = x[:, 3]
        ph = x[:, 4]
        curing_time = x[:, 5]
        metal_a_type = x[:, 6]
        metal_a_concentration = x[:, 7]
        metal_b_type = x[:, 8]
        molar_ratio = x[:, 9]
        condition = x[:, 10]
        
        # Simulate uniformity (0-1)
        uniformity = (0.3 + 0.02 * formula + 0.05 * concentration + 0.01 * temperature + 0.01 * soak_time + 0.02 * ph + 0.01 * curing_time)
        uniformity += 0.1 * torch.sin(formula + concentration + temperature)
        
        # Simulate coverage (0-1)
        coverage = (0.4 + 0.03 * formula + 0.04 * concentration + 0.02 * temperature + 0.02 * soak_time + 0.01 * ph + 0.01 * curing_time)
        coverage += 0.15 * torch.cos(temperature + ph + curing_time)
        
        # Simulate adhesion (0-1)
        adhesion = (0.35 + 0.02 * formula + 0.03 * concentration + 0.02 * temperature + 0.01 * soak_time + 0.02 * ph + 0.01 * curing_time)
        adhesion += 0.1 * torch.sin(soak_time + ph + curing_time)
        
        # Add metal oxide effects
        oxide_effect = 0.1 * (metal_a_concentration / 50.0) * (1.0 + 0.5 * (1.0 - torch.abs(metal_a_type - metal_b_type) / 20.0))
        uniformity += oxide_effect * 0.1
        coverage += oxide_effect * 0.15
        adhesion += oxide_effect * 0.2
        
        # Apply condition effects
        condition_effect = torch.where(condition == 3, 0.1, 0.0)  # Condition 3 (both) gets a bonus
        uniformity += condition_effect
        coverage += condition_effect
        adhesion += condition_effect
        
        # Normalize to [0, 1]
        uniformity = torch.clamp(uniformity, 0.0, 1.0).unsqueeze(1)
        coverage = torch.clamp(coverage, 0.0, 1.0).unsqueeze(1)
        adhesion = torch.clamp(adhesion, 0.0, 1.0).unsqueeze(1)
        
        # Avoid local optima: if coverage is too high, add some noise
        for i in range(coverage.shape[0]):
            if coverage[i, 0] > 0.95:
                # Add small noise to prevent getting stuck
                coverage[i, 0] -= torch.rand(()) * 0.1
                adhesion[i, 0] += torch.rand(()) * 0.1
        
        return torch.cat([uniformity, coverage, adhesion], dim=-1).to(self.device)
    
    def _compute_hypervolume(self):
        """Compute hypervolume"""
        partitioning = NondominatedPartitioning(ref_point=self.ref_point, Y=self.Y)
        return partitioning.compute_hypervolume().item()
    
    def _save_iteration_data(self, record):
        """Save iteration data"""
        filename = f"{self.output_dir}/tkg_optimization_history_{self.experiment_id}.json"
        with open(filename, 'a') as f:
            json.dump(record, f)
            f.write("\n")
    
    def save_experiment_data(self, x, y):
        """Save experiment data to CSV"""
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        x_cpu = x.cpu().numpy()
        y_cpu = y.cpu().numpy()
        
        new_rows = []
        for i in range(x_cpu.shape[0]):
            data = {name: val for name, val in zip(self.param_names, x_cpu[i])}
            data.update({
                "Uniformity": y_cpu[i, 0],
                "Coverage": y_cpu[i, 1],
                "Adhesion": y_cpu[i, 2],
                "Timestamp": timestamp
            })
            new_rows.append(data)
        
        new_data = pd.DataFrame(new_rows, columns=self.history.columns)
        if self.history.empty:
            self.history = new_data
        else:
            self.history = pd.concat([self.history, new_data], ignore_index=True)
        
        filename = f"{self.output_dir}/tkg_experiment_{self.experiment_id}.csv"
        self.history.to_csv(filename, index=False)
    
    def get_pareto_front(self):
        """Calculate Pareto front"""
        pareto_mask = torch.ones(self.Y.shape[0], dtype=torch.bool)
        for i in range(self.Y.shape[0]):
            for j in range(self.Y.shape[0]):
                if i != j and torch.all(self.Y[j] >= self.Y[i]) and torch.any(self.Y[j] > self.Y[i]):
                    pareto_mask[i] = False
                    break
        return self.X[pareto_mask], self.Y[pareto_mask]
    
    def _record_iteration(self, iteration, candidates, acquisition_values=None):
        """Record iteration data"""
        pareto_x, pareto_y = self.get_pareto_front()
        record = {
            "iteration": iteration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "phase": self.phase,
            "X": self.X.cpu().numpy().tolist(),
            "Y": self.Y.cpu().numpy().tolist(),
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
        """Run optimization process"""
        self._set_seed(self.seed)
        
        # Generate initial samples
        X_init = self.generate_initial_samples()
        print(f"【INFO】Generated {X_init.shape[0]} initial samples")
        
        # Initial experiments
        print("=== Initial Experiments ===")
        for candidate in X_init:
            candidate = candidate.unsqueeze(0)
            
            if simulation_flag:
                y = self.simulate_experiment(candidate)
            else:
                # In real scenario, get human input
                y = self.get_human_input(candidate)
            
            self.X = torch.cat([self.X, candidate])
            self.Y = torch.cat([self.Y, y])
            self.save_experiment_data(candidate, y)
        
        self._record_iteration(iteration=0, candidates=X_init)
        
        # Main optimization loop
        print("=== Optimization Phase ===")
        standard_bounds = torch.zeros_like(self.bounds, device=self.device)
        standard_bounds[1, :] = 1.0
        
        for _ in range(n_iter):
            # Update current iteration counter
            self.current_iteration += 1
            print(f"\n【INFO】Iteration {self.current_iteration}/{n_iter}, Phase {self.phase}")
            
            # Check if we need to transition to phase 2
            if self.phase == 1 and self.current_iteration > self.phase_1_iterations:
                self.phase = 2
                print("【INFO】Transitioning to Phase 2: Complex systems enabled")
            
            try:
                # Initialize model
                mll, model = self.initialize_model()
                fit_gpytorch_mll(mll)
                
                # Generate candidates using taKG acquisition function
                acq_func = self._compute_trace_aware_knowledge_gradient(model, self.X)
                
                candidates, acq_values = optimize_acqf(
                    acq_function=acq_func,
                    bounds=standard_bounds,
                    q=self.batch_size,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                    options={"batch_limit": 5, "maxiter": 200, "seed": self.seed},
                    sequential=True
                )
                
                # Unnormalize and process candidates
                candidates = unnormalize(candidates, self.bounds)
                
                # Discretization
                for j in range(len(self.parameters)):
                    candidates[:, j] = torch.round(candidates[:, j] / self.steps[j]) * self.steps[j]
                    candidates[:, j] = torch.clamp(candidates[:, j], self.bounds[0, j], self.bounds[1, j])
                
                # Apply constraints
                candidates = self._apply_safety_constraints(candidates)
                candidates = self._apply_phase_constraints(candidates)
                
                # Simulate experiments
                if simulation_flag:
                    y_new = self.simulate_experiment(candidates)
                else:
                    y_new = self.get_human_input(candidates)
                
                # Update data
                self.X = torch.cat([self.X, candidates])
                self.Y = torch.cat([self.Y, y_new])
                self.save_experiment_data(candidates, y_new)
                
                # Record iteration with the updated current_iteration
                self._record_iteration(
                    iteration=self.current_iteration,
                    candidates=candidates,
                    acquisition_values=acq_values,
                )
                
                # Compute hypervolume
                hv = self._compute_hypervolume()
                print(f"【INFO】Current hypervolume: {hv:.4f}")
                print(f"【INFO】Added {candidates.shape[0]} new samples")
            except Exception as e:
                print(f"【ERROR】Error in iteration {self.current_iteration}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print(f"\n【INFO】Optimization completed. Total samples: {self.X.shape[0]}")
        print(f"【INFO】Final hypervolume: {self.hypervolume_history[-1]:.4f}")
    
    def get_human_input(self, candidates):
        """Get human input for experiment results"""
        # This method would be used in real experiments to get actual measurements
        raise NotImplementedError("Human input method not implemented for simulation")

    def plot_pareto_front(self):
        """Plot Pareto front"""
        objectives = self.Y.cpu().numpy()
        pareto_x, pareto_y = self.get_pareto_front()
        pareto_front = pareto_y.cpu().numpy()
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all experiments
        ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                   c="gray", alpha=0.4, label="All experiments")
        
        # Plot Pareto front
        if len(pareto_front) > 0:
            ax.scatter(
                pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
                c='red', label="Pareto front"
            )
        
        ax.set_xlabel('Uniformity')
        ax.set_ylabel('Coverage')
        ax.set_zlabel('Adhesion')
        ax.set_title('Pareto Front of Glass Metallization Optimization')
        ax.legend()
        ax.grid(True)
        
        # Save figure
        fig_name = f"{self.fig_dir}/tkg_pareto_front_{self.experiment_id}.png"
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"【INFO】Pareto front plot saved to: {fig_name}")
    
    def plot_hypervolume_convergence(self):
        """Plot hypervolume convergence history"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(self.hypervolume_history)), self.hypervolume_history, 'b-', marker='o', label='Hypervolume')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Hypervolume')
        ax.set_title('Hypervolume Convergence')
        ax.grid(True)
        ax.legend()
        
        # Save figure
        fig_name = f"{self.fig_dir}/tkg_hypervolume_convergence_{self.experiment_id}.png"
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"【INFO】Hypervolume convergence plot saved to: {fig_name}")
    
    def get_algorithm_info(self):
        """Get algorithm information"""
        return {
            "name": "Trace-Aware Knowledge Gradient (taKG)",
            "acquisition_function": "qLogExpectedHypervolumeImprovement",
            "phase": self.phase,
            "phase_description": "Phase 1: Simple systems (only organic or only oxide)" if self.phase == 1 else "Phase 2: Complex systems (both organic and oxide)",
            "phase_1_iterations": self.phase_1_iterations,
            "hyperparameters": {
                "batch_size": self.batch_size,
                "num_restarts": self.num_restarts,
                "raw_samples": self.raw_samples,
                "n_init": self.n_init
            },
            "optimization_objectives": ["Uniformity", "Coverage", "Adhesion"]
        }
    
    def get_experiment_stats(self):
        """Get experiment statistics"""
        pareto_x, pareto_y = self.get_pareto_front()
        return {
            "total_experiments": self.X.shape[0],
            "total_iterations": len(self.iteration_history),
            "pareto_solutions": len(pareto_x),
            "current_phase": self.phase,
            "hypervolume": self._compute_hypervolume() if self.X.shape[0] > 0 else 0,
            "objectives": {
                "uniformity": {
                    "min": self.Y[:, 0].min().item() if self.Y.shape[0] > 0 else 0,
                    "max": self.Y[:, 0].max().item() if self.Y.shape[0] > 0 else 0,
                    "mean": self.Y[:, 0].mean().item() if self.Y.shape[0] > 0 else 0
                },
                "coverage": {
                    "min": self.Y[:, 1].min().item() if self.Y.shape[0] > 0 else 0,
                    "max": self.Y[:, 1].max().item() if self.Y.shape[0] > 0 else 0,
                    "mean": self.Y[:, 1].mean().item() if self.Y.shape[0] > 0 else 0
                },
                "adhesion": {
                    "min": self.Y[:, 2].min().item() if self.Y.shape[0] > 0 else 0,
                    "max": self.Y[:, 2].max().item() if self.Y.shape[0] > 0 else 0,
                    "mean": self.Y[:, 2].mean().item() if self.Y.shape[0] > 0 else 0
                }
            }
        }
    
    def get_heatmap_data(self, param1_idx=0, param2_idx=1, n_grid=20):
        """Generate heatmap data for two parameters"""
        if self.X.shape[0] == 0:
            return None
        
        # Create grid for the two parameters
        param1_min, param1_max = self.bounds[0, param1_idx].item(), self.bounds[1, param1_idx].item()
        param2_min, param2_max = self.bounds[0, param2_idx].item(), self.bounds[1, param2_idx].item()
        
        param1_grid = torch.linspace(param1_min, param1_max, n_grid, device=self.device)
        param2_grid = torch.linspace(param2_min, param2_max, n_grid, device=self.device)
        
        # Create meshgrid
        param1_mesh, param2_mesh = torch.meshgrid(param1_grid, param2_grid, indexing='ij')
        
        # Initialize input tensor with default values (mean of each parameter)
        default_params = (self.bounds[0] + self.bounds[1]) / 2
        X_grid = default_params.repeat(n_grid * n_grid, 1)
        
        # Fill in the grid values for the two parameters
        X_grid[:, param1_idx] = param1_mesh.flatten()
        X_grid[:, param2_idx] = param2_mesh.flatten()
        
        # Apply constraints
        X_grid = self._apply_safety_constraints(X_grid)
        
        # Normalize input
        X_normalized = normalize(X_grid, self.bounds)
        
        # Get model predictions
        mll, model = self.initialize_model()
        with torch.no_grad():
            posterior = model.posterior(X_normalized)
            mean = posterior.mean
            variance = posterior.variance
        
        # Reshape to grid
        mean_grid = mean.reshape(n_grid, n_grid, 3)
        variance_grid = variance.reshape(n_grid, n_grid, 3)
        
        return {
            "param1_name": self.param_names[param1_idx],
            "param2_name": self.param_names[param2_idx],
            "param1_grid": param1_grid.cpu().numpy().tolist(),
            "param2_grid": param2_grid.cpu().numpy().tolist(),
            "mean": mean_grid.cpu().numpy().tolist(),
            "variance": variance_grid.cpu().numpy().tolist(),
            "objectives": ["Uniformity", "Coverage", "Adhesion"]
        }
    
    def get_trace_data(self):
        """Get trace data for visualization"""
        return {
            "iteration_history": self.iteration_history,
            "hypervolume_history": self.hypervolume_history,
            "experiment_data": {
                "X": self.X.cpu().numpy().tolist(),
                "Y": self.Y.cpu().numpy().tolist(),
                "param_names": self.param_names,
                "objectives": ["Uniformity", "Coverage", "Adhesion"]
            },
            "pareto_front": {
                "X": self.get_pareto_front()[0].cpu().numpy().tolist(),
                "Y": self.get_pareto_front()[1].cpu().numpy().tolist()
            }
        }