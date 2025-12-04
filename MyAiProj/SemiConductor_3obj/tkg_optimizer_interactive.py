import sqlite3
import torch
import pandas as pd
from datetime import datetime
from src.tkg_optimizer import TraceAwareKGOptimizer
from config import OUTPUT_DIR, FIGURE_DIR


class DatabaseManager:
    """Database interaction class, responsible for all SQLite database operations"""
    
    def __init__(self, db_path):
        self.db_path = db_path
    
    def _get_connection(self):
        """Get database connection"""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            print(f"【ERROR】Database connection failed: {e}")
            raise
    
    def get_project(self, proj_name):
        """Get project information, return dictionary format for easy access by column name"""
        conn = self._get_connection()
        try:
            # Use row_factory to return dictionary format data
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM AlgoProjInfo WHERE ProjName = ?", (proj_name,))
            result = cursor.fetchone()
            # Convert sqlite3.Row object to dictionary
            return dict(result) if result else None
        finally:
            conn.close()
    
    def insert_samples(self, proj_name, iter_id, samples, is_initial=False):
        """Insert sample data"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Get the BatchNum value for the project
            cursor.execute("SELECT BatchNum FROM AlgoProjInfo WHERE ProjName = ?", (proj_name,))
            batch_num_result = cursor.fetchone()
            if not batch_num_result:
                print(f"【ERROR】Failed to get BatchNum for project '{proj_name}'")
                return False
            batch_num = batch_num_result[0]
            
            # Insert samples, generate ExpID (integer type) for each sample
            for i, sample in enumerate(samples):
                # Ensure i+1 does not exceed batch_num
                exp_id = i + 1 if (i + 1) <= batch_num else batch_num
                cursor.execute('''
                INSERT INTO BayesExperData (
                    ExpID, ProjName, IterId, Formula, Concentration, Temperature, SoakTime, PH, CreateTime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (exp_id, proj_name, iter_id, sample[0], sample[1], sample[2], sample[3], sample[4], create_time))
            
            # Update AlgoGenId
            cursor.execute('''
            UPDATE AlgoProjInfo 
            SET AlgoGenId = ?, IterId = ?
            WHERE ProjName = ?
            ''', (iter_id, iter_id, proj_name))
            
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"【ERROR】Failed to insert samples: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_experiment_data(self, proj_name, iter_id):
        """Get experiment data for a specific iteration"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT Formula, Concentration, Temperature, SoakTime, PH, Coverage, Uniformity, Adhesion 
            FROM BayesExperData 
            WHERE ProjName = ? AND IterId = ?
            ''', (proj_name, iter_id))
            return cursor.fetchall()
        finally:
            conn.close()
    
    def update_experiment_data(self, proj_name, iter_id, exp_data):
        """Update experiment data"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for data in exp_data:
                cursor.execute('''
                UPDATE BayesExperData 
                SET Coverage = ?, Adhesion = ?, Uniformity = ?, UpdateTime = ?
                WHERE ProjName = ? AND IterId = ? 
                  AND Formula = ? AND Concentration = ? 
                  AND Temperature = ? AND SoakTime = ? AND PH = ?
                ''', (
                    data.get('Coverage'), data.get('Adhesion'), data.get('Uniformity'), update_time,
                    proj_name, iter_id,
                    data.get('Formula'), data.get('Concentration'), 
                    data.get('Temperature'), data.get('SoakTime'), data.get('PH')
                ))
            
            # Update AlgoRecevId to indicate successful data return
            cursor.execute('''
            UPDATE AlgoProjInfo 
            SET AlgoRecevId = ?
            WHERE ProjName = ?
            ''', (iter_id, proj_name))
            
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"【ERROR】Failed to update experiment data: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()


class InteractiveTraceAwareKGOptimizer(TraceAwareKGOptimizer):
    """Interactive version of TraceAwareKGOptimizer with database interaction"""
    
    def __init__(self, output_dir, fig_dir, proj_name, db_path, seed=42, device=None):
        """
        Initialize the interactive Trace-Aware KG Optimizer with database interaction
        
        Args:
            output_dir: Directory to save output files
            fig_dir: Directory to save figures
            proj_name: Project name in the database
            db_path: Path to the SQLite database
            seed: Random seed for reproducibility
            device: Torch device to use
        """
        super().__init__(output_dir, fig_dir, seed=seed, device=device)
        
        # Database configuration
        self.proj_name = proj_name
        self.db_manager = DatabaseManager(db_path)
        self.current_iteration_id = 1
        
        print(f"【INFO】Initialized InteractiveTraceAwareKGOptimizer for project '{proj_name}'")
    
    def get_human_input(self, candidates):
        """Get human input for experiment results from database
        
        Args:
            candidates: Tensor of shape (batch_size, num_params) containing the candidate solutions
            
        Returns:
            Tensor of shape (batch_size, num_objectives) containing the measured objectives
        """
        batch_size = candidates.shape[0]
        num_objectives = 3  # Uniformity, Coverage, Adhesion
        
        print(f"【INFO】Processing {batch_size} candidates for iteration {self.current_iteration_id}")
        
        # Insert candidates into database
        candidates_np = candidates.cpu().numpy()
        success = self.db_manager.insert_samples(
            self.proj_name, 
            self.current_iteration_id, 
            candidates_np, 
            is_initial=self.current_iteration_id == 1
        )
        
        if not success:
            raise RuntimeError(f"Failed to insert samples into database for iteration {self.current_iteration_id}")
        
        # Wait for experiment data to be available in the database
        print("【INFO】Waiting for experiment data for iteration " + str(self.current_iteration_id) + "...")
        print("【INFO】Please run the experiment and update the results in the database.")
        print("【INFO】Once done, press Enter to continue...")
        input()  # Wait for user confirmation
        
        # Get experiment data from database
        exp_data = self.db_manager.get_experiment_data(self.proj_name, self.current_iteration_id)
        
        if len(exp_data) == 0:
            raise RuntimeError(f"No experiment data found in database for iteration {self.current_iteration_id}")
        
        # Process experiment data
        y = torch.zeros((batch_size, num_objectives), dtype=torch.float64, device=self.device)
        
        for i, data in enumerate(exp_data):
            if i >= batch_size:
                break
                
            # Extract objectives (order: Coverage, Uniformity, Adhesion in database)
            # Convert to our internal order: Uniformity, Coverage, Adhesion
            uniformity = float(data[6])  # 7th column in database
            coverage = float(data[5])     # 6th column in database
            adhesion = float(data[7])     # 8th column in database
            
            y[i] = torch.tensor([uniformity, coverage, adhesion], dtype=torch.float64, device=self.device)
        
        # Validate all candidates have results
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise RuntimeError(f"Invalid experiment data received for iteration {self.current_iteration_id}")
        
        # Increment iteration ID
        self.current_iteration_id += 1
        
        return y
    
    def load_existing_data(self):
        """Load existing experiment data from database"""
        print(f"【INFO】Loading existing experiment data for project '{self.proj_name}'...")
        
        # Get all experiment data for the project
        conn = self.db_manager._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT Formula, Concentration, Temperature, SoakTime, PH, Coverage, Uniformity, Adhesion, IterId
            FROM BayesExperData 
            WHERE ProjName = ?
            ORDER BY IterId, ExpID
            ''', (self.proj_name,))
            exp_data = cursor.fetchall()
            
            if len(exp_data) == 0:
                print(f"【INFO】No existing experiment data found for project '{self.proj_name}'")
                return
            
            # Process data
            X_list = []
            Y_list = []
            max_iter_id = 0
            
            for data in exp_data:
                # Extract parameters (order: Formula, Concentration, Temperature, SoakTime, PH)
                params = list(data[:5])
                
                # Add default values for additional parameters
                # Original parameters: Formula, Concentration, Temperature, SoakTime, PH
                # New parameters: Formula, Concentration, Temperature, SoakTime, PH, CuringTime, 
                # MetalA, MetalAConc, MetalB, MolarRatio, Condition
                curing_time = 20  # Default curing time
                metal_a_type = 1  # Default metal A type
                metal_a_concentration = 30  # Default metal A concentration
                metal_b_type = 0  # Default: no metal B
                molar_ratio = 5  # Default molar ratio
                condition = 1  # Default: organic only
                
                full_params = params + [curing_time, metal_a_type, metal_a_concentration, 
                                      metal_b_type, molar_ratio, condition]
                
                # Extract objectives
                uniformity = float(data[6])
                coverage = float(data[5])
                adhesion = float(data[7])
                
                # Extract iteration ID
                iter_id = data[8]
                if iter_id > max_iter_id:
                    max_iter_id = iter_id
                
                X_list.append(full_params)
                Y_list.append([uniformity, coverage, adhesion])
            
            # Convert to tensors
            X_tensor = torch.tensor(X_list, dtype=torch.float64, device=self.device)
            Y_tensor = torch.tensor(Y_list, dtype=torch.float64, device=self.device)
            
            # Update optimizer state
            self.X = X_tensor
            self.Y = Y_tensor
            self.current_iteration_id = max_iter_id + 1
            
            # Update history dataframe
            timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            new_rows = []
            for i in range(X_tensor.shape[0]):
                data = {name: val for name, val in zip(self.param_names, X_tensor[i].cpu().numpy())}
                data.update({
                    "Uniformity": Y_tensor[i, 0].item(),
                    "Coverage": Y_tensor[i, 1].item(),
                    "Adhesion": Y_tensor[i, 2].item(),
                    "Timestamp": timestamp
                })
                new_rows.append(data)
            
            if not self.history.empty:
                self.history = pd.concat([self.history, pd.DataFrame(new_rows)], ignore_index=True)
            else:
                self.history = pd.DataFrame(new_rows)
            
            print(f"【INFO】Loaded {len(exp_data)} existing experiment samples")
            print(f"【INFO】Next iteration ID: {self.current_iteration_id}")
            
        finally:
            conn.close()
    
    def optimize(self, n_iter=5, use_existing_data=True):
        """Run optimization process with database interaction
        
        Args:
            n_iter: Number of iterations to run
            use_existing_data: Whether to load existing data from database
        """
        # Load existing data if requested
        if use_existing_data:
            self.load_existing_data()
        
        # Run optimization with simulation_flag=False to use real experiment data
        super().optimize(n_iter=n_iter, simulation_flag=False)


# Example usage
if __name__ == "__main__":
    # Default database path (update with your actual database path)
    DEFAULT_DB_PATH = r"D:\Parameter\Meta\DB\sampleData.db"
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Interactive Trace-Aware KG Optimizer")
    parser.add_argument("--proj_name", type=str, required=True, help="Project name in the database")
    parser.add_argument("--db_path", type=str, default=DEFAULT_DB_PATH, help="Path to the SQLite database")
    parser.add_argument("--n_iter", type=int, default=5, help="Number of iterations to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = InteractiveTraceAwareKGOptimizer(
        output_dir=OUTPUT_DIR,
        fig_dir=FIGURE_DIR,
        proj_name=args.proj_name,
        db_path=args.db_path,
        seed=args.seed,
        device=torch.device(args.device) if args.device == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    )
    
    # Run optimization
    optimizer.optimize(n_iter=args.n_iter, use_existing_data=True)
