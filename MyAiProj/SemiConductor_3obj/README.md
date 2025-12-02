# Formula Optimization Project

## Overview

This project is designed to optimize formula based on a set of parameters. The optimization process uses a 
Bayesian optimization approach to find the best combination of parameters that maximizes the objective functions. 
The results are visualized using a Pareto front plot.

## Features

- **Parameter Optimization**: Optimize a formula based on a set of parameters.
- **Bayesian Optimization**: Utilize Bayesian optimization to find the best parameter combinations.
- **Pareto Front Plotting**: Visualize the optimization results using a Pareto front plot.
- **Command-Line Interface**: Use `argparse` to allow customization of the number of iterations via command-line arguments.

### Installation
1. **Setup Conda Environment:**
    Create an environment with
    ```bash
    conda create -n SemiConductor python=3.11
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Script

To run the optimization script, use the following command:

- `<number_of_iterations>`: The number of iterations for the optimization process. Default is 5 if not specified.

### Example

To run the script with 10 iterations:

```bash
python main.py --n_iter 10
```
