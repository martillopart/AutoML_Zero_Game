
# Getting Started Guide

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Bazel**: Version 3.7.2. Follow the installation instructions [here](https://docs.bazel.build/versions/main/install.html).
- **Compiler**: g++ version 9 or newer.
- **Python**: Version 3.9 or newer.
- **Numpy**: Install using the command `pip install numpy`.

## Installation

1. Install Bazel as described in the prerequisites.
2. Make sure g++ (version 9 or newer) is installed on your system.

## Running the Demos

To run the demo, follow these steps:

1. Run the standard script to verify its correct execution:
   ```bash
   ./run_demo.sh
   ```
2. Ensure the evaluation script works as expected:
	```bash
	./run_evaluation.sh
	```
## Setting Up Your Environment
-   Ensure Python 3.9 or newer is installed on your machine.
-   Install numpy by running:
	```bash
	pip install numpy
	```
## Playing the Game

To start the game script, run:
start the game script, run:

	`python3 Game.py` 

### Input Formats

Allowed input operations are as follows:

-   **SCALAR_CONST_SET**: Format `s<number> = <number>` (integer or decimal, can be negative)
-   **VECTOR_INNER_PRODUCT**: Format `s<number> = dot(v<number>, v<number>)` (dot product operation of two vectors)
-   **SCALAR_DIFF**: Format `s<number> = s<number> - s<number>` (subtraction of one s variable from another)
-   **SCALAR_PRODUCT**: Format `s<number> = s<number> * s<number>` (multiplication of one s variable by another)
-   **SCALAR_VECTOR_PRODUCT**: Format `v<number> = s<number> * v<number>` (multiplication of a vector by a scalar)
-   **VECTOR_SUM_OP**: Format `v<number> = v<number> + v<number>` (addition of two vectors)

At least one operation of each type ("Setup", "Predict", "Learn") must be entered.

### Ending the Input

Once at least one operation has been entered in the "Learn" phase, you can input `Stop` to end the input phase and begin the algorithm evaluation.

### Post-Evaluation

After evaluating the algorithm, the evaluation result is saved. The right to enter the algorithm then passes to the second player.

### Winning the Game

The player whose algorithm's fitness is higher wins. Algorithm fitness in the context of linear regression is assessed using RMSE (Root Mean Square Error), where a lower RMSE indicates better model fitness.

## Workflow

1.  In `Game.py`, `enter_alg` is called to allow entering the algorithm.
    `enter_alg(MAX_OP = 10)` 
    
2.  `check_input(user_input)` checks if the user input matches the regex patterns of allowed operations. If not, `find_closest_pattern_match` suggests the closest valid operation format.
3.  After all inputs, the algorithm evaluation proceeds in two steps:
    -   First, pseudo-code is converted to protobuf instructions for Bazel.
    -   Then, a Bazel script is called to evaluate the algorithm on a linear regression task, executing the code from file `run_evaluation_experiment.cc`.
  ##
This documentation provides a comprehensive guide for setting up, running demos, and playing the game, along with input formats and evaluation criteria.
