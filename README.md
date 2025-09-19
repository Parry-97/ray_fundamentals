# Project Overview

This project is a collection of hands-on scripts for educational purposes, used to learn and experiment with the Ray framework, its AI libraries, and the overall ecosystem. The examples focus on leveraging Ray for distributed machine learning tasks, covering both core Ray concepts and its integration with popular libraries like XGBoost and PyTorch.

**Technologies:**

- **Core:** Python, Ray
- **Machine Learning:** XGBoost, PyTorch, scikit-learn
- **Data Handling:** pandas, numpy, pyarrow
- **Environment Management:** uv

**Architecture:**

The project is structured as a series of standalone Python scripts, each demonstrating a specific aspect of the Ray framework:

- `ray_core.py`: Introduces fundamental Ray concepts like remote functions (tasks) and object references.
- `ray_advanced.py`: Delves into more advanced features such as the distributed object store and runtime environments.
- `ray_ai.py`: Showcases an end-to-end machine learning workflow using Ray to train an XGBoost model.
- `ray_torch.py`: Demonstrates how to use Ray Train for distributed training of a PyTorch model.

# Building and Running

This project uses `uv` for managing dependencies.

**1. Install Dependencies:**

```bash
uv sync
```

**2. Run the Examples:**

Each Python script can be executed individually to see the corresponding Ray functionality in action.

- **Core Ray Concepts:**

  ```bash
  python ray_core.py
  ```

- **Advanced Ray Concepts:**

  ```bash
  python ray_advanced.py
  ```

- **XGBoost with Ray:**

  ```bash
  python ray_ai.py
  ```

- **PyTorch with Ray Train:**

  ```bash
  python ray_torch.py
  ```

# Development Conventions

- **Coding Style:** The code follows the standard PEP 8 style guide for Python.
- **Testing:** While there are no formal test files, the scripts include assertions and print statements to verify the correctness of the operations.
- **Contributions:** (TODO: Add contribution guidelines if this were a collaborative project).
- **Version Control:** This repository is also used for experimenting with the `jj` version control system as a potential alternative to `git`.

