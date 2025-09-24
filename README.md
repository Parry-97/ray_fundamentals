# Ray Fundamentals Learning Repository 🚀

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ray 2.46.0](https://img.shields.io/badge/ray-2.46.0-orange.svg)](https://ray.io/)
[![uv](https://img.shields.io/badge/uv-enabled-green.svg)](https://docs.astral.sh/uv/)
[![Learning](https://img.shields.io/badge/purpose-educational-brightgreen.svg)](#)

> 📖 **Hands-on learning repository following the [Anyscale Introduction to Ray Course](https://courses.anyscale.com/courses/intro-to-ray)**

This repository contains practical Python implementations and examples designed to complement the Anyscale "Introduction to Ray" course. Each script demonstrates key concepts from distributed computing with Ray, progressing from basic tasks to advanced machine learning workflows.

## 📚 Table of Contents

- [Course Overview](#-course-overview)
- [Learning Path](#-learning-path)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Key Ray Concepts](#-key-ray-concepts)
- [Running Examples](#-running-examples)
- [Documentation](#-documentation)
- [Utilities & Troubleshooting](#-utilities--troubleshooting)
- [Additional Resources](#-additional-resources)

## 📖 Course Overview

This repository implements examples for the following **Anyscale course modules**:

| Course Module                 | Repository Files                     | Key Concepts                                            |
| ----------------------------- | ------------------------------------ | ------------------------------------------------------- |
| **🏗️ Ray Core Fundamentals**  | `ray_core.py`, `ray_actors.py`       | Remote functions, ObjectRefs, Actors                    |
| **⚡ Ray Core Advanced**      | `ray_advanced.py`                    | Object Store, Runtime Environments, Resource Management |
| **🤖 Ray AI Libraries**       | `ray_ai.py`                          | XGBoost integration, Distributed ML workflows           |
| **🔥 Ray Train with PyTorch** | `ray_torch.py`                       | Distributed training, Model parallelism                 |
| **🧪 Testing & Utilities**    | `ray_minimal_test.py`, shell scripts | Memory management, Cleanup utilities                    |

### 🎯 Learning Objectives

After working through this repository, you'll understand:

- Ray's distributed computing model and core abstractions
- How to write scalable remote functions and stateful actors
- Object store patterns and memory management strategies
- Integration of Ray with popular ML libraries (XGBoost, PyTorch)
- Best practices for distributed training and model serving
- Troubleshooting and resource optimization techniques

## 🛤️ Learning Path

### 🟢 **Beginner Level (Start Here)**

_Estimated time: 2-3 hours_

1. **📋 Prerequisites Check**

   ```bash
   python --version  # Should be 3.11+
   uv --version      # Package manager
   ```

2. **🚀 Ray Basics** - `ray_minimal_test.py`
   - Verify Ray installation
   - Understand basic Ray initialization
   - Simple remote functions

3. **🔧 Core Concepts** - `ray_core.py`
   - Remote functions (`@ray.remote`)
   - Object references (`ray.get`, `ray.put`)
   - Common patterns and anti-patterns

### 🟡 **Intermediate Level**

_Estimated time: 3-4 hours_

4. **👥 Stateful Actors** - `ray_actors.py`
   - Actor lifecycle and state management
   - Actor handles and communication
   - Use cases for actors vs tasks

5. **⚡ Advanced Features** - `ray_advanced.py`
   - Distributed object store
   - Runtime environments
   - Resource allocation and fractional resources
   - Nested tasks and patterns

### 🔴 **Advanced Level**

_Estimated time: 4-5 hours_

6. **🤖 ML Workflows** - `ray_ai.py`
   - Ray integration with XGBoost
   - Distributed data processing
   - Model training and evaluation

7. **🔥 Distributed Training** - `ray_torch.py`
   - Ray Train with PyTorch
   - Distributed data parallel training
   - Checkpointing and metrics

### 🏁 **Checkpoints**

- ✅ Can create and call remote functions
- ✅ Understand ObjectRefs and object store
- ✅ Can implement and use Ray actors
- ✅ Familiar with runtime environments
- ✅ Can integrate Ray with ML libraries
- ✅ Understand distributed training patterns

## 📁 Project Structure

```
ray_fundamentals/
├── 📄 README.md                 # This comprehensive guide
├── 📦 pyproject.toml           # Project dependencies & config
├── 🐍 Python Learning Modules:
│   ├── ray_minimal_test.py     # ✅ Installation verification
│   ├── ray_core.py             # 🏗️ Remote functions & ObjectRefs
│   ├── ray_actors.py           # 👥 Stateful actors & communication
│   ├── ray_advanced.py         # ⚡ Object store & runtime environments
│   ├── ray_ai.py               # 🤖 XGBoost ML workflow
│   └── ray_torch.py            # 🔥 PyTorch distributed training
├── 🛠️ Utility Scripts:
│   ├── cleanup_ray.sh          # 🧹 Clean Ray temp files
│   └── run_ray_safe.sh         # 🛡️ Run with memory limits
└── 📚 Documentation:
    └── docs/
        ├── ray_resources.md    # CPU/GPU resource allocation
        └── ray_runtime_notes.md # Runtime environment deep-dive
```

### 📋 Detailed File Descriptions

| File                  | Purpose                     | Key Concepts                                           | Prerequisites     |
| --------------------- | --------------------------- | ------------------------------------------------------ | ----------------- |
| `ray_minimal_test.py` | **🧪 Verify setup**         | Ray initialization, basic remote functions             | Python basics     |
| `ray_core.py`         | **🏗️ Foundation concepts**  | `@ray.remote`, `ray.get()`, `ray.put()`, anti-patterns | None              |
| `ray_actors.py`       | **👥 Stateful computing**   | Actor classes, state management, handles               | `ray_core.py`     |
| `ray_advanced.py`     | **⚡ Advanced patterns**    | Object store, runtime envs, resources, nested tasks    | `ray_actors.py`   |
| `ray_ai.py`           | **🤖 ML integration**       | XGBoost + Ray, distributed ML workflows                | ML basics, pandas |
| `ray_torch.py`        | **🔥 Distributed training** | Ray Train, PyTorch DDP, checkpointing                  | PyTorch knowledge |

## 🚀 Quick Start

### **System Requirements**

- **Python:** 3.11 or higher
- **Memory:** 4GB+ RAM recommended
- **OS:** Linux, macOS, or Windows with WSL

### **1. Setup Environment**

```bash
# Clone or navigate to this repository
cd ray_fundamentals

# Install dependencies using uv (recommended)
uv sync

# Alternative: using pip
pip install -r requirements.txt
```

### **2. Verify Installation**

```bash
# Test Ray installation with minimal example
uv run ray_minimal_test.py
# or: python ray_minimal_test.py
```

**Expected output:**

```
Ray initialized successfully!
Available resources: {'CPU': 1.0, 'memory': 256000000}
Simple task result: 84
Small matrix test passed: True
Ray shutdown successfully!
```

### **3. Start Learning**

Begin with the [Learning Path](#-learning-path) above, starting with `ray_core.py`:

```bash
uv run ray_core.py
```

## 🧠 Key Ray Concepts

### **🔧 Remote Functions (Tasks)**

```python
@ray.remote
def compute_task(data):
    return process(data)

# Schedule task execution
future = compute_task.remote(my_data)
result = ray.get(future)  # Retrieve result
```

**Files:** `ray_core.py`, `ray_advanced.py`

### **👥 Actors (Stateful Workers)**

```python
@ray.remote
class StatefulWorker:
    def __init__(self):
        self.state = {}

    def update(self, key, value):
        self.state[key] = value

# Create actor instance
worker = StatefulWorker.remote()
worker.update.remote("key", "value")
```

**Files:** `ray_actors.py`

### **🗃️ Object Store**

```python
# Store large objects once, reference many times
large_data = ray.put(massive_dataset)
results = [process_data.remote(large_data) for _ in range(10)]
```

**Files:** `ray_advanced.py`

### **🤖 ML Integration**

```python
# Distributed training with Ray Train
from ray.train.xgboost import XGBoostTrainer
trainer = XGBoostTrainer(
    datasets={"train": train_dataset},
    params={"objective": "reg:squarederror"}
)
result = trainer.fit()
```

**Files:** `ray_ai.py`, `ray_torch.py`

## ▶️ Running Examples

### **Basic Execution**

```bash
# Method 1: Using uv (recommended)
uv run <script_name>.py

# Method 2: Direct python execution
python <script_name>.py

# Method 3: With custom memory limits
./run_ray_safe.sh  # Runs ray_advanced.py with memory constraints
```

### **Memory-Safe Execution**

For systems with limited RAM:

```bash
# Use the provided safe execution script
./run_ray_safe.sh

# Or set environment variables manually
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_memory_usage_threshold=0.6
python ray_advanced.py
```

### **Example Execution Sequence**

```bash
# 1. Verify installation
uv run ray_minimal_test.py

# 2. Learn core concepts
uv run ray_core.py

# 3. Explore actors
uv run ray_actors.py

# 4. Advanced patterns
./run_ray_safe.sh  # runs ray_advanced.py

# 5. ML workflows
uv run ray_ai.py

# 6. Distributed training
uv run ray_torch.py

# 7. Cleanup (if needed)
./cleanup_ray.sh
```

## 📚 Documentation

The `docs/` directory contains additional learning resources:

### **📄 Available Documentation**

| Document               | Description                        | Key Topics                                          |
| ---------------------- | ---------------------------------- | --------------------------------------------------- |
| `ray_resources.md`     | **CPU/GPU Resource Management**    | `num_cpus`, `num_gpus`, resource allocation         |
| `ray_runtime_notes.md` | **Runtime Environments Deep Dive** | Environment isolation, pip vs uv, Docker containers |

### **📖 Reading Order**

1. Start with code examples
2. Reference `ray_resources.md` when working with `ray_advanced.py`
3. Review `ray_runtime_notes.md` for production deployment insights

## 🛠️ Utilities & Troubleshooting

### **🧹 Cleanup Scripts**

| Script            | Purpose                    | Usage               |
| ----------------- | -------------------------- | ------------------- |
| `cleanup_ray.sh`  | Remove Ray temporary files | `./cleanup_ray.sh`  |
| `run_ray_safe.sh` | Execute with memory limits | `./run_ray_safe.sh` |

### **🚨 Common Issues & Solutions**

| Issue              | Symptoms                  | Solution                                   |
| ------------------ | ------------------------- | ------------------------------------------ |
| **Memory errors**  | Ray crashes, OOM kills    | Use `run_ray_safe.sh` or reduce data sizes |
| **Port conflicts** | "Address already in use"  | Run `ray stop` or `./cleanup_ray.sh`       |
| **Import errors**  | Module not found          | Ensure `uv sync` completed successfully    |
| **Slow startup**   | Long initialization times | Clean temp files with `cleanup_ray.sh`     |

### **🔍 Debugging Tips**

```bash
# Check Ray status
ray status

# View Ray dashboard (if available)
# Open browser to: http://localhost:8265

# Monitor system resources
htop  # or top on macOS

# Check disk usage
df -h /tmp  # Ray uses /tmp by default
```

### **⚙️ Configuration**

**Environment Variables:**

```bash
# Memory management
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_memory_usage_threshold=0.6

# Disable warnings
export RAY_DISABLE_IMPORT_WARNING=1

# Custom temp directory
export RAY_TMPDIR=/path/to/custom/tmp
```

## 🔗 Additional Resources

### **📚 Official Documentation**

- [Ray Documentation](https://docs.ray.io/) - Comprehensive official docs
- [Ray GitHub Repository](https://github.com/ray-project/ray) - Source code and issues
- [Anyscale Platform](https://www.anyscale.com/) - Managed Ray platform

### **🎓 Learning Resources**

- [Ray Tutorial](https://docs.ray.io/en/latest/ray-overview/getting-started.html) - Getting started guide
- [Ray Design Patterns](https://docs.ray.io/en/latest/ray-core/patterns/) - Common usage patterns
- [Ray Examples](https://github.com/ray-project/ray/tree/master/python/ray/examples) - Official examples repository

### **🏗️ Architecture & Best Practices**

- [Ray Architecture](https://docs.ray.io/en/latest/ray-core/overview.html) - System design overview
- [Performance Tips](https://docs.ray.io/en/latest/ray-core/performance-tips.html) - Optimization guidelines
- [Memory Management](https://docs.ray.io/en/latest/ray-core/memory-management.html) - Memory optimization strategies

### **🚀 Production Deployment**

- [Ray Clusters](https://docs.ray.io/en/latest/cluster/getting-started.html) - Multi-node setup
- [Kubernetes](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) - K8s integration
- [Docker](https://docs.ray.io/en/latest/ray-overview/docker.html) - Containerization guide

---

### **📝 Development Notes**

- **Coding Style:** Follows PEP 8 with extensive inline documentation
- **Version Control:** Uses `jj` (Jitijiji) as an experimental alternative to Git
- **Package Management:** Primary dependency management via `uv` for faster installs
- **Testing Strategy:** Executable examples with assertions and print statements for validation

### **🤝 Contributing**

This is a personal learning repository, but suggestions and improvements are welcome! Feel free to:

- Report issues or errors in examples
- Suggest additional Ray concepts to explore
- Share alternative approaches or optimizations

---

**🎯 Happy Learning!** Start your Ray journey with the [Learning Path](#-learning-path) and dive into distributed computing! 🚀
