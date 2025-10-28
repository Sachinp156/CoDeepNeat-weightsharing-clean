# 🧬 Optimizing Deep Neural Networks Using Evolutionary Algorithms

### Author: [Sachin Parimanumkuzhi](mailto:sachinpv156@gmail.com)  
Master of Computer Science, The University of Adelaide (2024–2025)

---

## 📘 Project Overview

This project extends the **Keras-CoDeepNEAT** framework to improve the efficiency and stability of **Neural Architecture Search (NAS)** through *module-species-level weight sharing* and *gradient conflict mitigation*.

**CoDeepNEAT**, originally proposed by *Miikkulainen et al., 2019*, evolves neural network **modules** and **blueprints** using evolutionary algorithms.  
However, the baseline framework retrains similar architectures from scratch, leading to redundant computation and slower convergence.

This repository introduces several key innovations to optimize the evolutionary process:

---

## 🚀 Key Contributions

### 🧩 1. Module-Species-Level Weight Sharing
- Modules are grouped into *species* via **K-Means clustering** based on structural similarity.  
- When a module is first created, it only stores its **configuration**, not instantiated TensorFlow layers.  
- When assembled into a blueprint, layers are **instantiated** and stored in a **global registry** indexed by `(species_id, position_tag)`.  
- Subsequent modules of the same species **reuse shared layers** directly from the registry → eliminating redundant training and improving convergence.

### ⚙️ 2. Gradient Conflict Mitigation (MAGIC-T + PCGrad)
- **MAGIC-T (Xu et al., 2022)** limits child architectures to **one mutation per generation**, enabling smooth and stable shared updates.
- **PCGrad (Yu et al., 2020)** projects one gradient onto the non-conflicting component of another:
  \[
  g_i = g_i - \frac{g_i \cdot g_j}{\|g_j\|^2} g_j \quad \text{if } g_i \cdot g_j < 0
  \]
  → avoids destructive interference during shared layer updates.

### 🌿 3. Progressive Pruning
- Periodically removes the bottom **10–15 %** of underperforming modules based on fitness.
- Keeps the search space efficient and focused on high-performing module species.

### 📈 4. Integrated Experimental Pipeline
- Extends the original **Keras-CoDeepNEAT** runner to support:
  - `--share-mode module`
  - `--magic-t`
  - `--pcgrad-every`
  - `--no-clear-registry-before-final`
  - `--ab-compare` for baseline vs. shared evaluation

---

## 🧪 Experimental Setup

| Parameter | Value |
|------------|--------|
| Dataset | COVID-19 Radiography Dataset (4 classes: COVID, Normal, Viral Pneumonia, Lung Opacity) |
| Population Size | 6 |
| Generations | 10 |
| Training Epochs / Generation | 10 |
| Share Mode | `module` |
| Gradient Handling | MAGIC-T + PCGrad |
| Baseline Comparison | Standard CoDeepNEAT (no sharing, no PCGrad) |

Run example:
```bash
python example_scripts/run_covid_cxr.py \
  --ab-compare \
  --generations 10 --training-epochs 10 \
  --share-mode module --magic-t \
  --pcgrad-every 8 --pcgrad-sample-frac 0.33 \
  --ema-beta 0.10 \
  --ea-batch-size 32 --max-steps-per-epoch 60 \
  --no-clear-registry-before-final \
  --warm-start-final

