---
title: >-
  [Paper Note] A Differentiable Model of Supply-Chain Shocks
description: >-
  [NeurIPS 2025 (Workshop: Differentiable Systems and Scientific ML)][supply chain] A JAX-based differentiable Agent-Based Model (ABM) of supply chains (~1…
tags:
  - "NeurIPS 2025 (Workshop: Differentiable Systems and Scientific ML)"
  - "supply chain"
  - "differentiable simulation"
  - "agent-based model"
  - "JAX"
  - "GPU acceleration"
  - "variational inference"
date: 2026-05-08
content_hash: cd5fd2d51544fec3
---

# A Differentiable Model of Supply-Chain Shocks

**Conference**: NeurIPS 2025 (Workshop: Differentiable Systems and Scientific ML)
**arXiv**: [2511.05231](https://arxiv.org/abs/2511.05231)
**Code**: None
**Area**: Other
**Keywords**: supply chain, differentiable simulation, agent-based model, JAX, GPU acceleration, variational inference

## TL;DR
A JAX-based differentiable Agent-Based Model (ABM) of supply chains (~1,000 firms) that combines GPU parallelization and automatic differentiation to achieve Bayesian parameter calibration three orders of magnitude faster than conventional ABC, paving the way for shock-propagation modeling in global supply-chain networks.

## Background & Motivation

1. **Background**: Modeling shock propagation in supply chains has grown increasingly important in the wake of Covid-19 and the Russia–Ukraine war. Established approaches include the Leontief input-output framework (comparative-static analysis) and Agent-Based Models (ABMs, bottom-up dynamic simulation). ABMs can capture inventory adjustment, time-varying recovery, and other dynamic features, making them a natural choice for shock-propagation modeling.
2. **Limitations of Prior Work**:
   - The likelihood function of ABMs is analytically intractable, so conventional calibration relies on Approximate Bayesian Computation (ABC)—repeated sampling compared against summary statistics—which scales poorly in high-dimensional parameter spaces.
   - Surrogate-model approaches introduce approximation error; neural simulation-based inference (SBI) offers amortized inference but cannot exploit gradient information.
   - Traditional ABM implementations are CPU-serial; calibration requires tens of thousands of forward simulations, entailing prohibitive computational cost.
3. **Key Challenge**: ABMs involve discrete stochasticity and control-flow structures that resist direct differentiation, while high-dimensional parameters (one set per firm) render gradient-free methods highly inefficient.
4. **Goal**:
   - Implement a supply-chain ABM as a JAX differentiable program supporting automatic differentiation.
   - Leverage GPU tensorization for large-scale parallel simulation.
   - Replace ABC with Generalized Variational Inference (GVI) for high-dimensional Bayesian calibration.
5. **Key Insight**: Exploit JAX's AD + GPU acceleration + NumPyro probabilistic programming ecosystem to recast ABM calibration as a gradient-based optimization problem.
6. **Core Idea**: Express the supply-chain ABM as a JAX differentiable program and use GPU parallelism + automatic differentiation to accelerate calibration by three orders of magnitude.

## Method

### Overall Architecture

- **Input**: A directed production network of $M$ firms, a technology-coefficient matrix $\mathbf{A}$, and an exogenous shock process.
- **Simulator**: At each time step, firms receive and place orders, produce (subject to inventory and capacity constraints), and update inventories.
- **Calibration**: Given macroeconomic observations $\mathbf{y}$, infer the posterior distribution over firm-level latent parameters $\mathbf{n}$ (inventory levels).

### Key Designs

1. **Supply-Chain ABM**:
   - **Function**: Simulate shock propagation through the production network.
   - **Mechanism**: Each firm $i$ maintains a target inventory $S_{ij}^{\text{target}} = n_i S_{ij}(0)$; output is the minimum of demand, capacity, and input constraints: $x_i(t) = \min\{D_i(t-1),\, z_i(t)\, f(S_{ji}(t))\}$. Shocks are modeled via a productivity recovery process: $z_i(t) = 1 - \delta_i \exp(-\lambda_i(t-t^*)^+)$.
   - **Design Motivation**: ARIO-type models, while parsimonious, capture the core dynamics of inventory depletion → output decline → upstream/downstream cascades.

2. **JAX Tensorization**:
   - **Function**: Replace serial firm-by-firm simulation with GPU-parallel tensor operations.
   - **Mechanism**: All firm states (inventories, orders, output) are unified as tensors; single-step updates become matrix operations. JAX's `vmap`/`pmap` handles batch parallelism automatically.
   - **Design Motivation**: Conventional Python ABMs loop over firms sequentially. The GPU implementation yields massive speedups at 3,000 firms—CPU time grows sharply while GPU time remains nearly constant.

3. **Generalized Variational Inference (GVI) Calibration**:
   - **Function**: Infer the posterior over latent parameters via gradient-based optimization.
   - **Mechanism**: Minimize $q^*(\mathbf{n}) = \arg\min_{q \in \mathcal{Q}} \mathbb{E}_{\mathbf{n} \sim q}[\ell(\mathbf{y}; \mathbf{n})] + D_{\text{KL}}(q(\mathbf{n}) \| p(\mathbf{n}))$, where $\ell$ is an L2 loss and $q$ is a Gaussian variational family. JAX's AD computes gradients of the ELBO with respect to variational parameters $\phi$, which are optimized via SGD.
   - **Design Motivation**: GVI exploits gradient information for efficient search in high-dimensional parameter spaces; ABC is essentially infeasible in a 2,000-dimensional space (two parameters per firm).

### Loss & Training

- **Forward simulation**: Differentiable in JAX, supporting AD.
- **Loss**: L2 (simulated vs. observed macro time series) + KL prior regularization.
- **Optimization**: Stochastic Variational Inference (SVI) with SGD via the NumPyro probabilistic programming framework.

## Key Experimental Results

### Main Results — GPU vs. CPU Speed Comparison

| Number of Firms | CPU (Ryzen 9 9950X) | GPU (RTX 5090) | Speedup |
|----------------|---------------------|----------------|---------|
| 100 | Moderate | Very fast | ~10× |
| 1,000 | Very slow | Very fast | ~100×+ |
| 3,000 | Extremely slow | Nearly unchanged | >1,000× |

GPU runtime remains nearly constant as firm count increases (parallel capacity not yet saturated), while CPU runtime grows sharply.

### Ablation Study — SVI vs. ABC Calibration Efficiency

| Method | After 300 Model Evaluations | After 30,000 Evaluations | Notes |
|--------|-----------------------------|--------------------------|-------|
| SVI (gradient-based) | Low loss | Very low loss | Surpasses ABC at 30,000 evals after only 300 |
| ABC (gradient-free) | High loss | Moderate loss | Highly inefficient in high dimensions |

After 300 model evaluations, SVI achieves lower in-sample and out-of-sample loss than ABC after 30,000 samples—an approximate 100× improvement (still ~50× after accounting for gradient computation overhead).

### Key Findings

- **Three orders of magnitude of acceleration**: The combination of GPU parallelism and AD-based gradients makes ABM calibration at the 1,000-firm scale feasible where it was previously intractable.
- **Efficient GPU utilization**: The GPU remains unsaturated at 3,000 firms, suggesting the approach can scale to substantially larger networks.
- **High value of gradient information**: In a 2,000-dimensional parameter space, gradient-based methods outperform gradient-free methods by approximately two orders of magnitude.

## Highlights & Insights

- **Differentiable ABM paradigm**: The paper demonstrates the feasibility and efficiency advantages of implementing economic ABMs as differentiable programs. This paradigm is transferable to other ABM domains such as epidemiology and transportation (the paper cites analogous work in epidemiology).
- **GVI as a replacement for ABC**: Using variational inference instead of ABC for high-dimensional ABM calibration preserves uncertainty quantification while gaining gradient-based acceleration—a significant methodological advance in ABM calibration.
- **Scalability to global supply networks**: The paper concludes that the approach can scale to simulate global supply networks, including price dynamics, logistics, and network reorganization.

## Limitations & Future Work

- **Workshop paper with limited experiments**: Validation is performed only on synthetic data (sample true parameters → generate observations → calibrate to recover them); no real supply-chain data are used.
- **Simplified model**: The production function is Leontief (no input substitutability); price mechanisms, firm entry/exit, and multi-product settings are not considered.
- **Handling of discrete stochasticity**: The paper does not detail how discrete stochasticity in the ABM (e.g., order allocation) is handled; relaxation approximations may have been employed.
- **Future directions**:
  - End-to-end calibration using real input-output tables (e.g., WIOD global data) and real shock events (e.g., Covid-19 shutdowns).
  - Incorporation of price adjustment mechanisms and inventory substitution elasticities.
  - Exploration of structured variational distributions (e.g., normalizing flows capturing inter-firm correlations).

## Related Work & Insights

- **vs. Chopra et al. (2023)**: A pioneer in differentiable epidemiological ABMs; the present paper applies the same paradigm to economic supply-chain modeling.
- **vs. conventional ABC calibration**: ABC is effective in low dimensions but infeasible at high dimensions; the proposed GVI+AD approach represents a qualitative leap.
- **vs. surrogate-model approaches**: Surrogate methods require additional training and introduce approximation error; the present paper differentiates through the original ABM directly, with no approximation.

## Rating

- Novelty: ⭐⭐⭐⭐ First application of differentiable ABMs to supply-chain modeling; methodological contribution is clear.
- Experimental Thoroughness: ⭐⭐⭐ Workshop-scale experiments; validation on synthetic data only.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear; method description is concise.
- Value: ⭐⭐⭐⭐ Provides a viable technical path for large-scale economic ABM calibration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Exact Learning of Arithmetic with Differentiable Agents](exact_learning_of_arithmetic_with_differentiable_agents.md)
- [\[NeurIPS 2025\] Note 7: Value-Guided Search - Efficient Chain-of-Thought Reasoning](polymath_evaluating_mathematical_reasoning_in_multilingual_contexts.md)
- [\[NeurIPS 2025\] Scalable GPU-Accelerated Euler Characteristic Curves: Optimization and Differentiable Learning for PyTorch](scalable_gpu-accelerated_euler_characteristic_curves_optimization_and_differenti.md)
- [\[CVPR 2026\] DiffBMP: Differentiable Rendering with Bitmap Primitives](../../CVPR2026/others/diffbmp_differentiable_rendering_with_bitmap_primitives.md)
- [\[NeurIPS 2025\] Weight Weaving: Parameter Pooling for Data-Free Model Merging](weight_weaving_parameter_pooling_for_data-free_model_merging.md)

</div>

<!-- RELATED:END -->
