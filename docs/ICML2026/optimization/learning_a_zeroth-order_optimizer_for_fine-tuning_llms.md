---
title: >-
  [Paper Note] Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs
description: >-
  [ICML 2026][Optimization][Zeroth-order optimization] This paper introduces ZO Fine-tuner: it utilizes a "per-block lightweight neural network PertNN" to automatically learn the perturbation variance for each parameter bl…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Zeroth-order optimization"
  - "MeZO"
  - "L2L"
  - "block-diagonal perturbation"
  - "memory-efficient fine-tuning"
date: 2026-05-08
content_hash: 1a3fde345e2ed940
---

# Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs

**Conference**: ICML 2026  
**arXiv**: [2510.00419](https://arxiv.org/abs/2510.00419)  
**Code**: https://github.com/ASTRAL-Group/ZO_Fine_tuner (Available)  
**Area**: Optimization Algorithms / Efficient LLM Fine-tuning / Learning to Learn  
**Keywords**: Zeroth-order optimization, MeZO, L2L, block-diagonal perturbation, memory-efficient fine-tuning  

## TL;DR
This paper introduces ZO Fine-tuner: it utilizes a "per-block lightweight neural network PertNN" to automatically learn the perturbation variance for each parameter block of an LLM, upgrading the fixed $\mathcal{N}(0, I)$ in MeZO to a block-adaptive non-uniform distribution. On OPT-30B, the auxiliary network costs <2MB, yet outperforms existing zeroth-order baselines in 82.1% of 28 model-task pairs (4 LLMs × 7 datasets), achieving "train once, reuse across tasks and derivative models."

## Background & Motivation

**Background**: As LLM sizes explode, the optimizer states and backward activations of first-order optimizers like Adam consume approximately 12× the inference memory. Even with PEFT methods like LoRA or Prefix-Tuning, backpropagation still imposes a significant memory burden. MeZO (Malladi et al., 2023) introduced traditional ZO-SGD to LLM fine-tuning: estimating gradients with only two forward passes as $g\!\approx\!\tfrac{\mathcal{L}(\theta+\epsilon u)-\mathcal{L}(\theta-\epsilon u)}{2\epsilon}u,\ u\!\sim\!\mathcal{N}(0,I)$, reducing training memory to near-inference levels. Subsequent works like HIZOO, LOZO, MeZO-SVRG, ZO-AdamU, and ZO-DAP have manually designed more complex update rules on top of MeZO.

**Limitations of Prior Work**: These improvements rely on manual heuristics or mathematical approximations and still require extensive hyperparameter searches beyond the learning rate. Crucially, they all maintain an *isotropic* $\mathcal{N}(0, I)$ sampling distribution shared across all parameters. However, the quality of ZO gradient estimation depends on the local landscape; for an LLM with massive dimensional differences and highly non-uniform Hessians, applying the "same noise to all parameters" wastes the perturbation budget on suboptimal directions.

**Key Challenge**: While the L2L approach is naturally suited for adapting perturbation distributions, it faces two major hurdles in LLMs: (i) backpropagating through PertNN requires storing many activations; (ii) learning an auxiliary network for every parameter results in $O(d^2)$ complexity, which is unsustainable for 30B models. Additionally, L2L on small models suffers from poor transferability, often serving only a single model-task pair per training run.

**Goal**: To scale L2L to LLMs while achieving (a) memory and speed overhead comparable to MeZO, and (b) "train PertNN once on a base LLM, reuse across different tasks and derivative checkpoints."

**Key Insight**: The authors leverage the empirical finding by Zhang et al. (2024b) that Transformer Hessians exhibit a roughly block-diagonal structure (where embedding, Q, K, V, and projection layers naturally form parameter blocks). This suggests that adapting perturbation variance at the "block" level is sufficient to match the true curvature; LLaMA-8B contains only 291 parameter blocks, significantly fewer than 8 billion parameters.

**Core Idea**: Use "one PertNN per block" to learn a **block-diagonal perturbation covariance** $\Sigma_t\!=\!\mathrm{diag}(\sigma_t^{(1)} I_{d_1},\dots,\sigma_t^{(n)} I_{d_n})$, replacing MeZO's $u\!\sim\!\mathcal{N}(0,I)$ with $u\!\sim\!\mathcal{N}(0,\Sigma_t\Sigma_t^\top)$. PertNN is trained differentiably using first-order fine-tuning trajectories as "meta-supervision."

## Method

### Overall Architecture
The deployment phase of ZO Fine-tuner consists of MeZO's two forward passes, with one addition: before sampling perturbations at each step, $n$ lightweight PertNNs calculate the current block-wise standard deviations $\sigma_t^{(i)}$ to form $\Sigma_t$. Perturbations are sampled via reparameterized $u_t=\widetilde\Sigma_t z_t,\ z_t\!\sim\!\mathcal{N}(0,I_d)$, and LLM parameters are updated following the MeZO formula. PertNN itself is learned during the "meta-training" phase along the first-order fine-tuning trajectory of the LLM; it is frozen during deployment.

PertNN takes a set of **status summaries** that are nearly task- and model-agnostic: the previous perturbation variance $\sigma_{t-1}^{(i)}$, current block parameter mean/variance $\mathrm{Mean}_t^{(i)}, \mathrm{Var}_t^{(i)}$, and the two losses recorded at the previous step $\boldsymbol{\ell}_{t-1}$. This task-agnostic input provides the foundation for PertNN's transferability across datasets and derivative checkpoints.

### Key Designs

1. **Block-diagonal Adaptive Perturbation + Compact PertNN**:
    - **Function**: Replaces MeZO's fixed isotropic Gaussian with non-uniform Gaussians where variance is shared within Transformer parameter blocks.
    - **Mechanism**: An independent small network runs for each $i$-th block: $\sigma_t^{(i)}=\mathrm{PertNN}^{(i)}(\boldsymbol{\ell}_{t-1},\sigma_{t-1}^{(i)},\mathrm{Mean}_t^{(i)},\mathrm{Var}_t^{(i)};\omega^{(i)})$, forming $\Sigma_t=\mathrm{diag}(\sigma_t^{(1)}I_{d_1},\dots,\sigma_t^{(n)}I_{d_n})$. Reparameterization $u_t=\Sigma_t z_t$ ensures the perturbation process is differentiable w.r.t. $\omega$, enabling gradient-based training of PertNN.
    - **Design Motivation**: Theorem 3.1 proves that under the block-diagonal Hessian assumption, "block-wise adaptive variance" yields a tighter loss descent upper bound than MeZO. Since "number of blocks $\ll$ number of parameters" (e.g., 291 blocks for LLaMA-8B), this categorization adds negligible memory—the total size of all PertNNs for OPT-30B is <2MB in FP16.

2. **Variance Normalization: Decoupling "Perturbation Shape" and "Effective Learning Rate"**:
    - **Function**: Ensures $\Sigma_t$ only dictates the "relative sizes between blocks," while the global step size remains controlled by a unified learning rate $\eta$.
    - **Mechanism**: From $\mathbb{E}[\hat g]\!\approx\!\mathbb{E}[u_t u_t^\top]\nabla\mathcal{L}$, non-uniform variance changes the effective learning rate to $\eta\cdot\tfrac{\|u_t\|^2}{d}$. Since $u_t=\Sigma_t z_t \Rightarrow \mathbb{E}\|u_t\|^2=\|\Sigma_t\|_F^2$, the authors enforce $\|\Sigma_t\|_F^2=\|I_d\|_F^2=d$ at each step (i.e., $\widetilde\Sigma_t=\tfrac{\sqrt{d}}{\|\Sigma_t\|_F}\Sigma_t$). In high dimensions, $\|u_t\|$ concentrates around $\sqrt{d}$, pinning the effective learning rate.
    - **Design Motivation**: Without normalization, the variance learned by PertNN would interfere with the step size, making tuning unstable. Table 2 shows that adding normalization alone reduces the loss from 0.395 to 0.307 on LLaMA-8B/SQuAD.

3. **L2L Training Framework + Periodic Reset**:
    - **Function**: Treats the "one-step updated LLM loss" as a differentiable meta-objective for PertNN and prevents PertNN from overfitting to low-loss regions.
    - **Mechanism**: A first-order optimizer (SGD/Adam) first generates an LLM trajectory $\{\theta_0^k\}$ for task $\mathcal{T}$. At each $\theta_0^k$, a ZO update produces $\theta_1^k$. The meta-loss is $\mathcal{L}_{\text{ZO}}(\omega)=\mathcal{L}(\theta_0^k-\eta\hat g(\theta_0^k, \omega))$, with gradients backpropagated from $\theta_1^k$ to $\omega$ via reparameterization. Multi-task shuffle training is used, and the LLM is reset to its pre-fine-tuned state every fixed number of steps to re-cover high-loss regions.
    - **Design Motivation**: First-order (FO) trajectories naturally provide samples of $\theta$ at various loss levels without additional sampling costs. However, FO training eventually enters flat regions; if PertNN only sees low-loss inputs, it may fail in high-loss areas. Periodic reset addresses this bias—the "Reset+Normalize" combination improves Qwen-14B/SST2 accuracy from 0.800 to 0.935.

### Loss & Training
- LLM Update: $\theta_{t+1}=\theta_t-\eta_1\hat g_t$, where $\hat g_t$ is sampled using the normalized $\widetilde\Sigma_t$.
- PertNN Update: $\omega_{t+1}=\omega_t-\eta_2\partial\mathcal{L}_{\text{ZO}}/\partial\omega_t$, trained cumulatively along FO trajectories.
- In practice, meta-training is performed only once on COPA (small and smooth loss). The remaining 27 (model, dataset) pairs reuse PertNN *zero-shot*, directly verifying the "train once, reuse widely" claim.

## Key Experimental Results

### Main Results
4 LLMs (LLaMA-3.2-1B / LLaMA-3.1-8B / Qwen2.5-14B / OPT-30B) across 7 datasets (COPA, SST-2, CB, SQuAD, WSC, BoolQ, DROP), compared against MeZO / MeZO-Adam(U) / HIZOO / LOZO. Results are reported at optimal learning rates:

| Model | Method | SST-2 Loss/Acc | SQuAD Loss/F1 | BoolQ Loss/Acc | DROP Loss/F1 |
|-------|--------|----------------|---------------|----------------|--------------|
| LLaMA-3.2-1B | MeZO | 0.29 / 0.90 | 0.48 / 0.75 | 0.63 / 0.63 | 1.16 / 0.29 |
| LLaMA-3.2-1B | **ZO FT** | **0.14 / 0.93** | **0.37 / 0.78** | **0.58 / 0.66** | **1.03 / 0.35** |
| LLaMA-3.1-8B | MeZO | 0.29 / 0.92 | 0.32 / 0.89 | 0.42 / 0.78 | 0.69 / 0.64 |
| LLaMA-3.1-8B | **ZO FT** | **0.18 / 0.94** | **0.31 / 0.90** | **0.34 / 0.87** | **0.54 / 0.66** |
| Qwen2.5-14B | MeZO | 0.21 / 0.88 | 0.24 / 0.88 | 0.23 / 0.84 | 0.45 / 0.66 |
| Qwen2.5-14B | **ZO FT** | 0.24 / **0.94** | **0.22 / 0.91** | 0.29 / **0.89** | **0.40 / 0.70** |
| OPT-30B | MeZO | 0.38 / 0.89 | 0.59 / 0.74 | 0.60 / 0.66 | 1.66 / 0.31 |
| OPT-30B | **ZO FT** | **0.35** / 0.87 | **0.56 / 0.77** | 0.61 / **0.67** | **1.59 / 0.31** |

Overall, ZO Fine-tuner achieved the lowest loss in **82.1%** and the highest accuracy in **75.0%** of the 28 pairs. It improved average accuracy by +2.5% over MeZO. These gains stem from a single meta-training on COPA, representing a strict OOD transfer test.

Transfer to derivative models (Table 4, PertNN trained on LLaMA-3.1-8B transferred to LLaMA-3.1-8B-Instruct): SST2 MeZO 0.276/0.92 → ZO FT **0.164/0.95**. Transfer to long-sequence reasoning (Table 5, Qwen-14B fine-tuned on MetaMathQA): GSM8K MeZO 81.4 → ZO FT **85.6**.

### Ablation Study
Table 2 (Normalization and Periodic Reset, loss / acc):

| Configuration | LLaMA-8B/SST2 | Qwen-14B/SST2 | LLaMA-8B/SQuAD |
|------|---------------|---------------|----------------|
| Base | 0.398 / 0.874 | 0.409 / 0.800 | 0.395 / 0.840 |
| +Reset | 0.389 / 0.881 | 0.404 / 0.810 | 0.368 / 0.856 |
| +Normalize | 0.306 / 0.920 | 0.389 / 0.844 | 0.307 / 0.899 |
| +Reset+Normalize | **0.179 / 0.941** | **0.240 / 0.935** | **0.307 / 0.905** |

Table 3 (Parameter Sharing Granularity): layer-wise vs block-wise, LLaMA-8B/SST2 0.23/0.92 → **0.18/0.94**, confirming that "block-wise sharing" aligned with Hessian block-diagonality is superior.

### Key Findings
- **Normalization is the primary contributor**: Adding it alone reduced loss by 20-25% across most tasks, confirming that non-uniform variance otherwise distorts the effective learning rate.
- Reset provides smaller gains alone but synergizes with Normalize, pushing Qwen-14B/SST2 accuracy from 0.844 to 0.935. It resolves secondary biases like "insufficient coverage of high-loss regions."
- ZO Fine-tuner is more robust to learning rates (Figure 3), converging deeper at smaller learning rates, implying that block-adaptive perturbation + normalization acts as an implicit per-block preconditioner.
- The memory cost is negligible: <2MB for OPT-30B, while the speed overhead is just one PertNN forward pass per step.

## Highlights & Insights
- The key to scaling L2L to LLMs is not a "stronger auxiliary network" but **reducing the learning target from $d$-dimensions to $n$-dimensions (blocks)**. This reduction is justified by the geometric foundation of Hessian block-diagonality.
- Using FO fine-tuning trajectories as meta-training data is highly efficient, providing a full spectrum of samples from "high-loss start" to "low-loss end" with zero extra sampling overhead.
- The Normalization section presents an **important sanity check** for L2L/adaptive optimizers: when learning both "direction" and "magnitude," applying a budget normalization forces the network to learn direction while handing magnitude back to the learning rate.
- "Train once on COPA, reuse on 28 models/tasks" moves L2L optimizers from a research curiosity to a practical paradigm of shipping a pretrained tuner with each base model.

## Limitations & Future Work
- Meta-training PertNN still requires a one-time FO "teacher" trajectory, which is a cost for the base model provider but potentially unrepeatable for downstream users without FO capabilities.
- Experiments focused on GLUE/SuperGLUE styles. While MetaMathQA shows promise, the effectiveness for RLHF, longer reasoning chains, or multimodal LLMs remains an open question.
- Current block partitioning depends on standard Transformer structures; efficacy on non-standard architectures like MoE or SSMs requires re-verifying Theorem 3.1.
- The absence of a direct Pareto comparison with PEFT (e.g., LoRA) under identical memory budgets is noted, although FO Adam upper bounds are provided.

## Related Work & Insights
- **vs MeZO (Malladi et al., 2023)**: MeZO uses fixed $\mathcal{N}(0,I)$; this work learns a block-wise adaptive $\Sigma_t$, outperforming it in 82.1% of pairings with negligible memory cost.
- **vs HIZOO (Zhao et al., 2025)**: HIZOO estimates Hessian information manually; this work fits Hessian-aware variance via L2L, offering better transferability.
- **vs LOZO / Low-rank Gradient Approximation**: These methods compress ZO estimates; this work pursues "structured perturbations" (block-diagonal $\Sigma$). The two are orthogonal and combinable.
- **Transferable Design Idea**: The "Hessian block-diagonal $\rightarrow$ per-block parameter sharing" pattern can be applied to other tasks like adaptive learning rates or adaptive clipping.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Successfully extends the L2L framework to LLM-scale ZO fine-tuning with solid theoretical grounding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 4 models, 7 datasets, derivative model transfer, and dual ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear progression from motivation to theorems and implementation.
- **Value**: ⭐⭐⭐⭐ The "ship a pretrained finetuner" paradigm has strong potential for edge-side or memory-constrained fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](../../ICCV2025/optimization/zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)
- [\[ICML 2026\] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)
- [\[ICML 2026\] Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization](turning_stale_gradients_into_stable_gradients_coherent_coordinate_descent_with_i.md)
- [\[ICML 2026\] Distilling Linearized Behavior into Non-Linear Fine-Tuning for Effective Task Arithmetic](distilling_linearized_behavior_into_non-linear_fine-tuning_for_effective_task_ar.md)
- [\[ICML 2026\] HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation](ho-sfl_hybrid-order_split_federated_learning_with_backprop-free_clients_and_dime.md)

</div>

<!-- RELATED:END -->
