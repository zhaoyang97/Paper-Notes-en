---
title: >-
  [Paper Note] Zeroth-Order Fine-Tuning of LLMs in Random Subspaces
description: >-
  [ICCV 2025][Optimization][Zeroth-order optimization] This paper proposes SubZero (random **Sub**space **Zero**th-order), which estimates gradients in random subspaces via per-layer low-rank perturbations…
tags:
  - "ICCV 2025"
  - "Optimization"
  - "Zeroth-order optimization"
  - "LLM fine-tuning"
  - "random subspace"
  - "low-rank perturbation"
  - "memory-efficient"
date: 2026-05-08
content_hash: 2db0942de733b570
---

# Zeroth-Order Fine-Tuning of LLMs in Random Subspaces

**Conference**: ICCV 2025
**arXiv**: [2410.08989](https://arxiv.org/abs/2410.08989)  
**Code**: [https://github.com/zimingyy/SubZero](https://github.com/zimingyy/SubZero)  
**Area**: Optimization
**Keywords**: Zeroth-order optimization, LLM fine-tuning, random subspace, low-rank perturbation, memory-efficient

## TL;DR

This paper proposes SubZero (random **Sub**space **Zero**th-order), which estimates gradients in random subspaces via per-layer low-rank perturbations, significantly reducing gradient variance and angular error in zeroth-order optimization, enabling memory-efficient LLM fine-tuning at a cost close to inference.

## Background & Motivation

Fine-tuning large language models (LLMs) typically relies on first-order optimizers (SGD/Adam), but as model scale grows, the memory overhead of backpropagation becomes prohibitive. MeZO was the first to introduce zeroth-order (ZO) optimization for LLM fine-tuning, estimating gradients using only forward passes at inference-level memory cost.

**Key Challenge**: The variance of ZO gradient estimates scales linearly with the perturbation dimensionality (i.e., the number of model parameters)—for LLMs with billions of parameters, this results in extremely high variance, severely hampering convergence speed and final performance.

Existing mitigation strategies have notable shortcomings:
- **Larger batch sizes**: lead to sharply increasing overhead in later training stages.
- **Sparse perturbations** (e.g., pruning masks in S-MeZO): heuristically selected without theoretical justification.
- **Random subspace methods** (e.g., S-RGF): require storing a $d \times q$ projection matrix ($q$ times the model size), which is completely infeasible for LLMs.

**Key Insight**: Exploit the layer-wise matrix structure of LLMs by constructing independent low-rank subspaces per layer ($\mathbf{U}_i \in \mathbb{R}^{m_i \times r}$, $\mathbf{V}_i \in \mathbb{R}^{n_i \times r}$), sampling perturbations only within an extremely small $r \times r$ space. The projection matrices are column-orthogonal and lazily updated, avoiding storage of large projection matrices while ensuring low variance.

## Method

### Overall Architecture

SubZero retains MeZO's paradigm of "two forward passes for gradient estimation," but replaces full-parameter-space Gaussian random vectors with per-layer low-rank matrices $\tilde{\mathbf{Z}}_i = \mathbf{U}_i \mathbf{Z}_i \mathbf{V}_i^\top$, where $\mathbf{Z}_i \in \mathbb{R}^{r \times r}$ is a low-dimensional Gaussian random matrix and $r \ll \min(m_i, n_i)$.

### Key Designs

1. **Per-Layer Low-Rank Perturbation**: For the $i$-th layer weight matrix $\mathbf{W}_i \in \mathbb{R}^{m_i \times n_i}$, column-orthogonal projection matrices $\mathbf{U}_i$ and $\mathbf{V}_i$ are obtained via QR decomposition of two Gaussian random matrices. The perturbed loss difference is:
    $\rho = \frac{\mathcal{L}(\mathcal{W} + \varepsilon\tilde{\mathcal{Z}}; \mathcal{B}) - \mathcal{L}(\mathcal{W} - \varepsilon\tilde{\mathcal{Z}}; \mathcal{B})}{2\varepsilon}$
   The gradient estimate for the $i$-th layer is $\hat{\nabla}\mathcal{L}(\mathbf{W}_i) = \rho \mathbf{U}_i \mathbf{Z}_i \mathbf{V}_i^\top$.

   **Design Motivation**: Compared to model-level projection (S-RGF's $\mathbf{P} \in \mathbb{R}^{d \times q}$), the layer-wise projection matrices are block-diagonal, equivalent to $\mathbf{P} = \text{bdiag}(\mathbf{V}_1 \otimes \mathbf{U}_1, \cdots, \mathbf{V}_l \otimes \mathbf{U}_l)$, satisfying $\mathbf{P}^\top \mathbf{P} = \mathbf{I}_q$ without requiring storage of the full $d \times q$ matrix. Experiments show that column-orthogonal matrices significantly outperform Gaussian random projection matrices (Table 5, 74.0% vs. 67.5% on RTE).

2. **Lazy Subspace Update**: Projection matrices $\mathbf{U}_i, \mathbf{V}_i$ are regenerated every $F$ steps (default $F=1000$) and reused in between. Overly frequent updates increase QR decomposition overhead and restrict subspace exploration; overly infrequent updates lead to stale subspaces. Ablation experiments (Table 7) show $F=1000$ is the preferred choice.

3. **Non-Square Reshape Strategy**: LoRA's low-rank matrix $\mathbf{A}_i \in \mathbb{R}^{m_i \times k}$ (where $k \ll m_i$) is too tall and narrow to find a smaller $r \ll k$ for constructing low-rank perturbations. The solution reshapes $\mathbf{A}_i$ into a near-square matrix $\mathbf{A}'_i \in \mathbb{R}^{m'_i \times k'}$ (preserving the total number of elements), then applies low-rank perturbations on the reshaped matrix. Ablations (Table 8) confirm this strategy is critical for PEFT settings: prompt tuning accuracy improves from 74.2% to 89.1%.

### Loss & Training

- SGD (without momentum) is used as the base optimizer by default, maintaining the same memory efficiency as MeZO.
- A **norm alignment** trick is adopted: low-rank perturbations are scaled by $\mu = \sqrt{mn/r^2}$ to match the norm of full-dimensional perturbations, enabling direct reuse of MeZO's learning rate and perturbation scale hyperparameters.
- With in-place operations and per-layer parameter updates, the memory overhead is nearly identical to inference.

## Key Experimental Results

### Main Results (OPT-13B, SuperGLUE 11 tasks)

| Method | SST-2 | RTE | CB | BoolQ | WSC | WIC | MultiRC | COPA | ReCoRD | SQuAD | DROP | AVG Δ |
|--------|-------|-----|----|-------|-----|-----|---------|------|--------|-------|------|--------|
| MeZO(FT) | 92.1 | 71.5 | 71.4 | 74.4 | 61.5 | 60.0 | 60.1 | 87.0 | 82.0 | 84.2 | 31.2 | 0% |
| SubZero(FT) | 92.1 | 74.0 | 73.2 | 75.3 | **65.4** | **60.8** | 61.0 | 88.0 | **82.3** | **84.5** | **32.0** | **+1.89%** |
| MeZO(LoRA) | 92.2 | 74.4 | 69.6 | 75.2 | 64.4 | 59.7 | 58.2 | 87.0 | 82.0 | 82.9 | 31.0 | 0% |
| SubZero(LoRA) | **93.8** | **75.5** | 71.4 | **76.1** | **65.4** | **60.3** | **60.3** | **89.0** | 81.9 | **83.7** | 31.3 | **+1.57%** |

### Ablation Study

Performance of LLaMA2-7B and OPT-1.3B under different fine-tuning configurations:

| Model | Setting | MeZO | SubZero | SGD |
|-------|---------|------|---------|-----|
| LLaMA2-7B | FT | 64.3 | **71.4** | 69.6 |
| LLaMA2-7B | Prompt | 60.7 | **66.1** | 69.6 |
| OPT-1.3B | FT | 92.3 | **93.4** | 93.2 |
| OPT-1.3B | Prompt | 85.9 | **89.1** | 90.7 |

SubZero improves over MeZO by 7.1% on LLaMA-7B full fine-tuning, even surpassing SGD.

### Key Findings

- **Significantly improved gradient quality** (Fig. 1): SubZero's gradient cosine similarity with the expected gradient is substantially higher than MeZO's, with markedly lower variance.
- **Negligible memory overhead**: On OPT-13B, SubZero uses only 1.73% more memory than MeZO (26.53 vs. 26.08 GB), whereas S-RGF requires 23.8 GB (on RoBERTa-large experiments).
- **Manageable time overhead**: The additional cost of QR decomposition is less than 9% across all OPT model sizes.

## Highlights & Insights

- The core insight is precise: gradients during LLM fine-tuning rapidly concentrate in low-dimensional subspaces, providing natural justification for low-rank perturbations.
- The reshape strategy is simple yet critical for PEFT scenarios, addressing the extreme aspect ratio of LoRA matrices.
- Theoretical guarantees are complete: the paper proves proximity of the gradient estimate to the backpropagation gradient in the subspace (Theorem 5b) and establishes a convergence rate of $\mathcal{O}(d/\epsilon)$.

## Limitations & Future Work

- No systematic evaluation combining with second-order ZO optimizers (e.g., HiZOO) or momentum-based ZO methods (e.g., ZO-AdaMU).
- Convergence rate still depends on parameter dimension $d$, though the constant factor is reduced via subspace projection.
- The update frequency $F$ and rank $r$ currently require manual tuning.
- Theoretical analysis is based on a quadratic loss assumption; its alignment with actual LLM loss landscapes remains to be validated.

## Related Work & Insights

- **MeZO** is the most direct baseline; SubZero extends it by introducing structured low-rank perturbations.
- **GaLore** (gradient low-rank projection) exploits similar low-rank observations from a first-order optimization perspective.
- SubZero is orthogonal to and composable with PEFT methods such as LoRA—it can be applied directly to fine-tune LoRA adapter parameters.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Per-layer low-rank perturbation combined with lazy updates is an elegant and effective design.
- **Theoretical Depth**: ⭐⭐⭐⭐ — Complete gradient approximation and convergence analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive comparisons across multiple models, fine-tuning settings, and tasks.
- **Practicality**: ⭐⭐⭐⭐⭐ — Plug-and-play, with memory overhead on par with inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Improving the Straight-Through Estimator with Zeroth-Order Information](../../NeurIPS2025/optimization/improving_the_straight-through_estimator_with_zeroth-order_information.md)
- [\[NeurIPS 2025\] PROFIT: A Specialized Optimizer for Deep Fine Tuning](../../NeurIPS2025/optimization/profit_a_specialized_optimizer_for_deep_fine_tuning.md)
- [\[ICLR 2026\] Converge Faster, Talk Less: Hessian-Informed Federated Zeroth-Order Optimization](../../ICLR2026/optimization/converge_faster_talk_less_hessian-informed_federated_zeroth-order_optimization.md)
- [\[ICCV 2025\] Federated Continual Instruction Tuning](federated_continual_instruction_tuning.md)
- [\[ICCV 2025\] Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data](federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data.md)

</div>

<!-- RELATED:END -->
