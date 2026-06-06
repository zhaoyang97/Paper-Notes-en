---
title: >-
  [Paper Note] SAME: Stabilized Mixture-of-Experts for Multimodal Continual Instruction Tuning
description: >-
  [ICML2026][Multimodal VLM][SAME] SAME explicitly decomposes "catastrophic forgetting" in multimodal continual instruction tuning (MCIT) within MoE-LoRA into two independent sources: router drift and expert drift. It empl…
tags:
  - "ICML2026"
  - "Multimodal VLM"
  - "SAME"
  - "MCIT"
  - "router drift"
  - "expert drift"
  - "spectral-aware update"
  - "curvature-aware Riemannian scaling"
  - "adaptive expert activation"
date: 2026-05-08
content_hash: 42a6e8e21f3e6bc9
---

# SAME: Stabilized Mixture-of-Experts for Multimodal Continual Instruction Tuning

**Conference**: ICML2026  
**arXiv**: [2602.01990](https://arxiv.org/abs/2602.01990)  
**Code**: https://github.com/LAMDA-CL/Prism  
**Area**: Multimodal VLM / Continual Learning / MoE-LoRA  
**Keywords**: SAME, MCIT, router drift, expert drift, spectral-aware update, curvature-aware Riemannian scaling, adaptive expert activation  

## TL;DR
SAME explicitly decomposes "catastrophic forgetting" in multimodal continual instruction tuning (MCIT) within MoE-LoRA into two independent sources: router drift and expert drift. It employs spectral-aware subspace constraints to update routers, utilizes historical input covariance for Riemannian preconditioning to protect experts, and removes redundant updates via task-level adaptive freezing. The method consistently outperforms current MoE continual learning SOTA on CoIN, UCIT, and the authors' TriGap long-sequence benchmark.

## Background & Motivation

**Background**: MLLMs (e.g., LLaVA, Qwen-VL) show impressive performance on static multi-tasks via instruction tuning. However, in real-world deployment, tasks arrive sequentially, leading to MCIT (Multimodal Continual Instruction Tuning). Modern approaches integrate MoE + LoRA into FFN layers, leveraging sparse routing for "expert specialization" to resist forgetting (e.g., MoELoRA, CL-MoE, HiDe-LLaVA).

**Limitations of Prior Work**: The authors conduct diagnostic experiments (Fig. 1) by saving router and expert snapshots after each task in an 8-task sequence. Testing on Task 1 revealed: (a) as new tasks arrive, the router's activation distribution for Task 1 inputs drifts significantly, meaning **the same input is routed to different experts**; (b) even when retraining the router on Task 1 data (with frozen experts), Task 1 accuracy continues to decline and routing entropy decreases—indicating that **the experts themselves are drifting** and have lost their original Task 1 functionality.

**Key Challenge**: In MoE-based continual learning, "forgetting" is the superposition of two decoupled sources: (i) **router drift**—router updates cause old task samples to be mismatched to new experts; (ii) **expert drift**—shared experts are repeatedly overwritten by new task gradients, erasing original functions. Previous methods failed to treat these separately, often lead to a "whack-a-mole" problem where fixing one causes the other to fail.

**Goal**: Design independent stabilization mechanisms for both the router and the experts, complemented by a training efficiency component to reduce redundant updates, creating an end-to-end rehearsal-free solution.

**Key Insight**: The authors borrow two classical ideas from continual learning: "gradient subspace projection" and "natural gradient/Fisher metrics." The former addresses the router (preserving old task subspaces), while the latter protects experts (preconditioning under historical input geometry).

**Core Idea**: Maintain an uncentered covariance $\mathbf{C}^t$ of historical inputs as a proxy for "past geometry." Router gradients are projected onto high-energy subspaces of this covariance, while expert gradients are scaled via Riemannian preconditioning using $(\mathbf{C}^{t-1})^{-1}$, combined with task-level adaptive expert freezing.

## Method

### Overall Architecture
The backbone is LLaVA-v1.5-7B + CLIP-L/14-336, with LoRA experts inserted into the LLM's FFN layers. Given input $\mathbf{x}\in\mathbb{R}^d$, the output is $\mathbf{h}=\mathbf{W}_0\mathbf{x}+\sum_{i=1}^n \omega_i \mathbf{B}_i\mathbf{A}_i\mathbf{x}$, where $\omega_i=\mathrm{Softmax}(\mathbf{W}_G\mathbf{x})_i$ is the router output. Workflow for task $t$: (1) Online maintenance of input covariance $\mathbf{C}^t$ with top-$k$ SVD to obtain $\mathbf{V}_\parallel, \mathbf{V}_\perp$; (2) Constraint of $\mathbf{W}_G$ updates via spectral-aware rules; (3) Expert gradient preconditioning using $(\mathbf{C}^{t-1})^{-1}$; (4) Temporary freezing of experts based on utility-importance differentials.

### Key Designs

1.  **Spectral-aware Routing**:
    - **Function**: Restricts router updates to "newly added input directions" for the current task while remaining nearly invariant in directions relied upon by old tasks.
    - **Mechanism**: Maintains a cross-task sliding covariance $\mathbf{C}^t=(\alpha_{t-1}\mathbf{C}^{t-1}+n_t\hat{\mathbf{C}}^t)/\alpha_t$, storing only top-$k$ principal components such that the cumulative energy ratio $\sum_{i=1}^k\sigma_i^2/\sum_{i=1}^d\sigma_i^2\geq \delta$. SVD yields $\mathbf{C}^t=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$, partitioned into $\mathbf{V}_\parallel=\mathbf{V}[:,:k]$ (key directions for current/past tasks) and $\mathbf{V}_\perp=\mathbf{V}[:,k:]$ (near-zero variance subspace). Updates in new directions are scaled by $g(\boldsymbol{\Sigma})$ (using inverse sliding average $\hat\sigma_i$ as $\alpha_i$): $\Delta\mathbf{W}_\parallel^t=\Delta\mathbf{W}_G^t\mathbf{V}_\parallel g(\boldsymbol{\Sigma})\mathbf{V}_\parallel^\top$. Old task directions are directly projected: $\Delta\mathbf{W}_\perp^t=\Delta\mathbf{W}_G^t\mathbf{V}_\perp\mathbf{V}_\perp^\top$. Final update: $\Delta\mathbf{W}_G^t=\Delta\mathbf{W}_\parallel^t+\Delta\mathbf{W}_\perp^t$.
    - **Design Motivation**: Since $\mathbf{C}^t\propto \mathbf{X}\mathbf{X}^\top$, then $\mathbf{V}_\perp^\top\mathbf{X}^{old}\approx \mathbf{0}$, ensuring $\Delta\mathbf{W}_\perp^t \mathbf{X}^{old}\approx \mathbf{0}$—preserving predictions for old tasks. Simultaneous re-weighting within $\mathbf{V}_\parallel$ based on "local context $\hat\sigma_i$" avoids treating all new directions equally, addressing the drift observed in Fig. 1(a-c).

2.  **Curvature-aware Riemannian Scaling**:
    - **Function**: Forces expert updates to take smaller steps in "historically frequent input directions" to avoid overwriting learned functions.
    - **Mechanism**: In a rehearsal-free setting, functional degradation is approximated as $\Delta_{degrad}\triangleq \mathbb{E}_{\mathbf{x}\sim \mathcal{D}_{<t}}[\|\Delta\mathbf{W}_i \mathbf{x}\|^2]=\mathrm{tr}(\Delta\mathbf{W}_i\mathbf{C}^{t-1}\Delta\mathbf{W}_i^\top)$. Optimizing $\min_{\Delta\mathbf{W}_i}\mathcal{L}+\lambda\max(0,\Delta_{degrad}-\epsilon)$ leads to Riemannian updates $\Delta\mathbf{W}_i=-\eta\nabla_{\mathbf{W}_i}\mathcal{L}(\mathbf{C}^{t-1})^{-1}$. The term $(\mathbf{C}^{t-1})^{-1}$ is approximated using the damped pseudo-inverse from the spectral stage: $(\mathbf{C}^{t-1})^{-1}\approx \mathbf{V}_k(\boldsymbol{\Sigma}_k+\mu\mathbf{I})^{-1}\mathbf{V}_k^\top+\frac{1}{\mu}(\mathbf{I}-\mathbf{V}_k\mathbf{V}_k^\top)$.
    - **Design Motivation**: High variance directions (large $\sigma_i$) result in small $(\sigma_i+\mu)^{-1}$, "compressing" the update. Low variance or new directions use the $1/\mu$ default scale. This follows the principle of natural gradients: automatically applying "gravity" in directions frequently used in the past. 

3.  **Adaptive Expert Activation**:
    - **Function**: Temporarily freezes experts that are "low utility for the current task but high importance for historical tasks" to save computation and prevent interference.
    - **Mechanism**: Maintains two running averages—current task utilization $\mathcal{U}(i)$ (mean routing weight per batch) and historical importance $\mathcal{F}^{pre}(i)$ (approximated via routing-weighted input energy $\omega_i(\mathbf{x})\|\mathbf{x}\|^2$). Min-max normalization yields $\tilde{\mathcal{U}}(i)$ and $\tilde{\mathcal{F}}^{pre}(i)$. Activation score: $\mathrm{Score}(i)=\tilde{\mathcal{U}}(i)-\tilde{\mathcal{F}}^{pre}(i)$. Experts with $\mathrm{Score}(i)<\tau_{score}$ are frozen during the current task's training.
    - **Design Motivation**: Top-$k$ routing is sample-level and scatters single-task gradients across many experts. Task-level freezing focuses updates on the few experts truly needed, strengthening compartmentalization and reducing training time and VRAM usage.

### Loss & Training
Standard next-token CE for multimodal instruction tuning. LoRA rank=8, 1 epoch per task, bs=6, lr=2e-4 with cosine decay and 0.03 warmup, 8×RTX 5090. All constraints are implemented by modifying gradient update rules without additional loss terms. Overall training cost is nearly identical to vanilla MoELoRA.

## Key Experimental Results

### Main Results

CoIN 8-task benchmark (aligned with original paper reports):

| Method | ScienceQA | ImageNet | REC | OCR-VQA | Average |
|---|---|---|---|---|---|
| MoELoRA (Chen 2024) | 62.02 | 37.21 | 33.22 | 65.75 | 50.58 |
| SEFE (Chen 2025) | 75.35 | 83.10 | 16.75 | 66.25 | 58.57 |
| HiDe-LLaVA (Guo 2025a) | 73.20 | 69.28 | 59.18 | 64.76 | 63.95 |
| **SAME (Ours)** | **78.35** | **90.21** | **59.87** | 63.59 | **66.82** |

TriGap (10-task) long-sequence benchmark:

| Method | DocVQA | IconQA | FloodNet | Average |
|---|---|---|---|---|
| MoE-LoRA | 37.49 | 43.43 | 90.41 | 44.45 |
| CL-MoE | 36.79 | 52.70 | 80.09 | 44.11 |
| ModalPrompt | 38.23 | 44.73 | 71.52 | 40.15 |
| **SAME** | **43.87** | **64.03** | 81.09 | **46.53** |

On UCIT (6 tasks), SAME leads with an average accuracy of 67.12% (vs ModalPrompt at 65.52%).

### Ablation Study

| Configuration | CoIN Avg Acc | Note |
|---|---|---|
| Baseline (MoELoRA) | 50.58 | No constraints on router/experts |
| + Spectral-aware Routing | 61.32 | Stable routing, +10.74 |
| + Curvature-aware Scaling | 65.89 | Experts not overwritten, +4.57 |
| + Adaptive Expert Activation | 66.82 | Task-level freeze +0.93, saves time/VRAM |

### Key Findings
- **Router stability provides the largest single gain**: Fig. 3 shows that with spectral-aware routing, Task 1's expert activation distribution barely drifts during training, corresponding to the +10.74 gain.
- **Expert drift exists independently**: Using a "re-routing protocol" (frozen experts + retrained router), Fig. 4 shows that as the expert version increases, Task 1 accuracy still drops; curvature scaling significantly slows this decay.
- **Format drift is a silent killer**: On the ScienceQA → TextVQA → ImageNet sequence, 70.6% of "errors" were merely case changes ("a" vs "A"). SAME flattens such format drift.
- **Adaptive freezing is a free lunch**: While gaining +0.93 accuracy, it saves an average of 32.1 minutes and 2.3 K MiB VRAM per GPU per task.

## Highlights & Insights
- **Microscopic decomposition of forgetting**: Unlike previous work that vaguely cited "parameter drift," this paper uses a re-routing control experiment to isolate router drift and expert drift as independent observable signals. This "diagnosis → decoupling → divide-and-conquer" paradigm is highly valuable.
- **Unified covariance $\mathbf{C}^t$ usage**: The same cumulative covariance is reused for router subspace partitioning, expert Riemannian metrics, and adaptive freezing input energy approximation, reducing storage overhead to nearly zero.
- **Natural anti-forgetting from a Riemannian perspective**: $(\mathbf{C}^{t-1})^{-1}$ preconditioning ensures that "historically used directions" automatically gain resistance to change, equivalent to natural gradient descent under the Fisher Information metric.
- **Empirical evidence of format drift**: The analysis of case-sensitivity errors suggests that much "forgetting" in LLM continual learning is superficial output style shifts rather than knowledge loss.

## Limitations & Future Work
- **Known task boundaries**: Covariance updates and subspace partitioning currently assume task IDs are known.
- **Covariance maintained only at the routing layer**: There is a potential small mismatch between FFN input distribution and router input distribution.
- **Hyperparameter sensitivity**: Parameters like $\delta, \rho, \mu, \lambda, \epsilon, \tau_{score}$ require tuning.
- **Limited to LoRA-on-FFN**: Extending the method to attention layers or vision encoders is necessary for further validation.

## Related Work & Insights
- **vs MoELoRA / CL-MoE / HiDe-LLaVA**: These rely on routing sparsity/auxiliary losses or hierarchical distillation. SAME intervenes directly in update rules with zero auxiliary loss and zero rehearsal.
- **vs Replay-LoRA / SEFE**: Replay methods require storing old data; SAME is rehearsal-free, using covariance summaries to preserve "past geometry."
- **vs O-LoRA / GEM / OGD**: While sharing the "orthogonal gradient projection" concept, SAME upgrades it to the MoE dimension—managing both parameter-dimensional projection and routing distribution stability.

## Rating
- Novelty: ⭐⭐⭐⭐ Effective decoupling of MoE forgetting sources.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive benchmarks plus diagnostic protocols and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical flow and clear problem decomposition.
- Value: ⭐⭐⭐⭐ Rehearsal-free with low engineering cost.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts](toward_structural_multimodal_representations_specialization_selection_and_sparsi.md)
- [\[ICML 2026\] Decentralized Instruction Tuning: Conflict-Aware Splitting and Weight Merging](decentralized_instruction_tuning_conflict-aware_splitting_and_weight_merging.md)
- [\[ICLR 2026\] Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts](../../ICLR2026/multimodal_vlm/capacity-aware_inference_mitigating_the_straggler_effect_in_mixture_of_experts.md)
- [\[AAAI 2026\] MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment](../../AAAI2026/multimodal_vlm/mcmoe_completing_missing_modalities_with_mixture_of_experts_for_incomplete_multi.md)
- [\[CVPR 2026\] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](../../CVPR2026/multimodal_vlm/modes_accelerating_mixture-of-experts_multimodal_large_language_models_via_dynam.md)

</div>

<!-- RELATED:END -->
