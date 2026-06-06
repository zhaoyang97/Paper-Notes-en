---
title: >-
  [Paper Note] AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning
description: >-
  [ICML2026][LLM Evaluation][Zeroth-order fine-tuning] AGZO identifies that the row space of linear layer gradients is constrained by the forward activation subspace. Based on this…
tags:
  - "ICML2026"
  - "LLM Evaluation"
  - "Zeroth-order fine-tuning"
  - "activation subspace"
  - "low-rank perturbation"
  - "memory-efficient training"
  - "LLM optimization"
date: 2026-05-08
content_hash: bc5a642bcd545c69
---

# AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning

**Conference**: ICML2026  
**arXiv**: [2601.17261](https://arxiv.org/abs/2601.17261)  
**Code**: None  
**Area**: LLM Efficiency / Zeroth-Order Optimization  
**Keywords**: Zeroth-order fine-tuning, activation subspace, low-rank perturbation, memory-efficient training, LLM optimization  

## TL;DR
AGZO identifies that the row space of linear layer gradients is constrained by the forward activation subspace. Based on this, it perturbs parameters only along activation-guided low-rank directions during zeroth-order fine-tuning, improving gradient alignment and downstream performance while maintaining memory usage levels close to MeZO.

## Background & Motivation

**Background**: LLM downstream adaptation typically relies on backpropagation fine-tuning, but backpropagation requires saving forward activations, making memory a bottleneck for long sequences and large batches. Zeroth-order (ZO) optimization offers an alternative: estimating update directions via function value differences only from forward passes without saving activations. Consequently, its memory usage remains close to inference levels, making it suitable for resource-constrained devices or consumer-grade GPUs.

**Limitations of Prior Work**: Representational methods like MeZO use random Gaussian perturbations in the full parameter space to estimate gradients. Although LOZO introduces low-rank perturbations, the low-rank factors are still sampled randomly and are data-independent. These methods treat the model as a black box, ignoring significant structural information generated during the forward process, which leads to many query budgets being spent on directions largely irrelevant to the true gradient.

**Key Challenge**: ZO fine-tuning aims to bypass the memory overhead of backpropagation, but aligning a single difference direction in a high-dimensional parameter space with the true gradient is extremely difficult with purely random perturbations. The problem is not the use of low-rank structures, but whether the low-rank subspace correlates with the true gradient structure of the current batch.

**Goal**: The authors aim to construct more informative zeroth-order perturbation directions by utilizing forward activations, allowing ZO methods to approach first-order gradient updates without significantly increasing memory overhead.

**Key Insight**: The paper derives from the linear layer gradient formula: for weight $W_\ell$, the true gradient is the product of the upstream gradient matrix and the input activation matrix $\nabla_{W_\ell} f = Q_\ell H_\ell^\top$. This indicates that the row space of the gradient is contained within the subspace spanned by the activations. Activations are not merely irrelevant intermediate quantities but geometric constraints that determine the gradient direction.

**Core Idea**: Extract the principal directions of the activation matrix on-the-fly during each forward pass and restrict zeroth-order perturbations to this activation-guided low-rank subspace, replacing blind sampling in the full space with "direction-aware random perturbations."

## Method

### Overall Architecture

AGZO is designed for full-parameter zeroth-order fine-tuning. Similar to MeZO, it first calculates the loss $f_0=f(W;B)$ on current parameters $W$, then applies a small perturbation $W+\mu\Delta$ to calculate the perturbed loss $f_+=f(W+\mu\Delta;B)$. Finally, the update is estimated by $(f_+-f_0)/\mu$ multiplied by the perturbation direction. The distinction lies in the construction of the perturbation $\Delta$.

For each linear layer, AGZO captures the input activation matrix $H_\ell$ during the normal forward pass. It utilizes a lightweight power iteration to approximate the top $r$ principal directions of $H_\ell H_\ell^\top$, yielding an orthogonal basis $A_\ell\in\mathbb{R}^{d_{in}\times r}$. Perturbations are then sampled exclusively within this subspace: for linear layers, $\Delta_\ell=R_\ell A_\ell^\top$, where $R_\ell$ is a Gaussian random left factor; for non-linear parameters, the method falls back to standard Gaussian perturbations. In the main experiments, $r=1$ is used to concentrate the single ZO sampling on the strongest activation direction.

AGZO does not save full activations for backpropagation. Subspace extraction is completed while the activations are available; after extracting the small basis matrix $A_\ell$, $H_\ell$ is released. Perturbations are regenerated using random seeds. Thus, it only stores an additional $d_{in}\times r$ basis per layer compared to MeZO, which is significantly smaller than the weight matrix dimensions $d_{out}\times d_{in}$.

### Key Designs

1. **Gradient-Activation Subspace Analysis**:
	- Function: Explains why forward activations can guide zeroth-order perturbations.
	- Mechanism: Linear layer weight gradients satisfy $\nabla_{W_\ell} f(W;B)=Q_\ell H_\ell^\top$, meaning the gradient row space is constrained by the column space of $H_\ell$. The authors projected true gradients onto the principal activation subspace in GPT-2/SST-2 and found that a rank of approximately 10 achieves a cosine similarity near 1. The singular value spectra of both gradients and activations decay rapidly.
	- Design Motivation: If the majority of the true gradient energy resides in the principal activation directions, random full-space perturbation is inefficient. ZO should utilize the geometric structure exposed by the forward pass rather than relying on pure black-box sampling.

2. **Online Activation Subspace Extraction and Low-Rank Perturbation**:
	- Function: Constructs data-dependent perturbations without backpropagation.
	- Mechanism: Given the activation matrix $H$ and target rank $r$, a test matrix $\Omega$ is sampled to compute $Y=H\Omega$, followed by QR orthogonalization and $H(H^\top Q)$ power iteration to obtain basis $A$. The perturbation is defined as $\Delta_\ell=R_\ell A_\ell^\top$, confining its row space to the principal activation subspace.
	- Design Motivation: Direct SVD computation is expensive and increases memory pressure. Power iteration requires only a few matrix multiplications; the paper shows $K=3$ iterations achieve alignment comparable to exact SVD.

3. **Maintaining forward-only Memory Profile**:
	- Function: Enhances perturbation quality without compromising the core memory benefits of ZO.
	- Mechanism: AGZO saves only the low-dimensional basis and the random seed for each layer, discarding full activations immediately. During the update, the perturbation is regenerated: $W_\ell-\mu\Delta_\ell$ is recovered first, followed by the execution of $W_\ell\leftarrow W_\ell-\eta g\Delta_\ell$. Main experiments set $r=1$ to minimize basis storage.
	- Design Motivation: Saving activations for backpropagation to obtain more accurate directions would defeat the purpose of ZO fine-tuning. The value of AGZO lies in compressing transient information from the forward pass into small subspace descriptions.

### Loss & Training

Theoretically, AGZO can be viewed as optimizing a subspace-smoothed objective. The authors prove that the expected value of the estimator equals the gradient of the smoothed objective projected onto $A_\ell A_\ell^\top$, and the bias disappears linearly with $\mu$ when the true gradient row space is supported by $A_\ell$. Furthermore, in a noiseless setting, the expected cosine similarity between AGZO and the true gradient includes the term $\|GA\|_F/\|G\|_F$, representing the gradient energy captured by the subspace. Alignment is strictly superior to MeZO as long as the upstream gradient energy is not abnormally concentrated in activation directions with small singular values.

In the experiments, all ZO methods are trained for 20,000 steps. The Qwen3 model utilizes a perturbation scale of $\mu=10^{-7}$, while Pangu-1B uses $\mu=10^{-4}$ in BF16 to resist numerical noise. AGZO, MeZO, and LOZO share the same code framework, data processing, and evaluation pipelines. The first-order (FO) baseline is trained for 1,000 steps when memory permits.

## Key Experimental Results

### Main Results

| Model / Task | FO | AGZO | MeZO | LOZO | Zero | ICL | Conclusion |
|------------|----|------|------|------|------|-----|------|
| Qwen3-0.6B SST-2 | 0.904 | 0.877 | 0.858 | 0.870 | 0.540 | 0.510 | AGZO is closest to FO |
| Qwen3-0.6B CB | 0.946 | 0.892 | 0.803 | 0.760 | 0.410 | 0.570 | Significant gain for low-resource NLI |
| Qwen3-0.6B RTE | 0.808 | 0.772 | 0.732 | 0.743 | 0.599 | 0.722 | Superior to both ZO baselines |
| Qwen3-4B SST-2 | OOM | 0.892 | 0.875 | 0.866 | 0.649 | 0.887 | AGZO remains trainable when FO is infeasible |
| Qwen3-4B SQuAD | OOM | 0.876 | 0.870 | 0.869 | 0.583 | 0.555 | Small but stable lead on QA |
| Pangu-1B BoolQ | 0.751 | 0.730 | 0.699 | 0.696 | 0.695 | 0.735 | Effective on BF16/edge models |

### Ablation Study

| Analysis Item | Setting | Key Metric | Explanation |
|------|------|---------|------|
| Gradient Alignment | Qwen3-0.6B / SST-2 | AGZO consistently higher than MeZO | Empirically supports the theoretical directional alignment advantage |
| Cross-platform Validation | Pangu-1B GPU training, NPU evaluation | NPU Avg: AGZO 0.709, MeZO 0.703, LOZO 0.667 | Activation-guided ZO transfers to Ascend NPU environments |
| Comparison with LoRA | Qwen3-0.6B | AGZO stronger than LoRA on SST-2/CB/BoolQ | AGZO is a forward-only alternative, not a full PEFT replacement |
| Throughput | Qwen3-0.6B steps/s | Same magnitude as MeZO/LOZO, but power iteration adds compute | Trades moderate speed loss for better direction quality |
| Rank Ablation | Qwen3-0.6B / SST-2 | rank 1: 0.877, rank 4: 0.870, rank 16: 0.863 | Higher rank dilutes instantaneous perturbation quality in single-query settings |

### Key Findings
- AGZO is the strongest among ZO methods on most tasks. Notably, it improves the CB score of Qwen3-0.6B from MeZO's 0.803 to 0.892, indicating that the activation subspace significantly aids low-resource reasoning tasks.
- On Qwen3-4B, where FO cannot run due to memory constraints, AGZO remains trainable and generally outperforms MeZO/LOZO, demonstrating the practical value of forward-only fine-tuning.
- Memory curves indicate that AGZO's footprint is essentially identical to MeZO/LOZO and far lower than FO. On Pangu-1B, while FO OOMs with long contexts and large batches, AGZO can run at length 2048 with a batch size of 64.
- Diagnostics of exact SVD vs. power iteration show that the cosine similarity for $K=3$ is 0.0123, which is close to the 0.0124 of exact SVD and significantly higher than MeZO (0.0015) and LOZO (0.0014).

## Highlights & Insights
- The key insight of the paper is elegant: zeroth-order optimization is not synonymous with a complete black box. Even without backpass, forward activations reveal the gradient row space, and this structure can be exploited at a low cost.
- The distinction between AGZO and LOZO is crucial. Both employ low-rank structures, but LOZO's directions are data-independent random directions, while AGZO's directions are derived from current batch activations. Thus, "low-rank" is not the sole source of gain; activation alignment is the core.
- The result that rank 1 is actually best is interesting. It suggests that the bottleneck of single-query finite difference is not about maximizing subspace coverage, but about concentrating the random direction into high-energy directions under limited queries.
- These methods do not necessarily replace LoRA. LoRA trains a small number of adapters via backpropagation, while AGZO updates original parameters via forward-only passes. Future work could explore activation-guided ZO for training adapters or selective layer updates.

## Limitations & Future Work
- The main gains of AGZO are built on the structural constraint of linear layer gradients by the activation subspace; it falls back to standard Gaussian perturbations for non-linear parameters, underutilizing their structural information.
- Although the memory footprint is close to MeZO, power iteration and QR orthogonalization increase computational overhead. Throughput data indicates it is within an acceptable range, but extreme low-compute devices might require lighter approximations.
- The rank is fixed at 1 in main experiments, suggesting the current strategy is specialized for single queries; adaptive ranks across different layers or tasks have not yet been systematically explored.
- The experimental models extend only up to Qwen3-8B in supplementary results, leaving a gap before reaching tens of billions or larger scales. Activation spectra, numerical perturbation scales, and communication costs may shift the conclusions on ultra-large models.
- ZO methods require 20,000 steps to achieve results, meaning optimization efficiency still lags behind first-order methods. Future work could combine better difference estimation, control variables, PEFT, and hybrid FO/ZO training.

## Related Work & Insights
- **vs MeZO**: MeZO achieves ultra-low memory via full-space Gaussian perturbations but has blind directions; AGZO retains the forward-only form while improving perturbation quality using principal activation subspaces.
- **vs LOZO**: LOZO uses random low-rank perturbations based on the prior of low-rank gradients; AGZO further aligns these low-rank directions with the current batch's activations, resulting in higher cosine similarity to the true gradient.
- **vs First-Order Fine-Tuning (FO)**: FO usually performs best but incurs high memory costs for activation storage, causing OOM on Qwen3-4B in the experimental setup; AGZO sacrifices some performance for runnability.
- **vs LoRA**: LoRA reduces trainable parameters and retains backpropagation; AGZO eliminates backpropagation and updates original parameters. They address different memory bottlenecks and can potentially be combined.
- **vs Low-dimensional Fine-tuning Theory**: This work continues the observation of low intrinsic dimensions in fine-tuning but shifts from static random subspaces to dynamic activation subspaces extracted per batch.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Modifying ZO perturbation directions through the activation-gradient geometric relationship is a concise idea and more structured than random low-rank methods.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers Qwen3, Pangu, GPU/NPU, memory, throughput, LoRA, rank, and power iteration ablations; verification on larger models is still relatively sparse.
- Writing Quality: ⭐⭐⭐⭐☆ Theoretical and algorithmic explanations are clear with a complete chain of formulas; despite many tables, main conclusions are easy to grasp.
- Value: ⭐⭐⭐⭐☆ Highly practical for memory-constrained LLM fine-tuning, especially for exploring forward-only full-parameter adaptation, though training steps and compute overhead still need optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning](../../AAAI2026/llm_evaluation/low-rank_curvature_for_zeroth-order_optimization_in_llm_fine-tuning.md)
- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](../../ICLR2026/llm_evaluation/towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)
- [\[ICML 2026\] Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum](beyond_log_likelihood_probability-based_objectives_for_supervised_fine-tuning_ac.md)
- [\[ICML 2026\] Whose Alignment? Comparing LLM Process Alignment Across Diverse Organizational Decision Contexts](whose_alignment_comparing_llm_process_alignment_across_diverse_organizational_de.md)
- [\[ICML 2026\] Spherical Steering: Geometry-Aware Activation Rotation for Language Models](spherical_steering_geometry-aware_activation_rotation_for_language_models.md)

</div>

<!-- RELATED:END -->
