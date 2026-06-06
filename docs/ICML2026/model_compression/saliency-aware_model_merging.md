---
title: >-
  [Paper Note] Saliency-Aware Model Merging
description: >-
  [ICML 2026][Model Compression][model merging] SA-Merging adapts the SynFlow connectivity score from structured pruning to the data-free model merging scenario. For each expert's task vector…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "model merging"
  - "task vector"
  - "SynFlow"
  - "connectivity saliency"
  - "LoRA merging"
date: 2026-05-08
content_hash: e55ea27c8acccda3
---

# Saliency-Aware Model Merging

**Conference**: ICML 2026  
**arXiv**: [2606.00511](https://arxiv.org/abs/2606.00511)  
**Code**: Not released  
**Area**: Model Compression / Model Merging / Data-free Parameter Selection  
**Keywords**: model merging, task vector, SynFlow, connectivity saliency, LoRA merging  

## TL;DR
SA-Merging adapts the SynFlow connectivity score from structured pruning to the data-free model merging scenario. For each expert's task vector, it calculates the "end-to-end path sensitivity × aggregate direction consistency" as saliency. By iteratively removing low-saliency updates, it pushes data-free merging performance close to the level of test-time adaptation across vision, language, and LoRA multi-task benchmarks.

## Background & Motivation

**Background**: Starting from foundation models like CLIP, ViT, LLaMA, and T5, the community has trained a vast number of task-specific fine-tuned experts. Directly merging them into a unified model is a prominent research direction. Task Arithmetic represents each expert as a task vector $\tau_n = \theta_n - \theta_0$ followed by linear summation. Methods like TIES, DARE, PCB, and WUDI build on this by introducing magnitude pruning, sign election, and sparsification to reduce interference.

**Limitations of Prior Work**: These data-free methods almost exclusively assume that "parameters are independent and identically distributed (i.i.d.)"—where the importance of each weight is determined by its own absolute value. However, the functionality of deep networks is emergent through hierarchical cascades. A large-magnitude update might have zero effect on the final output if "bottlenecked" by small weights in subsequent layers. Conversely, a small update on a high-capacity path could be crucial. Selecting by magnitude alone (top-k) might prune small weights on critical paths while retaining large weights on dead ends, causing merged models to significantly lag behind Multi-Task Learning (MTL).

**Key Challenge**: Merging requires "equivalent functional compression of tasks," whereas magnitude provides only local parameter information, lacking global signals such as inter-layer coupling and cross-expert directional consistency. Furthermore, data-free merging cannot utilize any forward or backward gradient data to estimate "functional importance."

**Goal**: To calculate a saliency score for each coordinate of every task vector that accounts for inter-layer coupling under strictly data-free conditions (no task samples, no calibration sets), and to perform iterative pruning based on this. Additionally, the framework should seamlessly transition to LoRA experts without destroying low-rank structures.

**Key Insight**: The authors observe that SynFlow (Tanaka et al. 2020) from structured pruning provides a "data-free measure of end-to-end connectivity." While originally used for single-model pruning, it can be adapted as a saliency measure for task vectors. Another observation is "cross-expert consensus direction"—if an expert's update at a specific coordinate opposes the majority of other experts, it is likely noise rather than a valid update.

**Core Idea**: Structural sensitivity is measured using SynFlow-style connectivity gradients, modulated by the dot product with the sum of all task vectors (the "consensus direction") to obtain saliency $\mathcal{S}_n$. Refined through iterative top-k masking, the remaining task vectors are summed to obtain the merged model.

## Method

### Overall Architecture
The input consists of foundation parameters $\theta_0$ and $N$ fine-tuned experts $\{\theta_n\}$, converted into task vectors $\tau_n := \theta_n - \theta_0$. The process enters a loop of $T$ iterations. In each iteration: first, the current aggregate direction $\tau^* = \sum_i \tau_i$ is calculated; then, for each expert, the connectivity score $\mathcal{R}_n$ is computed to obtain structural sensitivity via gradients relative to $\tau_n$. This is multiplied by $\tau^*$ to derive saliency $\mathcal{S}_n$. Masks $m_n$ are generated based on the top-$(1-p)$ values within the tensor, updating $\tau_n \leftarrow m_n \odot \tau_n$. After $T$ iterations, all sparsified task vectors are added back to $\theta_0$. The entire process is data-free, using only parameters as input.

### Key Designs

1.  **Connectivity Saliency (SynFlow on task vectors)**:
    - **Function**: Uses a data-free scalar to measure "the contribution of each task vector coordinate to the end-to-end signal transmission capacity."
    - **Mechanism**: The network is viewed as $L$ sequential parameter blocks. The connectivity score is defined as $\mathcal{R}_n(\theta_0, \tau_n) = \mathbf{1}^\top (\prod_{l=1}^{L} |\theta_0^l + \tau_n^l|) \mathbf{1}$, and the gradient $\partial \mathcal{R}_n / \partial \tau_n$ is computed. This is essentially equivalent to "how many strong paths this coordinate participates in." Thus, a large update sandwiched between small weights in adjacent layers receives a near-zero gradient and naturally drops in importance ranking.
    - **Design Motivation**: Applying single-model SynFlow to merging scenarios provides a structural importance measure for free; it is orthogonal to magnitude/sign signals and serves as a new merging basis.

2.  **Consensus Direction Modulation for Implicit Sign Election**:
    - **Function**: Adds a "cross-expert consensus" layer on top of structural sensitivity to suppress isolated or opposing updates as noise.
    - **Mechanism**: Saliency is defined as $\mathcal{S}_n := \frac{\partial \mathcal{R}_n}{\partial \tau_n} \odot \sum_{i=1}^{N} \tau_i$. If an expert's update at a coordinate has a different sign than the aggregate direction, the product is negative or minimal. Only updates that are both "structurally important" and "aligned with the majority consensus" receive high saliency.
    - **Design Motivation**: Unlike TIES, which uses explicit sign voting, this method "dissolves" sign election into the saliency itself. This preserves multiplicative smoothing (unlike the hard cuts of voting) and avoids new hyperparameters while naturally coupling with connectivity sensitivity.

3.  **Iterative Saliency Pruning + Rank-wise LoRA Variant**:
    - **Function**: Gradually compresses each expert to the target sparsity over multiple rounds and extends the mechanism to LoRA experts.
    - **Mechanism**: Following SynFlow’s iterative pruning, masks are taken per tensor based on top-$(1-p)$ (avoiding global sorting that might prune entire small layers). After $T$ rounds, the retention rate is approximately $(1-p)^T$. The paper sets $T=10$ and a target retention of 10%, thus $p=0.2$. For LoRA, $\Delta W_n^l = s B_n^l A_n^l$ is treated as the task vector, but pruning is performed on rank-1 components rather than matrix elements. The saliency of the $k$-th rank component is $s_{n,k}^l = |\gamma_{n,k}^l \eta_{n,k}^l|$, where $\gamma_{n,k}^l = (b_{n,k}^l)^\top G_n^l a_{n,k}^l$ is structural sensitivity and $\eta_{n,k}^l = (b_{n,k}^l)^\top \overline{\Delta W}^l a_{n,k}^l$ is consistency with the aggregate update. Ranks are selected by saliency, and pruning is done via $B_n^l \leftarrow B_n^l \mathrm{Diag}(m_n^l)$ and $A_n^l \leftarrow \mathrm{Diag}(m_n^l) A_n^l$, preserving the LoRA structure.
    - **Design Motivation**: One-shot pruning ignores the drift of inter-layer dependencies during sparsification. Iterative pruning allows masks to self-align through prune-re-evaluate cycles. The rank-wise LoRA extension avoids destroying the low-rank structure that would occur with element-wise pruning.

### Loss & Training
The method has no training loss; the entire merging process involves zero backpropagation on task data and zero hyperparameter search. Computing $\partial \mathcal{R}_n / \partial \tau_n$ requires only a single automatic differentiation pass on parameters. For LoRA, gradients can be calculated directly on low-rank factors via $\partial \mathcal{R}/\partial B = s G (A)^\top$ and $\partial \mathcal{R}/\partial A = s B^\top G$ without explicitly expanding $\Delta W$.

## Key Experimental Results

### Main Results
Evaluations cover 4 scenarios: 8-task vision suites for CLIP ViT-B/32, B/16, and L/14; 8-task GLUE for RoBERTa-Base/Large; LoRA merging for Flan-T5-base; and decoder (instruction/math/code) merging suites.

| Dataset | Metric | Ours (SA-Merging) | Prev. SOTA data-free (WUDI) | Gain |
|---------|--------|-------------------|---------------------------|------|
| CLIP ViT-B/32 (8-task vision avg) | Top-1 acc | 85.9 | 85.2 | +0.7 |
| CLIP ViT-L/14 (8-task vision avg) | Top-1 acc | 93.4 | 92.6 | +0.8 |
| GLUE RoBERTa-Base | Norm. Avg. | 87.1 | 85.3 | +1.8 |
| GLUE RoBERTa-Large | Norm. Avg. | 90.2 | 88.8 | +1.4 |

Note: On CLIP ViT-L/14, SA-Merging's 93.4 is marginally higher than Traditional MTL (93.5) and exceeds all test-time/data-assisted methods (AdaMerging 90.8 / AdaMerging++ 91.0 / Representation Surgery 89.0), implying that strictly data-free merging can finally stand on equal footing with sample-dependent methods.

### Ablation Study
| Configuration | Key Findings | Description |
|---------------|--------------|-------------|
| Full SA-Merging | 8-task vision avg ≈ 85.9 (B/32) | Structural sensitivity + Consensus + Iteration |
| Magnitude only + Consensus | Significant drop to TIES levels | 退化 (Degrades) to weighted TIES without structural sensitivity |
| Structural sens. only (no consensus) | Intermediate performance | Cannot suppress conflicting updates from different experts |
| One-step pruning (T=1) | Weaker than T=10 | Confirms value of iterative refinement |
| Different pruning rates $p$ | Larger $T$ is more stable; trend agnostic to $p$ | Performance increases monotonically with $T$ (Figure 3b) |

### Key Findings
- Structural sensitivity $\partial \mathcal{R}_n / \partial \tau_n$ has low correlation with traditional magnitude ranking, confirming it captures "functional importance beyond magnitude" and complements existing magnitude-based methods.
- The number of iterations $T$ is the most robust knob: it shows a monotonic "the larger, the better" trend across vision and language tasks without evidence of mask overfitting.
- Rank-wise saliency for LoRA makes data-free LoRA merging significantly superior to naive element-wise pruning and preserves the rank structure for direct deployment.

## Highlights & Insights
- Reinterpreting "SynFlow from structured pruning" as "data-free saliency for task vectors" is a lightweight yet practical cross-domain transfer, providing a new merging basis that can be stacked with magnitude/sign methods.
- The design of consensus modulation $\odot \sum_i \tau_i$ is elegant—it integrates TIES-style sign election into the saliency multiplier, removing the need for explicit sign voting and thresholds while allowing magnitude to decide weight strength.
- The rank-wise saliency for LoRA is highly engineering-friendly: using dot products like $(b_{n,k}^l)^\top G a_{n,k}^l$ allows reuse of auto-diff on low-rank factors without materializing $\Delta W$, costing almost nothing even for very large models.

## Limitations & Future Work
- The connectivity score $\mathcal{R}_n$ relies on the product of $|\cdot|$, which can lead to numerical explosion or underflow in deep networks; log-domain normalization is needed (the paper lacks detailed implementation specifics on this).
- The consensus direction $\sum_i \tau_i$ is a simple sum; it can be biased by dominant experts when variance is high (e.g., LLM experts from vastly different domains). Introducing weighted or robust aggregation is a natural next step.
- Experiments focus on medium-scale merging with ~8 experts. Whether iterative pruning overhead and mask drift remain controllable when $N$ scales to dozens or hundreds of experts (modular/sparse merging) remains to be verified.

## Related Work & Insights
- **vs TIES-Merging / DARE / PCB**: While these use magnitude pruning and sign election/random dropping, this work replaces magnitude with structural sensitivity and integrates sign election into saliency. It is "complementary rather than a replacement."
- **vs WUDI-Merging**: Both are recent data-free SOTAs. WUDI uses task vector weighting, while this work focuses on "which coordinates participate in merging." They are orthogonal; SA-Merging consistently outperforms WUDI.
- **vs AdaMerging / Representation Surgery**: These rely on unlabeled test inputs for test-time tuning. SA-Merging's data-free results on ViT-L/14 catch up to or exceed these data-assisted methods, highlighting the potential of structural saliency signals.

## Rating
- Novelty: ⭐⭐⭐⭐ SynFlow-to-merging transfer is clever, consensus design is concise, LoRA rank-wise variant is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers Vision, GLUE, LoRA, and Decoders with comprehensive baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear derivations, unified notation, and well-explained design motivations.
- Value: ⭐⭐⭐⭐ Adds a new stackable basis for data-free model merging; LoRA extension is meaningful for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training](decouple_searching_from_training_scaling_data_mixing_via_model_merging_for_large.md)
- [\[ICML 2026\] When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging](when_shared_knowledge_hurts_spectral_over-accumulation_in_model_merging.md)
- [\[ICML 2026\] FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models](frism_fine-grained_reasoning_injection_via_subspace-level_model_merging_for_visi.md)
- [\[ICCV 2025\] Task Vector Quantization for Memory-Efficient Model Merging](../../ICCV2025/model_compression/task_vector_quantization_for_memory-efficient_model_merging.md)

</div>

<!-- RELATED:END -->
