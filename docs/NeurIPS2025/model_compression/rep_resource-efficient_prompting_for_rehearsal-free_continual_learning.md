---
title: >-
  [Paper Note] REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning
description: >-
  [NeurIPS 2025][Model Compression][Continual Learning] REP reduces training time by up to 51% and memory consumption by up to 41% for prompt-based rehearsal-free continual learning methods, with negligible accuracy loss…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Continual Learning"
  - "Resource Efficiency"
  - "Prompt Learning"
  - "Token Merging"
  - "Layer Dropping"
date: 2026-05-08
content_hash: fd3429b3fc459c94
---

# REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning

**Conference**: NeurIPS 2025
**arXiv**: [2406.04772](https://arxiv.org/abs/2406.04772)  
**Code**: N/A  
**Area**: Model Compression / Continual Learning
**Keywords**: Continual Learning, Resource Efficiency, Prompt Learning, Token Merging, Layer Dropping

## TL;DR

REP reduces training time by up to 51% and memory consumption by up to 41% for prompt-based rehearsal-free continual learning methods, with negligible accuracy loss, via three complementary techniques: fast prompt selection using a lightweight surrogate model, Adaptive Token Merging (AToM), and Adaptive Layer Dropping (ALD).

## Background & Motivation

Continual Learning (CL) trains models sequentially on multiple tasks, with catastrophic forgetting as the core challenge. Prompt-based rehearsal-free methods (e.g., L2P, DualPrompt, CODA-Prompt) adapt a frozen pretrained ViT by learning a small number of prompt parameters per task, avoiding the storage of past data and making them suitable for edge device deployment.

**Limitations of Prior Work**:

**High cost of prompt selection**: Typically requires a forward pass through the backbone (e.g., ViT-L) to compute query features for prompt retrieval, adding up to 28% additional computation time.

**Expensive prompt updates**: Despite the frozen backbone, each mini-batch still requires a full forward and backward pass to optimize prompts and the classification head, necessitating storage of all intermediate activations.

**Strict edge device constraints**: Device memory is typically 1–8 GB, and computational efficiency directly affects energy consumption and device longevity.

**Core Insights**:
- The prompt selection stage tolerates large approximation errors — a full-scale backbone is unnecessary.
- Different layers of the frozen backbone contribute unequally to new tasks — shallow layers are more important (more diverse attention distances), while deep layers tend to be global and homogeneous.

## Method

### Overall Architecture

The REP framework comprises three complementary techniques:
1. Replacing the backbone with a lightweight surrogate model for prompt selection.
2. AToM: adaptively merging redundant tokens during prompt updates.
3. ALD: adaptively dropping deep layers during prompt updates.

Each technique targets a distinct computational bottleneck and can be used independently or in combination.

### Key Designs

1. **Prompt Selection via Lightweight Surrogate Model**: A compact ViT-Ti (5.8M parameters) replaces ViT-L (307M parameters) for computing query features. Since ViT-Ti has a lower feature dimensionality $d < D$, a fixed random projection $\phi$ maps the low-dimensional features back to the original $D$-dimensional space for prompt matching:
    $p^*_{\text{efficient}} = \underset{p_k \in P}{\text{argmax}} \frac{\langle \phi(q_{\text{efficient}}(x_i^j)), p_k \rangle}{\|\phi(q_{\text{efficient}}(x_i^j))\| \|p_k\|}$
   This strategy is empirically shown to preserve approximately 97% of representational similarity (measured by CKA).

2. **Adaptive Token Merging (AToM)**: AToM introduces two key differences from standard Token Merging (ToMe):

    - **Protecting Prompt Tokens**: Standard ToMe does not distinguish between prompt and non-prompt tokens, which dilutes task-specific information in prompts and can cause gradient explosion. AToM excludes prompt tokens from the merging process.
    - **Progressive Scheduler**: Standard ToMe merges a uniform number $n$ of tokens per layer. AToM uses a progressive schedule:
    $r'(l) = \min(\delta \times (l-1), r_{\max})$
      where $\delta = r_{\max}/(L-1)$. Fewer tokens are merged in shallow layers (preserving important local and task-specific information) and more in deep layers (where information is already global and highly redundant). The default setting is $r_{\max} = 2n$.

3. **Adaptive Layer Dropping (ALD)**: Unlike uniform random dropping (Progressive Layer Dropping, PLD), ALD jointly considers temporal and spatial dimensions, leveraging feedback from AToM to guide layer dropping decisions. The layer retention probability is:
    $\theta_{t,l} = \alpha(l) \times ((1-\bar{\theta})\exp(-\gamma \cdot t) + \bar{\theta})$
   where $\alpha(l)$ is adjusted based on the number of tokens merged at that layer: when the cumulative merged token count exceeds threshold $\tau$ (typically in deep layers), $\alpha(l) = 0.9$ (more likely to drop); otherwise $\alpha(l) = 1$ (retain). This ensures shallow layers are preferentially retained while deep layers are more aggressively dropped.

### Loss & Training

REP does not modify the loss functions of the underlying CL methods. The standard framework is:
$$L = L_{\text{class}}(f_{\text{update}}(x_i^j), y_i^j) + \epsilon_1 L_{\text{prompt}}(p^*, q(x_i^j)) + \epsilon_2 L_{\text{aux}}$$

REP optimizes only the computational path — determining which tokens participate in computation and which layers execute forward/backward passes — without altering the loss or learning objectives.

## Key Experimental Results

### Main Results (7 prompt methods × 3 ViT backbones × 3 datasets)

| Model | Method | Dataset | w/o REP Acc | w/ REP Acc | Time Speedup | Memory Saving |
|------|------|--------|------------|-----------|---------|---------|
| ViT-L | L2P | Split ImageNet-R | 75.6 | 75.3 | 1.9× | 1.4× |
| ViT-L | DualPrompt | Split ImageNet-R | 71.2 | 70.6 | 2.0× | 1.4× |
| ViT-L | HiDe-Prompt | Split ImageNet-R | 78.7 | 78.0 | 1.8× | 1.2× |
| ViT-L | ConvPrompt | Split ImageNet-R | 79.1 | 78.5 | 1.3× | 1.3× |
| ViT-B | HiDe-Prompt | Split ImageNet-R | 64.5 | 64.4 | 1.7× | 1.7× |

Accuracy degradation ranges from 0.0–1.2% (Split CIFAR-100), 0.1–1.1% (Split ImageNet-R), and 0.0–0.8% (Split PlantDisease). In some cases REP even improves accuracy (e.g., L2P + ViT-L on PlantDisease: 75.9% → 81.1%).

### Ablation Study

| Configuration | Acc | Iter. Time (ms) | Memory (GB) | Notes |
|---------|------|-------------|---------|------|
| Full REP-L2P | 75.3 | 240 | 4.5 | Best balance |
| w/o AToM+ALD | 74.8 | 349 | 5.5 | Joint contribution is large |
| w/ ToMe (replaces AToM) | 70.2 | 275 | 3.7 | Accuracy drops 5.1% |
| w/ PLD (replaces ALD) | 73.3 | 259 | 4.5 | PLD yields 2% accuracy gap |
| Random Drop-25% | 70.6 | 398 | 6.5 | Uniform dropping infeasible |
| ALD only | 75.8 | 401 | 6.5 | Adaptive dropping preserves accuracy |

### Hyperparameter Sensitivity

| Token Merge Count $n$ | Acc | Time (ms) | Memory (GB) |
|--------------|------|---------|---------|
| 4 | 75.3 | 256 | 5.2 |
| 8 (default) | 75.3 | 240 | 4.5 |
| 10 | 73.6 | 228 | 4.1 |

| Retention Prob. $\bar{\theta}$ | Acc | Time (ms) |
|------------|------|---------|
| 0.1 | 72.9 | 217 |
| 0.5 (default) | 75.3 | 240 |
| 0.9 | 74.3 | 282 |

### Key Findings

- The core of AToM lies in protecting prompt tokens — standard ToMe causes gradient explosion, while excluding prompt tokens from merging stabilizes gradients.
- ALD leverages the token merge count from AToM for joint decision-making, enabling coordinated optimization across spatial and temporal dimensions.
- The finding that shallow layers are more important than deep layers for new tasks holds consistently across diverse prompt methods and backbone architectures.
- REP generalizes to non-prompt methods (SLCA, RanPAC), reducing training time by 37–48% and memory by up to 48%.
- Larger models (ViT-L) benefit more from REP, as they offer greater room for optimization.

## Highlights & Insights

- **Cost-accuracy analysis drives design**: Rather than compressing blindly, REP performs in-depth analysis of attention distances to identify distinct optimization opportunities in the prompt selection and update stages.
- **AToM vs. ToMe — the critical distinction**: Protecting prompt tokens and applying a progressive schedule are two simple modifications that yield a 5% accuracy gap, revealing the unique requirements of token merging in the CL setting.
- **AToM–ALD coupling**: ALD does not independently schedule its dropping parameters; instead, it uses the merge count from AToM as feedback, enabling natural coupling between the two techniques.
- **Broad applicability**: Validated across 7 prompt methods, 3 backbones, and 3 datasets, with further extension to 2 non-prompt methods, demonstrating strong reproducibility.

## Limitations & Future Work

- The surrogate model requires maintaining an additional ViT-Ti, increasing deployment complexity.
- Hyperparameters of AToM and ALD ($n$, $\bar{\theta}$, $\tau$, $\alpha$) require tuning for different model scales.
- Evaluation is limited to image classification; applicability to vision-language, detection, and other tasks remains unknown.
- Whether random projection consistently preserves 97% CKA similarity across all settings requires further verification.
- The theoretical guarantees for layer dropping are weak — the approach relies on empirical observations rather than formal analysis.

## Related Work & Insights

REP addresses a gap in resource efficiency for prompt-based CL methods. Unlike resource-aware CL methods based on CNNs such as BudgetCL and CarM, REP focuses on the ViT architecture. Its core insight — that shallow ViT layers are more sensitive to new tasks than deep layers — is consistent with findings in ViT pretraining research, but is applied here for the first time to computational optimization in continual learning. The design of protecting prompt tokens in AToM can be generalized to any method that employs learnable tokens within a Transformer.

## Rating

- Novelty: ⭐⭐⭐⭐ The prompt-token protection in AToM and the coupled ALD design are novel, though individual components are relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 methods × 3 models × 3 datasets, extensive ablations, and extension to non-prompt methods.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated and experiments are systematically organized.
- Value: ⭐⭐⭐⭐ Addresses a practical bottleneck for deploying prompt-based methods on edge devices; high engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](../../ICCV2025/model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)
- [\[CVPR 2026\] Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge](../../CVPR2026/model_compression/critical_patch-aware_sparse_prompting_with_decoupled_training_for_continual_lear.md)
- [\[NeurIPS 2025\] Train with Perturbation, Infer after Merging: A Two-Stage Framework for Continual Learning](train_with_perturbation_infer_after_merging_a_two-stage_framework_for_continual_.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[NeurIPS 2025\] A*-Thought: Efficient Reasoning via Bidirectional Compression for Low-Resource Settings](a-thought_efficient_reasoning_via_bidirectional_compression_for_low-resource_set.md)

</div>

<!-- RELATED:END -->
