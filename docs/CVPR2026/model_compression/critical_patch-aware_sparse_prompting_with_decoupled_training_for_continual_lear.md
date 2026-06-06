---
title: >-
  [Paper Note] Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge
description: >-
  [CVPR 2026][Model Compression][Continual Learning] This paper proposes CPS-Prompt, a framework that combines task-aware critical patch sampling (CPS) and decoupled prompt-classifier training (DPCT) to achieve approximate…
tags:
  - "CVPR 2026"
  - "Model Compression"
  - "Continual Learning"
  - "Edge Devices"
  - "Prompt-based CL"
  - "Token Reduction"
  - "Training Efficiency"
date: 2026-05-08
content_hash: 23486ddfe60c6625
---

# Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge

**Conference**: CVPR 2026
**arXiv**: [2604.07399](https://arxiv.org/abs/2604.07399)  
**Code**: [https://github.com/laymond1/cps-prompt](https://github.com/laymond1/cps-prompt)  
**Area**: Model Compression / Continual Learning
**Keywords**: Continual Learning, Edge Devices, Prompt-based CL, Token Reduction, Training Efficiency

## TL;DR
This paper proposes CPS-Prompt, a framework that combines task-aware critical patch sampling (CPS) and decoupled prompt-classifier training (DPCT) to achieve approximately 1.6× reduction in training-time memory and computation for prompt-based continual learning on edge devices, with only ~2% accuracy degradation.

## Background & Motivation

**Background**: Continual learning (CL) on edge devices (home robots, drones, smartphones) requires adapting to new tasks under constrained memory and compute budgets. Prompt-based continual learning (PCL) achieves parameter-efficient learning by freezing a ViT backbone and learning lightweight prompts, yet existing work primarily focuses on accuracy and inference efficiency.

**Limitations of Prior Work**: PCL methods such as C-Prompt achieve high accuracy but incur enormous training-time memory overhead (4.3× relative to the proposed method), making them unsuitable for memory-constrained edge deployment. OS-Prompt simplifies the two-stage pipeline but still exhibits high peak memory during backpropagation.

**Key Challenge**: Existing token reduction methods (ToMe, PatchDropout) are task-agnostic; when combined with PCL, they discard task-critical patches, causing severe accuracy degradation.

**Goal**: Achieve significant training-time memory and compute savings within the two-stage PCL architecture while maintaining competitive accuracy.

**Key Insight**: Leverage attention weights and value signals from the last layer of the frozen query encoder to estimate patch importance for task-aware sparsification, and eliminate the representational misalignment between sparse training and full-patch inference through decoupled training.

**Core Idea**: Task-aware patch sampling + decoupled prompt/classifier training = training efficiency + accuracy preservation.

## Method

### Overall Architecture
CPS-Prompt follows the standard two-stage PCL architecture: (1) a frozen query encoder $f_q$ performs a forward pass to generate task cues; (2) a prompt-injected backbone $f_p$ performs classification. The CPS module is inserted between the two stages to select critical patches, and the DPCT strategy trains the prompt and classifier in two separate phases.

### Key Designs

1. **Critical Patch Sampling (CPS)**:

    - **Function**: Extracts the class-to-patch attention weights $A^L_{\text{cls},j}$ and the L2 norm of value vectors $\|V^L_j\|_2$ from the last layer of the query encoder, and computes a criticality score:
    $s_j = A^L_{\text{cls},j} \cdot \|V^L_j\|_2$
    - **Mechanism**: Attention weights reflect each patch's contribution to the class representation, while the value norm captures feature saliency. Their product forms a comprehensive importance score. After temperature-scaled softmax:
    $p_j = \frac{\exp(s_j/\tau)}{\sum_i \exp(s_i/\tau)}$
      $k = \lfloor(1-r) \cdot N\rfloor$ patches are selected via multinomial sampling without replacement.
    - **Design Motivation**: The frozen backbone's prior knowledge enables training-free, task-aware sparsification; stochastic sampling introduces diversity to avoid overfitting; temperature $\tau$ controls the trade-off between determinism and exploration.

2. **Decoupled Prompt and Classifier Training (DPCT)**:

    - **Function**: Divides $E$ epochs into two phases — the first $\lfloor \lambda \cdot E \rfloor$ epochs jointly optimize prompt $\phi$ and classifier $\theta$ using sparse-patch inputs; the remaining epochs freeze the prompt and fine-tune only the classifier using full-patch inputs.
    - **Mechanism**:
    $\mathcal{L}_p = \mathcal{L}(f_p(\mathbf{X}_{\text{sampled}}; \theta, \phi), y)$
    $\mathcal{L}_{\text{cls}} = \mathcal{L}(f_p(\mathbf{X}_{\text{full}}; \theta, \phi), y), \quad \text{(}\phi \text{ frozen)}$
    - **Design Motivation**: Training with sparse patches causes the learned prompt representations to misalign with full-patch inference; fine-tuning the classifier separately on full patches eliminates this misalignment. Freezing the prompt also prevents gradients from propagating back through it, reducing computational overhead.

3. **Temperature-Controlled Stochastic Sampling vs. Deterministic Top-$k$**:

    - Experiments demonstrate that stochastic sampling outperforms deterministic Top-$k$ selection, as controlled randomness promotes exploration diversity during training, improving generalization to new tasks.

### Loss & Training
- Standard cross-entropy loss is used throughout.
- Both training phases use the Adam optimizer with cosine learning rate decay starting at 0.001.
- Optimal hyperparameters: patch reduction ratio $r=0.4$, temperature $\tau=0.1$.

## Key Experimental Results

### Main Results

| Dataset | Metric | CPS-Prompt | C-Prompt (SOTA) | CODA-Prompt | Notes |
|--------|------|-----------|-----------------|-------------|---------|
| CIFAR-100 | ACC↑ | 66.89 | **68.34** | 67.06 | Gap of only 1.45% |
| ImageNet-R | ACC↑ | 49.96 | **53.32** | 50.24 | Gap of 3.36% |
| CUB-200 | ACC↑ | 52.85 | 52.64 | **53.96** | Comparable to CODA |

Efficiency comparison (measured on Jetson Orin Nano):

| Method | Peak Memory | Training Time | Energy Consumption |
|------|------------|------------|---------|
| CPS-Prompt | **1×** | **1×** | **1×** |
| CODA-Prompt | ~1.6× | ~1.5× | ~1.6× |
| C-Prompt | ~4.3× | ~3.1× | ~3.3× |

### Ablation Study

| Configuration | ACC↑ (ImageNet-R) | Memory | Training Time | Notes |
|------|-----------------|------|---------|------|
| CODA-Prompt baseline | 50.24 | 440MB | 1788s | Baseline |
| + PD (random patch drop) | 45.32 | 253MB | 1388s | Large accuracy drop |
| + CPS (task-aware) | 47.16 | 253MB | 1389s | +1.8% over PD |
| + PD + DPCT | 47.96 | 253MB | 1126s | DPCT recovers accuracy |
| + CPS + DPCT (full) | **49.28** | **253MB** | **1126s** | Best configuration |

### Key Findings
- CPS and DPCT provide complementary gains: CPS improves patch quality while DPCT eliminates representational misalignment.
- Even with memory reduction exceeding 60%, CPS-Prompt retains over 90% of baseline accuracy.
- Stochastic sampling outperforms deterministic Top-$k$ particularly at low phase ratios.
- Temperature $\tau=0.1$ (sharper distribution) yields the best performance across all datasets.

## Highlights & Insights
- **Genuine edge deployment perspective**: Comprehensive real-device measurements (memory, time, energy) are conducted on a Jetson Orin Nano, rather than reporting only theoretical FLOPs.
- **Task-aware token reduction**: The method elegantly repurposes the query forward pass already present in the two-stage PCL architecture, introducing zero additional training overhead.
- **Simplicity of decoupled training**: Training the prompt and classifier separately with sparse and full patches respectively is a straightforward yet effective design.

## Limitations & Future Work
- Validation is limited to ViT-Tiny/16; performance on larger models (ViT-Base/Large) remains unknown.
- The patch reduction ratio $r=0.4$ is fixed; dynamic adaptive strategies are not explored.
- Only the class-incremental setting is considered; task-incremental and domain-incremental scenarios are not addressed.
- No comparison is made against more recent VLM-based CL methods.

## Related Work & Insights
- Comparisons with ToMe (token merging) and PatchDropout confirm that task-agnostic token reduction performs poorly in PCL settings.
- The motivation behind DPCT is analogous to the train-inference inconsistency problem in knowledge distillation.
- The CPS approach may generalize to other ViT downstream tasks that require token reduction.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of task-aware patch sampling and decoupled training is novel, though each individual module offers limited standalone technical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, real edge hardware, comprehensive ablations, and efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with complete algorithm diagrams and pseudocode.
- **Value**: ⭐⭐⭐⭐ Practically meaningful for edge continual learning, though the overall scope is relatively niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Elastic Weight Consolidation Done Right for Continual Learning](elastic_weight_consolidation_done_right_for_continual_learning.md)
- [\[NeurIPS 2025\] REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning](../../NeurIPS2025/model_compression/rep_resource-efficient_prompting_for_rehearsal-free_continual_learning.md)
- [\[CVPR 2026\] MEMO: Human-like Crisp Edge Detection Using Masked Edge Prediction](memo_human-like_crisp_edge_detection_using_masked_edge_prediction.md)
- [\[ICML 2026\] Energy-Structured Low-Rank Adaptation for Continual Learning](../../ICML2026/model_compression/energy-structured_low-rank_adaptation_for_continual_learning.md)
- [\[ICLR 2026\] IDER: IDempotent Experience Replay for Reliable Continual Learning](../../ICLR2026/model_compression/ider_idempotent_experience_replay_for_reliable_continual_learning.md)

</div>

<!-- RELATED:END -->
