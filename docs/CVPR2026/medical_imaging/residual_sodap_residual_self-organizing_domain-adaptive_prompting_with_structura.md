---
title: >-
  [Paper Note] Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning
description: >-
  [CVPR 2026][Medical Imaging][Continual Learning] This paper proposes the Residual SODAP framework, which jointly addresses prompt-side representation adaptation and classifier-side knowledge preservation through: α-entma…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Continual Learning"
  - "Domain-Incremental Learning"
  - "Prompt Learning"
  - "Catastrophic Forgetting"
  - "Knowledge Distillation"
date: 2026-05-08
content_hash: 255f217c58e52ede
---

# Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning

**Conference**: CVPR 2026
**arXiv**: [2603.12816](https://arxiv.org/abs/2603.12816)  
**Code**: N/A  
**Area**: Medical Imaging
**Keywords**: Continual Learning, Domain-Incremental Learning, Prompt Learning, Catastrophic Forgetting, Knowledge Distillation

## TL;DR

This paper proposes the Residual SODAP framework, which jointly addresses prompt-side representation adaptation and classifier-side knowledge preservation through: α-entmax sparse prompt selection with residual aggregation, data-free statistical distillation with pseudo-feature replay, prompt usage pattern drift detection (PUDD), and uncertainty-weighted multi-loss balancing. The framework achieves state-of-the-art performance on medical domain-incremental learning benchmarks.

## Background & Motivation

Continual learning (CL) faces the challenge of catastrophic forgetting, which is particularly severe in domain-incremental learning (DIL) settings where neither task identifiers nor historical data are available. Prompt-based continual learning (PCL) adapts to new domains by freezing the backbone and training only prompts, but suffers from two core limitations:

**Insufficient prompt selection mechanisms**:
   - Hard selection (Top-k): limits expressiveness and prevents gradient propagation through the selection process
   - Soft selection (Softmax): assigns non-zero weights to irrelevant prompts, leading to noise accumulation

**Neglect of classifier structure**: Existing PCL methods focus solely on prompt/prompt-pool design. However, through cross-combination diagnostic experiments (backbone × classifier), the authors find that even when backbone representations remain intact, the classifier layer exhibits significant performance degradation as domain-incremental training progresses (Fig. 1). This demonstrates that forgetting stems not only from representation drift but also from decision boundary instability.

## Method

### Overall Architecture

Residual SODAP comprises four core modules: (1) α-entmax residual prompt selection; (2) statistics-based knowledge preservation with pseudo-replay; (3) Prompt Usage Pattern Drift Detection (PUDD); and (4) uncertainty-weighted loss balancing.

### Key Designs

1. **α-Entmax Residual Prompt Selection**:

    - **Query augmentation**: At each Transformer layer $l$, the current CLS token $\mathbf{q}^{(l)}$, the global initial CLS $\mathbf{g}$, and a retrieval signal $\mathbf{r}^{(l)}$ obtained via MHA from a learnable memory bank $(\mathbf{M}_K, \mathbf{M}_V)$ are concatenated and passed through a bottleneck adapter to produce an augmented query $\tilde{\mathbf{q}}^{(l)}$. The memory bank is updated in a gradient-free manner via EMA (write operation) to maintain training stability.

    - **Sparse selection**: The augmented query is projected into a bottleneck space and its cosine similarity with prompt keys is computed to obtain logits, which are normalized via **α-entmax** ($\alpha=1.5$) in place of softmax:
    $[\alpha\text{-entmax}(\boldsymbol{\ell})]_j = \left[\frac{\alpha-1}{\alpha}(\ell_j - \tau(\boldsymbol{\ell}))\right]_+^{\frac{1}{\alpha-1}}$
    α-entmax assigns exact zero weights to low-scoring prompts, eliminating noise from irrelevant prompts while keeping the full pool differentiable.

    - **Frozen/active residual combination**: Starting from Stage 2, the prompt pool is partitioned into a frozen set $\mathcal{F}$ and an active set $\mathcal{A}$. Independent α-entmax routing is performed over each set, and the results are combined in residual form:
    $\mathbf{p}_{\text{out}}^{(l)} = \mathbf{p}_{\mathcal{F}}^{(l)} + \lambda_r \mathbf{p}_{\mathcal{A}}^{(l)}, \quad \lambda_r = 0.1$
    The frozen set serves as a stable foundation preserving prior knowledge, while the active set contributes only residual fine-tuning for new domain adaptation.

2. **Statistical Knowledge Preservation**:

    - **Saving knowledge assets at stage transitions**: The current classification head is frozen as a teacher; class-level feature statistics $(\boldsymbol{\mu}_c, \boldsymbol{\sigma}_c^2)$ are computed using the Welford online algorithm in a single pass, with memory efficiency and numerical stability.

    - **Real-feature distillation**: Current-batch real features are passed through both the teacher and student heads, and KL divergence is used for alignment:
    $\mathcal{L}_{\text{real}} = \text{KL}\left(\text{softmax}(\mathbf{z}_t/T) \| \text{softmax}(\mathbf{z}_s/T)\right) \cdot T^2$

    - **Pseudo-feature replay**: $K$ pseudo-features are sampled from stored class statistics as $\tilde{\mathbf{f}}_k \sim \mathcal{N}(\boldsymbol{\mu}_{c_k}, \text{diag}(\boldsymbol{\sigma}_{c_k}^2))$ (with uniformly sampled class indices to avoid underrepresentation of minority classes). After applying stop-gradient, pseudo-features are passed through both teacher and student heads to compute the distillation loss $\mathcal{L}_{\text{pseudo}}$. This preserves classifier decision boundaries without storing any raw data.

3. **Prompt Usage Pattern Drift Detection (PUDD)**:

    - Two drift signals are extracted: (i) selection entropy $H_t$ (short-term fluctuations increase upon domain shift as the distribution is re-adjusted); (ii) usage-set IoU ($\text{IoU}_t = |\mathcal{S}_t \cap \mathcal{S}_t^{\text{ref}}| / |\mathcal{S}_t \cup \mathcal{S}_t^{\text{ref}}|$, where low IoU indicates activation of different prompts).

    - Drift score fusion:
    $D_t = \alpha \cdot \frac{|H_t - \bar{H}_t|}{\sigma_{H,t} + \epsilon} + \beta \cdot \left(\frac{1}{\max(\text{IoU}_t, \eta)} - 1\right)$

    - **Drift-proportional pool expansion**: The number of newly added prompts is $E = \text{clamp}\left(\lfloor|\mathcal{A}| \cdot \bar{D}/D_{\max}\rfloor, E_{\min}, E_{\max}\right)$, yielding modest expansion for weak drift and greater expansion for strong drift.

### Loss & Training

The total loss is automatically balanced via uncertainty weighting (Kendall et al.):

$$\mathcal{L}_{\text{total}} = \sum_i \left(e^{-s_i} \mathcal{L}_i + s_i\right)$$

where $s_i = \log \sigma_i^2$ is a learnable log-variance. Loss terms with high uncertainty (large variance) are automatically down-weighted, and the regularization term $s_i$ prevents the degenerate solution of diverging variance. $s_i$ is initialized to 0 (equal weighting at start) and clipped to $[-3, 6]$.

Auxiliary losses include: a diversity loss $\mathcal{L}_{\text{div}}$ (suppressing similarity between frequently co-activated prompt values) and a norm regularization $\mathcal{L}_{\text{norm}}$ (constraining the active prompts to serve only as residuals).

## Key Experimental Results

### Main Results

Comparison on three DIL benchmarks (no Task-ID, no external data storage):

| Dataset | Metric | Residual SODAP | Prev. SOTA (OS-Prompt++) | Gain |
|--------|------|---------------|---------------------|------|
| DR (Diabetic Retinopathy) | AvgACC↑ | **0.850** | 0.769 | +0.081 |
| DR | AvgF↓ | **0.047** | 0.113 | -0.066 |
| Skin Cancer | AvgACC↑ | **0.760** | 0.725 | +0.035 |
| Skin Cancer | AvgF↓ | 0.031 | 0.063 | -0.032 |
| CORe50 | AvgACC↑ | **0.995** | 0.983 | +0.012 |
| CORe50 | AvgF↓ | **0.003** | 0.014 | -0.011 |

### Ablation Study

| Configuration | Key Metric | Remarks |
|------|---------|------|
| α-entmax vs. Softmax | Significant AvgACC improvement | Sparse selection eliminates noise from irrelevant prompts |
| w/o residual combination | Performance degradation | Frozen/active residual structure is critical for knowledge preservation |
| w/o PUDD | Fixed prompt pool size | Unable to adaptively adjust capacity for new domains |
| w/o pseudo-feature replay | Classifier degradation | Confirms the existence of classifier-level forgetting |
| w/o uncertainty weighting | Requires manual tuning | Automatic weighting simplifies hyperparameters and improves stability |

### Key Findings

- Classifier-level forgetting is an overlooked yet significant source of failure in PCL; jointly optimizing prompts and classifiers yields substantial gains.
- α-entmax ($\alpha=1.5$) achieves the optimal balance between softmax (expressive but noisy) and sparsemax (extremely sparse).
- Statistics-based pseudo-feature replay effectively preserves classifier decision boundaries with zero data storage.

## Highlights & Insights

- This work is the first to systematically address both representation adaptation and classifier knowledge preservation simultaneously within PCL.
- The PUDD drift detection scheme cleverly leverages information inherent in prompt selection patterns themselves, without requiring a separate domain discriminator.
- The data-free distillation approach combining Welford online statistics with diagonal Gaussian pseudo-replay is elegantly simple yet effective.
- Uncertainty weighting eliminates the need for manual tuning of multiple loss terms.

## Limitations & Future Work

- Validation is limited to classification tasks and has not been extended to dense prediction tasks such as segmentation or detection.
- Prompt pool expansion is monotonically increasing with no compression or pruning mechanism, potentially causing parameter bloat in long-term deployment.
- The PUDD threshold $\theta$ and window size $W$ remain as hyperparameters with limited adaptivity.
- The diagonal Gaussian assumption may be insufficient to capture complex class-conditional feature distributions.

## Related Work & Insights

- Compared against PCL methods including OS-Prompt, Coda-Prompt, and Dual-Prompt, highlighting the advantages of sparse selection and classifier preservation.
- The pseudo-feature replay idea is generalizable to other data-free continual learning scenarios.
- Drift detection combined with dynamic prompt expansion can be applied to adaptive architecture adjustment in online learning systems.

## Rating

- Novelty: ⭐⭐⭐⭐ Multiple innovative components work synergistically; the analysis perspective on classifier-level forgetting is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three benchmarks, multiple baselines, complete ablations, and three independent runs.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are detailed and the motivation for each module is clearly articulated.
- Value: ⭐⭐⭐⭐ Directly applicable to medical DIL scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](apex_adaptive_visual_prompting.md)
- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)
- [\[CVPR 2026\] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)
- [\[AAAI 2026\] ConSurv: Multimodal Continual Learning for Survival Analysis](../../AAAI2026/medical_imaging/consurv_multimodal_continual_learning_for_survival_analysis.md)
- [\[AAAI 2026\] Learning with Preserving for Continual Multitask Learning](../../AAAI2026/medical_imaging/learning_with_preserving_for_continual_multitask_learning.md)

</div>

<!-- RELATED:END -->
