---
title: >-
  [Paper Note] Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning
description: >-
  [CVPR 2026 (PHAROS-AIF-MIH Workshop)][Medical Imaging][Fair Diagnosis] An attention-based MIL model is built upon a ConvNeXt-Base backbone, employing a gradient reversal layer (GRL) to adversarially eliminate gender info…
tags:
  - "CVPR 2026 (PHAROS-AIF-MIH Workshop)"
  - "Medical Imaging"
  - "Fair Diagnosis"
  - "Chest CT"
  - "Multiple Instance Learning"
  - "Gradient Reversal Layer"
  - "Lung Disease Classification"
date: 2026-05-08
content_hash: af29b6ff11ab758b
---

# Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning

**Conference**: CVPR 2026 (PHAROS-AIF-MIH Workshop)  
**arXiv**: [2603.12988](https://arxiv.org/abs/2603.12988)  
**Code**: [GitHub](https://github.com/ADE-17/cvpr-fair-chest-ct)  
**Area**: Medical Imaging  
**Keywords**: Fair Diagnosis, Chest CT, Multiple Instance Learning, Gradient Reversal Layer, Lung Disease Classification

## TL;DR

An attention-based MIL model is built upon a ConvNeXt-Base backbone, employing a gradient reversal layer (GRL) to adversarially eliminate gender information from scan representations. Combined with focal loss ($\gamma=2$) + label smoothing ($\varepsilon=0.1$), subgroup oversampling, and 5-fold ensemble, the proposed method achieves a mean competition score of 0.685±0.030 on a four-class lung disease diagnosis task over 889 chest CT scans. The female macro-F1 (0.691) slightly exceeds the male macro-F1 (0.679), validating that GRL effectively closes the fairness gap.

## Background & Motivation

**Background**: Deep learning has achieved remarkable progress in automated chest CT analysis, enabling large-scale screening for pulmonary malignancies and COVID-19. However, fairness research indicates that models tend to encode and amplify demographic biases present in training data, leading to systematically inferior diagnostic outcomes for disadvantaged groups.

**Limitations of Prior Work**: The CVPR 2026 PHAROS-AIF-MIH challenge dataset (889 CT scans: 734 training / 155 validation, four classes: Healthy / COVID-19 / Adenocarcinoma / Squamous Cell Carcinoma) exhibits severe cross-group imbalance—only 18 female squamous cell carcinoma cases versus 91 male cases—while CT depth varies widely from fewer than 20 to over 800 slices. The competition metric is the average male and female macro-F1: $P = \frac{1}{2}(\text{MacroF1}_\text{male} + \text{MacroF1}_\text{female})$, directly penalizing gender unfairness.

**Key Challenge**: Three intertwined challenges are identified: (1) sparse volumetric signals—only a few slices among hundreds contain lesions, causing mean pooling to be overwhelmed by healthy slices; (2) demographic imbalance—extreme scarcity of female squamous cell carcinoma cases leads to severe underrepresentation under standard training; (3) gender as an implicit shortcut—even without explicit gender input, models can encode gender information from body habitus and acquisition parameters, coupling it with disease co-occurrence statistics.

**Goal**: To simultaneously address signal sparsity, subgroup imbalance, and gender encoding within an end-to-end framework, enabling gender-fair four-class lung disease diagnosis.

**Key Insight**: CT volumes are treated as bags of slices under the MIL paradigm for automatic informative-slice selection; GRL is adopted for adversarial gender disentanglement; and a fairness-aware protocol is applied to balance subgroups.

**Core Idea**: Attention-based MIL aggregates informative slices; GRL eliminates gender shortcuts; subgroup oversampling closes the fairness gap.

## Method

### Overall Architecture

Input CT volumes (capped at $M=32$ slices) are processed by ConvNeXt-Base, which extracts a $D$-dimensional embedding per slice. A two-layer MLP attention network computes per-slice weights that are used to produce a weighted-sum scan-level representation $H$. This representation is subsequently fed into a 4-class disease classification head and a gender adversarial head (binary classifier) connected via a GRL, with both heads trained jointly end-to-end. At inference time, a 5-fold full ensemble combined with horizontal-flip TTA and out-of-fold (OOF) threshold optimization is applied.

### Key Designs

1. **Attention-Based MIL Aggregation**

    - **Function**: Learns which slices in a variable-length CT sequence contain diagnostic information and aggregates them into a scan-level representation via weighted summation.
    - **Mechanism**: ConvNeXt-Base (classification head removed) extracts per-slice embeddings $h_i = f_\text{enc}(x_i) \in \mathbb{R}^D$; a two-layer MLP produces importance scores $s_i = a(h_i; \theta_a)$, which are softmax-normalized and used to compute $H = \sum_i w_i h_i$. Padding positions are masked out in the attention computation. During training, volumes with $N>M$ slices are randomly sampled; at inference, uniform sampling is used to preserve spatial coverage.
    - **Design Motivation**: Mean pooling dilutes diagnostic signals with healthy slices, while max pooling is sensitive to artifacts. The attention mechanism serves as a learnable compromise between the two and requires no slice-level annotations.

2. **GRL-Based Gender Debiasing**

    - **Function**: Erases gender-predictive information from the scan representation, preventing the model from exploiting gender as a diagnostic shortcut.
    - **Mechanism**: A GRL followed by a two-layer MLP binary classifier $z_\text{gen} = c(\mathcal{R}_\lambda(H))$ is attached to $H$. The GRL acts as identity in the forward pass and negates and scales gradients by $\lambda_\text{adv}$ in the backward pass. The total loss is $\mathcal{L} = \mathcal{L}_\text{disease} + \lambda_\text{adv} \cdot \mathcal{L}_\text{gender}$; the gender head is trained to predict gender, while the reversed gradients force the backbone to discard gender information.
    - **Design Motivation**: The backbone can implicitly encode gender features from body habitus and acquisition parameters. GRL represents the minimally invasive fairness constraint—it leaves the primary task architecture unchanged and only introduces an adversarial branch.

3. **Fairness-Aware Training Protocol**

    - **Function**: A multi-pronged approach to ensure that severely imbalanced subgroups (only 18 female squamous cell carcinoma cases) are not neglected during training.
    - **Mechanism**: (a) 5-fold CV stratified by (class, gender) into 8 subgroups, ensuring all subgroups appear in each fold; (b) WeightedRandomSampler substantially increases the sampling weight of female squamous cell carcinoma cases so that nearly every batch contains this subgroup; (c) two-stage fine-tuning—the backbone is frozen for the first 5 epochs to train only the attention and the two heads (LR=1e-3), followed by full unfreezing (backbone LR=1e-5, heads LR=1e-4, cosine annealing).
    - **Design Motivation**: No single strategy suffices under extreme imbalance—oversampling prevents class collapse, stratified folds ensure fair evaluation, and the two-stage schedule allows the attention mechanism to stabilize before the backbone is updated.

### Loss & Training

- Disease loss: focal loss ($\gamma=2, \alpha=0.25$) + label smoothing ($\varepsilon=0.1$), $\tilde{p}_t = (1-\varepsilon)p_t + \varepsilon/C$
- Gender loss: binary cross-entropy
- AdamW ($\beta_1=0.9, \beta_2=0.999$, WD=0.05); gradient accumulation $K=4$ (effective batch size=16); 50 epochs; single RTX A4000 GPU
- Inference: 5-fold soft logit voting + horizontal-flip TTA; OOF per-class threshold optimization over a dense grid $\mathcal{T} \subset [0.05, 0.95]$

## Key Experimental Results

### Main Results — Per-Fold Validation

| Fold | Competition Score P | Male macro-F1 | Female macro-F1 | F1-Adeno | F1-Squamous |
|------|---------------------|---------------|-----------------|----------|-------------|
| 0 | 0.698 | 0.673 | 0.722 | 0.807 | 0.258 |
| 1 | **0.727** | 0.754 | 0.699 | 0.796 | 0.378 |
| 2 | 0.674 | 0.658 | 0.690 | 0.692 | 0.500 |
| 3 | 0.688 | 0.743 | 0.634 | 0.803 | 0.303 |
| 4 | 0.637 | 0.565 | 0.709 | 0.681 | 0.389 |
| Mean±Std | 0.685±0.030 | 0.679±0.068 | 0.691±0.030 | 0.756±0.057 | 0.366±0.083 |

### OOF Global Ensemble Results

| Model | P | M-F1 | F-F1 | F1-A | F1-G | F1-Cov |
|-------|---|------|------|------|------|--------|
| OOF Global Mean | 0.683 | 0.679 | 0.688 | 0.755 | 0.366 | 0.813 |
| OOF ± | 0.032 | 0.066 | 0.029 | 0.056 | 0.083 | 0.070 |

### Ablation Study (Qualitative Pathway)

| Design Choice | Challenge Addressed | Improvement |
|---------------|---------------------|-------------|
| Mean → Max Pooling | Sparse tumor signals diluted by healthy slices | Restores positive prediction capability for sparse tumor slices |
| Max → Attention-MIL | Noise from background and boundary slices | Learns to dynamically ignore empty lung regions, improving robustness |
| + Subgroup oversampling | Extreme cross-group scarcity (only 18 female squamous cases) | Prevents class collapse; substantially improves Female macro-F1 |
| + GRL | Entanglement between tumor and gender features | Closes the fairness gap (P=0.685, F-F1≈M-F1) |

### Key Findings

- GRL successfully disentangles gender from tumor features: Female macro-F1 (0.691) slightly exceeds Male (0.679), confirming that the model no longer relies on gender bias.
- Squamous cell carcinoma achieves the lowest F1 (0.366±0.083); the fundamental bottleneck is data scarcity (only 18 female cases) rather than a methodological deficiency.
- 5-fold ensemble + TTA effectively mitigates the drag of high-variance folds (e.g., Fold 4 at 0.637).
- OOF threshold optimization is more robust than direct argmax, yielding a global competition score of 0.683 without leakage risk.

## Highlights & Insights

- GRL serves as a minimally invasive yet effective fairness constraint—it leaves the primary task architecture unchanged and only adds an adversarial branch. This "minimally invasive fairness" paradigm is transferable to any medical imaging task requiring debiasing.
- Under extreme subgroup imbalance, the combination of WeightedRandomSampler + focal loss + label smoothing constitutes a viable remedy—no single strategy suffices, and a multi-pronged approach is necessary to prevent class collapse.
- Two-stage fine-tuning (stabilizing the attention head first, then unfreezing the backbone) is critical for MIL training stability.
- OOF threshold optimization is underappreciated—directly tuning thresholds on the validation set in small-data regimes risks overfitting, whereas OOF provides a leakage-free global estimate.

## Limitations & Future Work

- Squamous cell carcinoma F1 is only 0.366±0.083, constrained by the scarcity of 18 female squamous cases; the authors suggest using diffusion models to generate synthetic CT scans to augment rare subgroups.
- Ablation analysis is presented as a qualitative pathway description rather than a quantitative step-by-step table; precise numerical drops upon removing individual components are not reported.
- Only gender is considered as a sensitive attribute; other fairness dimensions such as age and ethnicity are not addressed.
- Only 32 slices are sampled per volume, potentially discarding critical lesions in volumes with 800+ slices.
- Neither 3D convolutions nor z-axis positional encodings are employed, neglecting inter-slice spatial continuity.
- Validation is conducted solely on a single challenge dataset (889 cases); external generalizability remains unknown.

## Related Work & Insights

- **vs. Ilse et al. (ICML 2018) Attention-MIL**: This work extends the attention-MIL framework by incorporating a GRL adversarial branch and a fairness-aware training protocol, advancing from weakly supervised aggregation to fairness-aware diagnosis.
- **vs. Ganin & Lempitsky (2015) GRL**: Originally proposed for domain adaptation to remove domain-specific features; this work repurposes it to eliminate gender features for demographic fairness.
- **vs. 3D CT classification (3D ResNet, etc.)**: This work adopts a 2D backbone + MIL aggregation strategy, which is better suited for scenarios with highly variable slice counts, at the cost of sacrificing z-axis spatial modeling.

## Rating

⭐⭐⭐

- **Novelty** ⭐⭐⭐: GRL and attention MIL are both existing components applied in combination; no architectural-level originality is introduced.
- **Experimental Thoroughness** ⭐⭐⭐: Ablation is qualitative; only a single challenge dataset is used.
- **Writing Quality** ⭐⭐⭐⭐: Method description is clear and systematic, with complete formulations and intuitive diagrams.
- **Value** ⭐⭐⭐: Provides an end-to-end template for fairness in medical AI, though depth is limited by the challenge report format.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Robust Fair Disease Diagnosis in CT Images](robust_fair_disease_diagnosis_in_ct_images.md)
- [\[CVPR 2026\] MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](milpf_multiple_instance_learning_on_precomputed_fe.md)
- [\[CVPR 2026\] Every Error has Its Magnitude: Asymmetric Mistake Severity Training for Multiclass Multiple Instance Learning](every_error_has_its_magnitude_asymmetric_mistake_severity_training_for_multiclas.md)
- [\[ACL 2026\] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection](../../ACL2026/medical_imaging/multi-view_attention_multiple-instance_learning_enhanced_by_llm_reasoning_for_co.md)
- [\[CVPR 2026\] EMAD: Evidence-Centric Grounded Multimodal Diagnosis for Alzheimer's Disease](emad_evidence-centric_grounded_multimodal_diagnosis_for_alzheimers_disease.md)

</div>

<!-- RELATED:END -->
