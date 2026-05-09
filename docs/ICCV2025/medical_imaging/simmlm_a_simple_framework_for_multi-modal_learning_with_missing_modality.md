---
title: >-
  [Paper Note] SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality
description: >-
  [ICCV 2025][Medical Imaging][multi-modal learning] This paper proposes SimMLM, a simple yet effective framework for multi-modal learning under missing modality conditions. It consists of a Dynamic Mixture of Modality Experts (DMoME) architecture and a More vs. Fewer (MoFe) ranking loss. SimMLM comprehensively outperforms state-of-the-art methods on brain tumor segmentation and multi-modal classification tasks with fewer parameters and lower computational cost, while providing interpretable modality importance estimates.
tags:
  - ICCV 2025
  - Medical Imaging
  - multi-modal learning
  - missing modality
  - mixture of experts
  - ranking loss
  - brain tumor segmentation
date: 2026-05-08
content_hash: 04bdd70ecea0db13
---

# SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality

**Conference**: ICCV 2025  
**arXiv**: [2507.19264](https://arxiv.org/abs/2507.19264)  
**Code**: [https://github.com/LezJ/SimMLM](https://github.com/LezJ/SimMLM)  
**Area**: Medical Imaging  
**Keywords**: multi-modal learning, missing modality, mixture of experts, ranking loss, brain tumor segmentation

## TL;DR

This paper proposes SimMLM, a simple yet effective framework for multi-modal learning under missing modality conditions. It consists of a Dynamic Mixture of Modality Experts (DMoME) architecture and a More vs. Fewer (MoFe) ranking loss. SimMLM comprehensively outperforms state-of-the-art methods on brain tumor segmentation and multi-modal classification tasks with fewer parameters and lower computational cost, while providing interpretable modality importance estimates.

## Background & Motivation

Although multi-modal learning can leverage complementary information to improve performance, real-world deployments frequently encounter **missing modalities at test time** due to hardware failures, environmental constraints, or data acquisition limitations. Existing solutions suffer from the following shortcomings:

**Imputation-based methods** (synthesizing missing modalities via generative models): Synthesis quality is difficult to guarantee, with issues such as hallucinations, adversarial vulnerability, and high computational overhead.

**Complex architecture methods** (e.g., mmFormer, ShaSpec): These require specialized network designs for cross-modal alignment and feature sharing, resulting in large parameter counts, poor flexibility, and the constraint that shared encoders require consistent dimensionality across all modalities.

**Existing MoE methods** (e.g., MoMKE): Each modality must pass through all experts, causing computational complexity to scale quadratically with the number of modalities.

The core philosophy of SimMLM is **simplicity is effectiveness**: each modality corresponds to exactly one expert network, a dynamic gating network adaptively adjusts contribution weights, and the MoFe ranking loss enforces the intuitive property that "more modalities should yield better performance."

## Method

### Overall Architecture

SimMLM consists of two components:
1. **DMoME architecture**: modality-specific expert networks and a gating network that handle complete or partial modality inputs.
2. **MoFe ranking loss**: randomly drops modalities during training to enforce the ordering constraint that more modalities lead to better performance.

### Key Designs

1. **Dynamic Mixture of Modality Experts (DMoME)**:

    - Each modality $m$ is assigned a dedicated expert network $E^m(\mathbf{x}^m; \theta_m)$ that produces a task output $\mathbf{o}^m$ (e.g., classification logits).
    - A gating network $G(\cdot; \phi)$ takes all available modalities as input and produces gating values $\{g^m\}_{m=1}^M$, which are normalized via softmax to obtain weights $w^m$.
    - Inputs for missing modalities are set to zero, and the corresponding gating values are set to $g_m = -\infty$ (yielding zero weight after softmax), naturally excluding missing modalities from the output.
    - The final output is the weighted combination: $\mathbf{o} = \sum_{m=1}^M w^m E^m(\mathbf{x}^m; \theta_m)$
    - **Key advantages**: Each modality passes through only its own expert (vs. MoMKE where each modality passes through all experts), resulting in linear rather than quadratic FLOP scaling; no architectural constraints are imposed on individual expert networks, enabling heterogeneous designs.

2. **More vs. Fewer (MoFe) Ranking Loss**:

    - Core intuition: when more modalities are available, model performance should not be worse than with fewer modalities.
    - At each training iteration, two modality subsets are sampled from the full set $\mathbf{x}^{full}$ such that $\mathbf{x}^+ \supseteq \mathbf{x}^-$.
    - The MoFe loss is defined as:
    $$\ell_{\text{MoFe}}(\mathbf{o}^+, \mathbf{o}^-, \mathbf{y}) = \max(0, \ell_{\text{task}}(\mathbf{o}^+, \mathbf{y}) - \ell_{\text{task}}(\mathbf{o}^-, \mathbf{y}))$$
    - Gradients are generated only when the task loss with more modalities exceeds that with fewer, encouraging the network to eliminate such counter-intuitive behavior.
    - **Novelty**: Ranking is applied at the loss level rather than the confidence level, directly regularizing the geometry of the loss landscape. This design is universally applicable to classification, segmentation, regression, and other tasks.

3. **Two-Stage Training**:

    - **Stage 1 (Independent Learning)**: Each modality expert is trained independently to prevent any single modality from dominating the learning process.
    - **Stage 2 (Collaborative Learning)**: Experts and the gating network are jointly trained with the MoFe loss. The total loss is:
    $$\ell_{\text{total}} = \ell_{\text{task}}(\mathbf{o}^+, \mathbf{y}) + \ell_{\text{task}}(\mathbf{o}^-, \mathbf{y}) + \lambda \ell_{\text{MoFe}}$$
    - Stage 1 supports parallel training, and pre-trained experts can be reused when new modalities are added.

### Loss & Training

- MoFe coefficient $\lambda = 0.1$ (robust to different values).
- BraTS 2018: nnUNet as expert networks (one per modality for 4 modalities); gating network uses a lightweight CNN + Linear; Adam (lr=0.01); task loss = Dice + BCE.
- UPMC Food-101: Inception V3 (image) + BERT (text) as experts; MLP gating; Adam (lr=0.0001); CE loss.
- avMNIST: LeNet-5 (image and audio); Adam (lr=0.001); CE loss.
- All experiments conducted on a single A100 GPU.

## Key Experimental Results

### Main Results

BraTS 2018 official evaluation set (average Dice score across 15 missing/complete modality configurations):

| Method | ET ↑ | TC ↑ | WT ↑ | #Params | FLOPS |
|--------|------|------|------|---------|-------|
| mmFormer | 59.85 | 72.97 | 82.94 | 106M | 748G |
| ShaSpec | 61.58 | 77.45 | 85.92 | 187.7M | 713G |
| MoMKE | 65.56 | 78.58 | 86.69 | 7.8M | 490G |
| DMoME (w/o MoFe) | 66.05 | 79.14 | 86.71 | 7.8M | 123G |
| **SimMLM (DMoME + MoFe)** | **67.16** | **80.20** | **87.67** | **7.8M** | **123G** |

Classification task results:

| Dataset | Modality Setting | ShaSpec | MoMKE | **SimMLM** |
|---------|-----------------|---------|-------|------------|
| UPMC Food-101 | Image only | 69.22 | 70.46 | **72.20** |
| | Text only | 86.55 | 86.59 | **87.20** |
| | Image + Text | 92.73 | 92.71 | **94.99** |
| avMNIST | Image only | 91.90 | 92.61 | **92.69** |
| | Audio only | 89.28 | 91.16 | **91.61** |
| | Image + Audio | 98.71 | 98.69 | **99.27** |

### Ablation Study

Calibration error comparison (BraTS 2018 validation set, average across 15 configurations):

| Method | ET ECE↓ | TC ECE↓ | WT ECE↓ | ET SCE↓ | TC SCE↓ | WT SCE↓ |
|--------|---------|---------|---------|---------|---------|---------|
| MoMKE | 3.46 | 4.10 | 4.04 | 14.82 | 11.35 | 7.69 |
| DMoME | 3.34 | 3.86 | 3.98 | 13.40 | 11.01 | 7.11 |
| DMoME + $\mathcal{L}_{Conf}$ | 3.17 | 3.86 | 3.61 | 14.80 | 11.58 | 6.81 |
| **DMoME + $\mathcal{L}_{MoFe}$** | **3.15** | **3.75** | **3.55** | **13.21** | **10.83** | **5.65** |

Counterintuitive Rate (CR) comparison:

| Method | ET CR↓ | TC CR↓ | WT CR↓ |
|--------|--------|--------|--------|
| MoMKE | 14.60 | 32.47 | 9.86 |
| DMoME (w/o MoFe) | 10.47 | 30.94 | 3.87 |
| **SimMLM** | **7.20** | **28.19** | **3.75** |

Performance with single modality T1ce only:

| Tumor | ShaSpec | MoMKE | **SimMLM** |
|-------|---------|-------|------------|
| ET | 73.29 | 72.30 | **78.58** |
| TC | 78.65 | 78.71 | **83.45** |
| WT | 73.82 | 74.00 | **80.02** |

### Key Findings

- SimMLM surpasses ShaSpec (187.7M / 713G) and mmFormer (106M / 748G) using only 7.8M parameters and 123G FLOPS, achieving an order-of-magnitude improvement in efficiency.
- The MoFe loss reduces the counterintuitive rate for ET from 14.60% to 7.20% (a 50.68% reduction), effectively enforcing the monotonicity of "more modalities → better performance."
- Analysis of gating weights shows that DMoME automatically identifies critical modalities: ET segmentation prioritizes T1ce, while WT segmentation prioritizes FLAIR and T2, consistent with clinical knowledge.
- When T1ce is missing, DMoME automatically shifts weight to T1 (the non-contrast counterpart of T1ce), aligning with clinical practice.
- The MoFe loss coefficient $\lambda$ consistently outperforms the baseline across the range $[0.01, 0.5]$, demonstrating robustness to this hyperparameter.
- The largest accuracy gain on UPMC Food-101 is observed under the all-modality setting (92.71 → 94.99), indicating that SimMLM's advantage is more pronounced in noisy environments.

## Highlights & Insights

- **Extreme simplicity and efficiency**: No special architectures, shared encoders, or missing data generation are required. SimMLM achieves state-of-the-art performance through the minimal combination of experts, gating, and a ranking loss — embodying Occam's Razor.
- **Elegant design of the MoFe ranking loss**: Constraints are imposed at the task loss level rather than the confidence level, making the loss naturally applicable to arbitrary tasks (segmentation, classification, regression). This also regularizes the geometry of the loss landscape and incidentally improves model calibration.
- **Interpretability** is a practically important property: gating weights directly inform clinicians of the relative importance of each modality, which is especially valuable when some modalities are unavailable.
- The DMoME design allows new modalities to be incorporated by training only a new expert and re-running the collaborative training stage, without retraining from scratch — a practically engineering-friendly property.

## Limitations & Future Work

- The two-stage training procedure, while well-motivated, increases training complexity.
- The gating network relies on lightweight CNN/MLP architectures; its modeling capacity for highly heterogeneous modalities (e.g., structured clinical data combined with medical imaging) remains to be verified.
- The MoFe loss assumes all modalities contribute positively (i.e., "more modalities → better performance"), which may not hold for noisy or redundant modalities.
- Experiments are limited to scenarios with 2–4 modalities; scalability to a larger number of modalities (e.g., 10+) has not been validated.
- The optimality of the training-time missing modality simulation strategy (random dropping) has not been thoroughly investigated.

## Related Work & Insights

- MoMKE is the most direct predecessor; SimMLM substantially reduces computational cost through dedicated per-modality experts (vs. MoMKE's shared-expert design).
- ShaSpec's approach of decoupling modality-shared and modality-specific features is complex and inflexible; DMoME performs fusion in the output space rather than the feature space, which is more general.
- The ranking learning philosophy underlying the MoFe loss bears conceptual similarity to preference ranking in RLHF.
- An important implication for clinical AI deployment: in resource-constrained environments, tolerance to partial modality absence combined with interpretable modality importance weights constitutes a clinically trustworthy AI system.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The MoFe ranking loss is a concise yet insightful contribution; DMoME, while simple, proves highly effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, 15 modality configurations, efficiency analysis, calibration analysis, interpretability analysis, and counterintuitive rate evaluation — comprehensively conducted.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated, methodology is concisely described, and analysis is thorough.
- **Value**: ⭐⭐⭐⭐⭐ Provides a general, efficient, and interpretable solution for missing modality handling with high practical deployment value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] CLoE: Expert Consistency Learning for Missing Modality Segmentation](../../CVPR2026/medical_imaging/cloe_expert_consistency_learning_for_missing_modality_segmentation.md)
- [\[CVPR 2026\] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](../../CVPR2026/medical_imaging/must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)
- [\[AAAI 2026\] Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](../../AAAI2026/medical_imaging/learning_cell-aware_hierarchical_multi-modal_representations.md)
- [\[ICLR 2026\] CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework](../../ICLR2026/medical_imaging/care_towards_clinical_accountability_in_multi-modal_medical_reasoning_with_an_ev.md)
- [\[NeurIPS 2025\] RAD: Towards Trustworthy Retrieval-Augmented Multi-modal Clinical Diagnosis](../../NeurIPS2025/medical_imaging/rad_towards_trustworthy_retrieval-augmented_multi-modal_clinical_diagnosis.md)

<!-- RELATED:END -->
