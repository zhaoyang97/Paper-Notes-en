---
title: >-
  [Paper Note] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model
description: >-
  [CVPR 2026][Medical Imaging][source-free domain adaptation] This paper proposes Tell2Adapt, a unified framework that leverages the generalized knowledge of a vision foundation model (BiomedParse) to generate high-quality…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "source-free domain adaptation"
  - "vision foundation model"
  - "medical image segmentation"
  - "pseudo label"
  - "prompt regularization"
date: 2026-05-08
content_hash: 2e877bb8de37d49d
---

# Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model

**Conference**: CVPR 2026
**arXiv**: [2603.05012](https://arxiv.org/abs/2603.05012)  
**Authors**: Yulong Shi, Shijie Li, Ziyi Li, Lin Qi
**Code**: [derekshiii/Tell2Adapt](https://github.com/derekshiii/Tell2Adapt)  
**Area**: Medical Imaging
**Keywords**: source-free domain adaptation, vision foundation model, medical image segmentation, pseudo label, prompt regularization

## TL;DR

This paper proposes Tell2Adapt, a unified framework that leverages the generalized knowledge of a vision foundation model (BiomedParse) to generate high-quality pseudo labels via Context-Aware Prompt Regularization (CAPR), followed by Visual Plausibility Refinement (VPR) to eliminate anatomically implausible predictions, enabling unified source-free unsupervised domain adaptation for medical image segmentation across 10 domain transfer directions and 22 anatomical targets.

## Background & Motivation

Source-Free Unsupervised Domain Adaptation (SFUDA) is critical for medical imaging deployment, where source domain data cannot be shared due to privacy constraints, requiring models to adapt using only unlabeled target domain data.

Existing SFUDA methods exhibit key limitations:
- **Task-specific design**: Most methods are tailored to specific transfer tasks with small domain gaps (e.g., MRI→MRI) and cannot scale into a unified multi-modal, multi-target framework.
- **Poor generalizability**: Performance degrades significantly under large domain gaps (e.g., CT→MRI) or when multiple anatomical targets are involved.
- **Low-quality pseudo labels**: Pseudo labels generated directly from source models are noisy on the target domain, severely limiting adaptation effectiveness.
- **Insufficient clinical reliability**: The lack of anatomical plausibility validation may produce clinically unacceptable false positives.

A core observation motivates this work: Vision Foundation Models (VFMs) such as BiomedParse, pretrained on large-scale biomedical data, possess broad cross-modal and cross-anatomical knowledge. The key challenge is how to effectively transfer this knowledge to lightweight deployment models while ensuring clinical reliability.

## Method

### Overall Architecture

Tell2Adapt comprises three core stages: **prompt regularization → pseudo label generation and knowledge distillation → visual plausibility refinement**. The overall pipeline first normalizes diverse text prompts via CAPR, then uses BiomedParse to generate high-quality pseudo labels, trains a lightweight nnUNet student model via knowledge distillation, and finally applies VPR to remove anatomically implausible prediction components.

### Stage 1: Context-Aware Prompt Regularization (CAPR)

BiomedParse is a prompt-driven segmentation model whose performance is highly sensitive to input prompt quality. In practice, however, text prompt formulations vary substantially across users and clinical contexts.

The core idea of CAPR:
1. A meta-prompt is designed to leverage an LLM (e.g., GPT-4) to translate diverse non-standard text prompts into canonical instructions understandable by BiomedParse.
2. Different expressions of the same anatomical concept (e.g., "left ventricle," "LV," "左心室") are mapped to a consistent canonical prompt.
3. The regularized prompts are directly fed into BiomedParse for inference to generate high-quality segmentation pseudo labels.

### Stage 2: Pseudo Label Generation and Knowledge Distillation

1. Using CAPR-regularized prompts, BiomedParse performs inference on target domain images to generate pseudo labels.
2. Pseudo labels are converted into nnUNet-compatible format (BiomedParse → nnUNet format conversion).
3. A lightweight nnUNet student model is trained via knowledge distillation, supervised by the pseudo labels.
4. nnUNet serves as the final deployment model, offering fast inference and low resource requirements suitable for clinical deployment.

Advantages of knowledge distillation:
- BiomedParse acts as a teacher model providing generalized knowledge, which the nnUNet student inherits.
- The student model requires no access to source domain data and is trained entirely on target domain pseudo labels.
- The lightweight design accommodates the computational constraints of real-world clinical environments.

### Stage 3: Visual Plausibility Refinement (VPR)

VPR leverages anatomical priors precomputed by BiomedParse to validate and refine the student model's predictions in a post-processing step.

Core algorithm:
1. Statistical priors (Beta distribution parameters) for the target modality and anatomical structure are loaded from `Anatomical_Priors.json`.
2. For each predicted connected component $p_i$, a log-space plausibility score is computed:
$$\log S(p_i) = \sum_{k=1}^{4} \left[ (\alpha_k-1)\log f_{i,k} + (\beta_k-1)\log(1-f_{i,k}) - \log B(\alpha_k, \beta_k) \right]$$
where $f_{i,k}$ denotes the $k$-th dimensional feature value of the connected component in the low-level visual feature space of the target image.
3. Connected components with scores below $\mu_S - 2\sigma_S$ are discarded as anatomically implausible false positives.
4. Components smaller than a minimum size threshold are also directly removed.

The supported modalities span a broad clinical range: CT (abdomen/thorax/liver), MRI (abdomen/cardiac/brain), chest X-ray, cardiac ultrasound, endoscopic polyps, fundus/dermoscopy/OCT, and others, covering 14 clinical scenarios.

## Key Experimental Results

### Experimental Setup
- **Evaluation scale**: 10 domain transfer directions, 22 anatomical targets — claimed to be one of the largest SFUDA evaluations to date.
- **Anatomical regions**: Brain (BraTS tumor segmentation), cardiac (M&Ms cardiac MRI), polyps (endoscopic polyp segmentation), abdomen (AMOS/CHAOS abdominal organs).
- **VFM teacher**: BiomedParse
- **Student model**: nnUNet
- **Metrics**: Dice Similarity Coefficient (DSC), Hausdorff Distance 95% (HD95)

### Table 1: Cross-modal Abdominal Organ Segmentation Dice (%) Comparison

| Method | Type | Liver | R.Kidney | L.Kidney | Spleen | Avg |
|---|---|---|---|---|---|---|
| Source Only | — | 58.2 | 47.6 | 46.1 | 42.3 | 48.6 |
| TENT | Test-time | 63.4 | 52.1 | 51.8 | 48.7 | 54.0 |
| AdaptSeg | UDA | 71.5 | 63.2 | 62.4 | 59.8 | 64.2 |
| DPL | SFUDA | 69.3 | 58.7 | 57.2 | 55.1 | 60.1 |
| ProSFDA | SFUDA | 74.8 | 65.3 | 64.1 | 62.4 | 66.7 |
| **Tell2Adapt** | **SFUDA** | **82.6** | **76.8** | **75.4** | **73.1** | **77.0** |

Tell2Adapt outperforms the second-best method (ProSFDA) by approximately 10% in average Dice, with more pronounced advantages under large domain gap transfer scenarios.

### Table 2: Cardiac MRI Segmentation and Ablation Results

| Configuration | LV | RV | Myo | Avg DSC | Avg HD95 |
|---|---|---|---|---|---|
| Source Only | 72.3 | 65.1 | 58.4 | 65.3 | 14.2 |
| ProSFDA | 81.2 | 74.6 | 69.3 | 75.0 | 8.7 |
| Tell2Adapt (w/o CAPR) | 83.1 | 77.2 | 71.8 | 77.4 | 7.4 |
| Tell2Adapt (w/o VPR) | 85.4 | 79.8 | 74.1 | 79.8 | 6.8 |
| **Tell2Adapt (Full)** | **87.6** | **82.3** | **76.5** | **82.1** | **5.6** |

Ablation analysis shows:
- CAPR contributes approximately 4.7% Dice improvement (demonstrating the significant impact of prompt regularization on VFM inference quality).
- VPR further improves Dice by approximately 2.3% while substantially reducing HD95 (effectively removing false positives and noise).
- The two modules are complementary; the full framework achieves optimal performance.

## Highlights & Insights

- **Unified framework design**: The first unified SFUDA framework covering 10 domain transfer directions and 22 anatomical targets, overcoming the "one task, one model" limitation of existing methods.
- **VFM knowledge transfer pipeline**: A complete transfer pathway of VFM → pseudo labels → knowledge distillation → lightweight model is proposed, leveraging the generalization capacity of large models while meeting the efficiency requirements of clinical deployment.
- **Prompt regularization via CAPR**: CAPR addresses the practical problem of inconsistent prompt formulations in prompt-driven VFMs by achieving canonical mapping through an LLM.
- **Anatomy-prior-driven post-processing**: VPR requires no additional training and effectively removes false positives using only statistical priors and low-level visual features, enhancing clinical reliability.
- **Broad modality coverage**: The anatomical prior library supports 14 clinical scenarios including CT, MRI, X-ray, ultrasound, endoscopy, fundus, dermoscopy, OCT, and pathology.

## Limitations & Future Work

- **Dependence on VFM quality**: The performance upper bound of the framework is constrained by BiomedParse's capabilities; pseudo label quality may be insufficient for rare anatomical structures or modalities with limited VFM coverage.
- **CAPR relies on LLM APIs**: Prompt regularization requires calling an LLM (e.g., GPT-4), posing deployment challenges in offline or restricted environments.
- **Pre-specified anatomical priors**: VPR's prior parameters must be precomputed from BiomedParse; adding new anatomical targets requires updating the prior library.
- **Lack of end-to-end training**: The three-stage pipeline design introduces complexity, and independently optimizing each stage may incur information transmission loss.
- **Computational cost**: The overall cost of BiomedParse inference combined with nnUNet distillation training remains relatively high, despite the lightweight nature of the final deployment model.

## Related Work & Insights

- **SFUDA methods**: TENT (test-time adaptation), DPL (domain pseudo labels), ProSFDA (progressive source-free domain adaptation) → mostly designed for specific domain gaps, lacking a unified framework.
- **Vision foundation models**: SAM (general-purpose segmentation foundation model), BiomedParse (biomedical segmentation foundation model) → this work uses BiomedParse as the knowledge source.
- **Knowledge distillation**: A classical paradigm for transferring large model knowledge to lightweight models → used here to train nnUNet from VFM pseudo labels.
- **Medical image segmentation DA**: AdaptSeg (adversarial training), UAMT (uncertainty-aware mean teacher) → most require source domain data or support only specific modalities.
- **Tell2Adapt's positioning**: By combining prompt regularization and anatomical prior refinement, VFM generalization knowledge is efficiently transferred to a lightweight deployment model, realizing the first unified multi-modal, multi-target SFUDA framework.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of VFM + CAPR + VPR is conceptually clear and novel; the designed pathway for transferring foundation model knowledge to the SFUDA setting is well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 10 domain transfer directions and 22 anatomical targets constitute an exceptionally large-scale evaluation within the SFUDA literature.
- Writing Quality: ⭐⭐⭐⭐ — The framework structure is clearly presented, though the three-stage design description is somewhat fragmented.
- Value: ⭐⭐⭐⭐ — A unified SFUDA framework has direct practical value for clinical deployment of medical imaging systems; code is publicly available.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Unsupervised Domain Adaptation with Target-Only Margin Disparity Discrepancy](unsupervised_domain_adaptation_with_target-only_margin_disparity_discrepancy.md)
- [\[CVPR 2026\] Reclaiming Lost Text Layers for Source-Free Cross-Domain Few-Shot Learning](reclaiming_lost_text_layers_for_source-free_cross-domain_few-shot_learning.md)
- [\[CVPR 2026\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semisupervised_framework.md)
- [\[CVPR 2026\] GaussianPile: A Unified Sparse Gaussian Splatting Framework for Slice-based Volumetric Reconstruction](gaussianpile_a_unified_sparse_gaussian_splatting_framework_for_slice-based_volum.md)
- [\[ICLR 2026\] Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning](../../ICLR2026/medical_imaging/uni-ntfm_a_unified_foundation_model_for_eeg_signal_representation_learning.md)

</div>

<!-- RELATED:END -->
