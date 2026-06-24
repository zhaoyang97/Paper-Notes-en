---
title: >-
  [Paper Note] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization
description: >-
  [CVPR2025][Medical Imaging][domain generalization] Proposes GenEval, which quantifies the causal coverage gap through the Domain Conformal Boundary (DCB) theory and integrates human expert knowledge with the MedGemma-4B vision-language model to achieve single-source domain generalization (SDG). It significantly outperforms existing methods on diabetic retinopathy grading (8 datasets) and seizure onset zone detection (2 datasets).
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "domain generalization"
  - "vision-language model"
  - "diabetic retinopathy"
  - "conformal inference"
  - "LoRA"
date: 2026-05-08
content_hash: 6601826d00ba9143
---

# Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization

**Conference**: CVPR2025  
**arXiv**: [2603.12369](https://arxiv.org/abs/2603.12369)  
**Code**: [GitHub](https://github.com/IMPACTLabASU/GenEval)  
**Area**: Medical Imaging  
**Keywords**: domain generalization, vision-language model, diabetic retinopathy, conformal inference, LoRA

## TL;DR

Proposes GenEval, which quantifies the causal coverage gap through the Domain Conformal Boundary (DCB) theory and integrates human expert knowledge with the MedGemma-4B vision-language model to achieve single-source domain generalization (SDG). It significantly outperforms existing methods on diabetic retinopathy grading (8 datasets) and seizure onset zone detection (2 datasets).

## Background & Motivation

Cross-domain generalization is a core challenge in medical image classification. In diabetic retinopathy (DR) grading, different datasets exhibit domain shifts due to acquisition equipment, population characteristics, and protocols. Recent theoretical studies have revealed two necessary conditions for domain generalization: (1) causal coverage, where the source domain contains all causal factors required for the target domain; and (2) source risk minimization. However, existing DG methods (such as SPSD-ViT) fail to consistently outperform the ERM baseline and cannot determine whether a new domain is out-of-support during deployment.

**Core Problem**: Causal gaps exist among different DR datasets. For instance, neovascularization is a critical causal factor for Grade 4 DR; it is present in EyePACS but absent in Messidor 1. Consequently, models trained on Messidor 1 cannot identify neovascularization in EyePACS.

**Core Idea**: Human expert knowledge can compensate for the causal gap between domains, facilitating better generalization when input into foundation vision-language models via a multi-modal (image + textual knowledge) approach. However, expert knowledge is qualitative and ambiguous, necessitating quantification and refinement.

## Method

### Theoretical Framework: Domain Conformal Boundary (DCB)

**Step 1 — Calculating DCB**:
- Define the robustness metric $\rho(\mathcal{K}(X_i), D^s)$: the average Mahalanobis distance between the causal factor estimation of data point $X_i$ and other samples in the source domain.
- Based on conformal inference, split the source domain into $I_T$ and $I_V$, compute the residual distribution, and obtain a distribution-free prediction interval $C$.
- If the robustness metric of a target domain sample falls within $C$, the sample shares the same causal factor relationship with the source domain (with probability $\geq 1-\alpha$).

**Step 2 — Source Domain Consistency Degree (SDCD)**:
- Calculate the proportion of target domain samples that fall within the source domain DCB.
- Prove that SDCD is positively correlated with SDG performance (Lemma 1), with a Pearson correlation coefficient of 0.692 ($p < 0.02$).

**Step 3 — Knowledge Refinement**:
- Detect lesions such as microaneurysms and hemorrhages from fundus images using YOLOv12, and quantify them into a 14-dimensional real-valued vector.
- Stepwise ablate knowledge components and compute changes in SDCD to select the knowledge subset that maximizes SDCD.
- Observe that removing neovascularization (due to the difficulty of YOLO detection) yields the optimal SDCD.

### GenEval Multi-modal Classification Engine

- **Foundation Model**: MedGemma-4B, pre-trained on massive medical image-text pairs.
- **Parameter-Efficient Fine-Tuning**: Employs LoRA ($\text{rank}=16$, $\alpha=16$), training only about 2.4% of the parameters (~95M / 4B).
- **Prompt Design**: Integrates refined expert knowledge into structured clinical prompts, combined with image inputs.
- **Zero-shot Prompt**: Detailed descriptions of the clinical criteria for DR grades 0-4.
- **Fine-tuning Prompt**: Role-play + systematic screening protocol + refined knowledge.

### Causal Factor Extraction

- **From Data**: Linearize non-linear dynamical systems based on Koopman theory, identifying sparse causal factors via STRIDGE regression.
- **From Knowledge**: Expert knowledge is expressed as propositional logic formulas, where truth-value evaluation provides the quantification of causal factor relationships.

## Key Experimental Results

**Single-Source Domain Generalization (SDG)**:

| Source → Target | Best Baseline | Baseline Acc | GenEval Acc | K+D SDCD |
|---------------|---------|---------|-------------|----------|
| Messidor → APTOS | SPSD-ViT | 48.3% | **56.0%** | 98.03% |
| Messidor → EyePACS | SPSD-ViT | 57.4% | **80.04%** | 94.94% |
| EyePACS → APTOS | SPSD-ViT | 75.1% | 73.16% | 99.84% |
| EyePACS → Messidor2 | DRGen | 65.4% | **80.5%** | 99.83% |

**Extended SDG (EyePACS → 6 Target Domains)**:

| Method | APTOS | DeepDR | FGADR | RLDL | Average |
|------|-------|--------|-------|------|------|
| DECO | 59.7 | 40.3 | 9.9 | 49.3 | 50.68 |
| **GenEval** | **73.2** | **59.2** | **56.9** | **67.6** | **66.2** |

**Comparison with VLM Methods (SDG F1)**:

| Method | APTOS | Messidor | Average |
|------|-------|----------|------|
| CLIP-DR | 46.3 | 47.3 | 46.8 |
| **GenEval** | **72.0** | **78.2** | **75.1** |

**Multi-Source Domain Generalization**: GenEval Average 79.21% vs SPSD-ViT 73.3% (+5.91%)

**SOZ Detection (Cross-Center SDG)**: GenEval Average F1 90.0% vs CuPKL 88.1%

## Highlights & Insights

- Outstanding theoretical contribution: DCB provides a distribution-free evaluation framework for causal coverage, and SDCD can predict SDG performance.
- For the first time, human expert knowledge is systematically quantified, refined, and integrated into VLMs for domain generalization.
- Remarkable improvement on the FGADR dataset, rising from DECO's 9.9% to 56.9%.
- The knowledge refinement strategy is theoretically guided (maximizing SDCD) rather than blindly stacked.
- Highly versatile framework, validated on two completely different tasks: DR and SOZ.

## Limitations & Future Work

- Assumes that the data-generating mechanism is continuously differentiable, whereas mutations or threshold effects may exist in practice.
- YOLOv12 exhibits insufficient detection capabilities for certain causal factors (e.g., neovascularization), limiting the accuracy of knowledge quantification.
- Fine-tuning and inference of MedGemma-4B require substantial computing resources (though LoRA lowers the barrier).
- Knowledge extraction depends on domain experts; generalizing to a new domain requires redefining knowledge.
- Some SDG experiments utilize baseline data from other literatures, which may lead to slightly inconsistent experimental setups.

## Related Work & Insights

- **SPSD-ViT** (Rao et al.): The state-of-the-art DG baseline, employing a self-distilled ViT, but yields minor improvements in SDG.
- **CLIP-DR**: A ranking-aware prompting method that adapts CLIP for DR classification; this work significantly outperforms it (+28.3% F1).
- **MedGemma-4B**: A specialized pre-trained medical VLM, adapted in this work via LoRA fine-tuning.
- **Conformal Inference** (Angelopoulos & Bates): The methodological foundation of the proposed DCB theory.
- **Koopman Theory**: A theoretical tool used to extract causal factor relationships from data.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (A complete system combining DCB theory, knowledge refinement, and VLM fusion, with extremely high innovation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (8+2 datasets, multidirectional comparison across SDG/MDG/VLM, and sensitivity analysis)
- Writing Quality: ⭐⭐⭐⭐ (Rigorous theoretical derivation, though the dense content makes it a heavy read)
- Value: ⭐⭐⭐⭐⭐ (Addresses the core challenge of SDG, providing direct guiding significance for medical AI deployment)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] LangDAug: Langevin Data Augmentation for Multi-Source Domain Generalization in Medical Image Segmentation](../../ICML2025/medical_imaging/langdaug_langevin_data_augmentation_for_multi-source_domain_generalization_in_me.md)
- [\[AAAI 2026\] Experience with Single Domain Generalization in Real World Medical Imaging Deployments](../../AAAI2026/medical_imaging/experience_with_single_domain_generalization_in_real_world_medical_imaging_deplo.md)
- [\[CVPR 2025\] Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)
- [\[CVPR 2025\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)
- [\[CVPR 2026\] Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning](../../CVPR2026/medical_imaging/mind_the_discriminability_trap_in_source-free_cross-domain_few-shot_learning.md)

</div>

<!-- RELATED:END -->
