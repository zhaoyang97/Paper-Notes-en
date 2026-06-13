---
title: >-
  [Paper Note] SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding
description: >-
  [Medical Imaging] This paper proposes SEED (Semantic Evaluation for Visual Brain Decoding), a composite evaluation metric combining three complementary measures — Object F1, Cap-Sim…
tags:
  - "Medical Imaging"
date: 2026-05-08
content_hash: 79392155042204b8
---

# SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding

## Metadata
- **Conference**: ICLR 2026
- **arXiv**: [2503.06437](https://arxiv.org/abs/2503.06437)
- **Code**: [https://github.com/Concarne2/SEED](https://github.com/Concarne2/SEED)
- **Area**: Others
- **Keywords**: brain decoding, evaluation metrics, fMRI, semantic similarity, visual attention, human evaluation

## TL;DR
This paper proposes SEED (Semantic Evaluation for Visual Brain Decoding), a composite evaluation metric combining three complementary measures — Object F1, Cap-Sim, and EffNet — which substantially outperforms all existing metrics in alignment with human evaluation.

## Background & Motivation
- Visual brain decoding (reconstructing visual stimuli from fMRI) has advanced significantly, with recent models approaching perfect scores on existing percentage-based metrics, giving the impression that the problem is nearly solved.
- **Upon closer inspection**: reconstructed images frequently lose critical semantic elements (e.g., a teddy bear replaced by a cat), yet existing metrics assign high scores, misleading the research community.
- **Three core problems with existing evaluation**:
  1. **Pool dependency**: Two-way identification metrics (AlexNet, CLIP, etc.) rely on comparison pools, making cross-model comparisons unfair.
  2. **Insufficient difficulty**: Two-way identification tasks are too simple; recent models have nearly saturated them.
  3. **Lack of human alignment**: Metrics based on abstract features deviate substantially from human intuition.

## Method

### Overall Architecture: Inspired by Human Visual Attention
Human visual attention is a two-stage process:
- **Stage 1**: Parallel processing of basic features (color, orientation, brightness) → corresponds to convolutional models such as EffNet.
- **Stage 2**: Focused attention binds features into coherent objects → a stage absent from existing metrics.

SEED integrates three complementary metrics to simulate complete visual perception:

### Metric 1: Object F1 (Simulating Object-Oriented Attention)
An open-vocabulary image grounding model (MM-Grounding-DINO) is used to detect 82 object categories:

$$\text{Object Recall}_t = \frac{\text{Number of categories shared by GT and reconstruction}}{\text{Number of categories in GT}}$$

$$\text{Object Precision}_t = \frac{\text{Number of categories shared by GT and reconstruction}}{\text{Number of categories in reconstruction}}$$

The threshold $t$ is swept from 0 to a cutoff value and averaged to eliminate threshold sensitivity:
$$\text{Object F1} = \frac{2}{\text{Object Recall}^{-1} + \text{Object Precision}^{-1}}$$

### Metric 2: Cap-Sim (Simulating Feature Binding)
An image captioning model (GIT) generates descriptions, and semantic similarity between descriptions is then compared:

$$\text{Cap-Sim} = \cos(e_{\text{text}}(c(I_{GT})), e_{\text{text}}(c(I_{recon})))$$

where $e_{\text{text}}$ denotes a Sentence Transformer and $c$ denotes GIT. This captures object attributes (pose, color), background, and other semantics missed by Object F1.

### Metric 3: EffNet (Capturing Global Structure)
$$\overline{\text{EffNet}} = \text{corr}(e_{\text{img}}(I_{GT}), e_{\text{img}}(I_{recon}))$$

An ImageNet-pretrained EfficientNet is used to capture more global and structural scene features.

### SEED Composite Score
$$\text{SEED} = \frac{\text{Object F1} + \text{Cap-Sim} + \overline{\text{EffNet}}}{3}$$

The three metrics are complementary: Object F1 checks for the presence of key objects, Cap-Sim captures high-level semantic detail, and EffNet captures global structure.

### Human Evaluation Data Collection
- 22 annotators rated 1,000 GT–reconstruction image pairs on a 5-point Likert scale.
- ICC(2, n) = 0.84 (p=0), indicating high inter-rater agreement.
- The dataset is publicly released.

## Key Experimental Results

### Main Results: Alignment with Human Evaluation (NSD + MindEye2)

| Metric | Pairwise Accuracy | Kendall τ | Pearson r |
|--------|------------------|-----------|-----------|
| PixCorr | 53.8% | .075 | .117 |
| SSIM | 54.5% | .090 | .112 |
| AlexNet(2) | 55.0% | .185 | .187 |
| AlexNet(5) | 49.5% | .236 | .258 |
| Inception | 63.8% | .330 | .475 |
| CLIP | 66.4% | .368 | .436 |
| EffNet | 78.0% | .559 | .748 |
| SwAV | 69.7% | .394 | .576 |
| Object F1 | 75.8% | .516 | .708 |
| Cap-Sim | 73.8% | .477 | .666 |
| **SEED** | **81.0%** | **.621** | **.813** |

> SEED substantially leads all three human-alignment metrics, achieving a pairwise accuracy of 81% and a Pearson r of 0.813.

### Cross-Dataset Validation (GOD + Mind-Vis)

| Metric | Pairwise Accuracy | Kendall τ | Pearson r |
|--------|------------------|-----------|-----------|
| CLIP | 62.6% | — | — |
| EffNet | ~70% | — | — |
| Object F1 | ~68% | — | — |
| **SEED** | **~73%** | — | **Best** |

> SEED's advantage remains consistent across different datasets and model combinations.

### Key Findings
1. Most commonly used metrics (PixCorr, SSIM, AlexNet) exhibit near-zero correlation with human evaluation.
2. EffNet is the best-performing single metric (Pearson 0.748), yet SEED further improves this to 0.813.
3. Object F1 and Cap-Sim individually also show high correlation with human evaluation.
4. Re-evaluating state-of-the-art models with SEED reveals that even models with "near-perfect" scores frequently confuse key objects.
5. Caption-based similarity evaluation (Cap-Sim) has never been previously proposed, despite its conceptual simplicity.

## Highlights & Insights
- **Revealing evaluation blind spots**: The paper challenges the illusion that brain decoding is nearly solved.
- **Neuroscience inspiration**: The two-stage visual attention model motivates the design of Object F1 + Cap-Sim.
- **Human evaluation benchmark**: Data from 1,000 pairs × 22 annotators is publicly released, providing a standard for future research.
- **Novelty of Cap-Sim**: The simplest idea — comparing image captions — had surprisingly never been explored before.

## Limitations & Future Work
- SEED focuses solely on semantic similarity and does not assess low-level visual quality (e.g., texture, color fidelity).
- Object F1 is constrained by the 82 object categories recognizable by the detection model.
- Cap-Sim depends on the quality of the image captioning model, which may produce hallucinated descriptions.
- The optimality of equal-weight averaging of the three metrics is not thoroughly analyzed.

## Related Work & Insights
- **Brain decoding models**: MindEye (Scotti et al., 2023/2024), NeuroPictor (Huo et al., 2024), BrainDiffuser (Ozcelik et al., 2023)
- **Image quality assessment**: SSIM (Wang et al., 2004), FID, LPIPS
- **Open-vocabulary detection**: Grounding DINO (Zhao et al., 2024)
- **Image captioning**: GIT (Wang et al., 2022)

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Cap-Sim is novel; problem formulation and solution are well-motivated.
- **Theoretical Depth**: ⭐⭐⭐ — Primarily empirically driven; lacks theoretical analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Large-scale human evaluation + comprehensive multi-metric comparison + cross-dataset validation.
- **Practical Value**: ⭐⭐⭐⭐⭐ — Directly improves evaluation standards for brain decoding; human evaluation data is publicly released.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)
- [\[AAAI 2026\] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](../../AAAI2026/medical_imaging/fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)
- [\[AAAI 2026\] GuideGen: A Text-Guided Framework for Paired Full-Torso Anatomy and CT Volume Generation](../../AAAI2026/medical_imaging/guidegen_a_text-guided_framework_for_paired_full-torso_anatomy_and_ct_volume_gen.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](../../AAAI2026/medical_imaging/small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[ICCV 2025\] ViCTr: Vital Consistency Transfer for Pathology Aware Image Synthesis](../../ICCV2025/medical_imaging/victr_vital_consistency_transfer_for_pathology_aware_image_synthesis.md)

</div>

<!-- RELATED:END -->
