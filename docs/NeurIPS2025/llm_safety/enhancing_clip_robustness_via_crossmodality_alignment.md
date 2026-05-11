---
title: >-
  [Paper Note] Enhancing CLIP Robustness via Cross-Modality Alignment
description: >-
  [NeurIPS 2025 (Spotlight)][LLM Safety][CLIP] This paper proposes COLA, a training-free framework that eliminates non-semantic noise by projecting adversarially perturbed image features onto the subspace spanned by text f…
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "LLM Safety"
  - "CLIP"
  - "adversarial robustness"
  - "optimal transport"
  - "cross-modality alignment"
  - "subspace projection"
date: 2026-05-08
content_hash: 62becf33ecf40961
---

# Enhancing CLIP Robustness via Cross-Modality Alignment

**Conference**: NeurIPS 2025 (Spotlight)
**arXiv**: [2510.24038](https://arxiv.org/abs/2510.24038)
**Code**: None
**Area**: Multimodal VLM / Adversarial Robustness
**Keywords**: CLIP, adversarial robustness, optimal transport, cross-modality alignment, subspace projection

## TL;DR

This paper proposes COLA, a training-free framework that eliminates non-semantic noise by projecting adversarially perturbed image features onto the subspace spanned by text features, and then employs optimal transport (OT) to perform fine-grained distribution-level image-text alignment. COLA achieves an average improvement of 6.7% in adversarial robust accuracy across 14 zero-shot classification benchmarks while preserving clean sample performance.

## Background & Motivation

Vision-language models such as CLIP achieve impressive zero-shot classification performance, yet are highly vulnerable to adversarial perturbations (e.g., PGD and CW attacks)—a perturbation of merely $\varepsilon=1/255$ suffices to reduce ImageNet accuracy from 62% to 1%. Existing defenses fall into three categories: (1) adversarial fine-tuning (TeCoA/PMG/FARE)—requires additional training and significantly degrades clean accuracy; (2) prompt optimization—similarly requires training; (3) test-time defenses (TTE/TTC)—incur large inference latency or require a stronger counter-attack budget.

All of these approaches overlook a fundamental issue: **CLIP inherently exhibits a modality gap between image and text features.** Due to the global matching paradigm of contrastive learning, image and text feature distributions reside in different regions of the embedding space. Adversarial perturbations dramatically amplify this misalignment—not only is global alignment disrupted (image features deviate from text semantic prototypes), but local structure also collapses (neighboring image features become scattered and lose coherence).

## Core Problem

How can the cross-modal misalignment caused by adversarial attacks be effectively corrected **without any training or model architecture modification**, thereby improving zero-shot adversarial robustness of CLIP?

The challenge operates at two levels: (1) adversarial perturbations push image features away from semantically relevant text-space directions (global misalignment); (2) even when the feature spaces are unified, local semantic mismatches persist because images contain background and other visual cues unrelated to LLM-generated textual descriptions (local misalignment).

## Method

### Overall Architecture

COLA (CrOss-modaLity Alignment) is a test-time inference framework whose pipeline consists of three steps:

1. **Global feature projection**: SVD is applied to the text feature matrix to extract a principal component subspace; adversarial image features are then projected onto this subspace to filter out perturbations in non-semantic directions.
2. **Local distribution modeling**: Each image and each class is modeled as a discrete distribution—images generate $N$ support points via multi-view augmentation, and classes generate $M$ support points via LLM-generated textual descriptions.
3. **OT-based alignment and classification**: Optimal transport computes the minimum transport cost from each image distribution to each class distribution; the class with the smallest transport cost is selected as the prediction.

The input is an adversarial image and the output is a class label. The entire process requires no training or model modification.

### Key Designs

1. **Text Subspace Projection (Global Feature Alignment)**: The features of all $K$ classes $\times$ $M$ textual descriptions are arranged into a matrix $Z \in \mathbb{R}^{d \times KM}$. SVD is applied to obtain the top-$C=256$ principal components $U_C$, defining the projection $\Pi(\hat{x}) = U_C U_C^\top \hat{x}$. The core insight is that an adversarial perturbation $\delta$ can be decomposed into an in-subspace component $\delta_\parallel$ and an out-of-subspace component $\delta_\perp$; the projection directly eliminates $\delta_\perp$, thereby restoring alignment between image and text features. The authors theoretically prove that cosine similarity distortion strictly decreases after projection.

2. **Entropy-Based Importance Weighting**: Rather than uniformly weighting the $N$ augmented views of an image and the $M$ textual descriptions, prediction entropy is used to measure the importance of each support point. Lower entropy implies higher prediction confidence and thus higher weight, allowing more reliable views and descriptions to contribute more in OT matching.

3. **OT Classification with Projected Cost Matrix**: The key innovation is embedding subspace projection into the OT cost matrix: $C_y^\Pi(n,m) = 1 - \cos(\Pi(\hat{x}_n), z_y^m)$. This allows global and local semantic alignment to be jointly optimized within a unified OT framework. The classification rule becomes $y = \arg\min_y \, d_\mathrm{OT}(P(x), Q_y(z); C_y^\Pi)$.

### Theoretical Guarantees

The paper provides two key theoretical results:

1. **Projection preserves pairwise similarity**: The cosine similarity distortion after projection satisfies $\Delta_\Pi \leq \Delta$, indicating that projection does not harm the semantic relationships among features.
2. **OT classifier margin increases**: $\gamma(C^\Pi) \geq \gamma(C)$, meaning the decision margin of the OT classifier under the projected cost matrix is no smaller than under the original cost matrix, implying better generalization.

## Key Experimental Results

### 9 Datasets + PGD/CW Attacks (ViT-B/32, $\varepsilon=1/255$)

| Method | Type | 9-datasets Clean | 9-datasets Robust (PGD) | 9-datasets Robust (CW) |
|--------|------|------------------|-------------------------|------------------------|
| CLIP | baseline | 59.5 | 2.4 | 3.5 |
| TeCoA | fine-tuning | 33.9 | 15.6 | 14.6 |
| PMG | fine-tuning | 35.2 | 16.1 | 15.1 |
| FARE | fine-tuning | 44.8 | 10.0 | 10.9 |
| TTC | test-time | 55.1 | 30.8 | 29.4 |
| **COLA** | **test-time** | **61.9** | **45.3** | **40.9** |

### ImageNet and Variants (PGD Attack)

| Dataset | CLIP Robust | TTC Robust | COLA Robust | Gain (vs TTC) |
|---------|-------------|------------|-------------|---------------|
| ImageNet | 1.1 | 40.0 | 50.0 | +10.0 |
| ImageNet-V2 | 0.0 | 15.4 | 22.7 | +7.3 |
| ImageNet-Sketch | 0.8 | 34.4 | 43.2 | +8.8 |
| ImageNet-A | 6.1 | 48.5 | 55.6 | +7.1 |
| ImageNet-R | 5.0 | 24.4 | 29.8 | +5.4 |
| **Average** | **2.6** | **32.5** | **40.3** | **+7.7** |

### COLA Stacked on Fine-Tuned Models (PGD Attack, 9-datasets Robust)

| Base Model | Original Robust | +TTC | +COLA |
|-----------|----------------|------|-------|
| TeCoA | 15.6 | 17.9 | **27.3** |
| PMG | 16.1 | 18.5 | **29.1** |
| FARE | 10.0 | 25.6 | **45.3** |

### Different Backbones (PGD Attack)

| Model | ImageNet Robust (TTC) | ImageNet Robust (COLA) |
|-------|-----------------------|------------------------|
| ViT-B/16 | 20.1 | **32.1** |
| ViT-L/14 | 21.9 | **57.7** |

### Inference Efficiency (ImageNet, ViT-B/32)

| Method | Time | Clean | Robust |
|--------|------|-------|--------|
| CLIP | 10 min | 62.1 | 1.1 |
| TTC | 40 min | 51.7 | 40.0 |
| COLA | 28 min | 62.8 | 50.0 |

### Ablation Study

- **Projected vs. original cost matrix**: OT with $C^\Pi$ outperforms OT with $C$ by 3.7% on ImageNet PGD robust accuracy (50.0 vs. 46.3), confirming the effectiveness of the subspace-projected cost.
- **Number of augmentations**: Marginal gains diminish beyond $N=5$ image augmentations; text descriptions saturate beyond $M=50$. The method is not sensitive to these hyperparameters.
- **Number of SVD components $C$**: Performance improves steadily as $C$ increases below 200; clean accuracy saturates and robust accuracy gains slow for $C>200$. The final setting uses $C=256$.
- **Similarity distribution visualization**: Image-text cosine similarity drops sharply after adversarial attack and recovers to near-original levels after projection.
- **Large attack budget $\varepsilon=4/255$**: While other methods' robust accuracy nearly collapses to zero, COLA maintains substantial robustness (more than 50% higher than TTC), demonstrating strong resilience.

## Highlights & Insights

- **Training-free and plug-and-play**: No training or architectural modification is required; COLA can be directly stacked on any fine-tuned CLIP model, making it highly deployment-friendly.
- **Elegant theoretical support**: Subspace projection reduces cosine distortion, and the projected cost matrix enlarges the OT decision margin—theory and experiments are highly consistent.
- **Unified framework for global and local alignment**: Subspace projection addresses global misalignment while OT distribution matching resolves local semantic mismatch; the two components are naturally integrated via the projected cost matrix.
- **Dominant advantage under large-$\varepsilon$ attacks**: At $\varepsilon=4/255$, other methods nearly fail while COLA remains effective, revealing a fundamental mechanistic advantage.
- **Transferable paradigm**: The idea of using text features as a "clean reference subspace" to purify adversarial image features is transferable to other cross-modal robustness tasks.

## Limitations & Future Work

- **Inherited pretraining bias**: The text subspace encodes dataset-specific priors, which may limit generalization to unseen linguistic or visual domains.
- **Robustness to adaptive attacks is unknown**: The paper primarily evaluates against standard attacks (PGD/CW/AutoAttack); if the attacker is aware of COLA's projection mechanism, adaptive attacks designed to circumvent it remain unstudied.
- **SVD computational overhead**: SVD must be performed over text features for all classes, and the cost may become significant at very large class counts (e.g., ImageNet-21K).
- **Classification-only scope**: The framework is designed around zero-shot classification; downstream tasks such as detection and segmentation would require additional design.
- **Rudimentary augmentation strategy**: Image augmentation relies on simple random cropping and flipping; stronger augmentation strategies may yield further improvements.

## Related Work & Insights

- **vs. TTC** (Test-Time Counterattacks): TTC defends by generating counter-attacks, requiring a larger counter-attack budget and slower inference (40 min vs. COLA's 28 min). COLA substantially outperforms TTC in robust accuracy across all settings while maintaining better clean accuracy.
- **vs. TeCoA/PMG/FARE** (adversarial fine-tuning): These methods require additional training and significantly harm clean performance. COLA requires no training and can be applied as a plug-and-play module on top of these fine-tuned models to further boost performance.
- **vs. general OT alignment methods** (PLOT/AWT): These methods optimize OT alignment during training. COLA is the first to introduce OT into the test-time adversarial defense setting, and achieves a larger decision margin than standard OT through the projected cost matrix.

- **Subspace projection for VLM hallucination mitigation**: VLM hallucination can be viewed as image features "drifting" from the text semantic space; a similar projection-based purification approach may prove effective.
- **OT + VLM combinations merit further exploration**: OT is naturally suited to modeling fine-grained correspondences between modalities and can be extended to robustness improvements in image-text retrieval, VQA, and beyond.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of subspace projection and OT alignment for adversarial robustness is novel, though individual components are not new in themselves.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 14 datasets, multiple attack types, multiple backbones, stacking on fine-tuned models, large attack budgets, comprehensive ablations, and runtime comparisons—extremely thorough.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear and theoretical proofs are complete, though the dense notation presents a moderate barrier on first reading.
- Value: ⭐⭐⭐⭐ Training-free with strong performance; directly relevant to practical deployment. NeurIPS Spotlight recognition is well deserved.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training](toxictextclip_text-based_poisoning_and_backdoor_attacks_on_clip_pre-training.md)
- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [\[NeurIPS 2025\] Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values](distributive_fairness_in_large_language_models_evaluating_alignment_with_human_v.md)
- [\[NeurIPS 2025\] Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties](trans-env_a_framework_for_evaluating_the_linguistic_robustness_of_llms_against_e.md)
- [\[NeurIPS 2025\] Enhancing Sample Selection Against Label Noise by Cutting Mislabeled Easy Examples](enhancing_sample_selection_against_label_noise_by_cutting_mislabeled_easy_exampl.md)

</div>

<!-- RELATED:END -->
