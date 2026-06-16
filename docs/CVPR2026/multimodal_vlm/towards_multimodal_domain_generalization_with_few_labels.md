---
title: >-
  [Paper Note] Towards Multimodal Domain Generalization with Few Labels
description: >-
  [CVPR 2026][Multimodal VLM][Paper Note] This paper defines and investigates the new Semi-Supervised Multi-modal Domain Generalization (SSMDG) problem, proposing a unified framework driven by consensus-based pseudo-labeling, disagreement-aware regularization, and cross-modal prototype alignment to achieve cross-domain generalization under sparse labeling.
tags:
  - CVPR 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: 096170571a3574c2
---
# Towards Multimodal Domain Generalization with Few Labels

**Conference**: CVPR 2026  
**arXiv**: [2602.22917](https://arxiv.org/abs/2602.22917)  
**Code**: [https://github.com/lihongzhao99/SSMDG](https://github.com/lihongzhao99/SSMDG)  
**Area**: Multi-modal VLM  
**Keywords**: Semi-supervised learning, Domain Generalization, Multi-modal fusion, Pseudo-labeling, Cross-modal prototype alignment

## TL;DR
This paper defines and investigates the new Semi-Supervised Multi-modal Domain Generalization (SSMDG) problem, proposing a unified framework driven by consensus-based pseudo-labeling, disagreement-aware regularization, and cross-modal prototype alignment to achieve cross-domain generalization under sparse labeling.

## Background & Motivation
**Background**: Multi-modal Domain Generalization (MMDG) assumes all source domain data are labeled; Semi-Supervised Multi-modal Learning (SSML) utilizes unlabeled data but ignores domain shift; Semi-Supervised Domain Generalization (SSDG) handles domain shift but is limited to single-modal inputs. These three directions each address only partial aspects of the problem.

**Limitations of Prior Work**: In practical scenarios, three challenges coexist: multi-modal data + few labels + domain shift. MMDG methods cannot leverage large-scale unlabeled data; SSML methods assume identical training and testing distributions; SSDG methods cannot exploit cross-modal complementarity.

**Key Challenge**: (a) How to obtain reliable pseudo-labels under conditions of low confidence and inter-modality disagreement; (b) how to learn representations that are simultaneously invariant to both modality and domain under limited supervision.

**Goal**: Construct an SSMDG benchmark and design a unified framework to simultaneously address pseudo-label reliability and the learning of domain-modality invariant representations.

**Key Insight**: Leverage the consensus between fusion predictions and single-modal predictions to filter reliable pseudo-labels, and use class prototypes as semantic anchors across domains and modalities.

**Core Idea**: Achieve robust generalization on sparsely labeled multi-modal multi-domain data through consensus-driven pseudo-label filtering and cross-modal prototype alignment.

## Method

### Overall Architecture
SSMDG aims to tackle three major challenges simultaneously: multi-modal inputs, sparse labels, and cross-domain shift. This paper decomposes the problem into two parts: first, converting massive unlabeled samples into usable supervisory signals, and second, ensuring the learned representations are stable across modalities and domains. To achieve this, it equips each modality with an encoder, followed by two sets of classification heads: individual single-modal classifiers and a fusion classifier that concatenates all modal features. During training, a batch is mixed from labeled and unlabeled pools, passing through three complementary components: Consensus-Driven Consistency Regularization (CDCR) to select credible pseudo-labels, Disagreement-Aware Regularization (DAR) to recover valuable samples missed by CDCR, and Cross-Modal Prototype Alignment (CMPA) to align features across domains and modalities to the same set of class prototypes. All three share the same encoders and are trained end-to-end. Unlabeled samples follow a **consensus-based routing** path: those meeting consensus go to CDCR, while those with high-confidence fusion but no consensus go to DAR. Pseudo-labeled features from both paths flow into CMPA for prototype alignment.

```mermaid
graph TD
    A["Input: Multi-modal samples from K source domains<br/>Video+Audio, few labels + many unlabeled"] --> B["Modal Encoders + Dual Heads<br/>Single-modal Classifiers + Fusion Classifier"]
    B -->|Weak augmentation prediction| C{"Fusion & Single-modal<br/>High confidence & Consensus?"}
    C -->|Yes (Consensus)| D["Consensus-Driven Consistency Regularization (CDCR)<br/>Adopt pseudo-label + Weak-strong consistency loss"]
    C -->|No, but Fusion is High Conf.| E["Disagreement-Aware Regularization (DAR)<br/>GCE loss utilizes disagreement hard cases"]
    D --> F["Cross-Modal Prototype Alignment (CMPA)<br/>Class prototype anchors + Translation networks"]
    E --> F
    B -->|Labeled sample supervision| F
    F --> G["Generalize to Unseen Target Domain"]
```

### Key Designs

**1. Consensus-Driven Consistency Regularization (CDCR): Multi-view endorsement instead of relying solely on fusion**

Common semi-supervised methods apply a confidence threshold to fusion predictions. However, in cross-domain settings, fusion predictions can be overconfident due to domain shift. CDCR requires "multi-view consensus": for an unlabeled sample, a pseudo-label is only adopted if the fusion prediction and at least one single-modal prediction both exceed a high-confidence threshold $\tau$ and point to the same category. Samples passing this filter enter the batch $\mathcal{B}_{\text{cdcr}}^u$, and a FixMatch-style consistency loss pulls strong augmentation views toward the pseudo-label:

$$\mathcal{L}_{\text{cdcr}} = \frac{1}{|\mathcal{B}_{\text{cdcr}}^u|}\sum\sum_{n\in\{v,a,f\}}\mathcal{H}(\hat{y}, \hat{p}_n^s)$$

where $n$ iterates through video, audio, and fusion branches, and $\hat{p}_n^s$ represents predictions under strong augmentation. Consensus acts as a free quality filter—decisions agreed upon by multiple independent views are more likely correct, blocking low-quality pseudo-labels.

**2. Disagreement-Aware Regularization (DAR): Utilizing "disagreeing" samples rather than discarding them**

Strict CDCR filtering may discard samples where the fusion prediction is confident but modalities are inconsistent. These are often informative hard cases near decision boundaries. DAR utilizes these "non-consensus but fusion high-confidence" samples using Generalized Cross Entropy (GCE) instead of standard cross-entropy to handle potential noise:

$$\mathcal{L}_{\text{GCE}} = (1-p_{\hat{y}}^q)/q$$

The parameter $q\in(0,1]$ tunes noise tolerance: $q\to 0$ approaches standard cross-entropy, while $q\to 1$ mimics the noise-robust MAE loss. This prevents individual noisy samples from dominating the gradient.

**3. Cross-Modal Prototype Alignment (CMPA): Using prototypes as anchors for domain and modality invariance**

To ensure representation space is invariant to both domains and modalities, CMPA maintains a class prototype for each "modality × category × domain" (updated via EMA of labeled features). It pulls features from any domain or modality toward their **in-domain prototype** and the **mean of other source domain prototypes** of the same class. This bypasses the need for explicit domain labels or exhaustive modality pairing. Additionally, CMPA trains cross-modal translation networks $t_{v\to a}$ and $t_{a\to v}$ to hallucinate missing modal features during inference.

### Loss & Training
The total loss combines the supervision term and the three components:

$$\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda_1\mathcal{L}_{\text{cdcr}} + \lambda_2\mathcal{L}_{\text{dar}} + \lambda_3\mathcal{L}_{\text{cmpa}}$$

Supervision loss $\mathcal{L}_{\text{sup}}$ is calculated for fusion and single-modal classifiers on labeled data. Consistency training follows the weak-strong augmentation paradigm (RandAugment for video, SpecAugment for audio).

## Key Experimental Results

### Main Results (5 labels per class)

| Method | Type | HAC Mean | EPIC Mean |
|------|------|----------|-----------|
| Source-only | Baseline | 42.39 | 29.46 |
| SimMMDG | MMDG | 44.39 | 31.11 |
| MDJA | MMDG | 44.28 | 31.51 |
| FixMatch (Video) | SSL | 48.74 | 32.54 |
| CGMatch (Video) | SSL | 49.10 | 33.42 |
| **Ours** | **SSMDG** | **55.82** | **38.15** |

### Ablation Study

| Configuration | HAC Mean | EPIC Mean |
|------|----------|-----------|
| Baseline | 42.39 | 29.46 |
| + CDCR | 49.15 | 33.80 |
| + CDCR + DAR | 52.30 | 35.90 |
| + CDCR + DAR + CMPA | **55.82** | **38.15** |
| w/o Consensus Filtering | 47.20 | 31.50 |

### Key Findings
- SSMDG significantly outperforms all MMDG methods (+11%), as the latter cannot utilize unlabeled data.
- Single-modal SSL methods (e.g., FixMatch on video) already surpass MMDG methods, highlighting the value of unlabeled data.
- CDCR contributes the most (+7%), with DAR adding 3% and CMPA adding 3%.
- Disabling consensus filtering and using all high-confidence pseudo-labels drops performance by 5%, validating the strategy.
- Cross-modal translation mitigates performance degradation in modality-missing scenarios.

## Highlights & Insights
- **Forward-looking Problem Definition**: Unifies three independent challenges into SSMDG and establishes the first benchmark.
- **Consensus-Driven Pseudo-labeling**: Incorporating inter-modality consistency verification beyond fusion confidence is a natural and effective innovation for multi-modal SSL.
- **DAR Strategy**: Instead of discarding uncertain samples, using noise-robust losses to partially utilize them reflects an insightful design philosophy.

## Limitations & Future Work
- Evaluation is limited to video-audio; vision-language or tri-modal scenarios remain unexplored.
- Threshold $\tau$ is uniform across domains; domain-adaptive thresholds might be superior.
- Prototypes are updated via EMA; attention-weighted or more adaptive prototype estimation could offer improvements.
- Scalability to large-scale video datasets has not yet been verified.

## Related Work & Insights
- **vs SimMMDG**: SimMMDG performs cross-modal alignment with full labels; Ours achieves this with few labels via pseudo-labels and prototype alignment.
- **vs FixMatch**: While FixMatch is the SSL standard for single modalities, CDCR leverages multi-modal consensus for more reliable pseudo-labels.

## Rating
- Novelty: ⭐⭐⭐⭐ New problem definition + sound unified framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks, multiple baselines, and modality-missing analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and methodology.
- Value: ⭐⭐⭐⭐ Fills a void in unexplored multi-challenge intersections.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Single Domain Generalization for Few-Shot Counting via Universal Representation Matching](../../CVPR2025/multimodal_vlm/single_domain_generalization_for_few-shot_counting_via_universal_representation_.md)
- [\[ICLR 2026\] Reasoning-Driven Multimodal LLM for Domain Generalization](../../ICLR2026/multimodal_vlm/reasoning-driven_multimodal_llm_for_domain_generalization.md)
- [\[CVPR 2026\] Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning](mind_the_discriminability_trap_in_source-free_cross-domain_few-shot_learning.md)
- [\[CVPR 2026\] Addressing Exacerbated Attention Sink for Source-Free Cross-Domain Few-Shot Learning](addressing_exacerbated_attention_sink_for_source-free_cross-domain_few-shot_lear.md)
- [\[CVPR 2026\] Pointing at Parts: Training-Free Few-Shot Grounding in Multimodal LLMs](pointing_at_parts_training-free_few-shot_grounding_in_multimodal_llms.md)

</div>

<!-- RELATED:END -->
