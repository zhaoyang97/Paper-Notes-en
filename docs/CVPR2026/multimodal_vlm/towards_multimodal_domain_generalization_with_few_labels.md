---
title: >-
  [Paper Note] Towards Multimodal Domain Generalization with Few Labels
description: >-
  [CVPR 2026][Multimodal VLM][Semi-supervised Learning] This paper defines and investigates the novel problem of Semi-Supervised Multimodal Domain Generalization (SSMDG), and proposes a unified framework integrating consensus-driven pseudo-labeling, disagreement-aware regularization, and cross-modal prototype alignment to achieve cross-domain generalization of multimodal models under limited annotation.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Semi-supervised Learning
  - Domain Generalization
  - Multimodal Fusion
  - Pseudo Labels
  - Cross-modal Prototype Alignment
date: 2026-05-08
content_hash: fde90cb1f73c1b45
---

# Towards Multimodal Domain Generalization with Few Labels

**Conference**: CVPR 2026
**arXiv**: [2602.22917](https://arxiv.org/abs/2602.22917)
**Code**: [https://github.com/lihongzhao99/SSMDG](https://github.com/lihongzhao99/SSMDG)
**Area**: Multimodal VLM
**Keywords**: Semi-supervised Learning, Domain Generalization, Multimodal Fusion, Pseudo Labels, Cross-modal Prototype Alignment

## TL;DR
This paper defines and investigates the novel problem of Semi-Supervised Multimodal Domain Generalization (SSMDG), and proposes a unified framework integrating consensus-driven pseudo-labeling, disagreement-aware regularization, and cross-modal prototype alignment to achieve cross-domain generalization of multimodal models under limited annotation.

## Background & Motivation
**Background**: Multimodal Domain Generalization (MMDG) assumes all source domain data are labeled; Semi-Supervised Multimodal Learning (SSML) exploits unlabeled data but ignores domain shift; Semi-Supervised Domain Generalization (SSDG) addresses domain shift but is restricted to unimodal inputs. Each direction addresses only part of the problem.

**Limitations of Prior Work**: In real-world scenarios, three challenges arise simultaneously—multimodal data, scarce labels, and domain shift. MMDG methods cannot leverage large amounts of unlabeled data; SSML methods assume identical training and test distributions; SSDG methods cannot exploit cross-modal complementarity.

**Key Challenge**: (a) How to obtain reliable pseudo labels under low confidence and inter-modal disagreement; (b) how to learn representations that are simultaneously invariant to modality and domain under limited supervision.

**Goal**: Establish an SSMDG benchmark and design a unified framework that jointly addresses pseudo-label reliability and domain-modality invariant representation learning.

**Key Insight**: The consensus between fusion predictions and unimodal predictions is leveraged to filter reliable pseudo labels, while class prototypes serve as semantic anchors across domains and modalities.

**Core Idea**: Achieve robust generalization on sparsely labeled multimodal multi-domain data through consensus-driven pseudo-label filtering and cross-modal prototype alignment.

## Method

### Overall Architecture
The model comprises modality-specific encoders, unimodal classifiers, and a fusion classifier. During training, samples are drawn from a joint pool of labeled and unlabeled data, processed through three complementary components: (1) Consensus-Driven Consistency Regularization (CDCR); (2) Disagreement-Aware Regularization (DAR); (3) Cross-Modal Prototype Alignment (CMPA).

### Key Designs

1. **Consensus-Driven Consistency Regularization (CDCR)**:

    - **Function**: Generates reliable pseudo labels for unlabeled data.
    - **Mechanism**: A pseudo label is accepted only when the fusion prediction and at least one unimodal prediction simultaneously satisfy a high-confidence threshold $\tau$ with consistent label assignment. A FixMatch-style consistency loss is then applied to qualifying samples: $\mathcal{L}_{\text{cdcr}} = \frac{1}{|\mathcal{B}_{\text{cdcr}}^u|}\sum\sum_{n\in\{v,a,f\}}\mathcal{H}(\hat{y}, \hat{p}_n^s)$
    - **Design Motivation**: Decisions consistent across multiple views are more reliable than single-view predictions; the consensus mechanism naturally filters out low-quality pseudo labels.

2. **Disagreement-Aware Regularization (DAR)**:

    - **Function**: Exploits "non-consensus" samples excluded by CDCR that still carry useful information.
    - **Mechanism**: For samples where the fusion prediction is highly confident but inter-modal labels are inconsistent, the standard cross-entropy loss is replaced by the Generalized Cross-Entropy (GCE) loss $\mathcal{L}_{\text{GCE}} = (1-p_{\hat{y}}^q)/q$, which is more robust to noisy labels.
    - **Design Motivation**: Non-consensus samples typically reside near decision boundaries; discarding them entirely wastes valuable information. The parameter $q$ of GCE controls tolerance to noise.

3. **Cross-Modal Prototype Alignment (CMPA)**:

    - **Function**: Constructs a domain- and modality-invariant representation space.
    - **Mechanism**: Learnable prototype vectors are maintained per class as semantic anchors, and features from each domain and modality are aligned toward the corresponding class prototype. A cross-modal translation network is additionally trained to handle modality-missing scenarios.
    - **Design Motivation**: Class prototypes provide stable reference points across domains and modalities, offering greater flexibility than direct domain or modality alignment without requiring domain labels.

### Loss & Training
Total loss: $\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda_1\mathcal{L}_{\text{cdcr}} + \lambda_2\mathcal{L}_{\text{dar}} + \lambda_3\mathcal{L}_{\text{cmpa}}$. The supervised loss is computed over both fusion and unimodal predictions on labeled data. Weak augmentation uses standard transformations; strong augmentation applies RandAugment (video) and SpecAugment (audio).

## Key Experimental Results

### Main Results (5 labels per class)

| Method | Type | HAC Mean | EPIC Mean |
|--------|------|----------|-----------|
| Source-only | Baseline | 42.39 | 29.46 |
| SimMMDG | MMDG | 44.39 | 31.11 |
| MDJA | MMDG | 44.28 | 31.51 |
| FixMatch (Video) | SSL | 48.74 | 32.54 |
| CGMatch (Video) | SSL | 49.10 | 33.42 |
| **Ours** | **SSMDG** | **55.82** | **38.15** |

### Ablation Study

| Configuration | HAC Mean | EPIC Mean |
|---------------|----------|-----------|
| Baseline | 42.39 | 29.46 |
| + CDCR | 49.15 | 33.80 |
| + CDCR + DAR | 52.30 | 35.90 |
| + CDCR + DAR + CMPA | **55.82** | **38.15** |
| w/o Consensus Filtering | 47.20 | 31.50 |

### Key Findings
- The SSMDG method substantially outperforms all MMDG baselines (+11%), as the latter cannot exploit unlabeled data.
- Unimodal SSL (FixMatch on video) already surpasses MMDG methods, underscoring the value of leveraging unlabeled data.
- CDCR contributes the largest gain (+7%); DAR adds an additional 3%, and CMPA a further 3%.
- Directly using all high-confidence pseudo labels without consensus filtering degrades performance by 5%, validating the necessity of the filtering strategy.
- In modality-missing scenarios (video-only or audio-only), cross-modal translation yields more graceful performance degradation.

## Highlights & Insights
- **Forward-Looking Problem Formulation**: The paper unifies three independently studied challenges into SSMDG and establishes the first benchmark. The intersection of these three threads represents a practically needed yet previously unexplored setting.
- **Consensus-Driven Pseudo Labeling**: Unlike threshold-based filtering that relies solely on the fusion prediction, incorporating inter-modal consistency verification further improves reliability—a natural and effective innovation for multimodal semi-supervised learning.
- **GCE for Non-Consensus Samples**: Rather than simply discarding uncertain samples, the method gently exploits them via a noise-robust loss, reflecting a design philosophy of "partial utilization over wasteful exclusion."

## Limitations & Future Work
- Validation is limited to the video-audio bimodal setting; vision-language or trimodal scenarios remain unexplored.
- The threshold $\tau$ is applied uniformly across all domains; domain-adaptive thresholds may yield better results.
- Class prototypes are updated via simple averaging; momentum updates or attention-weighted aggregation may offer improvements.
- Scalability on large-scale datasets (e.g., large-scale video classification) has not been verified.

## Related Work & Insights
- **vs. SimMMDG**: SimMMDG performs cross-modal alignment with fully labeled data; the proposed method achieves the same objective under limited labels via pseudo labeling and prototype alignment, making it more practical.
- **vs. FixMatch**: FixMatch is the standard unimodal SSL baseline; the proposed CDCR leverages multimodal consensus to produce more reliable pseudo labels.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel problem formulation with a well-motivated unified framework
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks, diverse baseline comparisons, and modality-missing experiments add value
- Writing Quality: ⭐⭐⭐⭐ Problem definition and method description are clear
- Value: ⭐⭐⭐⭐ Fills an unexplored gap at the intersection of three research lines; the benchmark offers community value

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Reasoning-Driven Multimodal LLM for Domain Generalization](../../ICLR2026/multimodal_vlm/reasoning-driven_multimodal_llm_for_domain_generalization.md)
- [\[CVPR 2026\] BiCLIP: Domain Canonicalization via Structured Geometric Transformation](biclip_domain_canonicalization_via_structured_geometric_transformation.md)
- [\[CVPR 2026\] FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression](flashcache_frequency_kv_cache_compression.md)
- [\[CVPR 2026\] Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling](conditional_factuality_controlled_llms_with_generalization_certificates_via_conf.md)
- [\[CVPR 2026\] Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models](activation_matters_test-time_activated_negative_labels_for_ood_detection_with_vi.md)

<!-- RELATED:END -->
