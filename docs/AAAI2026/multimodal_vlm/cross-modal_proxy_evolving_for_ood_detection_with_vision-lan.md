---
title: >-
  [Paper Note] Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models
description: >-
  [AAAI 2026][Multimodal VLM][OOD Detection] This paper proposes CoEvo, a training-free and annotation-free test-time framework that dynamically updates positive and negative proxy caches through a bidirectional sample-con…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "OOD Detection"
  - "VLM"
  - "Cross-modal Proxy Co-evolution"
  - "Zero-shot"
  - "Test-time Adaptation"
  - "CLIP"
  - "Negative Labels"
date: 2026-05-08
content_hash: 27ccde48e7d931ea
---

# Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models

**Conference**: AAAI 2026
**arXiv**: [2601.08476](https://arxiv.org/abs/2601.08476)  
**Code**: Not released  
**Area**: Multimodal VLM / OOD Detection
**Keywords**: OOD Detection, VLM, Cross-modal Proxy Co-evolution, Zero-shot, Test-time Adaptation, CLIP, Negative Labels

## TL;DR

This paper proposes CoEvo, a training-free and annotation-free test-time framework that dynamically updates positive and negative proxy caches through a bidirectional sample-conditioned text/visual proxy co-evolution mechanism. On ImageNet-1K, CoEvo improves AUROC by 1.33% and reduces FPR95 by 45.98% (from 18.92% to 10.22%) over the strongest negative-label baseline, achieving state-of-the-art zero-shot OOD detection.

## Background & Motivation

Vision-language models (e.g., CLIP) deployed in open-world settings require reliable OOD detection to reject unseen categories. Zero-shot OOD detection requires no annotated negative samples. The dominant "negative label" paradigm selects text labels semantically unrelated to in-distribution (ID) classes from lexicons such as WordNet as OOD proxies, determining ID/OOD membership by comparing the similarity of a test image to positive and negative labels.

However, static negative label design suffers from two fundamental limitations:

1. **Unmodeled Negative Space**: A globally fixed negative label set can only sparsely sample the vast semantic space outside ID classes, leaving many negative semantics relevant to current test samples uncovered.
2. **Modality Misalignment**: Visual features drift under test-time distribution shift while text negative labels remain unchanged, distorting the cross-modal similarity geometry and destabilizing decision thresholds.

AdaNeg partially addresses the first issue by adapting visual proxies, but text negative labels remain static — adaptation is unidirectional.

## Core Problem

How can bidirectional, sample-conditioned dynamic adaptation of both text and visual modal proxies be achieved without training or annotation, to support zero-shot OOD detection under test-time distribution shift?

## Method

### Overall Architecture

CoEvo maintains two modality-specific proxy caches at test time (a text cache and a visual cache), each containing positive and negative queues. The core is a **Proxy-Aligned Co-Evolution mechanism**: visual cues guide dynamic mining of text negative labels, and the updated text proxies in turn refine the visual decision boundary, forming a closed loop.

### Key Designs

1. **Text Proxy Cache**:

    - **Positive proxy queue** $\mathbf{T}_p$: CLIP text encodings of ID class names, kept fixed to preserve ID semantic anchors.
    - **Negative proxy queue** $\mathbf{T}_n$: initialized with $M$ negative class encodings selected from large-scale lexicons (WordNet/CSP), dynamically evolved during inference.

2. **Visual Proxy Cache**:

    - **Positive proxy queue** $\mathbf{V}_p \in \mathbb{R}^{K \times L \times D}$: stores $L$ visual instances per ID class, initialized from corresponding text encodings, with high-confidence ID samples incrementally enqueued.
    - **Negative proxy queue** $\mathbf{V}_n$: high-confidence OOD samples are enqueued; a priority queue strategy evicts low-confidence or stale samples.

3. **Proxy-Aligned Co-Evolution Mechanism** (core contribution):

    - **Text proxy evolution**: gated by a confidence margin mechanism (based on adaptive threshold $\delta$ and margin $\gamma$); for high-confidence samples:
        - Predicted OOD → retrieve text negative labels semantically close to the test sample (Near Negatives), tightening the local open-set boundary.
        - Predicted ID → retrieve text negative labels semantically far from the test sample (Far Negatives), expanding negative space coverage.
        - A deduplication constraint prevents redundant enqueuing.
    - **Visual proxy evolution**: after text proxy updates, the visual negative proxy queue expands to accommodate newly exposed OOD semantics; an entropy-based confidence measure determines whether to replace existing proxies, ensuring the queue always stores the highest-confidence samples.

4. **OOD Score Evolution**:

    - **Pre-evolution score** (for proxy updating): $\mathcal{S}^{\text{pre}} = \lambda \mathcal{S}_T^{\text{pre}} + (1-\lambda) \mathcal{S}_V^{\text{pre}}$, with higher text weight ($\lambda=0.8$), since visual proxies are sparse and unreliable at cold start.
    - **Post-evolution score** (for final decision): $\mathcal{S}^{\text{post}} = (1-\lambda) \mathcal{S}_T^{\text{post}} + \lambda \mathcal{S}_V^{\text{post}}$, with weights flipped — after evolution, visual proxies accumulate instance-level information and become more discriminative.
    - This **weight-flipping** strategy is a key design: at cold start, semantic priors are trusted; after sufficient evolution, visual evidence is trusted.

5. **Adaptive Threshold**: A data-driven threshold estimation approach (minimizing intra-class variance of ID/OOD scores) is adopted, providing greater robustness than fixed thresholds.

### Loss & Training

Completely training-free and annotation-free. The CLIP backbone parameters are not modified; only the proxy caches are evolved online at test time.

## Key Experimental Results

### Main Results on ImageNet-1K (CLIP ViT-B/16)

| Method | iNaturalist FPR95↓ | SUN FPR95↓ | Places FPR95↓ | Textures FPR95↓ | Avg. FPR95↓ | Avg. AUROC↑ |
|------|-------------------|------------|--------------|----------------|------------|------------|
| NegLabel | 1.91 | 20.53 | 35.59 | 43.56 | 25.40 | 94.21 |
| CSP | 1.54 | 13.66 | 29.32 | 25.52 | 17.51 | 95.76 |
| AdaNeg | 0.59 | 9.50 | 34.34 | 31.27 | 18.92 | 96.66 |
| **CoEvo-NegLabel** | **0.53** | **4.42** | **23.51** | **12.42** | **10.22** | **97.95** |
| **CoEvo-CSP** | 0.46 | 4.68 | 25.83 | 12.78 | 10.94 | 97.85 |

- FPR95 drops from 18.92% (AdaNeg) to 10.22%, a relative reduction of 45.98%.
- The largest improvement is on the Textures dataset (31.27% → 12.42%), which exhibits the most severe distribution shift.

### OpenOOD Benchmark (Near-OOD + Far-OOD)

| Method | Near-OOD FPR95↓ | Far-OOD FPR95↓ | ID ACC↑ |
|------|----------------|----------------|---------|
| AdaNeg | 67.51 | 17.31 | 67.13 |
| CoEvo-NegLabel | 64.64 | 15.24 | 66.83 |
| CoEvo-CSP | 66.88 | **14.47** | **67.36** |

- Substantial gains are observed in the Far-OOD setting; improvements are more modest for Near-OOD.

### Ablation Study

| Proxy Evolution Configuration | Avg. FPR95↓ | Avg. AUROC↑ |
|---------------|------------|------------|
| No evolution (NegLabel) | 24.97 | 94.56 |
| Text evolution only | 21.77 | 95.38 |
| Visual evolution only | 17.41 | 96.99 |
| **Bidirectional co-evolution** | **10.22** | **97.95** |

- Both text and visual evolution contribute independently; bidirectional combination yields super-additive gains.
- The standalone contribution of visual evolution (−7.56 FPR95) exceeds that of text evolution (−3.20 FPR95).

### Robustness to Data Imbalance (FPR95↓, ImageNet vs. SUN)

| ID:OOD Ratio | NegLabel | AdaNeg | CoEvo-NegLabel |
|------------|---------|--------|---------------|
| 1:100 | 23.00 | 28.00 | **17.00** |
| 1:1 | 21.55 | 8.01 | **5.27** |
| 100:1 | 19.69 | 17.40 | **14.77** |

- CoEvo achieves the best performance across all ratios and remains effective under extreme imbalance (100:1, only 100 OOD samples).

## Highlights & Insights

- **Bidirectional co-evolution is the core innovation**: The closed-loop design in which text guides visual adaptation and visual feedback refines text addresses the fundamental limitation of unidirectional adaptation in prior methods.
- **The pre/post weight-flipping strategy is elegant**: Trusting text priors at cold start and visual evidence after sufficient evolution allows adaptive modulation of cross-modal contributions.
- **Fully training- and annotation-free**: CoEvo is plug-and-play with any CLIP model and requires neither additional training nor OOD annotation.
- **A 45.98% reduction in FPR95 is highly significant** — directly impacting safety in real-world deployment.

## Limitations & Future Work

- **Limited Near-OOD improvement**: Gains are less pronounced in fine-grained OOD settings such as SSB-hard compared to Far-OOD scenarios.
- **Test-time computational overhead**: Each sample requires Top-N text candidate retrieval and dual-modal cache updates, introducing higher inference latency than static methods.
- **Dependence on initial negative label quality**: If the WordNet/CSP initial negative labels offer insufficient coverage, the evolution starts from a poor initialization.
- **Order sensitivity**: Online evolution results may be affected by the order of test samples, a concern not thoroughly discussed in the paper.
- **Validated only on CLIP ViT-B/16**: Performance on larger-scale VLMs (e.g., ViT-L, SigLIP) remains unknown.

## Related Work & Insights

- **vs. NegLabel/CSP (static negative labels)**: CoEvo addresses the sparsity and misalignment issues of static methods through dynamic proxy evolution, reducing FPR95 by approximately 60%.
- **vs. AdaNeg (unidirectional adaptation)**: AdaNeg adapts only visual proxies while text labels remain static; CoEvo achieves bidirectional adaptation with substantial additional gains (FPR95 from 18.92% to 10.22%).
- **vs. MCM (maximum softmax)**: MCM does not use negative labels and relies solely on ID class similarity, yielding performance well below negative-label methods.
- **vs. training-based methods (LoCoOp/LAPT)**: CoEvo as a training-free method surpasses training-based methods requiring prompt tuning.

The co-evolution paradigm generalizes naturally to other test-time adaptation scenarios for VLMs, including domain adaptation, continual learning, and open-set recognition. The weight-flipping strategy — trusting priors at cold start and data after evolution — represents a general-purpose design pattern for online learning. The proxy cache with entropy-based online updates can also be adapted for retrieval-augmented systems.

## Rating

- Novelty: ⭐⭐⭐⭐ — Bidirectional proxy co-evolution mechanism is novel; the weight-flipping design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — ImageNet-1K, OpenOOD, ablations, hyperparameter sensitivity, and imbalance analysis are all thoroughly covered.
- Writing Quality: ⭐⭐⭐⭐ — Problem analysis is thorough, mathematical derivations are clear, and algorithm pseudocode is complete.
- Value: ⭐⭐⭐⭐ — Direct value for safe VLM deployment; training-free nature facilitates practical adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models](../../CVPR2026/multimodal_vlm/activation_matters_test-time_activated_negative_labels_for_ood_detection_with_vi.md)
- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](../../ICCV2025/multimodal_vlm/negrefine_refining_negative_label-based_zero-shot_ood_detection.md)
- [\[CVPR 2026\] Mind the Way You Select Negative Texts: Pursuing the Distance Consistency in OOD Detection with VLMs](../../CVPR2026/multimodal_vlm/mind_the_way_you_select_negative_texts_pursuing_the_distance_consistency_in_ood_.md)
- [\[AAAI 2026\] Harnessing Vision-Language Models for Time Series Anomaly Detection](harnessing_vision-language_models_for_time_series_anomaly_detection.md)
- [\[AAAI 2026\] Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models](cross-modal_unlearning_via_influential_neuron_path_editing_i.md)

</div>

<!-- RELATED:END -->
