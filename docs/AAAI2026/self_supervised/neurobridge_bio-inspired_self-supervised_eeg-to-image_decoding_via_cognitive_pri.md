---
title: >-
  [Paper Note] NeuroBridge: Bio-Inspired Self-Supervised EEG-to-Image Decoding via Cognitive Priors and Bidirectional Semantic Alignment
description: >-
  [AAAI 2026][Self-Supervised Learning][EEG decoding] This paper proposes NeuroBridge, a framework that employs Cognitive Prior Augmentation (CPA, asymmetric augmentation to simulate perceptual variability) and a Shared Semantic Projector (SSP, bidirectional alignment into a unified semantic space). On the THINGS-EEG dataset under a 200-class zero-shot EEG-to-image retrieval task, the method achieves 63.2% Top-1 (+12.3%) and 89.9% Top-5 (+10.2%), substantially surpassing the existing state of the art.
tags:
  - AAAI 2026
  - Self-Supervised Learning
  - EEG decoding
  - cross-modal contrastive learning
  - cognitive prior augmentation
  - shared semantic projection
  - zero-shot retrieval
  - brain-computer interface
date: 2026-05-08
content_hash: ab5e147c28687221
---

# NeuroBridge: Bio-Inspired Self-Supervised EEG-to-Image Decoding via Cognitive Priors and Bidirectional Semantic Alignment

**Conference**: AAAI 2026
**arXiv**: [2511.06836](https://arxiv.org/abs/2511.06836)
**Authors**: Wenjiang Zhang, Sifeng Wang, Yuwei Su, Xinyu Li, Chen Zhang, Suyu Zhong
**Code**: [GitHub](https://github.com/feroooooo/NeuroBridge)
**Area**: Self-Supervised
**Keywords**: EEG decoding, cross-modal contrastive learning, cognitive prior augmentation, shared semantic projection, zero-shot retrieval, brain-computer interface

## TL;DR

This paper proposes NeuroBridge, a framework that employs Cognitive Prior Augmentation (CPA, asymmetric augmentation to simulate perceptual variability) and a Shared Semantic Projector (SSP, bidirectional alignment into a unified semantic space). On the THINGS-EEG dataset under a 200-class zero-shot EEG-to-image retrieval task, the method achieves 63.2% Top-1 (+12.3%) and 89.9% Top-5 (+10.2%), substantially surpassing the existing state of the art.

## Background & Motivation

### Problem Background
Visual neural decoding aims to reconstruct or infer perceived visual stimuli from brain activity, with important applications in brain-computer interfaces and AI. EEG has become a prominent non-invasive neuroimaging modality due to its high temporal resolution, low cost, and portability. Cross-modal contrastive learning is the dominant paradigm for EEG-based visual decoding.

### Limitations of Prior Work

**Dynamic Variability Gap**: EEG responses evoked by the same image in the same subject vary substantially due to fluctuations in attention, mental state, and physiological noise — different individuals focus on different semantic regions of the same image of a cat, producing divergent EEG responses.

**Static Intrinsic Gap**: EEG is a temporal, low-dimensional, noisy signal, whereas images are spatially structured, high-dimensional, and semantically dense. The modality gap is fundamental.

**Unidirectional Alignment Limitation**: Existing methods (e.g., NICE, ATM) predominantly adopt unidirectional alignment, mapping EEG into a frozen CLIP embedding space. However, CLIP's semantic space is shaped by visual-language data and may be semantically misaligned with the perceptual and cognitive processes reflected by EEG.

**Insufficient Augmentation Strategy**: UBP introduces blur priors only on the visual side, while Neural-MCRL performs semantic completion only on the EEG side; a comprehensive bimodal augmentation framework is lacking.

**Data Scarcity**: The scale of paired EEG data is far smaller than large-scale visual-language datasets.

### Core Motivation
Inspired by biological systems' perceptual variability and co-adaptive strategies, the paper simulates human cognitive variability through asymmetric augmentation and achieves collaborative alignment of EEG and image representations in a unified semantic space via bidirectional projection.

## Method

### Overall Architecture
During training: paired EEG-image data → CPA cognitive prior augmentation → encoding (frozen CLIP image encoder + trainable EEG encoder) → SSP projection into a shared semantic space → optimization with bidirectional contrastive loss. During inference: EEG embeddings are matched to images in a visual concept pool via cosine similarity.

### Module 1: Cognitive Prior Augmentation (CPA)

CPA simulates cognitive variability in human visual perception. Its core design is an **asymmetric augmentation strategy**:

- **Image side**: $K$ augmentation strategies (Gaussian blur, Gaussian noise, low resolution, mosaic) are applied to generate multiple views $X'_{I,k} = t_{I,k}(X_I)$; the encoded representations are averaged as $H_I = \frac{1}{K}\sum_{k=1}^{K} H_{I,k}$ to obtain a semantically aggregated representation.
- **EEG side**: Only a single augmentation (smoothing) is applied: $X'_E = t_E(X_E)$.
- **Design Motivation for Asymmetry**: The CLIP image encoder is well pretrained and can fully exploit multiple augmentations; the EEG encoder is trained from scratch, and excessive augmentation would corrupt the signal structure.

Key findings regarding augmentation selection: Gaussian blur, noise, low resolution, and mosaic are effective because they preserve high-level semantics while suppressing low-level pixel variations. Color jitter and grayscale conversion have negative effects, suggesting that human perception is sensitive to color information (consistent with neuroscience findings). Random cropping may remove critical semantic regions. On the EEG side, only smoothing is effective, as it reduces noise in the low-SNR signal; temporal shifts disrupt temporal dynamics.

### Module 2: Shared Semantic Projector (SSP)

SSP maps features from both modalities into a unified, trainable semantic space:

$$Z_I = p_I(H_I), \quad Z_E = p_E(H_E)$$

where $p_I$ and $p_E$ are two projection networks (a default 512-dimensional linear projection performs best). The key distinction is that **$p_I$ is also trainable** (rather than fixed), enabling bidirectional alignment instead of unidirectionally mapping EEG into the CLIP space. Learning a shared space is more flexible than directly leveraging a fixed CLIP space.

### Module 3: Modality-Aware Contrastive Learning

A bidirectional InfoNCE loss is adopted. The key design is **asymmetric normalization**: $\ell_2$ normalization onto the unit hypersphere is applied only to image features, while EEG feature magnitudes are unconstrained. This uses feature direction for semantic alignment and magnitude as a learnable confidence signal. Experiments show (Table 7) that this asymmetric strategy (Asym: 63.2%) substantially outperforms symmetric normalization (Sym: 46.4%) and no normalization (Plain: 54.4%).

### Training Details
- Dataset: THINGS-EEG (10 subjects, RSVP paradigm; training set: 1,654 concepts × 10 images × 4 repetitions = 16,540 samples/subject; test set: 200 concepts × 1 image × 80 repetitions).
- Default configuration: RN50 image encoder + EEGProject EEG encoder (2.44M parameters), batch = 1024, epochs = 50, lr = 1e-4, τ = 0.07.
- Intra-subject: 17 parieto-occipital electrodes (P7/P5/P3/P1/Pz/P2/P4/P6/P8/PO7/PO3/POz/PO4/PO8/O1/Oz/O2); Inter-subject: all 63 channels + TSConv encoder.

## Key Experimental Results

### Table 1: THINGS-EEG 200-Class Zero-Shot Retrieval (Intra-Subject Average)

| Method | Top-1 (%) | Top-5 (%) |
|--------|-----------|-----------|
| BraVL | 5.8 | 17.5 |
| NICE | 16.1 | 43.6 |
| ATM | 27.1 | 58.1 |
| CognitionCapturer | 33.3 | 60.6 |
| Neural-MCRL | 32.4 | 64.1 |
| VE-SDN | 37.2 | 70.0 |
| UBP | 50.9 | 79.7 |
| **NeuroBridge** | **63.2** | **89.9** |

NeuroBridge improves over the previous SOTA (UBP) by +12.3% Top-1 and +10.2% Top-5. On Subject 10, the method achieves up to 73.6% Top-1 and 97.1% Top-5. It also achieves state-of-the-art performance in the inter-subject setting (19.0% Top-1 vs. UBP 12.4%; 45.9% Top-5 vs. UBP 33.4%).

### Table 2: Ablation Study (Component Contributions)

| Image Prior | EEG Prior | SSP | Top-1 (%) | Top-5 (%) |
|:-----------:|:---------:|:---:|-----------|-----------|
| ✗ | ✗ | ✗ | 40.5 | 72.2 |
| ✓ | ✗ | ✗ | 60.0 | 89.1 |
| ✗ | ✓ | ✗ | 40.8 | 72.7 |
| ✗ | ✗ | ✓ | 41.5 | 73.5 |
| ✓ | ✗ | ✓ | 62.1 | 89.8 |
| ✓ | ✓ | ✗ | 60.8 | 89.8 |
| ✓ | ✓ | ✓ | 63.2 | 89.9 |

Image Prior contributes the most (+19.5% Top-1) and is the key driver of performance improvement. EEG Prior and SSP each contribute approximately 1% gain individually; the combination of all three achieves optimal performance.

### Supplementary Experiments
- **Number of augmentation views**: Performance increases from 1 to 4 image augmentation views (50.9%→62.1%) and degrades beyond 5 (58.5%); 4 is the optimal trade-off.
- **Projector design**: 512-dimensional linear projection is optimal; MLP and higher-dimensional projections lead to overfitting.
- **Normalization strategy**: Asymmetric normalization (image only) 63.2%; symmetric normalization 46.4%; no normalization 54.4%; reverse asymmetric (EEG only) 38.6%.
- **Batch size**: 1024 is optimal (63.2%), outperforming batch size 32 by +8.6%; 2048 slightly decreases to 62.2%.
- **Temperature parameter**: τ = 0.5 is optimal (63.6%); NeuroBridge is more robust to temperature variation than standard contrastive learning.
- **THINGS-MEG validation**: Also achieves state of the art on MEG data — intra-subject 32.2% Top-1 vs. UBP 26.7%; inter-subject 3.4% vs. UBP 2.2%.
- **Encoder generalizability**: Consistent gains are observed across all combinations of image encoders (RN50 to ViT-bigG-14) and EEG encoders (EEGNet/TSConv/ATM/EEGProject).

## Highlights & Insights

- **Bio-inspired asymmetric augmentation**: The design directly simulates human perceptual variability — multi-view image augmentation mimics different attentional foci, while light EEG augmentation preserves temporal dynamics — with clear grounding in cognitive science.
- **Bidirectional alignment breakthrough**: SSP breaks the convention of forcibly aligning EEG to a fixed CLIP space, learning a new shared space with greater flexibility.
- **Asymmetric normalization insight**: Normalizing only image features allows EEG magnitude to encode confidence — a simple yet critical design choice (+16.8% vs. symmetric normalization).
- **Large SOTA margin**: Top-1 improves by 12.3% on the competitive THINGS-EEG benchmark, with improvements over the previous SOTA observed across all 10 subjects.
- **Encoder generalizability**: The framework is compatible with different image/EEG encoder combinations and consistently yields improvements.
- **Open-source code**: Full code is publicly available for reproduction.

## Limitations & Future Work

- **Hand-crafted augmentations**: The augmentation strategies in CPA (Gaussian blur, low resolution, etc.) are manually selected and may not fully capture cognitive variability; adaptive or learnable augmentation is a direction for future work.
- **Dependence on pretrained visual encoders**: The frozen CLIP encoder may introduce visual-language biases; the semantics reflected by EEG may differ fundamentally from language-driven semantics.
- **Limited to zero-shot retrieval**: The method has not been evaluated on more challenging decoding tasks such as image generation or reconstruction.
- **Limited data scale**: Validation is restricted to THINGS-EEG (1,654 concepts) and THINGS-MEG; performance on larger-scale datasets remains unknown.
- **Limited contribution of EEG augmentation**: EEG Prior improves Top-1 by only 0.3%, indicating substantial room for exploration in EEG-side augmentation strategies.
- **Lack of theoretical explanation for asymmetric normalization**: The paper offers only speculative justification for why allowing EEG magnitude to freely encode confidence is effective, without formal theoretical analysis.

## Related Work & Insights

- **UBP (Wu et al. 2025)**: The previous SOTA; introduces blur priors to simulate early visual perception, with augmentation applied only on the visual side. NeuroBridge surpasses it on all subjects by an average of +12.3% Top-1.
- **Neural-MCRL (Li et al. 2024)**: Introduces intra-modal semantic completion for EEG, augmenting only the EEG side. NeuroBridge integrates both modalities, achieving +30.8% Top-1.
- **VE-SDN (Chen et al. 2024)**: Visually enhanced decoding network; 37.2% → 63.2%, +26.0% Top-1.
- **CognitionCapturer (Zhang et al. 2025)**: Cognitive capturing method; 33.3% → 63.2%, +29.9% Top-1.
- **NICE (Song et al. 2024)**: TSConv encoder with simple contrastive learning baseline; 16.1% → 63.2%.
- **CLIP/ALIGN/BLIP, etc.**: Large-scale visual-language pretraining succeeds by leveraging massive paired data, which is unavailable at comparable scale for EEG-image pairs, motivating stronger architectural priors (CPA + SSP).

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of asymmetric augmentation and bidirectional SSP is innovative, though each individual module is relatively intuitive on its own.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 10 subjects, multi-encoder validation, detailed ablations, THINGS-MEG generalization, and hyperparameter sensitivity analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, coherent bio-inspired narrative, and well-designed ablation experiments.
- Value: ⭐⭐⭐⭐ — Significant advancement in EEG visual decoding, though the hand-crafted augmentation strategy limits methodological depth.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Self-Supervised Inductive Logic Programming](self-supervised_inductive_logic_programming.md)
- [\[AAAI 2026\] Let the Void Be Void: Robust Open-Set Semi-Supervised Learning via Selective Non-Alignment](let_the_void_be_void_robust_open-set_semi-supervised_learning_via_selective_non-.md)
- [\[CVPR 2026\] Suppressing Non-Semantic Noise in Masked Image Modeling Representations](../../CVPR2026/self_supervised/suppressing_non-semantic_noise_in_masked_image_modeling_representations.md)
- [\[CVPR 2026\] Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](../../CVPR2026/self_supervised/text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)
- [\[AAAI 2026\] Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space](towards_llm-empowered_knowledge_tracing_via_llm-student_hierarchical_behavior_al.md)

</div>

<!-- RELATED:END -->
