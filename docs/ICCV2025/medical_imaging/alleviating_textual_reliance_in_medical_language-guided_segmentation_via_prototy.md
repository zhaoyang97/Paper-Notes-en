---
title: >-
  [Paper Note] Alleviating Textual Reliance in Medical Language-guided Segmentation via Prototype-driven Semantic Approximation
description: >-
  [ICCV 2025][Medical Imaging][Medical image segmentation] This paper proposes ProLearn, a framework that introduces a Prototype-driven Semantic Approximation (PSA) module to fundamentally alleviate textual reliance in med…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "Medical image segmentation"
  - "language-guided segmentation"
  - "prototype learning"
  - "textual reliance"
  - "semantic approximation"
date: 2026-05-08
content_hash: c35d85c4f126d54c
---

# Alleviating Textual Reliance in Medical Language-guided Segmentation via Prototype-driven Semantic Approximation

**Conference**: ICCV 2025
**arXiv**: [2507.11055](https://arxiv.org/abs/2507.11055)  
**Code**: [https://github.com/ShuchangYe-bib/ProLearn](https://github.com/ShuchangYe-bib/ProLearn)  
**Area**: Medical Image Segmentation / Multimodal Learning
**Keywords**: Medical image segmentation, language-guided segmentation, prototype learning, textual reliance, semantic approximation

## TL;DR
This paper proposes ProLearn, a framework that introduces a Prototype-driven Semantic Approximation (PSA) module to fundamentally alleviate textual reliance in medical language-guided segmentation. The prototype space is initialized from a small number of image-text pairs; thereafter, both training and inference require no text input. ProLearn maintains strong performance under 1% text availability (QaTa-COV19 Dice = 0.857), with parameters 1000× fewer than LLM-based solutions and inference speed 100× faster.

## Background & Motivation

Deep learning has driven progress in medical image segmentation, with U-Net and its variants (U-Net++, Attention U-Net, TransUNet, etc.) widely adopted. Recent multimodal segmentation methods that leverage clinical reports as auxiliary guidance (e.g., LViT, GuideSeg) have demonstrated superiority over unimodal approaches, as text provides explicit semantic descriptions of lesions.

However, language-guided segmentation suffers from an inherent **textual reliance** problem: (1) **Training stage** — the majority of medical segmentation datasets lack paired reports, leaving large amounts of image-only data unused; (2) **Inference stage** — the requirement for paired reports at inference restricts these methods to retrospective analysis, whereas in most clinical settings (e.g., preoperative planning, real-time surgical guidance, diagnostic decision-making), segmentation must be performed prior to report generation.

SGSeg, a prior work, attempts to bridge the text gap at inference by using LLMs (GPT-2, Llama3) to synthesize reports from images. However, this introduces billion-parameter LLMs, resulting in bloated models and slow inference that are unsuitable for edge devices and real-time applications, while the textual reliance during training remains unresolved.

The key insight of this paper is that the essential guidance in language-guided segmentation is not the complete clinical report (which is often verbose and contains irrelevant information), but rather the segmentation-relevant semantic features embedded within it. Furthermore, the semantic space of medical reports is inherently constrained — clinical reports follow standardized medical terminology with a relatively closed vocabulary. Based on this insight, a limited set of prototypes can discretely represent segmentation-relevant semantics.

## Method

### Overall Architecture
ProLearn = PSA module + Language-guided U-Net. The PSA module initializes a prototype space from a small number of image-text pairs in a one-time process. During both training and inference, a query-response mechanism retrieves approximate semantic guidance from the prototype space using image features alone, requiring no text input. The prototype space is dynamically updated throughout training.

### Key Designs

1. **PSA Initialization (one-time process)**:

    - **Function**: Constructs a queryable prototype space from $K$ available image-text paired samples.
    - **Mechanism**:
        - **Proxy label extraction**: A BioMedCLIP encoder extracts image features $e_i^I$ and text features $e_i^T$. Cross-attention weights from a separately trained Language-guided U-Net are used to filter high-attention tokens ($\alpha_j > \tau$), yielding shortened segmentation-relevant sentences $T_i^{\text{selected}}$, which are encoded into semantic features $e_i^{\text{sem}}$.
        - **Hierarchical clustering**: HDBSCAN clusters $e_i^{\text{sem}}$ into $N$ proxy labels, each representing a distinct segmentation-relevant semantic.
        - **Prototype space construction**: Within each text proxy label cluster $\mathcal{C}_i$, K-means further clusters image features $e_k^I$ into $M$ sub-clusters. The sample closest to each sub-cluster centroid (rather than the centroid itself, to reduce the influence of outliers) is selected as a prototype pair $(q_{ij}, r_{ij})$, forming a query space $\mathcal{S}^Q$ (image prototypes) and a response space $\mathcal{S}^R$ (text prototypes), each of dimension $N \times M \times D$.
    - **Design Motivation**: The two-level clustering strategy — coarse grouping by textual semantics followed by fine grouping by visual features — ensures that the prototype space is both compact and expressive of visual diversity beyond what text alone can capture.

2. **PSA Query-Response Mechanism**:

    - **Function**: Provides approximate semantic guidance for arbitrary images during training and inference.
    - **Mechanism**:
        - **Query**: Cosine similarity $s_{ij} = s(q^*, q_{ij})$ is computed between the encoded image feature $q^* = f_{\text{enc}}^I(I^*)$ and all prototypes in the query space; the top-$k$ most similar prototypes $Q^*$ are selected.
        - **Response**: The corresponding text response prototypes $R^*$ are retrieved and aggregated via softmax-normalized similarity weighting:
       $$r^* = \sum_{r_i \in R^*} w_i r_i, \quad w_i = \frac{\exp(s(q^*, q_i))}{\sum_{q_j \in Q^*} \exp(s(q^*, q_j))}$$
        - The response $r^*$ is fed into the U-Net decoder to guide segmentation.
    - **Design Motivation**: Prototype lookup operates in $\mathcal{O}(1)$ time (querying a fixed-size prototype space), compared to $\mathcal{O}(n)$ for autoregressive LLM generation. The module requires only 1M parameters versus 1.5B (GPT-2) or 7B (Llama3).

3. **Language-guided U-Net**:

    - **Function**: Performs image segmentation with PSA responses as semantic guidance.
    - **Mechanism**: A standard U-Net encoder extracts image features for PSA querying; the decoder receives the approximate semantic features from the PSA response to guide mask prediction.
    - **Design Motivation**: The existing language-guided segmentation architecture is reused, with the PSA module serving as a plug-and-play replacement for the text encoder.

### Loss & Training
- Standard segmentation losses (Dice Loss + CE Loss).
- The prototype space is dynamically updated during training rather than being fixed.
- Both image-text paired samples and image-only samples can be used jointly during training.

## Key Experimental Results

### Main Results: Language-guided Segmentation under Varying Text Availability

| Dataset | Model | 50% Text Dice | 10% Text Dice | 1% Text Dice |
|--------|------|------------|------------|-----------|
| QaTa-COV19 | LViT | 0.842 | 0.800 | 0.701 |
| QaTa-COV19 | GuideSeg | 0.863 | 0.840 | 0.733 |
| QaTa-COV19 | SGSeg | 0.864 | 0.842 | 0.731 |
| QaTa-COV19 | **ProLearn** | **0.867** | **0.858** | **0.857** |
| MosMedData+ | SGSeg | 0.746 | 0.695 | 0.345 |
| MosMedData+ | **ProLearn** | **0.754** | **0.742** | **0.722** |
| Kvasir-SEG | GuideSeg | 0.885 | 0.775 | 0.562 |
| Kvasir-SEG | **ProLearn** | **0.898** | **0.890** | **0.872** |

### Ablation Study / Efficiency Comparison

| Model | Parameters | Inference Time | Time Complexity |
|------|--------|---------|----------|
| ProLearn (PSA) | 1M | 4ms | $\mathcal{O}(1)$ |
| SGSeg (GPT-2) | 1.5B | 136ms | $\mathcal{O}(n)$ |
| SGSeg (Llama3) | 7B | 1.2s | $\mathcal{O}(n)$ |

### Key Findings
- At 1% text availability, ProLearn exhibits almost no performance degradation (QaTa Dice: 0.867→0.857), while GuideSeg drops sharply from 0.863 to 0.733 and SGSeg from 0.864 to 0.731.
- ProLearn with only 1% text even surpasses all unimodal methods trained with 100% image data (U-Net, U-Net++, Swin U-Net, etc.).
- ProLearn reduces parameters by 1000× and achieves 100–300× faster inference compared to LLM-based solutions, making it suitable for edge devices and real-time applications.
- Performance remains stable across a wide range of hyperparameter settings (number of candidates $k$, number of prototypes $M$).

## Highlights & Insights
- The core insight is highly inspiring: the semantic space of medical reports is inherently constrained (standardized terminology and a closed vocabulary), enabling discrete approximation via a finite prototype set without regenerating text for each sample.
- The PSA query-response mechanism provides an elegant cross-modal bridging solution: image query → selection of visually nearest prototypes → retrieval of corresponding textual semantics, enabling text-free inference.
- The two-level clustering design (HDBSCAN by textual semantics + K-means by visual features) ensures that the prototype space is both semantically coherent and visually diverse.
- The performance degradation curves (Figure 5) visually demonstrate the robustness advantage of ProLearn over competing methods as text availability decreases.

## Limitations & Future Work
- Prototype space initialization relies on the feature quality of BioMedCLIP; performance may degrade for medical domains not well covered by BioMedCLIP.
- Proxy label extraction requires a pre-trained Language-guided U-Net to obtain attention weights, increasing pipeline complexity.
- Online prototype space update strategies (e.g., incremental updates as new data arrives) are not explored.
- Validation is limited to COVID-19 lung and colorectal polyp scenarios; more complex multi-organ or multi-lesion settings remain to be investigated.

## Related Work & Insights
- **vs. SGSeg**: SGSeg uses LLMs to generate synthetic reports to bridge the text gap at inference, but still relies on text during training and introduces billion-parameter LLMs. ProLearn fundamentally decouples the dependency on text, with substantial advantages in parameter count and inference speed.
- **vs. LViT/GuideSeg**: These methods strictly require paired image-text inputs; performance degrades sharply when text is unavailable. ProLearn approximates semantics via prototypes, rendering performance nearly invariant to text availability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to address textual reliance in language-guided segmentation via prototype learning, with a deep and well-motivated insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, five text availability ratios, efficiency comparison, and hyperparameter analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is precise, the motivation chain is complete, and figures are clear.
- Value: ⭐⭐⭐⭐⭐ Offers significant clinical utility by resolving a core bottleneck hindering the deployment of language-guided segmentation in practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training](boosting_vision_semantic_density_with_anatomy_normality_modeling_for_medical_vis.md)
- [\[ICCV 2025\] Progressive Test Time Energy Adaptation for Medical Image Segmentation](progressive_test_time_energy_adaptation_for_medical_image_segmentation.md)
- [\[AAAI 2026\] Divide, Conquer and Unite: Hierarchical Style-Recalibrated Prototype Alignment for Federated Medical Segmentation](../../AAAI2026/medical_imaging/divide_conquer_and_unite_hierarchical_style-recalibrated_prototype_alignment_for.md)
- [\[NeurIPS 2025\] Magical: Medical Lay Language Generation via Semantic Invariance and Layperson-tailored Adaptation](../../NeurIPS2025/medical_imaging/magical_medical_lay_language_generation_via_semantic_invariance_and_layperson-ta.md)
- [\[CVPR 2026\] MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration](../../CVPR2026/medical_imaging/medkco_medical_vision-language_pretraining_via_knowledge-driven_cognitive_orches.md)

</div>

<!-- RELATED:END -->
