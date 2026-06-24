---
title: >-
  [Paper Note] Boltzmann Attention Sampling for Image Analysis with Small Objects
description: >-
  [CVPR 2025][Medical Imaging][Boltzmann Sampling] Proposes BoltzFormer, a novel transformer decoder architecture that dynamically samples sparse attention regions using a Boltzmann distribution to focus on small objects. Combining an annealing temperature schedule (exploration in early layers, exploitation in later layers) and the PiGMA multi-query aggregation module, it achieves a 3-12% improvement in Dice score compared to SOTA on small object segmentation (where objects occ…
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Boltzmann Sampling"
  - "Sparse Attention"
  - "Small Object Detection and Segmentation"
  - "Annealing Temperature Schedule"
  - "Text-Prompted Segmentation"
date: 2026-05-08
content_hash: d33779a44c319de6
---

# Boltzmann Attention Sampling for Image Analysis with Small Objects

**Conference**: CVPR 2025  
**arXiv**: [2503.02841](https://arxiv.org/abs/2503.02841)  
**Code**: [https://aka.ms/boltzformer](https://aka.ms/boltzformer)  
**Area**: Medical Images / Small Object Segmentation  
**Keywords**: Boltzmann Sampling, Sparse Attention, Small Object Detection and Segmentation, Annealing Temperature Schedule, Text-Prompted Segmentation

## TL;DR

Proposes BoltzFormer, a novel transformer decoder architecture that dynamically samples sparse attention regions using a Boltzmann distribution to focus on small objects. Combining an annealing temperature schedule (exploration in early layers, exploitation in later layers) and the PiGMA multi-query aggregation module, it achieves a 3-12% improvement in Dice score compared to SOTA on small object segmentation (where objects occupy <0.1% of the image area), while reducing attention computation by an order of magnitude.

## Background & Motivation

**Background**: General-purpose segmentation models like SAM/SAM2/SEEM have already achieved segmentation via prompts such as text, points, or boxes. In the medical field, models such as BiomedParse further support end-to-end detection and segmentation with text prompts.

**Limitations of Prior Work**: Small objects (e.g., lung nodules, tumor lesions) typically occupy $<0.1\%$ of the image area. Over $99\%$ of the computation in standard transformer global attention is wasted on irrelevant regions, which is both inefficient and introduces disturbing noise. Existing sparse attention methods (such as the fixed-threshold mask attention in Mask2Former) use rigid rules, which are unsuitable for small objects with uncertain locations.

**Key Challenge**: The locations of small objects are unknown in advance (especially when only text prompts are available), yet attention computation must cover the target region to detect them. How can attention be focused efficiently without knowing the target's location beforehand?

**Core Idea**: Analogy to reinforcement learning: the selection of attention regions is modeled as a Boltzmann sampling strategy, featuring high-temperature wide exploration in the early layers, and low-temperature precise exploitation in the later layers.

## Method

### Overall Architecture

Image encoder extracts multi-scale visual features + semantic map $\rightarrow$ Text encoder extracts text embeddings $\rightarrow$ $m$ learnable latent queries are first initialized via self-attention with text $\rightarrow$ $L$ layers of BoltzFormer blocks (each layer: Boltzmann sampling $\rightarrow$ sparse cross-attention $\rightarrow$ self-attention among queries + text) $\rightarrow$ PiGMA aggregates mask predictions from $m$ queries to produce the final output.

### Key Designs

1. **Boltzmann Attention Sampling**:

    - **Function**: Generates a spatial probability distribution for each query in each layer, sampling sparse attention regions from it.
    - **Mechanism**: The query $q_\ell^{(i)}$ undergoes an MLP transformation and then takes a dot product with the semantic map to obtain the pixel confidence score $U_{xy}$. This is normalized using a Boltzmann distribution: $p_{xy}(q_\ell^{(i)}) = \frac{\exp(U_{xy}/\tau_\ell)}{\int \exp(U_{x'y'}/\tau_\ell)}$. Subsequently, $N$ patches are sampled from this distribution to form the attention set $\mathcal{A}_\ell^{(i)}$, and the query performs cross-attention only within these sampled regions.
    - **Annealing Temperature Schedule**: $\tau_\ell = \tau_0 / (1 + \ell)$, where layer 0 has the highest temperature (most dispersed sampling/exploration), and the temperature cools down layer-by-layer (sampling becomes increasingly concentrated/exploitation).
    - **Design Motivation**: Wide exploration is needed in the early stages when the target's location is uncertain, while fine-grained feature extraction is required in the later stages once the region is locked. This is a direct analogy to the exploration-exploitation trade-off in reinforcement learning.

2. **Multi-Query Ensemble**:

    - **Function**: Uses $m$ queries to independently sample and update, sharing information via self-attention.
    - **Mechanism**: After Boltzmann sampling in each layer, all queries perform self-attention exchange with the text. Even if a query fails to hit the target initially, it can obtain information from other queries that succeeded.
    - **Effect**: $m=10$ is sufficient (showing a significant improvement over $m=1$), and $m>10$ yields no significant gain.

3. **PiGMA Aggregation Module**:

    - **Function**: Aggregates mask predictions from $m$ queries into a final high-resolution mask.
    - **Mechanism**: Two parallel paths: (1) Query Ensemble Prediction: averages $m$ masks; (2) Pixel Grounded Correction: a two-layer convolutional network upsamples the low-resolution prediction and refines the details using the original image pixels.
    - **Design Motivation**: The randomness of Boltzmann sampling might lead to unstable predictions for a single query; the ensemble and pixel-level correction improve robustness.

### Loss & Training

Supervised using Dice loss + BCE loss. Training data comes from $7$ datasets in total, including Medical Segmentation Decathlon, LIDC-IDRI, and AMOS22. Sampling only needs to cover $10\%$ of visual tokens (reducing computation by an order of magnitude compared to full attention).

## Key Experimental Results

### Main Results: Average Dice Scores on 7 Medical Segmentation Benchmarks

| Method | Average | LIDC | AMOS-CT | MSD-Lung | MSD-Panc |
|------|------|------|---------|----------|----------|
| SAM+Hiera-S (text) | 67.0 | 67.1 | 88.4 | 61.6 | 55.1 |
| SAM2+Hiera-S (text) | 65.6 | 65.4 | 88.2 | 59.8 | 52.8 |
| SEEM+Hiera-S (text) | 71.5 | 72.1 | 91.1 | 65.9 | 61.4 |
| BiomedParse (Pre-trained) | 73.0 | 73.8 | 91.9 | 66.1 | 60.2 |
| nnU-Net (35 experts) | 67.3 | 64.8 | 85.0 | 60.2 | 52.4 |
| **BoltzFormer+Hiera-S** | **73.8** | 73.3 | 91.3 | **70.4** | **63.7** |
| **BoltzFormer+FocalL** | **75.2** | **75.4** | **92.7** | 70.2 | **64.0** |

### Ablation Study: Small Objects vs. Large Objects

| Method | Small Objects (<1%) Dice | Large Objects (≥1%) Dice |
|------|:-:|:-:|
| SAM | 64.5 | 82.3 |
| SAM2 | 62.1 | 82.3 |
| SEEM | 68.9 | 87.1 |
| **BoltzFormer** | **71.4** | **87.5** |

### Key Findings

- **Largest Gain on Small Objects**: BoltzFormer vs. SEEM shows a $+2.5\%$ improvement on small objects ($71.4$ vs. $68.9$), whereas on large objects the improvement is only $+0.4\%$ ($87.5$ vs. $87.1$), demonstrating that the enhancement primarily comes from small objects.
- Only $10\%$ of attention tokens are needed to achieve optimal performance ($5\%$ also achieves $72.9$), reducing attention computational cost by an order of magnitude.
- An initial temperature of $\tau_0=1$ is optimal (balancing exploration and exploitation); temperatures that are too high ($2.0$) lead to excessive exploration and performance degradation.
- Text-conditional prior vs. unconditional: $+1.4\%$ Dice ($73.7$ vs. $72.3$), showing that text semantics assist query initialization to target the correct regions.
- Outperforms $35$ nnU-Net expert models ($75.2$ vs. $67.3$), handling all tasks with a single model.
- The complete failure rate is only $1.4\%$ (primarily extremely small objects of a few pixels or featuring low contrast).

## Highlights & Insights

- **Elegance of RL Analogy**: Analogizing the selection of attention regions to policy optimization (state = query, action = sampled region, policy = Boltzmann distribution) is highly intuitive. The annealing temperature schedule naturally achieves the exploration-exploitation trade-off.
- **Modular Design**: The Boltzmann sampling module can be plugged into any existing transformer decoder without relying on a specific backbone.
- **10% is Enough**: Achieving or even surpassing full-attention performance using only $10\%$ of visual tokens is highly valuable for large images (such as high-resolution medical imaging).
- **Intuitive Visualization**: The visualization in Fig. 4 showing the sampled regions in intermediate layers gradually converging from being scattered across the whole image to focusing on the target region is highly intuitive; even if the target is completely missed prior to layer 5, it can be rapidly corrected.

## Limitations & Future Work

- Verified only on 2D medical images, without extension to 3D volumetric data or natural images.
- Random sampling introduces inference uncertainty (the same input might lead to different sampling paths). Although multi-query ensembling mitigates this, it is not completely eliminated.
- Extremely small objects (a few pixels) remain unresolved—this is fundamentally an issue of insufficient information.
- The annealing schedule is fixed to $\tau_0/(1+\ell)$, and adaptive or learnable temperature scheduling has not been explored.

## Related Work & Insights

- **vs. Mask2Former**: Mask2Former uses predicted masks from upper layers for hard-thresholded attention, but the predictions are inconsistent and offer little benefit for small objects. BoltzFormer's probabilistic sampling is more flexible.
- **vs. MP-Former**: MP-Former uses GT masks with noise during training, but suffers from a severe discrepancy between training and inference distributions. BoltzFormer's sampling strategy remains consistent during training and inference.
- **vs. Deformable DETR**: Performs deformable convolution-style sparsification around reference points, which is more of a local operation. BoltzFormer samples globally across the entire image, enabling it to discover distant targets.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Introducing Boltzmann sampling into transformer attention is pioneering, the RL analogy is elegant, and the annealing schedule design is ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Overwhelmingly thorough with 7 datasets, 3 types of baselines, and 6 ablation studies (sampling type, temperature, sample size, query count, text conditions, and PiGMA).
- Writing Quality: ⭐⭐⭐⭐⭐ Clear diagrams (especially Fig. 1/2/4), rigorous methodology descriptions, and intuitive visualization results.
- Value: ⭐⭐⭐⭐⭐ Addresses a critical pain point in medical imaging—small object segmentation. The modular design is easy for the community to adopt.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sequential Attention-based Sampling for Histopathological Analysis](../../NeurIPS2025/medical_imaging/sequential_attention-based_sampling_for_histopathological_analysis.md)
- [\[CVPR 2025\] Interactive Medical Image Analysis with Concept-based Similarity Reasoning](interactive_medical_image_analysis_with_concept-based_similarity_reasoning.md)
- [\[CVPR 2025\] Multi-modal Vision Pre-training for Medical Image Analysis (BrainMVP)](multi-modal_vision_pre-training_for_medical_image_analysis.md)
- [\[CVPR 2025\] EquivAnIA: A Spectral Method for Rotation-Equivariant Anisotropic Image Analysis](equivania_a_spectral_method_for_rotation-equivariant_anisotropic_image_analysis.md)
- [\[CVPR 2025\] Unmasking Biases and Reliability Concerns in Convolutional Neural Networks Analysis of Cancer Pathology Images](unmasking_biases_and_reliability_concerns_in_convolutional_neural_networks_analy.md)

</div>

<!-- RELATED:END -->
