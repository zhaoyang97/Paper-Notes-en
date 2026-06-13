---
title: >-
  [Paper Note] Manifolds and Modules: How Function Develops in a Neural Foundation Model
description: >-
  [NeurIPS 2025][Self-Supervised Learning][Neural Foundation Models] This work opens the "black box" of a state-of-the-art neural activity foundation model (FNN) from a computational neuroscience perspective. By constructi…
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "Neural Foundation Models"
  - "Encoding Manifolds"
  - "Decoding Manifolds"
  - "Interpretability"
  - "Mouse Visual System"
date: 2026-05-08
content_hash: 170a1a5b019cbac6
---

# Manifolds and Modules: How Function Develops in a Neural Foundation Model

**Conference**: NeurIPS 2025
**arXiv**: [2512.07869](https://arxiv.org/abs/2512.07869)  
**Code**: [GitHub](https://github.com/cajal/fnn) (FNN Model) / [GitHub](https://github.com/dyballa/NeuralEncodingManifolds) (Analysis Tools)  
**Area**: Computational Neuroscience / Self-Supervised Learning
**Keywords**: Neural Foundation Models, Encoding Manifolds, Decoding Manifolds, Interpretability, Mouse Visual System

## TL;DR

This work opens the "black box" of a state-of-the-art neural activity foundation model (FNN) from a computational neuroscience perspective. By constructing decoding and encoding manifolds, the study reveals that each processing module (encoder, recurrent, readout) exhibits qualitatively distinct representational structures, and identifies critical discrepancies between the model and the biological visual system.

## Background & Motivation

**Background**: Neural foundation models (e.g., FNN) excel at fitting biological visual systems, yet their black-box nature limits mechanistic understanding of brain function.

**Limitations of Prior Work**: Existing evaluations primarily focus on unit-level output prediction accuracy, without examining whether internal representations correspond to those in biological systems.

**Key Challenge**: Even when a model achieves superior performance in predicting neural activity, its internal computational mechanisms may fundamentally differ from biological systems — high performance does not imply biological relevance.

**Goal**: To systematically analyze the internal representations at each processing stage of FNN and assess their similarity to the biological visual system.

**Key Insight**: Characterizing each artificial neuron's temporal response properties "neuron by neuron" — akin to an electrophysiologist — to construct decoding and encoding manifolds.

**Core Idea**: Jointly applying three complementary analysis techniques (decoding manifolds, encoding manifolds, decoding trajectories) to comprehensively evaluate the biological plausibility of a foundation model.

## Method

### Overall Architecture

Stimuli commonly used in mouse visual system studies (drifting gratings, optic flow, etc.; 88 unique sequences) are presented to FNN. Responses of 2,000 neurons per layer are extracted, and three representational analyses are constructed:
1. **Decoding Manifolds**: Trial embeddings in neural activity space
2. **Encoding Manifolds**: Neuron embeddings in stimulus–response space
3. **Decoding Trajectories**: Temporal evolution of neural activity for each stimulus

### Key Designs

1. **Decoding Manifold Construction**:

    - **Function**: Embeds trials in the coordinate space of neural activity, with each point representing a single stimulus trial
    - **Mechanism**: PCA is applied to stimulus- and time-averaged activity to generate 48 points (6 stimulus types × 8 directions)
    - **Significance**: Trials of the same stimulus should cluster together, reflecting the "readability" of brain states

2. **Encoding Manifold Construction**:

    - **Function**: Characterizes the topological relationships among neurons within a stimulus–response framework
    - **Mechanism**: A three-step pipeline: (1) non-negative tensor decomposition of the $N \times S \times T$ tensor to obtain neural factors; (2) construction of a weighted graph in neural encoding space using the IAN algorithm; (3) dimensionality reduction via diffusion maps to obtain the manifold
    - **Significance**: Reveals the global organizational topology of functionally similar neurons

3. **Decoding Trajectory Analysis**:

    - **Function**: Tracks the evolution of neural activity at each time step
    - **Quantitative Metrics**: "Tubularity" — decomposed into Tightness and Crossings
    - **Biological Comparison**: Evaluates whether the temporal dynamics of the artificial network resemble those of mouse retina/V1

### Classification Accuracy Evaluation

Leave-one-out 3-NN and logistic regression classifiers are applied to layer-wise activations to assess stimulus classification performance.

## Key Experimental Results

### Main Results

Stimulus classification accuracy per layer (Logistic Regression / 3-NN):

| Layer | Enc1 | Enc2 | Enc4 | Enc8 | Rec | RecOut | Readout | Out |
|-------|------|------|------|------|-----|--------|---------|-----|
| LR | 0.59 | 0.62 | 0.66 | 0.74 | 0.89 | 0.90 | 0.88 | 0.77 |
| 3-NN | 0.41 | 0.66 | 0.58 | 0.61 | 0.73 | 0.64 | 0.63 | 0.67 |

**The recurrent module achieves the highest classification accuracy** (0.89/0.90), substantially outperforming all encoder layers.

### Ablation Study (Representational Characteristics by Module)

| Module | Encoding Manifold | Decoding Trajectories | Biological Comparison |
|--------|------------------|-----------------------|----------------------|
| Encoder (Enc) | Clusters by feature map depth; non-selective "intensity arm" (β) present | Periodic stimuli yield ring-like structure; lacks stimulus-dependent temporal patterns | ❌ Large divergence from retina |
| Recurrent (Rec) | Diverse selectivity and temporal responses across regions | Stimulus-dependent trajectory bundles emerge; classification accuracy jumps | ⚠️ Similarities to V1, but high entanglement |
| Readout | Highly fragmented; each cluster originates from a single feature map | Within-map invariance, across-map diversity | ❌ Biological systems show greater within-type variability |

### Key Findings

- **The recurrent module is the critical inflection point**: Stimulus representations undergo a qualitative shift here — temporally distinct stimulus patterns are "pushed apart," and classification accuracy jumps from 0.74 to 0.89
- **Encoder exhibits edge-padding artifacts**: Padding artifacts at feature map borders produce non-selective "intensity arms" that distort representations
- **The readout module is "biologically implausible"**: It fits neural data via linear combinations of numerous self-similar feature maps rather than through a biologically credible mechanism
- Tubularity analysis quantifies that trajectory "Crossings" in FNN layers are significantly lower than in biological data ($p < 0.005$)
- The output layer achieves continuous representations through linear combinations — notably, the primary fitting of neural activity occurs within the readout module rather than across the entire network

## Highlights & Insights

- **First joint application of three analysis tools**: The combination of decoding manifolds, encoding manifolds, and decoding trajectories provides a multi-dimensional comparison that surpasses conventional pairwise or average-based methods such as RSA
- **A cautionary note for foundation models**: Even with excellent predictive performance, internal model mechanisms may deviate substantially from biology — high performance does not imply biological plausibility
- **Recurrent module ≈ general-purpose representation learning**: Its functional role is analogous to uniformity and alignment in self-supervised foundation models, suggesting broader interpretive relevance
- **The readout module as an "adapter"**: Concentrating the fitting of neural activity in the readout module implies that this architecture separates representation learning from data fitting

## Limitations & Future Work

- Only one foundation model (FNN) is analyzed; comparisons with other video-based foundation models are absent
- A limited stimulus set is used (though it elicits activations similar to those from training stimuli)
- Analysis is restricted to a single session/scan; cross-animal and cross-session validation is limited
- Encoding manifold analysis depends on sampling choices (2,000 neurons, 40 feature maps), and different sampling strategies may affect results
- The work does not explore how these findings could inform improvements to model architecture

## Related Work & Insights

- FNN employs a Gaussian readout + DenseNet encoder + ConvLSTM recurrent module, representing the current SOTA architecture for neural foundation models
- The encoding manifold methodology originates from Dyballa et al., with diffusion maps preserving the intrinsic geometry of the data
- Compared to conventional RSA (Representational Similarity Analysis), encoding/decoding manifolds provide richer global topological information
- **Future directions**: Upcoming neural foundation models may benefit from connecting a lightweight retina-like encoder directly to a recurrent module, while constraining feature dimensionality to match the diversity of biological cell types

## Rating

- Novelty: ⭐⭐⭐⭐ — The unique combination of three analysis tools offers a fresh perspective; however, the primary contribution lies in analysis rather than methodological innovation
- Experimental Thoroughness: ⭐⭐⭐ — Only one model is examined and the stimulus set is limited, though the depth of analysis is commendable
- Writing Quality: ⭐⭐⭐⭐ — The interdisciplinary writing challenge is handled well; figures are information-dense
- Value: ⭐⭐⭐⭐ — Makes an important contribution to understanding the biological relevance of neural foundation models, with meaningful implications for future architectural design

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] How 'Neural' is a Neural Foundation Model?](../../ICML2026/self_supervised/how_neural_is_a_neural_foundation_model.md)
- [\[NeurIPS 2025\] TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)
- [\[NeurIPS 2025\] Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings](hybrid_autoencoders_for_tabular_data_leveraging_model-based_augmentation_in_low-.md)
- [\[NeurIPS 2025\] Implicit Modeling for Transferability Estimation of Vision Foundation Models](implicit_modeling_for_transferability_estimation_of_vision_foundation_models.md)
- [\[NeurIPS 2025\] Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models](mitra_mixed_synthetic_priors_for_enhancing_tabular_foundation_models.md)

</div>

<!-- RELATED:END -->
