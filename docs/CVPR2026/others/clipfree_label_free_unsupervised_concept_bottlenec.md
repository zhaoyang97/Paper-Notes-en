---
title: >-
  [Paper Note] U-F²-CBM: CLIP-Free, Label Free, Unsupervised Concept Bottleneck Models
description: >-
  [CVPR 2026][concept bottleneck model] This paper proposes TextUnlock, a method that trains a lightweight MLP to project features from an arbitrary frozen visual classifier into the text embedding space—while preserving t…
tags:
  - "CVPR 2026"
  - "concept bottleneck model"
  - "CLIP-free"
  - "label-free"
  - "unsupervised"
  - "interpretability"
  - "TextUnlock"
date: 2026-05-08
content_hash: b733098ff17aa222
---

# U-F²-CBM: CLIP-Free, Label Free, Unsupervised Concept Bottleneck Models

**Conference**: CVPR 2026
**arXiv**: [2503.10981](https://arxiv.org/abs/2503.10981)  
**Code**: None (pseudocode provided in the paper)  
**Area**: Explainable AI / Concept Bottleneck Models
**Keywords**: concept bottleneck model, CLIP-free, label-free, unsupervised, interpretability, TextUnlock

## TL;DR
This paper proposes TextUnlock, a method that trains a lightweight MLP to project features from an arbitrary frozen visual classifier into the text embedding space—while preserving the original classifier's output distribution—requiring no CLIP, no annotations, and no linear probe training. Any legacy classifier can thereby be converted into an interpretable concept bottleneck model. Evaluated on 40+ architectures, the approach surpasses even supervised CLIP-based CBMs.

## Background & Motivation
Concept Bottleneck Models (CBMs) map dense features to human-interpretable concept activations, which are then linearly combined to predict class labels. Existing CBMs suffer from three key limitations: (1) they rely on CLIP to provide image–concept annotations, introducing CLIP's biases into legacy models; (2) they require manual annotations or image–text paired data; and (3) they require training a linear classifier to map concepts to classes. In practice, high-performing task-specific legacy models (e.g., DINO, BeiT) already exist, and reconstructing their reasoning via CLIP discards the original decision logic. The central question is: can any frozen classifier be converted into a CBM without CLIP and without annotations?

## Core Problem
How can an arbitrary visual classifier be transformed into an interpretable concept bottleneck model without relying on CLIP, without any image–concept or image–class annotations, and without altering the original classifier's inference process?

## Method

### Overall Architecture
The approach proceeds in two stages: (1) **TextUnlock**—an MLP is trained to map visual features into the text encoder space, using the original classifier's softmax distribution as the distillation target (not ground-truth labels); (2) **U-F²-CBM**—with the MLP frozen, a concept set is encoded by the text encoder to obtain a concept embedding matrix $C$; concept activations are derived via cosine similarity; and the concept-to-class classifier weights $W_{con}$ are derived unsupervisedly as $C \cdot U^T$ (concept–classname text similarity), without any supervised training.

### Key Designs
1. **TextUnlock distribution alignment**: The core loss is $\mathcal{L} = -\sum_i^K o_i \log \frac{e^{s_i}}{\sum_j e^{s_j}}$, where $o$ is the softmax output of the original classifier (not ground-truth labels) and $s$ is the cosine similarity between the MLP-projected features and text class-name embeddings. This is essentially knowledge distillation—not from a large model to a small one, but from the classifier's discrete distribution to its visual–linguistic correspondence distribution. Key properties: (a) no labels required; (b) the original classifier's inference is preserved (accuracy drop of only ~0.2%); (c) applicable to any architecture.

2. **Unsupervised concept-to-class mapping**: Conventional CBMs require training a linear probe $W_{con}$. This paper observes that concept embeddings $C$ and class-name embeddings $U$ reside in the same text space, enabling a direct text-to-text lookup: $W_{con} = C \cdot U^T$. The final CBM output is $S_{cn} = (\tilde{f} \cdot C^T) \cdot (C \cdot U^T) = \tilde{f} \cdot C^T C \cdot U^T$—i.e., the original classifier $\tilde{f} \cdot U^T$ scaled by the concept Gram matrix $C^TC$, which reduces to the original classifier when $C^TC = I$.

3. **On-the-fly CBM construction at inference**: The concept set can be swapped arbitrarily at inference time by simply re-encoding the concept set, with no retraining of any component. This flexibility is unavailable to CLIP-based methods.

### Loss & Training
- MLP: 3 layers ($n \to 2n \to 2n \to m$), LayerNorm + GELU + Dropout(0.5)
- Text encoder: MiniLM (all-MiniLM-L12-v1), $m=384$
- Concept set: 20K most common English words, filtered rigorously (class names, synonyms, hypernyms, etc. removed)
- Training: Adam lr=1e-4, cosine decay, single RTX 2080 Ti

## Key Experimental Results

| Method | Model | CBM Top-1↑ | Notes |
|---|---|---|---|
| LF-CBM | CLIP ResNet50 | 67.5 | Supervised + CLIP |
| DN-CBM | CLIP ResNet50 | 72.9 | Supervised + CLIP |
| CDM | CLIP ViT-B/16 | 79.3 | Supervised + CLIP |
| DCBM | CLIP ViT-L/14 | 77.9 | Supervised + CLIP |
| **U-F²-CBM** | ResNet50 | 78.1 | **Unsupervised + CLIP-free** |
| **U-F²-CBM** | EfficientNetv2-S | 83.0 | +5.1% over CLIP ViT-L/14 |
| **U-F²-CBM** | ConvNeXtV2-B@384 | **86.4** | SOTA |
| **U-F²-CBM** | BeiT-L/16 | 86.2 | |

- Accuracy loss after TextUnlock: average of only 0.2% across 40 models
- Zero-shot image captioning: CIDEr 17.9 / SPICE 6.9, surpassing ZeroCap and ConZIC
- Additional datasets: outperforms CLIP-based methods on Places365, EuroSAT, and DTD
- Concept intervention: removing class-relevant concepts causes ~20% accuracy drop, validating concept interpretability

### Ablation Study
- MLP ablations (mean/random/shuffled inputs): accuracy drops to ~0%, confirming the MLP learns a meaningful transformation
- MLP design: 2 layers + dim_factor=2 is optimal (75.80 vs. 72.48 for 1 layer)
- Text encoder choice: differences across Sentence-BERT variants are negligible (~0.05%)
- Text prompt robustness: gap between best and worst prompt is only 0.36%
- General concept set vs. LLM-generated concept set: general set outperforms by 2–3% (LLM-generated sets introduce spurious correlations)

## Highlights & Insights
- A "triple-free" CBM: CLIP-free + Label-free + Unsupervised—the first approach to simultaneously eliminate all three constraints in the CBM literature
- The distribution distillation insight is elegant: training the MLP on the original classifier's softmax distribution (rather than ground-truth labels) naturally preserves the inference logic
- Gram matrix perspective: a CBM is fundamentally the original classifier scaled by the concept Gram matrix—it reduces to the original classifier when the concept set is sufficiently complete
- Large-scale validation across 40+ architectures covering CNNs, Transformers, and hybrid designs
- On-the-fly concept set switching at inference—truly plug-and-play interpretability

## Limitations & Future Work
- Polysemy: class name "drake" may match the musician rather than the bird species, making performance dependent on concept set quality
- The MLP is trained using only class names—while intentionally avoiding information leakage, this may limit the completeness of the learned semantic space
- Concept activations are cosine similarities and lack calibration—high activation does not necessarily imply high semantic importance
- Downstream concept intervention still requires manual selection of intervention concepts, limiting automation

## Related Work & Insights
- **vs. LF-CBM / LaBo / CDM**: These supervised methods rely on CLIP for concept annotations and require linear probe training; U-F²-CBM is fully unsupervised and CLIP-free while achieving higher performance
- **vs. T2C**: T2C also maps features to the CLIP space but depends on CLIP supervision and discards the original classifier's distribution; TextUnlock preserves the original distribution
- **vs. DeVIL / LIMBER**: These methods require annotated image–text data to train autoregressive generators and alter the original model's inference process

The distribution distillation approach to visual–language spaces may generalize to other domains requiring interpretability (e.g., medical image classifiers). The concept Gram matrix perspective provides a unified mathematical framework for CBMs. The zero-shot captioning capability implies that any classifier can "verbalize" what it observes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The triple elimination of CLIP/labels/supervision represents a major breakthrough in CBM research; the Gram matrix perspective is elegant
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 40+ architectures, 4 datasets, comprehensive ablations, concept interventions, and zero-shot captioning
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, concise methodology, deep theoretical insights, and thorough appendix (16 sections)
- Value: ⭐⭐⭐⭐⭐ Liberates explainable AI from CLIP dependency; any legacy model can be plug-and-play converted to a CBM

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Stronger Normalization-Free Transformers](stronger_normalization-free_transformers.md)
- [\[AAAI 2026\] Cost-Free Neutrality for the River Method](../../AAAI2026/others/cost-free_neutrality_for_the_river_method.md)
- [\[AAAI 2026\] Radar-APLANC: Unsupervised Radar-based Heartbeat Sensing via Augmented Pseudo-Label and Noise Contrast](../../AAAI2026/others/radar-aplanc_unsupervised_radar-based_heartbeat_sensing_via_augmented_pseudo-lab.md)
- [\[ICLR 2026\] Bayesian Influence Functions for Hessian-Free Data Attribution](../../ICLR2026/others/bayesian_influence_functions_for_hessian-free_data_attribution.md)
- [\[ICLR 2026\] Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition](../../ICLR2026/others/training_deep_normalization-free_spiking_neural_networks_with_lateral_inhibition.md)

</div>

<!-- RELATED:END -->
