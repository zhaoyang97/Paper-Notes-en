---
title: >-
  [Paper Note] Data Distributional Properties as Inductive Bias for Systematic Generalization
description: >-
  [CVPR 2025][Multimodal VLM][Systematic Generalization] It is discovered that manipulating only the distributional properties of training data (diversity, burstiness, and latent intervention) can induce systematic generalization in multimodal masked language models. Specifically, increasing attribute diversity boosts out-of-distribution (OOD) shape prediction accuracy from 0.6% to 90%, requiring no modifications to the model architecture or training strategies.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Systematic Generalization"
  - "Data Distributional Properties"
  - "Inductive Bias"
  - "Disentangled Representation"
  - "Multimodal MLM"
date: 2026-05-08
content_hash: ba150a49e628b5fa
---

# Data Distributional Properties as Inductive Bias for Systematic Generalization

**Conference**: CVPR 2025  
**arXiv**: [2502.20499](https://arxiv.org/abs/2502.20499)  
**Code**: [https://github.com/fdelrio89/data-systematic](https://github.com/fdelrio89/data-systematic)  
**Area**: Multimodal VLM  
**Keywords**: Systematic Generalization, Data Distributional Properties, Inductive Bias, Disentangled Representation, Multimodal MLM

## TL;DR
It is discovered that manipulating only the distributional properties of training data (diversity, burstiness, and latent intervention) can induce systematic generalization in multimodal masked language models. Specifically, increasing attribute diversity boosts out-of-distribution (OOD) shape prediction accuracy from 0.6% to 90%, requiring no modifications to the model architecture or training strategies.

## Background & Motivation

**Background**: Systematic Generalization (SG) refers to the ability of a model to recombine learned concepts into novel scenarios—for example, recognizing a "red sphere" after encountering "red cubes" and "blue spheres" during training. This is a core capability of human cognition, yet deep learning models generally lack it.

**Limitations of Prior Work**: Existing efforts to improve SG mainly focus on model architecture design (e.g., disentangled VAEs, group-equivariant networks) and training strategies (contrastive learning, data augmentation), with little attention paid to the impact of the distributional properties of the training data itself.

**Key Challenge**: While models achieve nearly perfect in-distribution (ID) performance (99.6% shape accuracy), their out-of-distribution (OOD) performance—when encountering unseen attribute combinations during training—drops below random guess (0.6% vs. 33% random). This indicates that models learn severe spurious correlations (e.g., "specific color = specific shape").

**Goal**: To investigate which distributional properties of the training data can break spurious correlations and induce systematic generalization, and to understand the underlying mechanisms.

**Key Insight**: In CLEVR-like multimodal scenarios, three data distributional properties are systematically manipulated—diversity (cardinality of attribute values), burstiness (constraints on attribute values within a single sample), and latent intervention (random perturbation of a certain attribute during training)—to observe their impact on OOD generalization.

**Core Idea**: By increasing the diversity of attribute values in the training data, spurious correlations between attributes are broken, forcing the model to learn independent attribute encodings and thereby achieving zero-shot generalization of attribute recombinations.

## Method

### Overall Architecture
Experimental setup: CLEVR-like scenes containing 3-10 objects, each characterized by attributes of {shape, color, material, size}. Text queries describe the scene with some attributes masked, and the model predicts the masked attributes. Certain shape-color combinations are held out during training to serve as the OOD test set. The model is a simple Transformer encoder (256 dim, 4 layers, 4 heads) that simultaneously processes image patches and text tokens.

### Key Designs

1. **Diversity**:

    - **Function**: To increase the cardinality of the color attribute in the training data.
    - **Mechanism**: The RGB color space is uniformly partitioned into $n^3$ colors ($n \in \{2,3,4,5,6\}$), expanding the color set from 8 to 216. As the number of colors increases, the model can no longer predict shapes by memorizing the association of "color X = shape Y," forcing it to learn independent shape representations.
    - **Design Motivation**: Under 8 colors, the shape OOD accuracy is 0.6%, whereas under 216 colors, it leaps to 90.0%—an absolute improvement of 89.4 percentage points. Even when using only 25% of the training data, high diversity yields 81.0% accuracy, far exceeding the performance of the full dataset with low diversity (0.6%). This demonstrates that diversity is significantly more important than data volume.

2. **Burstiness**:

    - **Function**: To restrict the diversity of attribute values within a single sample.
    - **Mechanism**: With probability $p_{burst}$, the number of distinct colors in each image is restricted to at most 3. This disrupts the model's ability to predict a shape using other colors within the same sample—if an image consists entirely of red objects, the color provides no information about the shape.
    - **Design Motivation**: In the 64-color configuration, increasing burstiness from 0.0 to 1.0 improves the OOD accuracy from 48.5% to 63.3% (+14.8%), showing complementarity with diversity.

3. **Latent Intervention**:

    - **Function**: To disrupt spurious correlations by randomly perturbing irrelevant attributes during training.
    - **Mechanism**: Apply random hue jittering (ColorJitter) to the colors of all objects in the image, with intensity $\in \{0, 0.05, 0.1, 0.5\}$. This alters the color while preserving the shape, effectively serving as a causal intervention at the latent variable level.
    - **Design Motivation**: Under the 125-color configuration, a jitter intensity of 0.05 improves the OOD accuracy from 81.8% to 85.0% (+3.2%). The three approaches can be applied in combination.

### Loss & Training
Standard masked language modeling (MLM) loss with a masking probability of 0.15. The model is a simple Transformer (256 dim, 4 layers, 4 heads), optimized using Adam with lr=1e-4, batch size of 256, and trained for 1000 epochs.

## Key Experimental Results

### Main Results

| Number of Colors | Shape ID | Shape OOD | Change |
|--------|---------|---------|------|
| 8 | 99.6% | 0.6% | Baseline |
| 27 | 96.9% | 1.5% | - |
| 64 | 96.9% | 48.5% | +47.9 |
| 125 | 96.1% | 81.8% | +81.2 |
| 216 | 96.3% | **90.0%** | **+89.4** |

### Ablation Study

| Configuration | Shape OOD | Description |
|------|---------|------|
| 8-color full data | 0.6% | More in-distribution data is futile |
| 216-color 25% data | **81.0%** | Low volume with high diversity far outperforms high volume with low diversity |
| 64-color + Burstiness p=1.0 | 63.3% | Burstiness +14.8% |
| 64-color + Intervention j=0.5 | 63.8% | Latent intervention +15.3% |
| 8-color dim=32 | 0.0% | Decreasing capacity does not induce SG |
| 216-color dim=512 | 93.5% | High capacity + high diversity works better |

### Key Findings
- **Diversity is the dominant factor**: An absolute improvement of 89.4%, far exceeding burstiness (+14.8%) and latent intervention (+15%).
- **Data volume does not solve the SG problem**: Increasing in-distribution data even slightly decreases OOD accuracy (as the model memorizes spurious correlations more thoroughly).
- **Capacity bottleneck is not the mechanism**: Reducing the hidden dimension from 256 to 32 still yields 0% OOD accuracy. The improvement in SG is not driven by the model being forced to "compress" information.
- **NMI and parallelism are the underlying mechanisms**: Normalized Mutual Information (NMI) between attributes is negatively correlated with OOD accuracy (r=-0.79), while the parallelism (p-score) of attribute encodings in the representation space is positively correlated with OOD accuracy (r=0.73). Diversity promotes systematic generalization by reducing mutual information between attributes $\rightarrow$ facilitating parallel representations.
- **Cross-attribute generalization**: Enhancing the independence between color and shape simultaneously boosts the OOD accuracy of material (87.8 $\rightarrow$ 97.2%) and size (91.2 $\rightarrow$ 97.7%), showing that improvements in data distribution have a global effect.

## Highlights & Insights
- **"Achieving SG by modifying data rather than the model" is a profound discovery**: It suggests that many failures of SG may not stem from insufficient model capacity, but rather from spurious correlations residing in the training data distribution.
- **The causal chain of NMI $\rightarrow$ parallelism $\rightarrow$ SG** provides a novel mechanistic explanation for SG: models need to encode different attributes into parallel directions in the representation space (resembling the linear analogy relationships in word2vec). Diverse data naturally induces this structure.
- **Practical Takeaway**: When constructing multimodal training data, one should deliberately increase the range diversity of each attribute, rather than simply scaling up the volume of data.

## Limitations & Future Work
- Validated only on synthetic CLEVR-like data; attributes in real-world data are uncontrollable (making it hard to simply increase the "number of colors").
- The tested model is a minimal Transformer (256 dim, 4 layers); whether this generalizes to large-scale pretrained models remains unknown.
- Only the diversity of the color attribute was manipulated; the interaction effects of varying multiple attributes simultaneously are unexplored.
- Whether the conclusions derived from synthetic data can guide the construction of pretraining data for real VLMs requires substantial follow-up experimental validation.

## Related Work & Insights
- **vs. $\beta$-VAE / Disentanglement Methods**: These approaches promote disentangled representations by modifying model architectures and loss functions. This work demonstrates that similar effects can be achieved solely through the data distribution, without requiring additional regularization.
- **vs. Data Augmentation**: Traditional augmentations (flipping, cropping) do not alter the statistical relationships between attributes. In contrast, the "diversity" and "latent intervention" proposed in this work directly target the statistical structure of attribute distributions.
- **vs. Compositional Generalization (e.g., SCAN, COGS)**: These SG studies in the NLP domain mainly focus on structural improvements, whereas this work provides a complementary perspective from the data standpoint.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ This is the first work to systematically demonstrate that data distributional properties can serve as an inductive bias for SG, presenting a compelling absolute improvement of 89%.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ An exceptionally thorough ablation study (covering diversity, burstiness, intervention, capacity, and data volume), enriched by an in-depth mechanistic analysis of NMI and parallelism.
- Writing Quality: ⭐⭐⭐⭐⭐ Superb logical flow: phenomenon $\rightarrow$ manipulation $\rightarrow$ mechanistic explanation, with each step strongly backed by experimental evidence.
- Value: ⭐⭐⭐⭐ It makes significant theoretical contributions to the understanding of SG, although its transferability from synthetic data to the real world remains to be verified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] On the Out-of-Distribution Generalization of Multimodal Large Language Models](on_the_out-of-distribution_generalization_of_large_multimodal_models.md)
- [\[CVPR 2025\] Joint Vision-Language Social Bias Removal for CLIP](joint_vision-language_social_bias_removal_for_clip.md)
- [\[NeurIPS 2025\] NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints](../../NeurIPS2025/multimodal_vlm/navil_rethinking_scaling_properties_of_native_multimodal_large_language_models_u.md)
- [\[ICLR 2026\] DataProphet: Demystifying Supervision Data Generalization in Multimodal LLMs](../../ICLR2026/multimodal_vlm/demystifying_supervision_data_generalization_in_multimodal_lms.md)
- [\[CVPR 2025\] Single Domain Generalization for Few-Shot Counting via Universal Representation Matching](single_domain_generalization_for_few-shot_counting_via_universal_representation_.md)

</div>

<!-- RELATED:END -->
