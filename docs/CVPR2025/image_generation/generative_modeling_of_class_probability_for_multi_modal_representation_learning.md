---
title: >-
  [Paper Note] Generative Modeling of Class Probability for Multi-Modal Representation Learning
description: >-
  [CVPR 2025][Image Generation][Multimodal Alignment] CALM proposes a generative multimodal representation learning method based on class-anchor alignment. By introducing class labels from an independent dataset as anchors to bridge the modality gap between video and text, it models uncertainty using a cross-modal probabilistic variational autoencoder. This approach significantly outperforms existing methods across four benchmarks, particularly in out-of-domain evaluations.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Multimodal Alignment"
  - "Class-Anchor Probability Distribution"
  - "Variational Autoencoder"
  - "Video-Text Retrieval"
  - "Cross-Modal Generative Modeling"
date: 2026-05-08
content_hash: e09256c220a1b183
---

# Generative Modeling of Class Probability for Multi-Modal Representation Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.17417](https://arxiv.org/abs/2503.17417)  
**Code**: None  
**Area**: Multimodal VLM / Video-Text Retrieval  
**Keywords**: Multimodal Alignment, Class-Anchor Probability Distribution, Variational Autoencoder, Video-Text Retrieval, Cross-Modal Generative Modeling

## TL;DR
CALM proposes a generative multimodal representation learning method based on class-anchor alignment. By introducing class labels from an independent dataset as anchors to bridge the modality gap between video and text, it models uncertainty using a cross-modal probabilistic variational autoencoder. This approach significantly outperforms existing methods across four benchmarks, particularly in out-of-domain evaluations.

## Background & Motivation
1. **Background**: Multimodal understanding (specifically video-text alignment) typically relies on contrastive learning to project features from different modalities into a shared embedding space. Methods such as CLIP4Clip, TS2-NET, and X-pool have achieved promising results in video-text retrieval.
2. **Limitations of Prior Work**: Contrastive learning relies on strict definitions of positive and negative pairs. However, information imbalance exists between video and text—a single sentence can correspond to multiple different videos, and each video contains rich visual information that text cannot fully describe. This modality discrepancy often causes contrastive learning to fail in modeling the underlying data distribution.
3. **Key Challenge**: Direct pairwise comparison between modalities ignores partial matching and uncertainty. Discriminative methods (contrastive learning) naturally struggle to model data distribution variability and partial cross-modal information overlap.
4. **Goal**: (a) Achieve cross-modal alignment without relying on strict positive-negative pairs; (b) Model the uncertainty in cross-modal alignment; (c) Enhance cross-domain generalization capability.
5. **Key Insight**: Intra-modal relationships are simpler than cross-modal relationships (as they share similar statistical properties). Therefore, a set of input-independent "class anchors" can serve as a bridge to convert cross-modal alignment into probability distribution alignment over these anchors.
6. **Core Idea**: Construct a shared anchor space using class labels from an independent classification dataset, calculate the probability distributions of video and text relative to these anchors, and achieve alignment by reconstructing the text-anchor distribution from the video-anchor distribution using a VAE.

## Method

### Overall Architecture
CALM is built upon pre-trained CLIP encoders. The input video is processed by a CLIP vision encoder combined with temporal fusion to obtain the video feature $\mathbf{V}$, while the text is processed by a CLIP text encoder to obtain the sentence feature $\mathbf{S}$. Concurrently, class labels from an independent classification dataset (Charades, 157 classes) are extracted, converted into prompts ("The content of [label]"), and encoded via the CLIP text encoder to obtain the class anchor features $\mathbf{P}$. The cosine similarities between the video/text features and the anchors are calculated and normalized via softmax to obtain the probability distributions $\mathbf{V}_p$ and $\mathbf{S}_p$, respectively. Finally, a cross-modal probabilistic VAE reconstructs $\mathbf{S}_p$ from $\mathbf{V}_p$ to achieve alignment.

### Key Designs

1. **Class-Prompt Probability Distribution**:
    - **Function**: Generates probabilistic representations for each modality relative to a shared semantic space.
    - **Mechanism**: Selects $K=157$ class labels (from the Charades dataset), appends the prompt template "The content of [label_k]", and passes them through the CLIP text encoder with learnable positional embeddings to obtain the anchor $\mathbf{p}_k$. The cosine similarity between the modality features and the anchors is calculated, followed by a temperature-scaled softmax to yield probability distributions: $\mathbf{V}_p = \text{softmax}(\tau \cdot \cos(\bar{\mathbf{h}}^v, \mathbf{P}))$ and $\mathbf{S}_p = \text{softmax}(\tau \cdot \cos(\mathbf{h}_{CLS}^s, \mathbf{P}))$.
    - **Design Motivation**: Class anchors are independent of the input data, providing complementary semantic information. The text-anchor distribution represents intra-modality relations (simpler), whereas the video-anchor distribution represents cross-modality relations (more complex). Aligning the two reformulates the challenging cross-modal problem into a generative problem mapping the cross-modal distribution to the intra-modal distribution.

2. **Cross-Modal Probabilistic VAE**:
    - **Function**: Generates the text-anchor distribution from the video-anchor distribution while modeling the uncertainty of the alignment.
    - **Mechanism**: The encoder encodes $\mathbf{V}_p$ into a Gaussian latent variable $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ (using the reparameterization technique). The decoder reconstructs $\hat{\mathbf{S}}_p$ from $\mathbf{z}$. The ELBO is decomposed into a reconstruction term and a KL regularization term. Since the output is a probability distribution, cross-entropy is used for the reconstruction loss: $\mathcal{L}_{rec} = -\sum_k \mathbf{S}_p^{(k)} \log \hat{\mathbf{S}}_p^{(k)}$. The KL divergence regularizes the posterior distribution to be close to a standard Gaussian prior.
    - **Design Motivation**: Deterministic mapping cannot capture the inherent uncertainty between modalities (as the same video can correspond to different descriptions). The latent variable $\mathbf{z}$ of the VAE naturally models this one-to-many relationship. Through the conditional distribution $p(\mathbf{S}_p | \mathbf{V}_p)$, the model implicitly learns a joint representation of video and text.

3. **Temperature Scaling and Positional Embedding Enhancement**:
    - **Function**: Controls the sharpness of the probability distribution and distinguishes different class anchors.
    - **Mechanism**: The temperature parameter $\tau$ controls the concentration of the softmax output—higher temperatures produce smoother distributions, while lower temperatures produce sharper ones. A learnable positional embedding $\mathbf{e}_k^{pos}$ is also added to each anchor, allowing the model to distinguish the semantic roles of different anchors.
    - **Design Motivation**: Pure cosine similarity might generate overly concentrated or dispersed probabilities; the temperature parameter offers flexible control. Positional embeddings ensure that the anchors contain learnable structural information in addition to the semantic representations of class labels.

### Loss & Training
The total loss is defined as $\mathcal{L} = \mathcal{L}_{rec} + \alpha \mathcal{L}_{KL} + \mathcal{L}_{task}$, where $\alpha=0.1$ balances the reconstruction and KL divergence, and $\mathcal{L}_{task}$ is the downstream task loss (e.g., contrastive loss or captioning loss). A ViT-B/32 CLIP is used, with a latent space dimension of $d=256$. The AdamW optimizer is employed with a learning rate of 1e-5. Training is conducted for 5 epochs for the retrieval task and 20 epochs for the captioning task. Videos are uniformly sampled at 12 frames with a resolution of 224×224.

## Key Experimental Results

### Main Results

**Video Retrieval (In-Domain):**

| Dataset | R@1 | CALM | Prev. SOTA | Gain |
|--------|-----|------|----------|------|
| MSR-VTT | R@1 | **50.8** | 49.0 (DiffusionRet) | +1.8 |
| DiDeMo | R@1 | **51.1** | 47.8 (EMCL) | +3.3 |
| LSMDC | R@1 | **27.5** | 26.0 (T-MASS) | +1.5 |

**Video Retrieval (Out-of-Domain) — Key Highlights:**

| Train Set $\rightarrow$ Test Set | R@1 | CALM | Prev. SOTA | Gain |
|---------------|-----|------|----------|------|
| MSR-VTT $\rightarrow$ DiDeMo | R@1 | **41.2** | 37.3 (T-MASS) | +3.9 |
| MSR-VTT $\rightarrow$ LSMDC | R@1 | **21.4** | 19.6 (T-MASS) | +1.8 |
| DiDeMo $\rightarrow$ LSMDC | R@1 | **22.1** | 20.4 (T-MASS) | +1.7 |
| DiDeMo $\rightarrow$ MSR-VTT | R@1 | **41.7** | 39.7 (T-MASS) | +2.0 |

### Ablation Study

| Configuration | MSR-VTT R@1 | Description |
|------|-------------|------|
| CALM (full) | 50.8 | Full model |
| w/o class anchors | ~47-48 (estimated) | Degenerates to direct contrastive learning |
| w/o VAE (direct distribution alignment) | Lower | Unable to model uncertainty |
| w/o positional embedding | Lower | Anchors become indistinguishable |

### Key Findings
- CALM shows more pronounced advantages in out-of-domain evaluations (with an average R@1 improvement of 2.3%), indicating that the complementary semantic information provided by class anchors effectively enhances generalization capability.
- In the MSR-VTT $\rightarrow$ DiDeMo out-of-domain evaluation, the MnR of CALM drops from 26.3 (T-MASS) to 16.1 (an enormous improvement), indicating a massive boost in ranking quality.
- Using class labels from an independent dataset (Charades, 157 action categories) as anchors validates that anchors do not need to overlap with task data to be effective.
- The in-domain to out-of-domain performance drop of CALM (averaging ~11%) is comparable to CLIP4Clip, but its absolute performance is significantly higher.

## Highlights & Insights
- **Novel class-anchor bridging concept**: Instead of directly contrasting videos and texts, alignment is achieved indirectly through an independent semantic anchor space. This idea can be transferred to any cross-modal task with an inherent modality gap (such as audio-text or tactile-visual).
- **Paradigm shift from discriminative to generative**: Converts cross-modal alignment from "determining a match" to "generating one modal probability distribution from another," making it naturally suited to handle one-to-many mappings and partial matching.
- **Uncertainty modeling via VAE**: Unlike deterministic mappings, the variance of the latent variable $z$ explicitly captures the level of uncertainty in modality alignment, providing confidence information for downstream applications.
- **Strong out-of-domain generalization**: Operating as input-independent semantic bases, class anchors reduce overfitting to specific dataset distributions.

## Limitations & Future Work
- The number of class anchors $K=157$ is fixed and selected from Charades. The impact of larger or more diverse anchor sets (e.g., ImageNet-1K's 1000 classes) on performance has not been explored.
- Only ViT-B/32 CLIP is used; performance on larger models (such as ViT-L/14) has not been verified.
- The choice of latent space dimension $d=256$ for the VAE lacks sufficient ablation.
- Probability distributions are calculated using global average frame features, which loses fine-grained temporal details; frame-level anchor distributions could be explored.
- The impact of the choice of class anchors deserves deeper investigation to see if different category systems produce varying effects.

## Related Work & Insights
- **vs CLIP4Clip**: CLIP4Clip directly contrasts video and text features, relying on positive and negative pairs. CALM introduces anchors as intermediaries, avoiding the rigid pairing requirements of contrastive learning.
- **vs DiffusionRet**: DiffusionRet utilizes a diffusion model to establish a shared latent space but still relies on direct matching. The anchor method in CALM is lighter and generalizes better.
- **vs T-MASS**: T-MASS models text embeddings as support sets to capture uncertainty, whereas CALM models uncertainty more systematically at the level of probability distributions through a VAE.
- **vs UATVR**: UATVR performs distribution matching directly in the feature space, leading to a significant performance drop in out-of-domain evaluations; the anchor mechanism in CALM provides more stable cross-domain representations.

## Rating
- Novelty: ⭐⭐⭐⭐ The alignment of class-anchor probability distributions is a novel perspective, though the core components (VAE, CLIP) are mature technologies.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across four datasets with in-domain and out-of-domain testing, but ablation details are somewhat limited.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and comprehensive derivation of equations.
- Value: ⭐⭐⭐⭐ Significant improvements in out-of-domain generalization, providing valuable inspiration for multi-modal alignment research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling](mmar_towards_lossless_multi-modal_auto-regressive_probabilistic_modeling.md)
- [\[NeurIPS 2025\] Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification](../../NeurIPS2025/image_generation/latent_zoning_network_a_unified_principle_for_generative_modeling_representation.md)
- [\[CVPR 2025\] Symbolic Representation for Any-to-Any Generative Tasks](symbolic_representation_for_any-to-any_generative_tasks.md)
- [\[CVPR 2025\] Multi-Group Proportional Representation for Text-to-Image Models](multi-group_proportional_representations_for_text-to-image_models.md)
- [\[CVPR 2025\] SyncVP: Joint Diffusion for Synchronous Multi-Modal Video Prediction](syncvp_joint_diffusion_for_synchronous_multi-modal_video_prediction.md)

</div>

<!-- RELATED:END -->
