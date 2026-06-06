---
title: >-
  [Paper Note] Flowing Backwards: Improving Normalizing Flows via Reverse Representation Alignment
description: >-
  [AAAI 2026][Image Generation][Normalizing Flows] This paper proposes R-REPA (Reverse Representation Alignment), which creatively exploits the invertibility of Normalizing Flows to align intermediate features with visual…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Normalizing Flows"
  - "TARFlow"
  - "Representation Alignment"
  - "Reverse Alignment"
  - "ImageNet"
date: 2026-05-08
content_hash: 8390d7db65ef91eb
---

# Flowing Backwards: Improving Normalizing Flows via Reverse Representation Alignment

**Conference**: AAAI 2026
**arXiv**: [2511.22345](https://arxiv.org/abs/2511.22345)  
**Code**: None  
**Area**: Generative Models / Normalizing Flows / Representation Alignment
**Keywords**: Normalizing Flows, TARFlow, Representation Alignment, Reverse Alignment, Image Generation, ImageNet

## TL;DR

This paper proposes R-REPA (Reverse Representation Alignment), which creatively exploits the invertibility of Normalizing Flows to align intermediate features with visual foundation models along the generative (reverse) path. It further introduces a training-free classification algorithm, achieving new state-of-the-art results for normalizing flows on ImageNet 64×64 and 256×256 with a 3.3× training speedup.

## Background & Motivation

Normalizing Flows (NFs) are a class of generative models with **exact mathematical invertibility**: the forward path maps data to a latent space for density estimation, while the reverse path generates new samples from the latent space. This bidirectional structure inherently creates a natural synergy between representation learning and data generation—two sides of the same coin.

However, standard NFs optimize only maximum likelihood estimation (MLE) along the forward path, resulting in intermediate features that lack semantic meaning and limiting generation quality. The recent REPA method demonstrated the power of a "representation-first" strategy in diffusion models—aligning internal features of the denoising network with pretrained visual encoders to substantially improve training efficiency and generation quality.

The core question of this paper is: **Can the unique invertible structure of NFs be leveraged to design a superior representation alignment strategy?** Unlike diffusion models where only the forward path is operable, the reverse generative path of NFs offers entirely new alignment possibilities. The authors find that aligning features along the generative path ($z \to x$) is more effective than aligning along the encoding path ($x \to z$), achieving simultaneous improvements in both generation quality and discriminative capability.

## Method

### Overall Architecture

The method builds upon TARFlow (Transformer AutoRegressive Flow) and comprises three core contributions:

1. **Training-free classification algorithm**: Leverages the density estimation capability of conditional NFs to perform test-time classification via a single gradient step.
2. **Reverse Representation Alignment (R-REPA)**: Aligns intermediate NF features with visual foundation models along the generative path.
3. **Latent space extension**: Migrates TARFlow into a VAE latent space to enable high-resolution generation.

### Key Design 1: Training-Free Classification Algorithm

Conventional evaluation of NF discriminative capability requires training a separate linear classifier for each layer (linear probing), which is costly and indirect. This paper proposes directly utilizing the model's density estimation for classification:

1. Define classification logits $\boldsymbol{\lambda} \in \mathbb{R}^K$, initialized to zero.
2. Compute the weighted class embedding $\mathbf{e}_{\text{eff}} = \text{softmax}(\boldsymbol{\lambda})^T \mathbf{E}$.
3. Compute the conditional log-likelihood $\mathcal{L}(\boldsymbol{\lambda}) = \log p(\mathbf{x} | \mathbf{e}_{\text{eff}}; \theta)$.
4. Compute the gradient with respect to the logits and predict the class corresponding to the maximum gradient component.

The entire process requires only **a single forward and backward pass**, with no additional parameters to train. Experiments verify that the classification accuracy of this method matches the best-layer result of standard linear probing, making it a more efficient and principled semantic evaluation metric.

### Key Design 2: Systematic Exploration of Three Alignment Strategies

Given a pretrained frozen visual encoder $\Phi(\cdot)$, the alignment loss is defined as:

$$\mathcal{L}_{\text{align}}^{(t,l)}(\theta, \phi) = -\frac{1}{P} \sum_{p=1}^{P} \text{sim}\left(\mathbf{v}^{[p]}, [\text{Proj}_\phi(\mathbf{h}^{(t,l)})]^{[p]}\right)$$

where $\mathbf{h}^{(t,l)}$ denotes the feature at the $l$-th layer of the $t$-th block in TARFlow, and $\text{Proj}_\phi$ is a learnable MLP projection head. The authors systematically compare three gradient backpropagation strategies:

| Strategy | Gradient Path | Update Scope | Core Idea |
|----------|--------------|--------------|-----------|
| Forward (F-REPA) | Forward computation graph | All blocks before the aligned layer | Direct analogy to REPA |
| Detach (D-REPA) | Input gradient detached | Current block only | Analogous to timestep isolation in diffusion models |
| **Reverse (R-REPA)** | **Reverse generative computation graph** | **All blocks after the aligned layer** | **NF-unique: backprop through $f_\theta^{-1}$** |

R-REPA implementation: The forward path first computes $\mathbf{z} = f_\theta(\mathbf{x})$; after detaching $\mathbf{z}$, the reverse generation $f_\theta^{-1}$ is executed, and the alignment loss is computed on intermediate features of the reverse path. Gradients propagate through the generative path, updating only parameters after the aligned layer in the generative path, without interfering with forward density modeling.

### Key Design 3: Accelerated Pseudo-Reverse Implementation

The naive reverse path is autoregressive (each token depends on all preceding tokens) and cannot be parallelized. This paper designs an accelerated implementation:

1. During the forward pass, cache the input $\hat{\mathbf{x}}^{t-1} = \text{stop\_gradient}(\mathbf{x}^{t-1})$ for each block.
2. In the pseudo-reverse, use cached $\hat{\mathbf{x}}^{t-1}$ to provide autoregressive context, parallelizing the reverse computation.
3. Due to invertibility, the pseudo-reverse output is numerically identical to the cached values while constructing a valid reverse computation graph.

Speedup: approximately **50×** over the naive reverse implementation, with approximately **50% memory reduction**.

### Loss & Training

The total loss is a weighted sum of the standard NF loss and the alignment loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NF}} + \lambda_{\text{align}} \cdot \frac{1}{|\mathcal{A}|} \sum_{(t,l) \in \mathcal{A}} \mathcal{L}_{\text{align}}^{(t,l)}$$

Optimal configuration: alignment at layer 6 of Blocks 7 & 8 (i.e., the blocks that first process the latent variable $\mathbf{z}$ in the generative path), guiding the early generation stage to establish correct high-level image structure.

For 256×256 resolution, a pretrained VAE encoder compresses images into latent space; the NF models the latent space. During training, noise ($\sigma=0.20$) is added to the latent vectors, and generation involves sampling followed by score-based denoising.

## Experiments

### Ablation Study: Comparison of Three Alignment Strategies (ImageNet 64×64, 400K iter)

| Strategy | Alignment Position | FID ↓ | sFID ↓ | IS ↑ | Acc. (%) ↑ |
|----------|--------------------|-------|--------|------|------------|
| TARFlow baseline | — | 12.91 | 33.79 | 36.62 | 37.43 |
| Forward | All blocks | 12.25 | 37.97 | 40.85 | 46.97 |
| Detach | All blocks | 12.19 | 34.31 | 41.98 | 49.06 |
| Reverse | All blocks | 12.21 | 33.80 | 42.08 | 49.91 |
| Forward | Block 1&2 | 12.67 | 39.99 | 41.11 | 61.16 |
| Detach | Block 7&8 | 12.12 | 34.00 | 41.18 | 55.14 |
| **Reverse** | **Block 7&8** | **11.93** | **33.78** | 40.90 | 55.21 |
| **Reverse** | **Block 7&8, L6** | **11.71** | 33.68 | 44.31 | 57.35 |

Key Findings:
- The Forward strategy systematically degrades sFID (33.79→37.97), as unconstrained gradients in early blocks create conflicts between MLE and alignment objectives.
- The Reverse strategy consistently outperforms Detach on FID (11.93 vs. 12.12), as updating only the generative path does not interfere with density modeling.
- Aligning Blocks 7&8 (the position where the generative path first processes $\mathbf{z}$) yields the best FID, while Blocks 1&2 yield the best accuracy—a trade-off exists between generation quality and discriminative capability.

### Training Efficiency vs. Baseline (ImageNet 64×64 & 256×256)

| Model | Resolution | Training Iter. | FID ↓ | Acc. (%) ↑ |
|-------|------------|----------------|-------|------------|
| TARFlow | 64 | 1M | 11.76 | 39.97 |
| +R-REPA | 64 | 400K | 11.71 | 57.76 |
| +R-REPA | 64 | 1M | **11.25** | 57.02 |
| Latent-TARFlow | 256 | 1M | 13.05 | 40.22 |
| +R-REPA | 256 | 1M | **12.79** | **56.24** |

R-REPA surpasses the baseline's 1M-iteration result using only 400K iterations (FID 11.71 vs. 11.76), achieving a **3.3× training speedup**. Classification accuracy improves from 39.97% to 57.76% (+17.8 pp).

### Main Results: ImageNet 64×64

| Model Type | Model | FID ↓ | sFID ↓ |
|------------|-------|-------|--------|
| Diffusion | EDM | 1.36 | — |
| Diffusion | ADM | 2.09 | 4.29 |
| GAN | BigGAN | 4.06 | 3.96 |
| Consistency Model | iCT-deep | 3.25 | — |
| NF | TARFlow | 4.21 | 5.34 |
| **NF** | **+R-REPA (Ours)** | **3.69** | **4.34** |

R-REPA reduces the NF FID from 4.21 to 3.69 (−12.4%), surpassing BigGAN (4.06) for the first time, while requiring only **two sampling steps**.

### Main Results: ImageNet 256×256

| Model | FID ↓ | sFID ↓ | IS ↑ |
|-------|-------|--------|------|
| ADM | 4.59 | 5.25 | 186.70 |
| DiT | 2.27 | 4.60 | 278.24 |
| SiT | 2.06 | 4.50 | 270.30 |
| VAR | 1.73 | — | 350.2 |
| Latent-TARFlow | 5.15 | 6.78 | 243.49 |
| **+R-REPA + Patch1** | **4.18** | **4.96** | 240.8 |

On 256×256, FID improves from 5.15 to 4.18 (−18.8%), reaching a level comparable to ADM (4.59) while maintaining significantly higher inference efficiency than diffusion models.

## Highlights & Insights

1. **Architecture-aware method design**: R-REPA does not simply transplant REPA onto NFs; rather, it deeply exploits the unique invertibility of NFs to perform alignment on the reverse generative path. This "structure determines strategy" design philosophy is broadly instructive—different generative models warrant distinct enhancement schemes.

2. **Engineering ingenuity in the accelerated pseudo-reverse**: The naive autoregressive reverse path is unsuitable for training, but by caching forward features with gradient detachment, the $O(D)$ serial computation is converted into $O(1)$ parallel computation while preserving numerical consistency and correct gradient flow. This 50× speedup makes R-REPA practically viable.

3. **Training-free classification as a semantic probe**: More efficient and principled than linear probing—by directly leveraging the generative model's density estimation for discrimination, it validates the hypothesis that "generation implies understanding" and provides a standardized tool for evaluating the discriminative capability of NFs.

4. **Precise analysis of generative vs. encoding paths**: The analysis of why Forward REPA degrades sFID (early blocks are responsible for low-level spatial statistics; forcing high-level semantic alignment disrupts this) reflects a deep understanding of the internal division of labor within the model.

## Limitations & Future Work

1. **Persistent gap with diffusion models**: FID of 4.18 vs. SiT 2.06 / VAR 1.73 on ImageNet 256×256 indicates that NFs still trail in absolute generation quality.
2. **Dependence on pretrained visual encoders**: Alignment requires a frozen visual foundation model for supervision, introducing additional model dependencies and computational overhead.
3. **Trade-off between generation quality and discriminative capability**: Blocks 7&8 optimize FID while Blocks 1&2 optimize accuracy; the two objectives cannot be jointly maximized. The current configuration favors generation quality.
4. **Validated only on TARFlow**: R-REPA has not been tested on other NF architectures such as RealNVP or Glow; its generalizability remains to be verified.
5. **Resolution limitations**: Experiments are conducted only at 64×64 and 256×256; scalability to higher resolutions (512+) has not been explored.

## Related Work & Insights

| Direction | Representative Works | Relationship to This Paper |
|-----------|---------------------|---------------------------|
| NF architectures | TARFlow, JetFormer, FARMER | This paper builds upon TARFlow |
| Representation alignment for accelerated generation | REPA (diffusion models) | Direct inspiration; this paper proposes NF-specific reverse alignment |
| REPA extensions | REPA-E, LightningDiT, U-REPA | All operate on the forward path of diffusion models; invertible architectures not addressed |
| NF extensions | STARFlow | Concurrent work extending NF scale and task complexity |

## Rating

- **Novelty**: ⭐⭐⭐⭐ (Reverse-path alignment is an innovation unique to NFs, not a direct application of REPA)
- **Technical Contribution**: ⭐⭐⭐⭐ (Accelerated pseudo-reverse + training-free classification + systematic strategy comparison)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Multi-resolution, complete ablations, SOTA comparison, training efficiency analysis)
- **Practicality**: ⭐⭐⭐ (NFs are less mainstream in practice than diffusion models, but the methodological ideas have transfer value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Coupling Within: Flow Matching via Distilled Normalizing Flows](../../ICML2026/image_generation/the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)
- [\[NeurIPS 2025\] Amortized Sampling with Transferable Normalizing Flows](../../NeurIPS2025/image_generation/amortized_sampling_with_transferable_normalizing_flows.md)
- [\[ICLR 2026\] GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models](../../ICLR2026/image_generation/glass_flows_reward_alignment_diffusion.md)
- [\[AAAI 2026\] Multi-Metric Preference Alignment for Generative Speech Restoration](multi-metric_preference_alignment_for_generative_speech_restoration.md)
- [\[AAAI 2026\] Hyperbolic Hierarchical Alignment Reasoning Network for Text-3D Retrieval](hyperbolic_hierarchical_alignment_reasoning_network_for_text-3d_retrieval.md)

</div>

<!-- RELATED:END -->
