---
title: >-
  [Paper Note] GenHancer: Imperfect Generative Models are Secretly Strong Vision-Centric Enhancers
description: >-
  [ICCV 2025][Image Generation][CLIP Enhancement] This work identifies that "perfect image reconstruction does not always yield the best visual representations," and proposes GenHancer — a two-stage post-training method that uses only a lightweight randomly initialized denoiser (~1/10 the parameters of pretrained heavy denoisers) conditioned solely on the global [CLS] token. Through self-supervised reconstruction, GenHancer enhances CLIP's fine-grained visual perception, achieving a 6.0% improvement over DIVA on MMVP-VLM.
tags:
  - ICCV 2025
  - Image Generation
  - CLIP Enhancement
  - Generative Models
  - Fine-Grained Vision
  - Diffusion Feedback
  - Lightweight Denoiser
date: 2026-05-08
content_hash: bc1105c2ac5f99e7
---

# GenHancer: Imperfect Generative Models are Secretly Strong Vision-Centric Enhancers

**Conference**: ICCV 2025
**arXiv**: [2503.19480](https://arxiv.org/abs/2503.19480)
**Code**: [github (mashijie1028/GenHancer)](https://mashijie1028.github.io/GenHancer)
**Area**: Visual Representation Enhancement / Image Generation
**Keywords**: CLIP Enhancement, Generative Models, Fine-Grained Vision, Diffusion Feedback, Lightweight Denoiser

## TL;DR

This work identifies that "perfect image reconstruction does not always yield the best visual representations," and proposes GenHancer — a two-stage post-training method that uses only a lightweight randomly initialized denoiser (~1/10 the parameters of pretrained heavy denoisers) conditioned solely on the global [CLS] token. Through self-supervised reconstruction, GenHancer enhances CLIP's fine-grained visual perception, achieving a 6.0% improvement over DIVA on MMVP-VLM.

## Background & Motivation

### Problem Definition

Discriminative models such as CLIP excel at high-level semantic understanding but exhibit systematic deficiencies in fine-grained visual perception (e.g., orientation, color, quantity, viewpoint). These deficiencies propagate to Multimodal Large Language Models (MLLMs) that use CLIP as the visual encoder, limiting their performance on vision-centric tasks.

### Limitations of Prior Work

**Visual expert ensemble methods (e.g., Cambrian)**: Concatenate multiple visual encoders, increasing inference cost and architectural complexity.

**Diffusion feedback methods (e.g., DIVA)**: Leverage pretrained Stable Diffusion heavy denoisers as feedback signals to enhance CLIP, but suffer from:
- Dependence on pretrained heavy denoisers (SD's UNet/DiT) with large parameter counts
- Insufficient exploration of why and what type of generative model effectively enhances representations
- End-to-end training that introduces irrelevant information, potentially degrading enhancement quality

### Core Motivation

**Counter-intuitive findings**: Systematic experiments reveal four key facts:
1. More training iterations → better reconstruction, but potentially worse representations
2. Larger denoiser → better reconstruction, but not necessarily better representations
3. Adding a small number of local tokens → substantially improved reconstruction, but sharply degraded representations
4. Using a pretrained denoiser → better reconstruction, but weaker representations

**General philosophy**: Generative models simultaneously contain "useful knowledge" (visual patterns, details) and "irrelevant information" (feature space gaps). Effective enhancement requires maximizing the mutual information $I(V;G_1)$ from useful knowledge while minimizing $I(V;G_2)$ from irrelevant information.

## Method

### Overall Architecture

A two-stage post-training pipeline: Stage-1 freezes the CLIP ViT and trains the projector and denoiser (to eliminate feature space gaps); Stage-2 fine-tunes the CLIP ViT with LoRA (to learn fine-grained visual knowledge). Only the [CLS] token is used as the conditioning input to the denoiser.

### Key Designs

#### 1. **Conditioning Visual Token Selection — [CLS] Only**

- **Function**: Restricts conditioning input to CLIP's global class token exclusively, discarding all local patch tokens.
- **Mechanism**: Under the mutual information framework, self-supervised reconstruction is equivalent to maximizing $I(V;G)$. Including local tokens in the conditioning introduces direct correspondences to local image regions, making the reconstruction task trivially easy (information leakage) and reducing $I(V;G_1)$, preventing the ViT from learning useful information from the denoiser.
  $$\max_V I(V;G_1) - \lambda I(V;G_2) \Rightarrow \max_V I(V;G_1) + \lambda d(V;V_0)$$
  Experiments demonstrate that adding even 10% of local tokens causes a sharp drop in enhancement performance.
- **Design Motivation**: Using only the [CLS] token forces the ViT to compress all fine-grained visual information into the global representation, maximizing the efficiency of mutual information transfer. This finding holds for both continuous and discrete denoisers.

#### 2. **Two-Stage Training Strategy**

- **Function**: Decomposes training into two stages — first eliminating irrelevant information, then learning useful knowledge.
- **Mechanism**:
    - **Stage-1**: Freezes the CLIP ViT $\mathbf{v}_\theta$ and trains the projector $\mathbf{h}_\omega$ and denoiser $\mathbf{g}_\phi$. The denoiser acquires basic generative capability, while the projector learns to bridge the feature space gap, reducing $I(V;G_2)$.
    - **Stage-2**: Fine-tunes the CLIP ViT using LoRA (rank=16) to amplify $I(V;G_1)$ and enhance fine-grained representations. Experiments show that once Stage-1 is sufficiently trained, whether the denoiser and projector continue to be trained in Stage-2 has negligible impact.
    - End-to-end training (without Stage-1) degrades performance by more than 5% across all settings.
- **Design Motivation**: The denoiser is randomly initialized and lightweight; direct end-to-end training injects noisy gradients ($G_2$) into the ViT at early stages, damaging existing representations. The two-stage strategy is an elegant realization of $\min I(V;G_2)$.

#### 3. **Lightweight Denoiser and Generative Paradigms**

- **Function**: Demonstrates that a lightweight, randomly initialized denoiser suffices to achieve superior enhancement, applicable to both continuous and discrete generative paradigms.
- **Mechanism**:
    - **Continuous denoiser (Rectified Flow)**: Adopts a FLUX-style DiT architecture with only 2 MM-DiT + 4 Single-DiT blocks (~1/10 the parameters of the original FLUX), injecting [CLS] conditioning via adaptive layer normalization. The loss is a flow-matching regression objective:
    $$\mathcal{L}_c = \mathbb{E}_{t,\mathbf{x}} \|(\widetilde{\mathbf{x}_1} - \widetilde{\mathbf{x}_0}) - \mathbf{g}_\phi(\widetilde{\mathbf{x}_t}, t, \mathbf{h}_\omega \circ \mathbf{v}_\theta(\mathbf{x}))\|_2^2$$
    - **Discrete denoiser (Perceiver)**: Uses a 6-layer Perceiver to predict masked tokens over a VQ-GAN codebook, injecting [CLS] conditioning via cross-attention. The loss is cross-entropy:
    $$\mathcal{L}_d = \mathbb{E}_{\mathbf{x}} -\log \prod_{i=1}^L \mathbf{g}_\phi(s_i | s_{<i}, \mathbf{h}_\omega \circ \mathbf{v}_\theta(\mathbf{x}))$$
    - **Timestep sampling**: Proposes scaled Logit-Normal sampling $t = \text{sigmoid}(s \cdot \varepsilon)$, $\varepsilon \sim \mathcal{N}(0,1)$; with $s=1$, sampling is concentrated around intermediate timesteps, increasing reconstruction difficulty and amplifying $I(V;G_1)$.
- **Design Motivation**: The success of lightweight denoisers demonstrates that enhancement does not require perfect reconstruction capability — it only requires the ViT to learn sufficient visual patterns. Excessively large denoisers may in fact introduce more $G_2$.

### Loss & Training

- Trained on CC3M dataset, 1 epoch per stage
- AdamW optimizer: Stage-1 lr=1e-4, Stage-2 lr=1e-5
- LoRA rank=16 for CLIP ViT fine-tuning
- Global batch size 256

## Key Experimental Results

### Main Results

**MMVP-VLM Benchmark (fine-grained visual perception, 9 visual pattern categories)**:

| CLIP Backbone | Method | Orient. | Attr. | State | Count | Pos. | Color | Struct. | Text | View. | Avg. |
|---------------|--------|---------|-------|-------|-------|------|-------|---------|------|-------|------|
| OpenAI L@224 | Original | 13.3 | 13.3 | 20.0 | 20.0 | 13.3 | 53.3 | 20.0 | 6.7 | 13.3 | 19.3 |
| | DIVA | 13.3 | 20.0 | 40.0 | 6.7 | 20.0 | 53.3 | 46.7 | 20.0 | 13.3 | 25.9 |
| | **Ours** | 13.3 | 33.3 | 33.3 | 20.0 | 6.7 | 73.3 | 46.7 | 20.0 | 40.0 | **31.9 (+6.0)** |
| MetaCLIP H@224 | Original | 6.7 | 13.3 | 60.0 | 13.3 | 6.7 | 53.3 | 26.7 | 13.3 | 33.3 | 25.2 |
| | **Ours** | 20.0 | 20.0 | 66.7 | 26.7 | 26.7 | 66.7 | 33.3 | 20.0 | 53.3 | **37.0 (+5.1)** |
| SigLIP SO@224 | Original | 26.7 | 20.0 | 53.3 | 40.0 | 20.0 | 66.7 | 40.0 | 20.0 | 53.3 | 37.8 |
| | **Ours** | 20.0 | 20.0 | 66.7 | 60.0 | 20.0 | 86.7 | 40.0 | 13.0 | 53.3 | **42.2 (+1.5)** |

### Ablation Study

**Conditioning token ratio ablation (MMVP-VLM, continuous denoiser)**:

| Condition | [CLS]+0% | [CLS]+10% | [CLS]+50% | [CLS]+100% |
|-----------|----------|-----------|-----------|------------|
| Reconstruction Quality | Lowest | ↑ | ↑↑ | Highest |
| Enhancement Performance | **Best** | Sharp drop | Very poor | Very poor |

**Timestep sampling comparison (s = scale factor, MMVP-VLM)**:

| Distribution | Scale | OpenAI@224 | OpenAI@336 | MetaCLIP@224 |
|--------------|-------|------------|------------|--------------|
| Uniform | N/A | 21.5 | 22.2 | 23.7 |
| **Logit-Normal** | **1.0** | **31.9** | **29.6** | **31.9** |

**End-to-end vs. two-stage**: End-to-end training degrades performance by >5% across all settings.

### Key Findings

1. **Perfect reconstruction ≠ good representations**: The key to enhancing CLIP lies not in generation quality, but in effectively transferring fine-grained knowledge.
2. **Lightweight denoisers suffice**: A FLUX-lite DiT with 2+4 blocks (~10% of parameters) outperforms DIVA, which uses the full SD UNet.
3. **Local tokens are detrimental**: Even 10% of local tokens leads to information leakage and training collapse.
4. **Two-stage training is essential**: End-to-end training introduces excessive irrelevant information due to feature space gaps.
5. **CLIP's original capabilities are preserved**: Zero-shot classification and retrieval performance changes by <0.3%.

## Highlights & Insights

1. **"Imperfection as perfection"**: The core finding is counter-intuitive — a perfect generative model is not required to enhance a discriminative model; in fact, overly strong generation capability may be harmful.
2. **Mutual information theoretical framework**: The decomposition of $I(V;G_1)$ vs. $I(V;G_2)$ elegantly explains all experimental observations.
3. **Extreme methodological simplicity**: Compared to DIVA, which requires the pretrained full SD UNet, GenHancer requires only a randomly initialized small model with 1 epoch of training.
4. **Unified continuous/discrete framework**: All three key design points apply equally to Rectified Flow and Perceiver, validating the universality of the underlying principles.
5. **Plug-and-play MLLM enhancement**: The enhanced CLIP can directly replace the visual encoder in LLaVA without architectural changes.

## Limitations & Future Work

1. **Evaluation limited to CLIP-family models**: Applicability to other visual encoders such as DINOv2 and EVA-CLIP remains unknown.
2. **Small training set (CC3M)**: Whether scaling to larger datasets (e.g., DataComp) would yield further improvements is unexplored.
3. **LoRA rank=16 optimality unverified**: No ablation over LoRA rank is provided.
4. **Discrete denoiser requires a VQ-GAN codebook**: There is an implicit dependency on codebook quality.
5. **MMVP-VLM benchmark is small** (135 pairs): Stability on larger-scale benchmarks remains to be verified.

## Related Work & Insights

- **Key distinction from DIVA**: DIVA uses a pretrained SD UNet with end-to-end training; GenHancer uses a randomly initialized lightweight denoiser with two-stage training.
- **Key distinction from ROSS**: ROSS incorporates reconstruction loss during MLLM training; GenHancer enhances CLIP independently, offering greater flexibility.
- **Broader implications**: These findings may generalize to other modalities such as audio and 3D — using lightweight generative feedback to enhance fine-grained perception in discriminative models.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The finding that imperfect generative models are more beneficial is profound and counter-intuitive; the mutual information framework offers theoretical elegance.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple CLIP backbones, both continuous and discrete paradigms, MLLM integration, and zero-shot preservation are evaluated, though the benchmark scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The argumentation is logically clear, with three key points building progressively; figures are well-designed.
- **Value**: ⭐⭐⭐⭐⭐ — Provides an extremely low-cost solution to visual deficiencies in MLLMs with strong inspirational value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning](genflowrl_shaping_rewards_with_generative_object-centric_flow_in_visual_reinforc.md)
- [\[ICCV 2025\] Mind the Gap: Aligning Vision Foundation Models to Image Feature Matching](mind_the_gap_aligning_vision_foundation_models_to_image_feature_matching.md)
- [\[NeurIPS 2025\] Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior](../../NeurIPS2025/image_generation/diff-icmh_harmonizing_machine_and_human_vision_in_image_compression_with_generat.md)
- [\[ICCV 2025\] Deeply Supervised Flow-Based Generative Models](deeply_supervised_flow-based_generative_models.md)
- [\[ICCV 2025\] DICE: Staleness-Centric Optimizations for Parallel Diffusion MoE Inference](dice_staleness-centric_optimizations_for_parallel_diffusion_moe_inference.md)

</div>

<!-- RELATED:END -->
