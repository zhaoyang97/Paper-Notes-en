---
title: >-
  [Paper Note] Imagine Before Concentration: Diffusion-Guided Registers Enhance Partially Relevant Video Retrieval
description: >-
  [CVPR 2026][Image Generation][Partially Relevant Video Retrieval] This paper proposes DreamPRVR, which adopts a coarse-to-fine "imagine before concentrate" strategy: a truncated diffusion model generates global semantic…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Partially Relevant Video Retrieval"
  - "Diffusion Models"
  - "Register Tokens"
  - "Cross-Modal Alignment"
  - "Global Context"
date: 2026-05-08
content_hash: b444f48ce809b2d9
---

# Imagine Before Concentration: Diffusion-Guided Registers Enhance Partially Relevant Video Retrieval

**Conference**: CVPR 2026
**arXiv**: [2604.03653](https://arxiv.org/abs/2604.03653)  
**Code**: [https://github.com/lijun2005/CVPR26-DreamPRVR](https://github.com/lijun2005/CVPR26-DreamPRVR)  
**Area**: Image Generation
**Keywords**: Partially Relevant Video Retrieval, Diffusion Models, Register Tokens, Cross-Modal Alignment, Global Context

## TL;DR

This paper proposes DreamPRVR, which adopts a coarse-to-fine "imagine before concentrate" strategy: a truncated diffusion model generates global semantic register tokens under text supervision, which are then fused into fine-grained video representations to suppress spurious local noise responses, achieving state-of-the-art performance on three PRVR benchmarks.

## Background & Motivation

**Background**: Partially Relevant Video Retrieval (PRVR) aims to retrieve untrimmed videos given a text query that describes only a portion of the video content. Existing methods (e.g., MS-SL, GMMFormer, HLFormer) primarily focus on segment-level modeling, employing sliding windows or Gaussian attention for local matching.

**Limitations of Prior Work**: The core issue is "query ambiguity" — a generic query may match the corresponding segment in the target video while also accidentally matching locally similar segments in irrelevant videos, producing spurious local spike responses. This can cause globally irrelevant videos to be erroneously ranked highly. Furthermore, the widely adopted Multiple Instance Learning (MIL) paradigm only rewards the best-matching segment, leaving other segments undertrained and lacking the contextual grounding needed to resolve ambiguity.

**Key Challenge**: Existing methods lack explicit global context modeling. The few works that do consider global information (e.g., HLFormer's semantic entailment, RAL's global uncertainty) treat global context as a training-only regularizer, without improving video embeddings at inference time.

**Goal**: (1) How to extract reliable global semantic representations from noisy, untrimmed videos; (2) How to leverage textual semantics to effectively supervise global representation generation; (3) How to incorporate global semantics into local video representations to suppress spurious responses.

**Key Insight**: Inspired by the register token concept in ViT, global register tokens are introduced to store holistic video semantics. Since directly extracting reliable registers from noisy videos is difficult, a diffusion model is employed for iterative refinement and generation.

**Core Idea**: A text-supervised truncated diffusion model iteratively generates global semantic registers starting from a video-centric distribution, which are then fused into local representations via attention to enhance fine-grained video features.

## Method

### Overall Architecture

DreamPRVR comprises four core components: (1) Text Semantic Structure Learning, which constructs an ordered text latent space and samples supervision signals; (2) truncated diffusion-based global register generation; (3) register-enhanced video representation learning; and (4) cross-modal similarity computation. The overall framework follows a variational inference paradigm, treating registers as latent variables.

### Key Designs

1. **Text Semantic Structure Learning (TSSL) + Text Perturbation Sampler (TPS)**:

    - **Function**: Constructs an ordered text latent space and generates diverse supervision signals to guide register generation.
    - **Mechanism**: TSSL consists of two losses: the Query Diversity Loss $L_{div}$ disperses query embeddings of different videos to increase semantic richness, while the Query Similarity Preservation Loss $L_{qsp}$ keeps query embeddings of the same video tightly clustered (treated as complementary positive views of the same global semantics). TPS explicitly models text uncertainty by applying controlled perturbations to whitened features as $\hat{q} = \alpha \cdot \bar{q} + \beta$, where $\alpha \sim \mathcal{N}(1, (\gamma\sigma_q)^2I)$, requiring no additional trainable parameters.
    - **Design Motivation**: Existing query diversity losses blindly separate all queries, ignoring intra-video query correlations. $L_{qsp}$ addresses this gap, enabling the latent space to be simultaneously compact within videos and discriminative across videos.

2. **Probabilistic Variational Sampler (PVS) + Diffusion Register Estimator (DRE)**:

    - **Function**: Generates clean global semantic registers from video features.
    - **Mechanism**: PVS first encodes video features into a probability distribution $p(r_T | V_v) \sim \mathcal{N}(\mu_v, \sigma_v^2 I)$, sampling a video-centric initial noise $r_T$ via reparameterization. DRE is a lightweight MLP-based diffusion module that starts from $r_T$ (rather than random Gaussian noise) and performs $T$ steps of iterative denoising under text supervision $\hat{q}$, ultimately generating the optimal registers $r_0$. The training objective follows the standard DDPM noise prediction formulation: $L_{dre} = \mathbb{E}_{t, \hat{q}_t, \epsilon}[\|\epsilon - \epsilon_\phi(\hat{q}_t, t, c)\|^2]$.
    - **Design Motivation**: Direct pooling or single-step mapping is insufficient to disentangle reliable semantics from redundant, noisy untrimmed videos. PVS provides a semantically informed starting point (truncated diffusion), while DRE progressively purifies semantics through iterative refinement. t-SNE visualizations confirm that registers evolve from disordered states to well-separated discriminative clusters.

3. **Register-Augmented Gaussian Attention Block (RAB)**:

    - **Function**: Integrates the generated global registers into local video representations.
    - **Mechanism**: Video tokens and registers are concatenated as $x = [V_o, r_0]$ and processed through a modified Gaussian attention mechanism: $\text{GA}(x) = \text{softmax}(\mathcal{M}_r + (\mathcal{M}_\sigma^g \odot \frac{x^q(x^k)^\top}{\sqrt{d_h}})) x^v$. An asymmetric attention mask $\mathcal{M}_r$ is applied: video tokens can attend to both registers and other video tokens, while registers attend only to video tokens. $N_a$ RABs are arranged in parallel, with outputs aggregated via MAIM. Registers are discarded after processing and do not participate in final similarity computation.
    - **Design Motivation**: The asymmetric mask design allows registers to provide global contextual information to video tokens while preventing information short-circuiting among registers themselves.

### Loss & Training

The total loss is: $L_{total} = L_{sim} + L_{tssl} + L_{pvs} + \lambda_{dre} L_{dre}$. $L_{sim}$ is the standard retrieval similarity loss (following MS-SL); $L_{tssl} = \lambda_d L_{div} + \lambda_q L_{qsp}$; $L_{pvs} = \lambda_{kl} L_{kl}$ (Gaussian prior constraint for PVS). The model is trained on a single A100-40G GPU using the Adam optimizer with a batch size of 128. The default number of diffusion steps is $T=10$, with 4–8 registers.

## Key Experimental Results

### Main Results

| Method | ActivityNet SumR | Charades SumR | TVR SumR |
|--------|-----------------|---------------|----------|
| MS-SL | 140.1 | 68.4 | 172.4 |
| GMMFormer | 146.0 | 72.9 | 176.6 |
| HLFormer | 154.9 | 78.7 | 187.7 |
| GMMFormerV2 | 154.9 | 78.2 | 189.1 |
| **DreamPRVR** | **156.1** | **80.0** | **193.1** |

**DreamPRVR individual metrics on Charades-STA**:

| Metric | R@1 | R@5 | R@10 | R@100 |
|--------|-----|-----|------|-------|
| HLFormer | 2.6 | 8.5 | 13.7 | 54.0 |
| **DreamPRVR** | **2.6** | **8.7** | **14.5** | **54.2** |

### Ablation Study

| Configuration | ActivityNet SumR | Charades SumR | TVR SumR | Note |
|---------------|-----------------|---------------|----------|------|
| Full DreamPRVR | **156.1** | **80.0** | **193.1** | Complete model |
| w/o registers | 153.4 | 76.8 | 187.0 | No global registers |
| w/ adaptive pooling | 151.9 | 78.1 | 191.4 | Simple pooling replacing diffusion |
| w/o DRE | 150.6 | 78.3 | 190.8 | No diffusion iterative refinement |
| w/o PVS | 154.9 | 77.6 | 190.9 | Random noise initialization |
| $L_{sim}$ only | 150.5 | 76.6 | 187.0 | Retrieval loss only |
| w/o $L_{tssl}$ | 151.3 | 76.9 | 191.1 | No text structure learning |

### Key Findings

- Removing registers causes Charades SumR to drop from 80.0 to 76.8 (−3.2) and TVR SumR from 193.1 to 187.0 (−6.1), confirming the value of global context.
- Adaptive pooling (−1.9) is substantially inferior to diffusion-based generation, demonstrating that simple aggregation is insufficient to extract reliable global semantics from noisy videos.
- PVS's video-centric initialization outperforms random noise initialization (Charades 80.0 vs. 77.6), validating the necessity of truncated diffusion.
- Performance improves steadily as diffusion steps $T$ increase from 2 to 10, then declines beyond $T>10$, suggesting that over-refinement may lead to overfitting.
- 4–8 registers are optimal; excessive registers introduce redundancy and degrade performance.
- t-SNE visualizations clearly show registers transitioning from an initially disordered state to compact video-level clusters.

## Highlights & Insights

- **Cognitive analogy of "imagine before concentrate"**: The diffusion generation process is analogized to the cognitive "imagination" phase (forming coarse-grained global perception), while fine-grained matching corresponds to the "concentration" phase — an elegant and intuitive conceptual design.
- **Efficient use of truncated diffusion**: Rather than large-scale diffusion models, a lightweight MLP with 6–8 registers and 10 diffusion steps achieves significant gains, demonstrating that the diffusion paradigm can be highly efficient for retrieval tasks with acceptable training and inference overhead.
- **Complementary design of the QSP loss**: Treating multiple queries from the same video as positive pairs rather than dispersing them independently is a principled correction to existing query diversity losses.

## Limitations & Future Work

- The method relies on pre-extracted I3D features and does not explore end-to-end training or stronger visual encoders (e.g., CLIP ViT).
- The number of registers and diffusion steps require dataset-specific tuning (4 for ActivityNet, 8 for TVR).
- The conditioning signal $c$ for the diffusion model is obtained via simple cross-attention from video features, which may lack sufficient richness.
- Future work could consider extending this framework to Video Corpus Moment Retrieval (VCMR).

## Related Work & Insights

- **vs. GMMFormer (Gaussian attention PRVR)**: DreamPRVR builds upon its Gaussian attention mechanism with register augmentation, achieving approximately +7 SumR improvement on Charades.
- **vs. HLFormer (hyperbolic space + semantic entailment)**: HLFormer's global context serves only as a training-time regularizer, whereas DreamPRVR's registers also participate in feature enhancement at inference time.
- **vs. DiffusionRet / DiffDis (diffusion-based retrieval)**: These works apply diffusion to model the joint query-candidate distribution, while DreamPRVR uses diffusion to generate global registers — representing a novel fusion of generative and discriminative paradigms.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of generating registers via diffusion for retrieval is novel, with an elegant conceptual design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, 12+ baselines, comprehensive ablations, efficiency analysis, and multiple visualizations.
- Writing Quality: ⭐⭐⭐⭐ The variational inference framework derivation is complete and the figures are clear.
- Value: ⭐⭐⭐⭐ Establishes a new generative-discriminative fusion paradigm for PRVR; the register approach is transferable to other tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OpenDPR: Open-Vocabulary Change Detection via Vision-Centric Diffusion-Guided Prototype Retrieval for Remote Sensing Imagery](opendpr_open-vocabulary_change_detection_via_vision-centric_diffusion-guided_pro.md)
- [\[CVPR 2026\] Learnability-Guided Diffusion for Dataset Distillation](learnability-guided_diffusion_for_dataset_distillation.md)
- [\[ICCV 2025\] DiffDoctor: Diagnosing Image Diffusion Models Before Treating](../../ICCV2025/image_generation/diffdoctor_diagnosing_image_diffusion_models_before_treating.md)
- [\[CVPR 2026\] SPDMark: Selective Parameter Displacement for Robust Video Watermarking](spdmark_selective_parameter_displacement_for_robust_video_watermarking.md)
- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
