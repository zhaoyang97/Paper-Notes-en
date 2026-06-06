---
title: >-
  [Paper Note] SweetTok: Semantic-Aware Spatial-Temporal Tokenizer for Compact Video Discretization
description: >-
  [ICCV 2025][Video Generation][video discretization] This paper proposes SweetTok, a video tokenizer that decouples spatial and temporal information compression via a Decoupled Query AutoEncoder (DQAE)…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "video discretization"
  - "spatial-temporal decoupling"
  - "vector quantization"
  - "language codebook"
date: 2026-05-08
content_hash: 7a9901ee28a02f3b
---

# SweetTok: Semantic-Aware Spatial-Temporal Tokenizer for Compact Video Discretization

**Conference**: ICCV 2025
**arXiv**: [2412.10443](https://arxiv.org/abs/2412.10443)  
**Code**: None  
**Area**: Video Generation
**Keywords**: video discretization, spatial-temporal decoupling, vector quantization, language codebook, video generation

## TL;DR

This paper proposes SweetTok, a video tokenizer that decouples spatial and temporal information compression via a Decoupled Query AutoEncoder (DQAE), and assigns codewords by part-of-speech through a Motion-enhanced Language Codebook (MLC). Using only 25% of the token count, SweetTok achieves 42.8% improvement in rFVD and 15.1% improvement in gFVD, attaining an optimal balance between compression ratio and reconstruction fidelity.

## Background & Motivation

Visual tokenizers are critical components in modern visual generation and understanding models. Current video tokenizers face two core challenges:

**Low compression ratio**: Conventional methods generate tokens based on 2D patches or 3D tubes, where each token corresponds to a specific spatial region, resulting in redundancy across both spatial and temporal dimensions. For example, OmniTokenizer requires 5,120 tokens to represent a 17-frame video clip. The recent LARP achieves high compression by adaptively querying flattened video patches, but directly flattening video tokens interleaves spatial and temporal information, increasing learning difficulty and degrading reconstruction performance.

**Detail loss under high compression**: Introducing pre-trained language embeddings as codebooks is a common strategy to compensate for compression loss; however, existing work primarily focuses on the image modality and neglects the relationship between **textual and motion information** in videos.

**Core Insight**: Static appearance and dynamic motion in videos are fundamentally distinct—the nature of spatial redundancy differs from that of temporal redundancy. The two should be **decoupled** and compressed separately, rather than handled jointly.

## Method

### Overall Architecture

SweetTok consists of two core components:

1. **Decoupled Query AutoEncoder (DQAE)**: Compresses spatial and temporal information independently via separate spatial and temporal queries.
2. **Motion-enhanced Language Codebook (MLC)**: Partitions the language codebook into spatial (nouns + adjectives) and temporal (verbs + adverbs) subsets based on part-of-speech.

### Key Designs

1. **Decoupled Query AutoEncoder (DQAE)**

   **Patchify Stage**: Given a video $x \in \mathbb{R}^{T \times H \times W \times 3}$, the first frame is selected as the spatial reference, and the remaining $T-1$ frames are used for temporal information. Two distinct patch kernels are applied:
    - Spatial: $\mathcal{P}_s$ of shape $p_h \times p_w$, transforming the first frame into $v_s \in \mathbb{R}^{1 \times 32 \times 32 \times D}$
    - Temporal: $\mathcal{P}_t$ of shape $p_t \times p_h \times p_w$, transforming subsequent frames into $v_t \in \mathbb{R}^{4 \times 32 \times 32 \times D}$

   **Spatial Tokenization**: The first-frame patches $v_s$ are compressed by the spatial encoder $\mathcal{E}_{DQAE_s}$ into $L_{spatial}=256$ learnable spatial queries $\mathbf{Q_s}$:
    $\mathbf{Z_{Q_s}} = \mathcal{E}_{DQAE_s}(\mathbf{Q_s}, v_s), \quad \tilde{\mathbf{Z}}_{Q_s} = \mathcal{Q}_{MLC}(\mathbf{Z_{Q_s}})$
   The spatial decoder then reconstructs the first-frame patches: $\tilde{v}_s = \mathcal{D}_{DQAE_s}(\mathbf{Q}_{v_s}, \tilde{\mathbf{Z}}_{Q_s})$

   **Temporal Tokenization**: Exploiting the substantial redundancy along the temporal dimension, **inter-frame residuals** $\Delta v_t^i = v_s^i - v_t^i$ are adopted as temporal inputs, since the spatial stage has already reconstructed the first frame. The residuals are compressed into $L_{temporal}=1024$ temporal queries:
    $\mathbf{Z_{Q_t}} = \mathcal{E}_{DQAE_t}(\mathbf{Q_t}, \Delta v), \quad \tilde{\mathbf{Z}}_{Q_t} = \mathcal{Q}_{MLC}(\mathbf{Z_{Q_t}})$

   **Decoding Strategy: Spatial First, Then Temporal**. The reconstructed first frame $\tilde{v}_s$ is replicated $t$ times and fed into the temporal decoder together with the quantized temporal residuals:
    $\tilde{v} = \mathcal{D}_{DQAE_t}([\tilde{v}_s \| \cdots \| \tilde{v}_s], \tilde{\mathbf{Z}}_{Q_t})$
   The final video is reconstructed through a pixel decoder $\mathcal{D}_{pixel}$.

   **Design Motivation**: Joint spatial-temporal compression increases the difficulty for the decoder to learn cross-frame motion of the same pixels. By decoupling the two, spatial queries capture static appearance and temporal queries capture dynamic variations without interference.

2. **Motion-enhanced Language Codebook (MLC)**

   **Core Idea**: The vocabulary is partitioned into four part-of-speech categories—nouns and adjectives (corresponding to static spatial information) and verbs and adverbs (corresponding to temporal motion information).

   Implementation details:
    - Candidate words are extracted from video captions in the dataset.
    - CLIP text encoder embeddings are extracted to construct the codebook $C \in \mathbb{R}^{L \times D}$.
    - A graph convolutional network $\mathcal{F}$ projects CLIP embeddings into the visual latent space, where graph edges are constructed from word co-occurrence pairs within the same caption.

   During quantization, spatial queries retrieve nearest neighbors from the noun+adjective codebook, while temporal queries retrieve from the verb+adverb codebook:
    $\hat{z}_s = \mathcal{F}(c_i), \quad i = \arg\min_{c_i \in C_{noun} \cup C_{adj}} \|z_s - \mathcal{F}(c_i)\|$
    $\hat{z}_t = \mathcal{F}(c_i), \quad i = \arg\min_{c_i \in C_{verb} \cup C_{adv}} \|z_t - \mathcal{F}(c_i)\|$

   Codebook sizes: 10,481 for spatial (5,078 nouns + 5,403 adjectives) and 11,139 for temporal (9,267 verbs + 1,872 adverbs).

3. **Flexibility of the Decoupled Design**

   The spatial branch can be fine-tuned independently on ImageNet to yield a strong image tokenizer. Furthermore, the encoded tokens inherently carry semantic information, enabling direct use for few-shot inference with LLMs.

### Loss & Training

The reconstruction loss comprises four terms: $\mathcal{L}_{rec} = \mathcal{L}_{L2} + \mathcal{L}_{Lpips} + \mathcal{L}_{vq} + \mathcal{L}_g$

- $\mathcal{L}_{L2}$: Pixel-level L2 reconstruction loss
- $\mathcal{L}_{Lpips}$: LPIPS perceptual loss
- $\mathcal{L}_{vq}$: Vector quantization commitment loss (spatial + temporal)
- $\mathcal{L}_g$: GAN adversarial loss

Training configuration: 8 × A100 GPUs, batch size 8, 1,000K iterations, Adam optimizer, cosine learning rate schedule (max 1e-4, min 1e-5).

## Key Experimental Results

### Main Results

Video reconstruction (UCF-101, 256×256):

| Tokenizer | #Tokens | rFVD↓ |
|-----------|---------|-------|
| OmniTok | 5120 | 42 |
| LARP-B | 1024 | 64 |
| LARP-L | 1024 | 35 |
| **SweetTok** | **1280** | **20** |
| SweetTok* (no compression) | 5120 | 11 |

Video generation (UCF-101, class-conditional):

| Tokenizer | Generator | #Tokens | gFVD↓ |
|-----------|-----------|---------|-------|
| OmniTok | AR, 650M | 5120 | 191 |
| LARP-L | AR, 632M | 1024 | 99 |
| **SweetTok** | **AR, 650M** | **1280** | **84** |
| SweetTok | AR, 1.9B | 1280 | 65 |

Image reconstruction (ImageNet, 256×256, 256 tokens):

| Tokenizer | rFID↓ |
|-----------|-------|
| TiTok | 1.71 |
| TokenFlow | 1.03 |
| **SweetTok** | **0.73** |

### Ablation Study

Comparison of compression methods:

| Method | #Tokens | rFVD↓ | Note |
|--------|---------|-------|------|
| Vanilla downsampling | 1280 | 227.65 | Linear interpolation performs poorly |
| Vanilla query (LARP) | 1024 | 35.15 | Unified compression after flattening |
| **DQAE (decoupled query)** | **1280** | **20.46** | Decoupling yields significant gains |

Codebook ablation:

| Method | rFVD↓ | Note |
|--------|-------|------|
| Baseline (no LC) | 29.45 | No language codebook |
| + LC (standard language codebook) | 24.80 | Semantic information is beneficial |
| **+ MLC (motion-enhanced)** | **20.46** | Motion vocabulary is critical |
| + Qwen-based MLC | 20.12 | Marginal gain from more complex LLM |

### Key Findings

- DQAE improves over the flattened query baseline by 42% (35.15→20.46), validating the effectiveness of decoupled compression.
- MLC reduces rFVD by an additional 17.5% over a standard language codebook, confirming that the motion-enhanced temporal codebook is essential.
- Visual semantic understanding: few-shot image classification achieves 90.8% (miniImageNet) and video action recognition achieves 90.1% (UCF-101), demonstrating that the encoded tokens capture rich semantic content.
- Scaling law behavior is observed: expanding the generator from 650M to 1.9B parameters reduces gFVD from 84 to 65.

## Highlights & Insights

- **Divide-and-conquer decoupled compression**: Spatial and temporal redundancies have fundamentally different characteristics; compressing them separately is more efficient than joint compression.
- **Part-of-speech-based codebook assignment**: Nouns and adjectives capture appearance; verbs and adverbs capture motion—this structured language-vision alignment is both intuitive and effective.
- **One tokenizer, multiple tasks**: A single model covers reconstruction, generation, image processing, and few-shot understanding.
- **Clever use of inter-frame residuals**: Temporal tokenization operates on residuals rather than raw frames, naturally eliminating temporal redundancy.

## Limitations & Future Work

- The ratio of spatial tokens (256) to temporal tokens (1,024) is fixed; different video content may have different optimal allocations.
- Only fixed-length videos (17 frames) are supported; adaptive-length support is worth exploring.
- The graph convolutional network for projecting text embeddings is relatively heavy; lightweight alternatives present an opportunity for optimization.
- A continuous (non-discrete) latent space variant has not been explored.

## Related Work & Insights

- TiTok's 1D token compression motivated the query-based compression paradigm, but lacked special treatment for the temporal dimension of videos.
- LARP's adaptive query mechanism is adopted and extended in this work into a decoupled variant.
- The part-of-speech codebook assignment strategy is potentially generalizable to other modalities such as audio (frequency-related words vs. rhythm-related words).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Both the decoupled design and the motion-enhanced codebook are original contributions
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across reconstruction, generation, image processing, and understanding, with detailed ablations
- Writing Quality: ⭐⭐⭐⭐ Clear structure and thorough comparisons
- Value: ⭐⭐⭐⭐⭐ Significant contribution to the video tokenizer field, achieving state-of-the-art in both compression efficiency and quality

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning](disentangled_world_models_learning_to_transfer_semantic_knowledge_from_distracti.md)
- [\[CVPR 2026\] TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models](../../CVPR2026/video_generation/tear_temporal-aware_automated_red-teaming_for_text-to-video_models.md)
- [\[ICCV 2025\] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)
- [\[CVPR 2026\] Semantic Satellite Communications for Synchronized Audiovisual Reconstruction](../../CVPR2026/video_generation/semantic_satellite_communications_for_synchronized_audiovisual_reconstruction.md)
- [\[CVPR 2026\] AutoCut: End-to-end Advertisement Video Editing Based on Multimodal Discretization and Controllable Generation](../../CVPR2026/video_generation/autocut_end-to-end_advertisement_video_editing_based_on_multimodal_discretizatio.md)

</div>

<!-- RELATED:END -->
