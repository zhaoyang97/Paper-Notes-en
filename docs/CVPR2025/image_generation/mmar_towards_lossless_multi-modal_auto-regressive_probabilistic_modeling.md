---
title: >-
  [Paper Note] MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling
description: >-
  [CVPR 2025][Image Generation][multi-modal auto-regressive] This work integrates continuous image representations and discrete text representations into a unified autoregressive probabilistic modeling framework for the first time. It avoids information loss by replacing VQ discretization with a lightweight diffusion head, and derives v-prediction as the optimal parameterization to address numerical error issues in low-precision training.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "multi-modal auto-regressive"
  - "continuous image tokens"
  - "diffusion head"
  - "v-prediction"
  - "joint probability modeling"
  - "lossless compression"
date: 2026-05-08
content_hash: 08c125ead7730578
---

# MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling

**Conference**: CVPR 2025  
**arXiv**: [2410.10798](https://arxiv.org/abs/2410.10798)  
**Code**: [GitHub](https://github.com/ydcUstc/MMAR)  
**Area**: Image Generation  
**Keywords**: multi-modal auto-regressive, continuous image tokens, diffusion head, v-prediction, joint probability modeling, lossless compression

## TL;DR

This work integrates continuous image representations and discrete text representations into a unified autoregressive probabilistic modeling framework for the first time. It avoids information loss by replacing VQ discretization with a lightweight diffusion head, and derives v-prediction as the optimal parameterization to address numerical error issues in low-precision training.

## Background & Motivation

**Background**: Supporting both image understanding and generation in multimodal large models is a popular research direction. Existing joint probability modeling methods fall into three categories:
1. **Discrete AR** (Chameleon, EMU3): VQ-VAE discretizes images → codebook size limits the information capacity of each token, leading to a demand for a large number of tokens (e.g., EMU3 requires 4096 tokens/image).
2. **Discrete Diffusion** (Show-o): Discrete tokens + diffusion modeling.
3. **Continuous Diffusion** (Transfusion, MonoFormer): Continuous representation + DiT diffusion modeling.

**Key Challenge**:
- **Discrete Methods**: Codebook size bottleneck leads to information loss, causing understanding performance to lag behind LLaVA, which uses continuous CLIP representations.
- **Continuous Diffusion Methods (Transfusion)**: Diffusion models encode image information across different noise levels. Feeding clean images only activates representation levels related to low-noise "image enhancement". The authors' experiments reveal that different understanding tasks achieve optimality at different non-zero noise levels (e.g., object localization tasks perform better at high noise levels), indicating that clean inputs cannot fully exploit the learned image modeling capability.

**Key Insight**: This work proposes a continuous AR method—directly performing autoregressive modeling on continuous image tokens and sampling continuous representations for each image patch through a lightweight diffusion head. Its autoregressive nature ensures that all image hidden representations are optimized simultaneously for both generation and understanding.

## Method

### Overall Architecture

MMAR extends a pretrained decoder-only LLM as follows:

- **Text Part**: Standard autoregression, softmax + cross-entropy loss.
- **Image Part**: Masked AR (bidirectional encoder-decoder), using a lightweight Diffusion MLP head at each patch position to sample continuous image tokens from the conditional representation $z_i = f_\theta(x_{<i})$.
- **Unified Modeling**: [text][img] → P(I|T); [img][text] → P(T|I); [img] → P(I).

### Key Designs

#### 1. Continuous Tokens + Diffusion Head

For an image token $x_i$, conditioned on the LLM backbone output $z_i$, a lightweight MLP diffusion model is used to predict:

$$L_i = \mathbb{E}_{x_i, \epsilon, t}[w_t \cdot \|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_i + \sqrt{1-\bar\alpha_t}\epsilon, t, z_i)\|^2]$$

Key Advantages:
- **Lossless Information**: The continuous tokenizer can compress a 128×128 patch into a single continuous token (requiring only 16 tokens/512×512 image, compared to 4096 in EMU3).
- **Decoupling Diffusion from Backbone**: The diffusion process is conducted solely within the MLP head of each patch. The backbone remains unaffected by noise levels, ensuring all hidden representations are fully available for understanding tasks.

#### 2. v-prediction Optimal Parameterization

It is derived from first principles of minimizing numerical errors that v-prediction is the optimal parameterization under low-precision training:

**Analysis of the Problem**: In bfloat16 training, the numerical error of floating-point representations is proportional to the magnitude of the value. Under $\epsilon$-prediction, the error term of DDIM sampling is:

$$| \sqrt{1-\bar\alpha_{t-1}} - \frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}}\sqrt{1-\bar\alpha_t} | \cdot \delta \sigma_t$$

When the SNR is extremely low ($\bar\alpha_t \to 0$), the error approaches infinity.

**Solution**: Employing the v-prediction parameterization $v_i^{(t)} = \sqrt{\bar\alpha_t}\epsilon - \sqrt{1-\bar\alpha_t}x_i$, which is theoretically proven to minimize numerical errors at all noise levels.

#### 3. Two-Stage Training Strategy

- **Stage 1 (Image Expert Pretraining)**: Large-scale medium-quality data (Capfusion-120M), mask ratio [0.7, 1.0], to learn diverse visual understanding.
- **Stage 2 (Image Expert Fine-tuning)**: Small-scale high-quality data (CC12M + LAION-aesthetics), mask ratio range expanded to [0, 1.0].

**Identified Problem**: Images generated at the end of Stage 1 exhibit "holes"—because the lower bound of the mask ratio was set to 0.7, preventing the model from fully learning scenarios with high completion rates (the last 30%). Stage 2 addresses this by adjusting the lower bound of the mask ratio to 0.

### Loss & Training

$$L = \sum_{i \in I_{img}} L_i - \sum_{i \in I_{txt}} \log p_\theta(x_i | x_{<i})$$

The image part utilizes v-prediction diffusion loss, while the text part employs standard cross-entropy.

## Key Experimental Results

### Main Results (Average of 18 Visual Understanding Benchmarks)

| Method | LLM | V-Token | AVE@18Und. |
|------|-----|---------|------------|
| Chameleon-7B | 7B scratch | vq-vae | 18.34 |
| Transfusion* | Qwen2-0.5B | vae | 28.26 |
| Show-o | Phi-1.5B | CLIP | 33.06 |
| VILA-U | LLaMA-2-7B | vq-vae | — |
| **MMAR-0.5B** | **Qwen2-0.5B** | **vae** | **34.56** |
| **MMAR-7B** | **Qwen2-7B** | **vae** | **48.25** |
| LLaVA-1.5 (Understanding only) | Vicuna-7B | CLIP | 47.08 |

MMAR-7B (without CLIP, 256x256 only) nearly matches LLaVA-1.5, which uses pretrained CLIP.

### Ablation Study

| Setting | MMB | MMEP | AVE@18Und. | FID-30K ↓ |
|------|-----|------|------------|-----------|
| MMAR-0.5B (v-pred) | 48.45 | 882.1 | 34.56 | 36.6 |
| w/ ε-prediction | 45.53 | 880.7 | 32.21 | 61.53 |
| Show-o-like (w/VQ) | 37.54 | 618.2 | 29.70 | 66.26 |
| Transfusion-like | 29.47 | 594.3 | 28.26 | 95.38 |

### Key Findings

1. **v-prediction vs \epsilon-prediction**: v-prediction comprehensively outperforms \epsilon-prediction in both understanding (+2.35 AVE) and generation (FID 36.6 vs 61.53).
2. **Continuous vs Discrete**: MMAR (continuous) significantly outperforms Show-o-like (VQ discrete), validating the information loss caused by discretization.
3. **MMAR vs Transfusion**: At equivalent scales, MMAR comprehensively outperforms Transfusion-like methods in both understanding and generation.
4. **Only 16 tokens/image**: Achieves comparable performance to EMU3 (which uses 4096 tokens) with an extremely small token count.
5. **Scalability**: Significant performance improvements from 0.5B to 7B models indicate that the method effectively scales with model and data size.

## Highlights & Insights

1. **Significant Theoretical Contribution**: Derives the optimality of v-prediction from first principles of minimizing numerical errors, offering a fresh understanding of diffusion model theory.
2. **Highly Minimalist Unified Framework**: The diffusion head is merely an MLP, decoupling the diffusion process from the backbone. This design is simple and efficient.
3. **In-depth Analysis of Transfusion's Limitations**: Reveals the fundamental flaw of continuous diffusion methods in understanding tasks through experiments (i.e., different noise levels encode different information).
4. **High Practical Value**: Representing a 512×512 image with only 16 continuous tokens carries significant implications for inference efficiency.

## Limitations & Future Work

1. **Gap in Generation Quality**: FID metrics have not reached the level of dedicated generation models, requiring additional high-quality data training.
2. **Evaluated only at 256x256 Resolution**: Performances under higher resolutions remain unexplored.
3. **Limitations of Masked AR**: The image part uses masked AR instead of causal AR, which is not fully consistent with text AR.
4. **EmbeddingViT Module**: Introducing an additional encoder module increases the overall parameter count.
5. **Complexity of Two-stage Training**: Different mask ratios and data mixtures require careful tuning.

## Related Work & Insights

- **MAR**: The diffusion head of MMAR is directly inspired by MAR, but generalizes it from ImageNet class generation to LLM + large-scale image-text data.
- **Transfusion**: A direct competitor to MMAR; experiments demonstrate its incomplete information utilization issues.
- **Chameleon / EMU3**: Representatives of discrete AR; MMAR demonstrates a more efficient alternative using continuous tokens.
- **Insight**: The concept of "decoupling the diffusion process from the backbone" can be extended to any scenario that requires modeling continuous distributions within an AR framework.

## Rating ⭐

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Engineering Practicality | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficient](collaborative_decoding_makes_visual_auto-regressive_modeling_efficient.md)
- [\[CVPR 2025\] HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation](hmar_efficient_hierarchical_masked_auto-regressive_image_generation.md)
- [\[CVPR 2025\] Generative Modeling of Class Probability for Multi-Modal Representation Learning](generative_modeling_of_class_probability_for_multi_modal_representation_learning.md)
- [\[CVPR 2025\] SyncVP: Joint Diffusion for Synchronous Multi-Modal Video Prediction](syncvp_joint_diffusion_for_synchronous_multi-modal_video_prediction.md)
- [\[CVPR 2025\] OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows](omniflow_any-to-any_generation_with_multi-modal_rectified_flows.md)

</div>

<!-- RELATED:END -->
