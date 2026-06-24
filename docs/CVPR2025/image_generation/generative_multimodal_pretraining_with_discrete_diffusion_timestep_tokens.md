---
title: >-
  [Paper Note] Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens
description: >-
  [CVPR 2025][Image Generation][Multimodal Large Language Models] DDT-LLaMA proposes using diffusion timestep encoding to learn discrete visual tokens (DDT) with a recursive structure. This endows visual token sequences with hierarchical dependencies similar to natural language, thereby achieving SOTA performance in both multimodal understanding and generation under a unified next-token-prediction framework.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Multimodal Large Language Models"
  - "Diffusion Timestep Tokens"
  - "Visual Tokenizer"
  - "Recursive Structure"
  - "Unified Understanding & Generation"
date: 2026-05-08
content_hash: cc1f777e5ec6f8ff
---

# Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens

**Conference**: CVPR 2025  
**arXiv**: [2504.14666](https://arxiv.org/abs/2504.14666)  
**Code**: [https://DDT-LLaMA.github.io/](https://DDT-LLaMA.github.io/) (Project Page)  
**Area**: Multimodal VLM / Image Generation  
**Keywords**: Multimodal Large Language Models, Diffusion Timestep Tokens, Visual Tokenizer, Recursive Structure, Unified Understanding & Generation

## TL;DR
DDT-LLaMA proposes using diffusion timestep encoding to learn discrete visual tokens (DDT) with a recursive structure. This endows visual token sequences with hierarchical dependencies similar to natural language, thereby achieving SOTA performance in both multimodal understanding and generation under a unified next-token-prediction framework.

## Background & Motivation

1. **Background**: Multimodal large language models (MLLMs) aim to perform visual understanding and generation simultaneously within a unified next-token-prediction paradigm. Existing approaches are categorized into cascaded architectures (e.g., EMU2, which trains separate understanding and generation modules and concatenates them) and tokenization methods (e.g., Emu3, which quantizes images into discrete tokens for joint training).
2. **Limitations of Prior Work**: All existing methods rely on spatial visual tokens (arranged in raster-scan order of image patches). However, spatial token sequences lack the recursive structure inherent in human language. Experiments show that shuffling the order of spatial tokens has almost no effect on LLM training convergence—a behavior starkly different from that of language tokens. This suggests that LLMs cannot model spatial tokens effectively.
3. **Key Challenge**: There is a conflict of objectives between understanding and generation: understanding is a many-to-one mapping (e.g., various photos of Corgis $\rightarrow$ "Corgi"), whereas generation is a one-to-all mapping (retaining all visual details). To excel at both in a unified framework, visual tokens must possess language-like modelability.
4. **Goal**: Design a visual tokenizer that embeds a recursive structure into visual tokens, enabling LLMs to process visual information as naturally as language.
5. **Key Insight**: During the forward process of diffusion models, as the timestep increases, Gaussian noise progressively destroys the visual attributes of the image. The authors leverage this "progressive attribute loss" process to learn a token sequence with a recursive compensation structure, where each new token compensates for the visual attributes lost at the current timestep.
6. **Core Idea**: Encode the "progressive information loss" corresponding to diffusion timesteps into a recursive visual token sequence, giving it hierarchical dependencies similar to natural language.

## Method

### Overall Architecture
The system consists of two parts: (1) The DDT Tokenizer, which encodes an image into a recursive sequence of discrete tokens; (2) DDT-LLaMA, an MLLM based on LLaMA3-8B trained with unified next-token prediction. During inference, the LLM predicts the visual tokens, which are then reconstructed into an image using a diffusion model as the decoder.

### Key Designs

1. **DDT Encoder (Diffusion Timestep Encoder)**:
    - **Function**: Encodes a noise-free image into a sequence of T recursive discrete tokens.
    - **Mechanism**: A Transformer encoder processes image patch tokens and T learnable query tokens. An MMDiT-style architecture with dual independent Transformers and shared attention (similar to SD3) is employed, and only the outputs of the query tokens $(\hat{V}_1, ..., \hat{V}_T)$ are retained. Crucially, the process is recursive: $f_{t+1}(x_0) = (f_t(x_0), V_{t+1})$, meaning the $(t+1)$-th token sequence appends a new token to the $t$-th sequence to compensate for the newly lost visual attributes from $x_t$ to $x_{t+1}$.
    - **Design Motivation**: Unlike spatial tokens, the order of DDT tokens has explicit semantics—moving from coarse-grained (global layout, dominant colors) to fine-grained (textures, edges), presenting a recursive nested structure similar to language.

2. **Vector Quantization**:
    - **Function**: Discretizes continuous encoder outputs into tokens from a fixed vocabulary.
    - **Mechanism**: A standard VQ module is used with a codebook size of 65,536 and a dimension of m=16. An EMA update strategy is applied to improve training stability. Two tricks are introduced to increase codebook usage: (a) a low dimension of m=16 (validated in literature); (b) monitoring dead entries and replacing them with random encoder outputs from training batches.
    - **Design Motivation**: Discrete tokens directly leverage the vocabulary expansion mechanism of LLMs, enabling truly unified training.

3. **Diffusion Decoder**:
    - **Function**: Reconstructs the image by guiding the denoising process based on DDT tokens.
    - **Mechanism**: Based on the MMDiT architecture. At timestep $t$, only the first $t$ tokens $(V_1, ..., V_t)$ are used as guiding conditions (the remaining are masked to 0). The decoder learns to reconstruct $\hat{x}_0 = d(x_t, t, (V_1, ..., V_t))$. The training objective combines the reconstruction loss under Rectified Flow and the standard commitment loss. During inference, images are generated from pure noise via a T-step DDPM process.
    - **Design Motivation**: Expanding token inputs naturally aligns with the timestep-based denoising of diffusion models—heavier noise requires more tokens to compensate for information loss.

### Loss & Training
Tokenizer training: Reconstruction loss $\mathcal{L} = \mathbb{E}[\|d(t\epsilon + (1-t)x_0, t, (V_1,...,V_t)) - x_0\|^2]$ + commitment loss. Trained only on ImageNet. Two-stage MLLM training: (1) Pre-training phase aligns DDT tokens with text tokens using 200M image-text pairs (Laion+Coyo), training on 512 Ascend 910B NPUs for about two weeks; (2) Instruction tuning phase trains on multimodal tasks using open-source datasets. The vocabulary is expanded with 65,536 visual codes.

## Key Experimental Results

### Main Results

**Text-to-Image Generation (MLLM-based Generalist)**

| Benchmark | Metric | DDT-LLaMA | Prev. SOTA (Emu3) | Gain |
|-----------|------|-----------|----------|------|
| GenEval | Overall↑ | **0.66** | 0.54 | +0.12 |
| GenEval | Counting↑ | **0.56** | 0.34 | +0.22 |
| GenEval | Position↑ | **0.39** | 0.17 | +0.22 |
| GenEval | ColorAttri↑ | **0.48** | 0.21 | +0.27 |
| T2I-CompBench | Color↑ | **0.728** | 0.611 | +0.117 |

**Zero-shot Image Editing**

| Benchmark | Metric | DDT-LLaMA | UltraEdit (Specialist) |
|-----------|------|-----------|----------|
| MagicBrush | L1↓ | **5.4** | 6.6 |
| MagicBrush | CVS↑ | **92.9** | 88.4 |

### Ablation Study

| Configuration | Description |
|------|------|
| Spatial tokens (shuffled sequence) | Training curve barely changes $\rightarrow$ not a good "visual language" |
| DDT tokens (shuffled sequence) | Training curve degenerates significantly $\rightarrow$ possesses language-like sequential dependence |
| Training tokenizer on ImageNet only | Sufficient, proving that recursive structure is more important than data scale |

### Key Findings
- The language validation experiment for DDT tokens is highly convincing: shuffling the DDT token sequence significantly degrades the LLM training curve (consistent with language behavior), whereas spatial tokens remain unaffected.
- Training the tokenizer solely on ImageNet outperforms tokenizers trained on large-scale datasets (Laion/Coyo), demonstrating that the recursive structure itself is key.
- Achieving an Overall score of 0.66 on GenEval is close to the 0.67 of the specialized T2I model DALL-E 3, vastly outperforming all other MLLM methods.
- Outperforming even specialized image editing models (like UltraEdit) on image editing tasks demonstrates the strengths of a unified framework.

## Highlights & Insights
- **Leveraging diffusion timesteps to construct recursive tokens is an insightful innovation**: It cleverly connects the progressive denoising process of diffusion models with the recursive structure of language, introducing a completely new visual quantization paradigm. Starting from the observation that "spatial tokens are not a good visual language," the authors analyze the reasons and propose a solution, making the overall research logic exceptionally complete.
- **Language validation experiment**: Evaluating the "language-likeness" of tokens through sequence perturbation experiments is a generalizable evaluation methodology that can be used to assess any tokenization scheme attempting to unify non-text modalities into LLMs.
- **Successfully training the tokenizer on ImageNet alone**: This shows that the structure of the tokens is more crucial than the scale of the training data, providing important guidance for future tokenizer design.

## Limitations & Future Work
- The token sequence length is tied to the number of diffusion timesteps T, requiring a trade-off between sequence length and reconstruction quality.
- The MMDiT-based tokenizer architecture is heavy, potentially yielding lower encoding efficiency compared to simple VQ-VAEs.
- Currently handles only image modalities; extending to higher-dimensional modalities like video and 3D remains to be explored.
- Training the MLLM demands massive computational resources (512 NPUs for two weeks), limiting its practicality.
- While visual understanding has improved, there is still a performance gap compared to specialized understanding models (such as LLaVA-1.6).

## Related Work & Insights
- **vs Emu3**: Emu3 uses standard spatial VQ tokens for unified training, whereas DDT-LLaMA replaces them with recursive timestep tokens, achieving significantly better results in both generation and understanding.
- **vs Transfusion**: Transfusion mixes autoregressive and diffusion objectives inside a single Transformer but still relies on spatial tokens. DDT's advantage is that the tokens themselves possess a recursive structure.
- **vs VILA-U**: VILA-U attempts to unify understanding and generation tokens but is also limited by the non-recursive nature of spatial tokens.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The insight of mapping diffusion timesteps to recursive visual language is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple tasks including generation, editing, and understanding.
- Writing Quality: ⭐⭐⭐⭐⭐ Deep motivation analysis and compelling language validation experiments.
- Value: ⭐⭐⭐⭐⭐ Offers a new tokenization paradigm for unified multimodal models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DoraCycle: Domain-Oriented Adaptation of Unified Generative Model in Multimodal Cycles](doracycle_domain-oriented_adaptation_of_unified_generative_model_in_multimodal_c.md)
- [\[CVPR 2026\] RebRL: Reinforcing Discrete Visual Diffusion Models with Rebalanced Timestep Credits](../../CVPR2026/image_generation/rebrl_reinforcing_discrete_visual_diffusion_models_with_rebalanced_timestep_cred.md)
- [\[ICML 2025\] Efficient Generative Modeling with Residual Vector Quantization-Based Tokens](../../ICML2025/image_generation/efficient_generative_modeling_with_residual_vector_quantization-based_tokens.md)
- [\[CVPR 2025\] VLOGGER: Multimodal Diffusion for Embodied Avatar Synthesis](vlogger_multimodal_diffusion_for_embodied_avatar_synthesis.md)
- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](../../ICCV2025/image_generation/timestep-aware_diffusion_model_for_extreme_image_rescaling.md)

</div>

<!-- RELATED:END -->
