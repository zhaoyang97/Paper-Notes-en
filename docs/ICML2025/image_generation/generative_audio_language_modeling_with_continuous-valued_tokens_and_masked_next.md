---
title: >-
  [Paper Note] Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction
description: >-
  [ICML 2025][Image Generation][Audio Generation] This paper studies causal language models for audio generation without using discrete tokens, leveraging token-wise diffusion to model the distribution of continuous-valued next-tokens, and proposes a masked next-token prediction task. With 193M parameters, it achieves performance comparable to SOTA diffusion models on AudioCaps.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Audio Generation"
  - "Continuous-Valued Tokens"
  - "Language Modeling"
  - "Token-wise Diffusion"
  - "Masked Next-Token Prediction"
date: 2026-05-08
content_hash: a83f89246f04da1a
---

# Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction

**Conference**: ICML 2025  
**arXiv**: [2507.09834](https://arxiv.org/abs/2507.09834)  
**Code**: Yes (Project Page)  
**Area**: Image Generation (Audio Generation)  
**Keywords**: Audio Generation, Continuous-Valued Tokens, Language Modeling, Token-wise Diffusion, Masked Next-Token Prediction

## TL;DR
This paper studies causal language models for audio generation without using discrete tokens, leveraging token-wise diffusion to model the distribution of continuous-valued next-tokens, and proposes a masked next-token prediction task. With 193M parameters, it achieves performance comparable to SOTA diffusion models on AudioCaps.

## Background & Motivation

**Background**: Autoregressive next-token prediction combined with Transformer decoders has become the de facto standard for LLMs, achieving great success in NLP. Extending this paradigm to audio has been a research hotspot.

**Limitations of Prior Work**: Audio is inherently a continuous signal. Extending it to autoregressive LMs faces unique challenges. Existing methods (such as AudioGen) rely on discretization (quantizing audio into discrete tokens via VQ-VAE, etc.), but the quantization process inevitably causes information loss, limiting generation quality.

**Key Challenge**: Discrete tokens are convenient for modeling with standard LMs (using cross-entropy loss) but lose fine details of audio; directly modeling the probability distribution of continuous-valued tokens faces issues like complex distributions and training difficulties.

**Goal**: To explore causal language models for audio generation without discrete tokens and propose new training tasks to improve performance.

**Key Insight**: Using token-wise diffusion to model the distribution of the next continuous-valued token, while innovatively introducing masked prediction into the causal LM framework.

**Core Idea**: A two-pronged approach utilizing token-wise diffusion to model continuous token distributions + masked next-token prediction in causal LMs, achieving efficient continuous audio generation.

## Method

### Overall Architecture

- **Input**: Text condition + continuous audio token sequence (extracted by a pretrained encoder)
- **Modeling**: Causal Transformer autoregressively predicts the next token, where the tokens are continuous-valued vectors.
- **Distribution Modeling**: The next-token at each position models its conditional distribution using a small diffusion model.
- **Output**: Decoded into an audio waveform after token-wise autoregressive sampling.

### Key Designs

1. **Token-wise Diffusion for Continuous Tokens**:

    - Unlike traditional LMs that use softmax + cross-entropy on discrete tokens, this paper uses a small diffusion model at each autoregressive step to model the next continuous-valued token distribution $p(x_{t+1} | x_{\leq t})$.
    - The Transformer decoder provides the conditional context, and the diffusion model denoises and generates the next token under this condition.
    - **Design Motivation**: The distribution of continuous token space can be multimodal and complex; diffusion models are ideal for modeling such distributions.

2. **Masked Next-Token Prediction (MNTP)**:

    - Innovatively integrates masked prediction into the causal LM framework.
    - During training, some tokens are randomly masked but the causal structure is maintained (only previous unmasked tokens are visible).
    - The model needs to predict the masked tokens, effectively incorporating bidirectional information into the causal framework.
    - **Design Motivation**: Standard next-token prediction only utilizes preceding context. MNTP forces the model to leverage longer-range dependencies, adapting BERT-style training to a causal setting.

3. **Lightweight Parameter Design**:

    - The Base model has only 193M parameters, and the Large model has 462M parameters.
    - Far smaller than AudioGen Base (285M) and Large (1B).
    - **Design Motivation**: To demonstrate the parameter efficiency advantages of the continuous token approach.

### Loss & Training

- Token-wise diffusion loss: Standard denoising score matching loss, trained independently at each token position.
- MNTP loss: An additional masked token prediction loss.
- The total loss is a weighted sum of the two.
- Uses a pretrained continuous audio encoder (e.g., a continuous version of EnCodec) to extract tokens.

## Key Experimental Results

### Main Results

| Model | Params | FAD↓ | KL↓ | Note |
|------|--------|------|-----|------|
| AudioGen Base | 285M | Baseline | Baseline | Discrete tokens |
| AudioGen Large | 1B | Baseline | Baseline | Discrete tokens |
| **Ours Base**| **193M** | **-20% (relative)** | **-40% (relative)** | Continuous tokens |
| **Ours Base + MNTP** | **193M** | **-41% (relative to AG-Base)** | - | +masked prediction |
| **Ours Large + MNTP** | **462M** | **-33% (relative to AG-Large)** | - | Comparable with SOTA diffusion models |

### Ablation Study

| Configuration | FAD↓ | Note |
|------|------|------|
| Discrete tokens (AudioGen) | Baseline | Standard quantization |
| Continuous tokens (w/o MNTP) | 20% improvement | Continuous token advantage |
| Continuous tokens + MNTP | 41% improvement | MNTP extra contribution ~20% |
| Standard next-token only | Intermediate | Validates MNTP complementarity |

### Key Findings

- The continuous token approach significantly outperforms the discrete token scheme under the same parameter size (FAD relative improvement of 20-40%).
- MNTP is effective in the causal LM framework, providing an additional ~20% relative improvement.
- Only 193M parameters are needed to reach the level of the 285M AudioGen, and 462M is comparable to SOTA diffusion models.
- Remarkable parameter efficiency: achieving better performance with less than half the parameters of AudioGen Large.

## Highlights & Insights

1. **Breaking the "Must Discretize" Assumption**: Proves that causal LMs can directly model continuous tokens.
2. **MNTP as an Effective Enhancement for Causal LMs**: Introduces the information gain of masked prediction into the autoregressive framework.
3. **Parameter Efficiency**: Continuous representations avoid the overhead of maintaining large discrete codebooks.
4. **Generality Potential**: This method can in principle be extended to other continuous modalities like video and music.

## Limitations & Future Work

1. Token-wise diffusion increases the computational cost of sampling per step (although the total parameters are fewer).
2. Inference latency may be higher than pure discrete token methods.
3. The choice and quality of the continuous token encoder set the performance upper boundary.
4. Has not yet been scaled to ultra-large scales (>1B parameters).

## Related Work & Insights

- AudioGen/AudioLDM series provide comparative baselines for discrete and description-based continuous methods in audio generation.
- MAR (Masked Autoregressive) has also explored similar masked + autoregressive combinations in the image domain.
- Insight: This method may serve as a general template for "LM paradigm generating continuous data".

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of continuous token LM + MNTP is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Adequately validated on AudioCaps, but could include more datasets and tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear
- Value: ⭐⭐⭐⭐ Provides a new parameter-efficient solution for audio generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PanoLlama: Generating Endless and Coherent Panoramas with Next-Token-Prediction LLMs](../../ICCV2025/image_generation/panollama_generating_endless_and_coherent_panoramas_with_next-token-prediction_l.md)
- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](../../NeurIPS2025/image_generation/continuous_diffusion_model_for_language_modeling.md)
- [\[ICML 2025\] Efficient Generative Modeling with Residual Vector Quantization-Based Tokens](efficient_generative_modeling_with_residual_vector_quantization-based_tokens.md)
- [\[ICML 2025\] Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots](hierarchical_masked_autoregressive_models_with_low-resolution_token_pivots.md)
- [\[CVPR 2026\] FVAR: Next-Focus Prediction for Visual Autoregressive Modeling](../../CVPR2026/image_generation/fvar_next-focus_prediction_for_visual_autoregressive_modeling.md)

</div>

<!-- RELATED:END -->
