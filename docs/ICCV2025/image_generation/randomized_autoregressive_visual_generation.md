---
title: >-
  [Paper Note] Randomized Autoregressive Visual Generation
description: >-
  [ICCV2025][Image Generation][Autoregressive image generation] This paper proposes Randomized AutoRegressive modeling (RAR): during standard autoregressive training, the input sequence is randomly permuted and gradually annealed back to raster-scan order, enabling the model to learn bidirectional context. RAR achieves a state-of-the-art FID of 1.48 on ImageNet-256 for autoregressive image generation while remaining fully compatible with the language model framework.
tags:
  - "ICCV2025"
  - "Image Generation"
  - "Autoregressive image generation"
  - "random permutation"
  - "bidirectional context"
  - "annealing strategy"
  - "ImageNet"
date: 2026-05-08
content_hash: 2ce348f7fec7422a
---

# Randomized Autoregressive Visual Generation

**Conference**: ICCV2025
**arXiv**: [2411.00776](https://arxiv.org/abs/2411.00776)  
**Code**: [bytedance/1d-tokenizer](https://github.com/bytedance/1d-tokenizer)  
**Area**: Image Generation
**Keywords**: Autoregressive image generation, random permutation, bidirectional context, annealing strategy, ImageNet

## TL;DR

This paper proposes Randomized AutoRegressive modeling (RAR): during standard autoregressive training, the input sequence is randomly permuted and gradually annealed back to raster-scan order, enabling the model to learn bidirectional context. RAR achieves a state-of-the-art FID of 1.48 on ImageNet-256 for autoregressive image generation while remaining fully compatible with the language model framework.

## Background & Motivation

### The Dilemma of Autoregressive Models in Visual Generation

Autoregressive (AR) models are the core framework of large language models (GPT, Llama, Gemini) and have achieved remarkable success in NLP. In visual generation, AR models (e.g., LlamaGen, Open-MAGVIT2) have also demonstrated competitive performance, yet still lag behind diffusion models and masked transformers. The root cause is:

- **Unidirectional context limitation**: Standard AR models use causal attention, where each token can only attend to preceding tokens, precluding bidirectional context modeling.
- **Intrinsic differences of visual signals**: Text has a natural left-to-right order, but images have no fixed token arrangement; visual signals are lower-level and more redundant, making bidirectional modeling more critical.
- **Order bias**: Although raster-scan order is the dominant choice, it introduces directional bias that limits the model's ability to learn dependencies along other directions.

### Limitations of Prior Work

- **VAR**: Replaces next-token prediction with next-scale prediction and introduces intra-scale bidirectional attention, but deviates from the standard AR paradigm.
- **MAR**: Generalizes the MaskGIT framework under an autoregressive definition, naturally introducing bidirectional attention, but is similarly incompatible with conventional language model frameworks.
- Although effective, these methods **break compatibility with language models**, which is unfavorable for building future unified multimodal models.

### Core Idea

Can bidirectional context learning be enhanced while **fully preserving** the AR model architecture and training paradigm? RAR answers affirmatively: by training with randomly permuted input sequences, every token is predicted under a variety of possible contexts, enabling the model to learn bidirectional representations. An annealing strategy then gradually transitions the model back to raster-scan order, yielding optimal generation quality.

## Method

### 1. Review of Standard Autoregressive Modeling

Given a discrete token sequence $\mathbf{x} = [x_1, x_2, \cdots, x_T]$, the standard AR model maximizes:

$$p_\theta(\mathbf{x}) = \prod_{t=1}^{T} p_\theta(x_t | x_1, \cdots, x_{t-1})$$

Each token depends only on preceding tokens, resulting in unidirectional context modeling.

### 2. Permutation Objective

The core contribution of RAR is to maximize the expected likelihood over all possible factorization orders:

$$p_\theta(\mathbf{x}) = \mathbb{E}_{\tau \sim \mathcal{S}_T} \left[ \prod_{t=1}^{T} p_\theta(x_{\tau_t} | x_{\tau_{<t}}) \right]$$

where $\mathcal{S}_T$ denotes the set of all permutations of the index sequence $[1,2,\cdots,T]$, and $\tau$ is a randomly sampled permutation. Since the model parameters $\theta$ are shared across all permutations, each token is exposed to a wide variety of contexts during training, allowing the model to learn bidirectional dependencies.

**Key distinction**: Unlike BERT/MaskGIT, which rely on mask tokens, RAR follows a permuted objective approach, training autoregressively over all possible factorization orders.

### 3. Target-aware Positional Embedding

Random permutation training introduces an ambiguity problem that standard positional encodings cannot resolve.

**Illustrative example**: Consider two permutations $\tau_a = [1,2,\cdots,T-1,T]$ and $\tau_b = [1,2,\cdots,T,T-1]$ (differing only in the last two positions). When predicting the second-to-last token, the input features are identical under both permutations, yet the prediction targets differ — making it impossible for the model to determine which token to predict.

**Solution**: An additional set of target-aware positional embeddings $\mathbf{p}_{ta} = [p_1, p_2, \cdots, p_T]$ is introduced. The positional embedding of the next token to be predicted is added to the current token:

$$\hat{\mathbf{x}}_\tau = [x_{\tau_1} + p_{\tau_2},\ x_{\tau_2} + p_{\tau_3},\ \cdots,\ x_{\tau_{T-1}} + p_{\tau_T},\ x_{\tau_T}]$$

This way, each prediction step is informed of the position index of the target token, eliminating the ambiguity introduced by permutations. After training, since the model has been annealed to raster order, the two sets of positional embeddings can be merged into one, incurring no additional parameters or computation at inference.

### 4. Randomness Annealing Strategy

Training exclusively with random permutations has two drawbacks:
- The permutation space is enormous ($256!> 10^{506}$ for 256 tokens), and the model may expend substantial capacity learning to handle diverse permutations rather than improving generation quality.
- Experiments show that raster-scan order remains the optimal generation order.

RAR therefore introduces an annealing parameter $r$ that controls the probability of using random permutations versus raster order:

$$r = \begin{cases} 1.0, & \text{if } epoch < start \\ 0.0, & \text{if } epoch > end \\ 1.0 - \frac{epoch - start}{end - start}, & \text{otherwise} \end{cases}$$

- At the beginning of training, $r=1$: fully random permutations are used to thoroughly learn bidirectional representations.
- $r$ linearly decays to 0, gradually transitioning the model toward raster-scan order.
- At the end of training, raster order is used exclusively, consistent with standard AR inference.

Optimal setting: 400 training epochs in total, $start=200$, $end=300$ (first 200 epochs: purely random; epochs 200–300: gradual annealing; final 100 epochs: purely raster).

## Key Experimental Results

### Experimental Setup

| Configuration | Details |
|------|------|
| VQ Tokenizer | MaskGIT-VQGAN, 16× downsampling, codebook size 1024 |
| Input Resolution | 256×256 → 256 discrete tokens |
| Dataset | ImageNet-1K training set (1.28M images) |
| Training | AdamW, batch size 2048, 400 epochs (250k steps) |
| Learning Rate | Linear warmup to 4e-4 (100 epochs), cosine decay to 1e-5 |
| Model Scale | RAR-B (261M) / RAR-L (461M) / RAR-XL (955M) / RAR-XXL (1.5B) |

### Main Results on ImageNet-256 (FID↓)

| Method | Type | Params | FID |
|------|------|--------|-----|
| DiT-XL/2 | Diffusion | 675M | 2.27 |
| MDTv2-XL/2 | Diffusion | 676M | 1.58 |
| MaskBit | Masked Trans. | 305M | 1.52 |
| MAR-H | Masked AR | 943M | 1.55 |
| VAR-d30-re | Scale AR | 2.0B | 1.73 |
| LlamaGen-3B-384 | AR | 3.1B | 2.18 |
| Open-MAGVIT2-XL | AR | 1.5B | 2.33 |
| **RAR-B** | **AR** | **261M** | **1.95** |
| **RAR-L** | **AR** | **461M** | **1.70** |
| **RAR-XL** | **AR** | **955M** | **1.50** |
| **RAR-XXL** | **AR** | **1.5B** | **1.48** |

### Key Findings

- **RAR-B (261M) outperforms LlamaGen-3B (3.1B)**: parameter count reduced by 91%, FID 1.95 vs. 2.18.
- **RAR-XXL (1.5B) achieves FID 1.48**: the first language-model-style AR generator to surpass the best diffusion models and masked transformers.
- **Annealing strategy is critical**: raster-only FID 3.08, random-only FID 3.01, optimal annealing (200, 300) FID 2.18.
- **Inference speed advantage**: RAR-XL generates 8.3 img/s, 11.9× faster than MaskBit (0.7) and 27.7× faster than MAR-H (0.3), owing to KV-cache compatibility.
- **Scan order ablation**: All 6 scan orders (raster, spiral, Z-curve, etc.) perform well, but raster scan remains optimal (FID 2.18 vs. second-best Z-curve at 2.29).

## Highlights & Insights

1. **Minimal yet highly effective improvement**: RAR requires no architectural changes — randomly permuting the input sequence during training and annealing back achieves a dramatic FID reduction from 3.08 to 1.48.
2. **Preserves language model compatibility**: Unlike VAR and MAR, RAR fully retains next-token prediction and causal attention, enabling direct reuse of LLM optimization techniques (KV-cache, vLLM, etc.).
3. **Annealing > pure random / pure raster**: Experiments clearly demonstrate that bidirectional context learning and fixed scan order each offer distinct advantages; the annealing strategy elegantly combines both.
4. **Elegant target-aware positional embedding design**: Permutation ambiguity is resolved by incorporating the positional information of the next token, with zero inference overhead (encodings are merged post-training).
5. **Outstanding parameter efficiency**: RAR-B (261M) surpasses competing methods with 1.5B+ parameters, demonstrating that bidirectional representation learning is more effective than simply scaling up the model.
6. **Connection to XLNet**: The permutation objective originates from XLNet in NLP; RAR successfully transfers it to visual generation, validating the value of cross-domain method transfer.

## Limitations & Future Work

1. **Full global context remains unattainable**: During generation, some tokens are always produced before others and cannot truly attend to the full context; the paper mentions resampling/refinement as potential remedies but leaves this unexplored.
2. **Tokenizer bottleneck**: The MaskGIT-VQGAN tokenizer used is relatively dated (codebook size 1024); replacing it with stronger tokenizers (e.g., TiTok, MAGVIT-v2) could yield further improvements.
3. **Validation limited to ImageNet 256×256**: The method has not been evaluated at higher resolutions or on other datasets.
4. **Sensitivity to annealing hyperparameters**: The choice of start and end epochs requires careful tuning, and the optimal range may differ under different total training durations.
5. **Permutation space sampling efficiency**: The $256!$ permutation space is far from fully explored; whether more efficient sampling strategies (e.g., curriculum-based permutations) exist warrants investigation.
6. **Joint text-image training not explored**: Despite emphasizing language model compatibility, no experiments on joint training with text are presented.

## Related Work & Insights

- **XLNet** [Yang+ 2019]: RAR's permutation objective is directly derived from XLNet's permutation language model in NLP, augmented with an annealing strategy and target-aware positional embeddings to suit visual tasks.
- **LlamaGen** [Sun+ 2024]: An AR approach that directly applies the Llama architecture to image generation; RAR substantially outperforms it within the same framework through training strategy innovation.
- **MAR** [Li+ 2024]: Also explores a random-order AR framework but finds that simply substituting random order yields limited gains; RAR's annealing strategy is the key breakthrough.
- **VAR** [Tian+ 2024]: Proposes next-scale prediction to enable intra-scale bidirectional attention, but at the cost of language model compatibility.
- **MaskGIT/MaskBit**: Non-autoregressive methods based on mask prediction that achieve excellent generation quality but suffer from slow inference and incompatibility with KV-cache.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Autoregressive Image Generation with Randomized Parallel Decoding](../../ICLR2026/image_generation/autoregressive_image_generation_with_randomized_parallel_decoding.md)
- [\[NeurIPS 2025\] InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](../../NeurIPS2025/image_generation/infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)
- [\[ICCV 2025\] Grouped Speculative Decoding for Autoregressive Image Generation](grouped_speculative_decoding_for_autoregressive_image_generation.md)
- [\[ICML 2025\] Continuous Visual Autoregressive Generation via Score Maximization](../../ICML2025/image_generation/continuous_visual_autoregressive_generation_via_score_maximization.md)
- [\[ICLR 2026\] BAR: Refactor the Basis of Autoregressive Visual Generation](../../ICLR2026/image_generation/bar_refactor_the_basis_of_autoregressive_visual_generation.md)

</div>

<!-- RELATED:END -->
