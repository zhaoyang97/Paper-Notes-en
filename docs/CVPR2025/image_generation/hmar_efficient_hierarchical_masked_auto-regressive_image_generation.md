---
title: >-
  [Paper Note] HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation
description: >-
  [CVPR 2025][Image Generation][Autoregressive Image Generation] HMAR reformulates the next-scale prediction of VAR into a Markov process (relying only on the cumulative reconstruction of the previous scale rather than all prior scales). It introduces multi-step masked generation within each scale to eliminate the conditional independence assumption. Coupled with a customized IO-aware block-sparse attention kernel, HMAR matches or exceeds VAR/DiT quality on ImageNet while achie…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Autoregressive Image Generation"
  - "Next-Scale Prediction"
  - "Masked Prediction"
  - "Markov Process"
  - "Efficient Attention"
  - "Loss Reweighting"
date: 2026-05-08
content_hash: d33104bb0c756f4b
---

# HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation

**Conference**: CVPR 2025  
**arXiv**: [2506.04421](https://arxiv.org/abs/2506.04421)  
**Code**: None  
**Area**: Image Generation / Autoregressive Generation  
**Keywords**: Autoregressive Image Generation, Next-Scale Prediction, Masked Prediction, Markov Process, Efficient Attention, Loss Reweighting

## TL;DR

HMAR reformulates the next-scale prediction of VAR into a Markov process (relying only on the cumulative reconstruction of the previous scale rather than all prior scales). It introduces multi-step masked generation within each scale to eliminate the conditional independence assumption. Coupled with a customized IO-aware block-sparse attention kernel, HMAR matches or exceeds VAR/DiT quality on ImageNet while achieving 2.5× training acceleration and a 3× reduction in inference memory.

## Background & Motivation

**Background**: Visual Auto-Regressive modeling (VAR) bridges the gap between autoregressive models and diffusion models in speed and quality through the next-scale prediction paradigm. VAR decomposes an image into K multi-resolution scales, generating a higher-resolution scale at each step conditioned on the tokens of all preceding scales.

**Limitations of Prior Work**: (1) **Quality Bottleneck**: VAR samples all tokens in a single parallel step within each scale, implicitly assuming conditional independence of tokens within the same scale, which leads to "over-smoothing" and cross-scale error accumulation; (2) **Efficiency Issues**: Conditioning on all prior scales causes the sequence length to grow super-linearly (5.84× longer than next-token at 256×256 resolution), and FlashAttention does not natively support the block-causal attention pattern of VAR; (3) **Lack of Flexibility**: The number of inference steps is fixed during training, and changing the steps requires retraining.

**Key Challenge**: There is significant room for improvement in VAR across quality, efficiency, and flexibility—the conditional independence assumption degrades quality, long sequences hinder efficiency, and fixed inference steps restrict flexibility.

**Key Insight**: It is observed that the cumulative reconstruction $\tilde{x}_{1:k}$ in VAR encoding already encapsulates all information from the first $k$ scales (analogous to a Laplacian pyramid). Therefore, next-scale prediction can be reformulated as a Markov process $p(r_k | \tilde{x}_{1:k-1})$. This leads to a block-diagonal attention pattern (5× sparser than block-causal) and allows for the integration of MaskGIT-like multi-step masked generation within each scale to model intra-scale token dependencies.

## Method

### Overall Architecture

HMAR consists of two sub-modules: (1) **Markovian Next-Scale Prediction Module**: modifies VAR's full-history conditioning to rely solely on the cumulative reconstruction of the previous scale, accelerated during training using an IO-aware block-diagonal attention kernel; (2) **Intra-Scale Masked Refinement Module**: employs multi-step masked generation within each scale to eliminate the conditional independence assumption, offering a controllable trade-off between quality and speed. The two modules are trained sequentially in stages.

### Key Designs

1. **Markovian Next-Scale Prediction**:

    - **Function**: Reduces the sequence length of next-scale prediction from super-linear to linear, enabling block-diagonal sparse attention.
    - **Mechanism**: Leverages the properties of VQ-VAE residual encoding, where the cumulative reconstruction $\tilde{x}_{1:k} = \sum_{j=1}^{k} \tilde{x}_j$ contains all information from the first $k$ scales. Consequently, $p(r_k | r_1,...,r_{k-1}) = p(r_k | \tilde{x}_{1:k-1})$, reformulating generation as a Markov process. In practice, an interpolation function scales $\tilde{x}_{1:k-1}$ to $H_{k-1} \times W_{k-1}$ to act as the condition, converting the attention pattern from block-causal to block-diagonal (achieving a 5× sparsity improvement).
    - **Design Motivation**: Attention analysis (Fig. 9) reveals that most attention in VAR is indeed concentrated on the immediate preceding scale, validating the rationality of the Markov assumption. Eliminating the need for a KV cache during inference directly reduces memory usage by 3×.

2. **Hierarchical Multi-Step Masked Generation**:

    - **Function**: Models intra-scale token dependencies to eliminate the conditional independence assumption of VAR.
    - **Mechanism**: At each scale $k$, an initial next-scale prediction yields $r_k^0$ (equivalent to VAR's single-step result). This is followed by iteratively refining the output using $M_k$ masked generation steps—each step randomly masks a portion of tokens and re-predicts them based on the unmasked tokens and the cumulative reconstruction of the previous scale. If $M_k=0$, it degenerates to VAR; if $M_k=H_k \times W_k$, it degenerates to next-token prediction. During the fine-tuning training phase, the mask ratio is uniformly sampled as $\gamma \sim \mathcal{U}(0,1)$. During inference, utilizing multi-step refinement at coarser scales improves FID, while multi-step refinement at finer scales enhances perceptual quality.
    - **Design Motivation**: VAR's parallel generation assumes conditional independence among tokens of the same scale, leading to over-smoothing and error accumulation in practice (Fig. 17). Masked generation offers a controllable trade-off between quality and speed.

3. **Multi-Scale Loss Reweighting**:

    - **Function**: Balances the training contributions of different resolution scales.
    - **Mechanism**: The uniform average loss of VAR results in the finest scale contributing 256 times more than the coarsest scale. HMAR introduces scale weights $w(k)$ such that $\sum w(k) = 1$. Experiments show that the learning difficulty of each scale approximately follows a log-normal distribution (Fig. 12). Thus, a log-normal weighting function is adopted as $w(k)$ to align the model capacity allocation with the learning difficulty distribution.
    - **Design Motivation**: Early errors at coarser scales accumulate and propagate to all subsequent scales (Fig. 17). Furthermore, the number of tokens contributed by different scales varies drastically, making uniform weighting suboptimal.

### Loss & Training

- **Phase 1 (Next-Scale)**: Cross-entropy loss with IO-aware window attention + log-normal loss reweighting.
- **Phase 2 (Masked Refinement)**: Addition of a mask prediction head, fine-tuned using $\mathcal{L}_{mask} = \sum_k \mathcal{L}(\gamma r_k | \bar{\gamma} r_k)$.
- Utilizes a pre-trained multi-scale VQ-VAE tokenizer from VAR.
- $K=10$ scales ($1 \times 1$ to $16 \times 16$), consistent with VAR.
- Uses top-k and top-p sampling during inference, defaulting to 14 steps (10 next-scale steps + a small number of mask steps per scale).

## Key Experimental Results

### Main Results

ImageNet 256×256 (cfg=not explicitly specified):

| Method | Type | FID↓ | IS↑ | Params | Steps |
|------|------|------|------|--------|-------|
| DiT-XL/2 | Diffusion | 2.27 | 278.2 | 675M | 250 |
| VAR-d16 | AR | 3.36 | 277.8 | 310M | 10 |
| VAR-d24 | AR | 2.15 | 312.4 | 1.0B | 10 |
| VAR-d30 | AR | 1.95 | 303.6 | 2.0B | 10 |
| **HMAR-d16** | **Hybrid** | **3.01** | **288.6** | 465M | 14 |
| **HMAR-d24** | **Hybrid** | **2.10** | **324.3** | 1.3B | 14 |
| **HMAR-d30** | **Hybrid** | **1.95** | **334.5** | 2.4B | 14 |

ImageNet 512×512:

| Method | FID↓ | IS↑ | Params |
|------|------|------|--------|
| DiT-XL/2 | 3.04 | 240.8 | 675M |
| VAR-d36 | 2.63 | 303.2 | - |
| **HMAR-d24** | Matches or exceeds | Higher IS | - |

### Efficiency Comparison

| Metric | HMAR vs VAR |
|------|-------------|
| Training Speed | **2.5× Faster** |
| Inference Speed | **1.75× Faster** |
| Inference Memory | **3× Lower** |
| Attention Computation | 10× Faster (IO-aware kernel) |

### Ablation Study

Loss weighting strategy (d16, 256×256):

| Weighting Strategy | FID↓ | IS↑ |
|---------|------|------|
| Uniform (VAR) | 3.36 | 277.8 |
| Linear | ~3.2 | ~280 |
| **Log-Normal** | **3.01** | **288.6** |

### Key Findings

- HMAR-d30 matches VAR-d30 on ImageNet 256×256 with an FID of 1.95, but improves IS from 303.6 to 334.5 (+31 points), offering a significant perceptual improvement in image quality.
- The Markovian reformulation increases training sequence sparsity by 5× (at 256×256 resolution), and the IO-aware kernel accelerates attention computation by 10×.
- Inference requires no KV cache, resulting in a 3× memory reduction, which makes large-scale models and high-resolution inference highly feasible.
- The number of mask steps can be flexibly adjusted during inference: more steps at coarser scales improve global structure (FID↓), while more steps at finer scales enhance detail (IS↑).
- Log-normal loss weighting reduces FID by approximately 0.35 and increases IS by around 11 compared to uniform weighting.
- HMAR can be applied zero-shot to inpainting, outpainting, and class-conditional editing, capabilities that VAR lacks.

## Highlights & Insights

1. **Elegant Markov Equivalence Derivation**: Leverages the mathematical properties of VQ-VAE residual quantization to prove that $p(r_k|r_{<k}) = p(r_k|\tilde{x}_{1:k-1})$, drawing an analogy to Laplacian/Gaussian pyramids. This formulation is theoretically concise and practically efficient.
2. **Simultaneous Improvements in Quality, Efficiency, and Flexibility**: While these three aspects typically present a trade-off, HMAR wins on all fronts through the Markovian reformulation and masked generation, representing a rare Pareto improvement.
3. **Customizable Sampling Schedule**: The number of mask steps can be independently adjusted at different scales without retraining, providing extreme flexibility for the trade-off between quality and speed.
4. **Custom IO-aware GPU Kernel**: The engineering contribution is equally vital—the Triton-implemented block-sparse attention kernel translates the paper's theoretical sparsity advantages into concrete practical performance gains.

## Limitations & Future Work

- The parameter count is approximately 30-50% larger than the corresponding VAR model (e.g., HMAR-d16 with 465M vs VAR-d16 with 310M) due to the addition of the mask prediction head.
- Validation is currently limited to class-conditional generation on ImageNet, with a lack of experiments on text-to-image generation.
- Although the Markov assumption is validated by attention analysis, it might lose information in extreme edge cases (e.g., when distant preceding scales contain critical global structures).
- The two-stage training process (next-scale training followed by mask fine-tuning) increases the complexity of the training pipeline.

## Related Work & Insights

- HMAR is a hierarchical integration of VAR and MaskGIT—VAR provides inter-scale causality, while MaskGIT provides intra-scale non-causal refinement.
- Comparison with HART (another VAR variant): HART employs continuous-value diffusion for intra-scale refinement, whereas HMAR utilizes discrete masked generation, which is much more efficient.
- Implications for Video VAR: The Markovian technique can be applied to the temporal dimension of video frames to reduce the memory over head of long video generation.
- The loss reweighting strategy can be generalized to any multi-scale or multi-stage generative models.

## Rating

⭐⭐⭐⭐⭐ (5/5)

- **Novelty** ⭐⭐⭐⭐⭐: Markov equivalence derivation + masked refinement + IO-aware kernel—the three contributions are tightly coupled and coherent.
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: Evaluated across quality, efficiency, and flexibility; comprehensive ablation studies; compared against various baselines like VAR, DiT, and MaskGIT.
- **Writing Quality** ⭐⭐⭐⭐⭐: Concise theoretical derivation, clear presentation of experiments, and overall excellent structure.
- **Value** ⭐⭐⭐⭐⭐: Faster, more memory-efficient, and more flexible than VAR without compromising quality, serving as a drop-in upgrade solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficient](collaborative_decoding_makes_visual_auto-regressive_modeling_efficient.md)
- [\[CVPR 2025\] MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling](mmar_towards_lossless_multi-modal_auto-regressive_probabilistic_modeling.md)
- [\[CVPR 2025\] Hierarchical Flow Diffusion for Efficient Frame Interpolation](hierarchical_flow_diffusion_for_efficient_frame_interpolation.md)
- [\[NeurIPS 2025\] Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation](../../NeurIPS2025/image_generation/distilled_decoding_2_onestep_sampling_of_image_autoregressiv.md)
- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](../../ICCV2025/image_generation/dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)

</div>

<!-- RELATED:END -->
