---
title: >-
  [Paper Note] Caracal: Causal Architecture via Spectral Mixing
description: >-
  [ICML 2026][Image Generation][FFT] Caracal replaces the $\mathcal{O}(L^2)$ attention in Transformers with an $\mathcal{O}(L \log L)$ Multi-Head Fourier (MHF) module. By employing a "pad-FFT-multiply-iFFT-truncate" pipeli…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "FFT"
  - "Attention Alternative"
  - "Causal Modeling"
  - "Long Sequence"
  - "SSM Comparison"
date: 2026-05-08
content_hash: e6bfc9fae729b042
---

# Caracal: Causal Architecture via Spectral Mixing

**Conference**: ICML 2026  
**arXiv**: [2605.00292](https://arxiv.org/abs/2605.00292)  
**Code**: See paper Appendix E  
**Area**: LLM Efficiency / Sequence Modeling / Long Context  
**Keywords**: FFT, Attention Alternative, Causal Modeling, Long Sequence, SSM Comparison

## TL;DR
Caracal replaces the $\mathcal{O}(L^2)$ attention in Transformers with an $\mathcal{O}(L \log L)$ Multi-Head Fourier (MHF) module. By employing a "pad-FFT-multiply-iFFT-truncate" pipeline, it achieves strict causal masking in the frequency domain. It removes positional encodings entirely and utilizes standard FFT operators (independent of custom CUDA kernels like Mamba), achieving performance comparable to Llama, Mamba, Mamba-2, and Jamba across scales from Tiny to Large.

## Background & Motivation
**Background**: Long-sequence modeling follows two main paths: Transformer attention (high expressivity but $\mathcal{O}(L^2)$ complexity and requiring positional encodings) and SSMs represented by Mamba (linear complexity but dependent on custom CUDA kernels, leading to poor portability). Spectral methods (FNet, AFNO, SPECTRE) offer $\mathcal{O}(L \log L)$ complexity but are mostly limited to encoder-only architectures due to the difficulty of implementing frequency-domain causal masking.

**Limitations of Prior Work**: (1) Sparse attention (Longformer/BigBird) sacrifices information coverage; (2) Positional encodings like RoPE, YaRN, and ALiBi are "patches" with limited extrapolation capabilities; (3) Mamba-like models require SSD-style operators that are difficult to debug and inconsistent across GPUs; (4) Existing spectral methods (FNet, Hyena) are either non-causal or use static position-based filters, lacking data-dependent mixing.

**Key Challenge**: The causality constraint of autoregressive generation naturally conflicts with the "global atomic operation" of FFT. While attention can zero out the upper-triangle of a weight matrix, FFT has no explicit weight matrix to mask. Achieving causality by running FFTs of length $t$ for each step $t$ results in $\mathcal{O}(L^2 \log L)$ complexity, slower than $\mathcal{O}(L^2)$.

**Goal**: (1) Enable FFT-based mixing to maintain causality in a single parallel forward pass during training; (2) Remove positional encodings while maintaining extrapolation capabilities; (3) Use only standard torch/numpy FFT operators without hardware dependencies; (4) Introduce data-dependent gating to compensate for the expressivity limitations of static FFT weights.

**Key Insight**: The authors start from the equivalence of "frequency domain multiplication = time domain causal convolution." By padding the input to $2L$, performing FFT, element-wise multiplication, iFFT, and then truncating back to $L$, the pipeline is mathematically equivalent to a strict causal convolution, yet all steps are completed via parallel FFTs.

**Core Idea**: Replace attention with a unified module featuring content-adaptive kernels, FFT acceleration, and frequency-domain causality, while retaining a small amount of sliding-window attention for local precision.

## Method

### Overall Architecture
Caracal's structure is nearly identical to GPT-2, with two modifications: (1) Global masked multi-head attention is replaced by the MHF module; (2) Positional encodings are removed (FFT sinusoidal bases inherently encode positional information). To preserve local precision, a Sliding-Window Attention (SWA) layer (window 256) is inserted every two MHF layers. The overall complexity remains $\mathcal{O}(L \log L + L \cdot W)$. Feed-forward, LN, and residual connections remain unchanged, allowing direct reuse of the existing Transformer ecosystem.

### Key Designs

1.  **Multi-Head Fourier (MHF) Module**:
    - **Function**: Achieves $\mathcal{O}(L \log L)$ global information mixing between tokens via frequency-domain multiplication, supporting autoregressive generation.
    - **Mechanism**: A 4-step pipeline. Step 1: Use causal depthwise 1D conv (kernel=3) to inject local inductive bias, compensating for local pattern loss after removing positional encodings. Step 2: After LayerNorm, project in parallel to a value stream $x_v = \text{Linear}_V(x_{norm})$ and a gate stream $x_g = \text{Conv1d}_{G2}(\sigma(\text{Linear}_{G1}(x_{norm})))$, where group convolutions ($n_{head}$ groups) handle intra-head channel interactions. Step 3: Zero-pad the sequence to $N=2L$, compute FFT for $V_{fft}$ and $G_{fft}$, and perform frequency-domain element-wise multiplication $X_{fft} = V_{fft} \odot G_{fft}$, equivalent to time-domain causal convolution $r_t = \sum_{j=0}^{t} v_j g_{t-j}$. Step 4: After iFFT, truncate back to length $L$ to remove "future" signals introduced by padding, followed by $\text{Linear}_O$.
    - **Design Motivation**: Reformulate "attention as a sum of data-dependent weights from query/key" into "data-dependent weights provided by the gate stream as a convolution kernel," retaining selectivity while avoiding SSM serial scans using standard FFT operators.

2.  **Frequency-Domain Causal Masking (pad-FFT-multiply-iFFT-truncate)**:
    - **Function**: Enables FFT to strictly satisfy the condition that "output $t$ depends only on inputs $\leq t$" while maintaining parallelism.
    - **Mechanism**: Pure FFT causality is a complex mathematical challenge as weights cannot be masked like in attention. The authors circumvent this by zero-padding a sequence of length $L$ to $2L$, performing the FFT/multiplication/iFFT, and retaining only the first $L$ elements. Since the circular convolution of a $2L$ FFT degenerates into a linear convolution $r_t = \sum_{j=0}^{t} v_j g_{t-j}$ when truncated to the first $L$ dimensions, dependencies on future tokens are automatically eliminated.
    - **Design Motivation**: Transform the "seemingly unsolvable" causality problem into a geometric arrangement of padding and truncation, essentially trading $2\times$ sequence length to complete causal convolution in one forward pass without running FFTs for each $t$ individually.

3.  **No Positional Encoding + Hybrid SWA Local Compensation**:
    - **Function**: Completely remove explicit positional encodings like RoPE/ALiBi while maintaining local resolution via SWA.
    - **Mechanism**: The FFT basis $e^{-i \frac{2\pi}{L} tj}$ contains built-in sequence position information, and downstream SWA layers do not require PE either. SWA is implemented via FlashAttention with a window of 256 to prevent cost explosion. A MHF:SWA ratio of 2:1 is used to balance global long-range dependencies and local phrase-level patterns.
    - **Design Motivation**: Modern PEs (RoPE, YaRN) are increasingly complex but fail to fully solve extrapolation; making the architecture inherently position-aware is theoretically better for arbitrary long contexts.

### Loss & Training
The model uses standard next-token prediction CE loss without external auxiliary losses. Training follows GPT-3 style hyperparameter settings (Tiny 63M → Large 724M), with all baselines utilizing hardware-optimized kernels (mamba_ssm for Mamba, FlashAttention for Llama).

## Key Experimental Results

### Main Results
Evaluation across 9 zero-shot common-sense reasoning and LM tasks, sweeping across 4 sizes:

| Size | Model | LMB ppl↓ | Avg acc↑ |
|------|-------|----------|----------|
| Tiny | Llama (64M) | 164.19 | 40.87 |
| Tiny | Mamba (66M) | 129.88 | 41.12 |
| Tiny | **Caracal (63M)** | 219.90 | **41.14** |
| Small | Llama (124M) | 79.94 | 43.02 |
| Small | Mamba (129M) | 86.33 | 43.60 |
| Small | Mamba2 (125M) | 100.76 | 42.64 |
| Small | **Caracal (120M)** | 92.05 | 43.35 |
| Medium | Llama (360M) | 32.65 | 47.07 |
| Medium | **Caracal (345M)** | 38.50 | 46.47 |
| Large | Llama (757M) | 24.92 | 48.73 |
| Large | **Caracal (724M)** | 29.39 | **49.01** |

Caracal's average accuracy is competitive with Llama, Mamba, and Jamba across all sizes, slightly exceeding Llama at the Large scale (49.01 vs 48.73).

### Ablation Study
Aligned with a broader range of baselines at 345M parameters, 15B tokens, and 4096 context length:

| Model | LMB ppl↓ | Avg acc↑ |
|-------|----------|----------|
| Transformer++ | 41.08 | 42.92 |
| RetNet | 49.73 | 42.54 |
| GLA | 43.02 | 44.09 |
| Mamba | 40.21 | 43.59 |
| Gated DeltaNet | 30.94 | 45.42 |
| Moneta | 29.31 | 46.45 |
| Yaad | 29.11 | 45.94 |

Caracal ranks in the top tier alongside Mamba and DeltaNet, significantly outperforming earlier models like Transformer++ and RetNet.

### Key Findings
- **Algorithmic "Middle Ground" replacing hardware tricks**: Trading SSM's $\mathcal{O}(L)$ for $\mathcal{O}(L \log L)$ maintains performance while drastically reducing implementation complexity since all operations use standard FFT.
- **High LMB ppl on Tiny (219.90)** is a weakness of Caracal—dynamic gating lacks sufficient fitting in small models; however, the Avg acc remains competitive, indicating that ppl $\neq$ task performance.
- **Removing PE without performance loss** suggests that implicit positional information in FFT bases is sufficient, potentially enabling better long-context extrapolation.
- **SWA is essential**: Ablations show pure MHF is weak on ARC-c; adding SWA at a 2:1 ratio recovers local modeling capabilities.

## Highlights & Insights
- **Mathematically elegant causal trick**: The pad-2L → FFT → multiply → iFFT → truncate pipeline adapts classic DSP techniques to generative LMs, successfully addressing a long-standing issue for Fourier-based generative models by pairing it with data-dependent gating.
- **Unified perspective on "Content-Adaptive Kernels"**: Attention, SSMs, and FFT are all viewed as different sources of weights for $r_t = \sum_j w_{tj} v_j$. Attention uses query/key, S4 is static, Mamba uses input-dependent states, and Caracal uses gate-generated content-aware filters. This framing clarifies the fundamental differences between these architectures.
- **Hardware independence** represents true engineering value. Caracal can be deployed on any hardware supporting FFT (including TPUs and specialized NPUs) without being locked into NVIDIA GPUs.

## Limitations & Future Work
- **Theoretically $\mathcal{O}(L \log L)$ is slower than SSM's $\mathcal{O}(L)$**, which may be disadvantageous at extreme context lengths (100k+ tokens); million-token experiments were not conducted.
- **Lack of explicit length extrapolation experiments**: The claim that "FFT bases inherently carry position" is theoretically argued but lacks zero-shot stretching results (e.g., 50k to 200k).
- **2L padding wastes half the computation**: Whether actual wall-clock throughput beats FlashAttention depends on specific FFT implementations; the paper did not report speed comparisons for short contexts (1k–4k).

## Related Work & Insights
- **vs Mamba/Mamba-2**: Both are attention alternatives, but Caracal is more portable without hardware kernels; performance is comparable in small-to-medium scales.
- **vs Hyena**: Hyena uses FFT, but its filters are position-based (generated via MLP from $t$) rather than content-aware; Caracal’s gate-driven filters are closer to Mamba’s selectivity.
- **vs FNet/FNO/AFNO**: These encoder-only models are non-causal and cannot perform generation; Caracal is among the first strictly causal FFT replacements.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of frequency-domain causality and content-aware gating is a novel and elegant realization for autoregressive LMs.
- **Experimental Thoroughness**: ⭐⭐⭐ Solid sweeps across sizes and baselines, but lacks hard data on extreme long contexts (≥32k) and training throughput.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logical progression from first principles of attention/FFT to causal masking dilemmas and the pad-truncate solution.
- **Value**: ⭐⭐⭐⭐ Provides a portable SSM alternative for non-NVIDIA hardware users, friendly for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Spectral Guidance for Flexible and Efficient Control of Diffusion Models](spectral_guidance_for_flexible_and_efficient_control_of_diffusion_models.md)
- [\[ICML 2026\] Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation](local_hessian_spectral_filtering_for_robust_intrinsic_dimension_estimation.md)
- [\[ICML 2026\] Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis](learning_general_causal_structures_with_hidden_dynamic_process_for_climate_analy.md)
- [\[ICML 2026\] AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching](ag-repa_causal_layer_selection_for_representation_alignment_in_audio_flow_matchi.md)
- [\[ICCV 2025\] Spectral Image Tokenizer](../../ICCV2025/image_generation/spectral_image_tokenizer.md)

</div>

<!-- RELATED:END -->
