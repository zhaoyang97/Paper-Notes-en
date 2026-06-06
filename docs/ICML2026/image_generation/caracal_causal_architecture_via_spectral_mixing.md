---
title: >-
  [Paper Note] Caracal: Causal Architecture via Spectral Mixing
description: >-
  [ICML 2026][Image Generation][FFT] Caracal replaces the $\mathcal{O}(L^2)$ attention in Transformers with an $\mathcal{O}(L \log L)$ Multi-Head Fourier (MHF) module…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "FFT"
  - "Attention Replacement"
  - "Causal Modeling"
  - "Long Sequence"
  - "SSM Comparison"
date: 2026-05-08
content_hash: e5fd8955fb51b683
---

# Caracal: Causal Architecture via Spectral Mixing

**Conference**: ICML 2026  
**arXiv**: [2605.00292](https://arxiv.org/abs/2605.00292)  
**Code**: See paper Appendix E  
**Area**: LLM Efficiency / Sequence Modeling / Long Context  
**Keywords**: FFT, Attention Replacement, Causal Modeling, Long Sequence, SSM Comparison

## TL;DR
Caracal replaces the $\mathcal{O}(L^2)$ attention in Transformers with an $\mathcal{O}(L \log L)$ Multi-Head Fourier (MHF) module, achieving strict causal masking in the frequency domain via a "pad-FFT-multiply-iFFT-truncate" pipeline. It completely removes positional encoding, relying solely on standard FFT operators (without custom CUDA kernels like Mamba), and matches the performance of Llama / Mamba / Mamba-2 / Jamba across all model scales from Tiny to Large.

## Background & Motivation
**Background**: There are two mainstream approaches for long-sequence modeling: Transformer attention (high expressiveness but $\mathcal{O}(L^2)$ complexity and requires positional encoding); and SSMs like Mamba (linear complexity but relies on custom CUDA kernels, limiting portability). Fourier-based methods (FNet, AFNO, SPECTRE) offer $\mathcal{O}(L \log L)$ complexity, but due to the difficulty of causal masking in the frequency domain, they are mostly limited to encoder-only settings.

**Limitations of Prior Work**: (1) Sparse attention (Longformer/BigBird) sacrifices information coverage; (2) Positional encodings like RoPE/YaRN/ALiBi are "patches" with limited extrapolation; (3) Mamba-like models require SSD-style custom operators, which are hard to implement/debug and behave inconsistently across GPUs; (4) Existing spectral methods (FNet, Hyena) are either non-causal or use static, position-based filters, lacking data-dependent mixing.

**Key Challenge**: The causality constraint in autoregressive generation conflicts with the "global atomic operation" of FFT—attention can zero out the upper triangle of the weight matrix, but FFT lacks an explicit weight matrix to mask. Enforcing causality by running an FFT of length $t$ for each $t$ is slower than $\mathcal{O}(L^2)$ (specifically, $\mathcal{O}(L^2 \log L)$).

**Goal**: (1) Enable FFT-based mixing to maintain causality in a single parallel forward pass during autoregressive training; (2) Remove positional encoding while retaining extrapolation; (3) Use only standard torch/numpy FFT operators, avoiding hardware dependencies like Mamba; (4) Introduce data-dependent gating to compensate for the limited expressiveness of static FFT weights.

**Key Insight**: The authors leverage the equivalence between "multiplication in the frequency domain = causal convolution in the time domain": pad the input to $2L$ → FFT → element-wise multiplication → iFFT → truncate back to $L$. This pipeline is mathematically equivalent to a strict causal convolution, with all steps performed using parallel FFTs.

**Core Idea**: Replace attention with a unified module of "content-adaptive convolution kernel × FFT acceleration × frequency-domain causality," while retaining a small amount of sliding-window attention for local precision.

## Method

### Overall Architecture
Caracal is structurally almost identical to GPT-2, with only two modifications: (1) global masked multi-head attention is replaced by the MHF module; (2) positional encoding is removed (the sinusoidal basis of FFT inherently encodes position). To preserve local precision, a Sliding-Window Attention (SWA) layer (window size 256) is inserted after every two MHF layers, keeping overall complexity at $\mathcal{O}(L \log L + L \cdot W)$. Feed-forward, LayerNorm, and residual connections remain unchanged, allowing direct reuse of the existing Transformer ecosystem.

### Key Designs

1. **Multi-Head Fourier (MHF) Module**:

    - **Function**: Achieves $\mathcal{O}(L \log L)$ global token mixing in the frequency domain, supporting autoregression.
    - **Mechanism**: Four-step pipeline. Step 1: Causal depthwise 1D conv (kernel=3) injects local inductive bias to compensate for the loss of local patterns after removing positional encoding. Step 2: After LayerNorm, project to value stream $x_v = \text{Linear}_V(x_{norm})$ and gate stream $x_g = \text{Conv1d}_{G2}(\sigma(\text{Linear}_{G1}(x_{norm})))$ in parallel; the gate stream uses group conv ($n_{head}$ groups) for intra-head channel interaction. Step 3: Zero-pad the sequence to $N=2L$, perform FFT to obtain $V_{fft}, G_{fft}$, then element-wise multiply in the frequency domain $X_{fft} = V_{fft} \odot G_{fft}$, which is equivalent to a causal convolution in the time domain $r_t = \sum_{j=0}^{t} v_j g_{t-j}$. Step 4: iFFT, truncate back to length $L$ to remove "future" pseudo-signals introduced by padding, then pass through $\text{Linear}_O$.
    - **Design Motivation**: Reformulate "attention as data-dependent weighted sum via query/key" into "gate stream as data-dependent convolution kernel," preserving selectivity while avoiding the serial scan of SSMs, and using only standard FFT operators throughout.

2. **Frequency-Domain Causal Mask (pad-FFT-multiply-iFFT-truncate)**:

    - **Function**: Ensures FFT maintains strict causality—output at $t$ depends only on inputs $\leq t$—while retaining parallelism.
    - **Mechanism**: Pure FFT causality is mathematically challenging—unlike attention, weights cannot be masked. The authors circumvent this by zero-padding the sequence of length $L$ to $2L$, performing FFT, multiplying by the gate, iFFT, and then keeping only the first $L$ elements. The $2L$-length FFT corresponds to circular convolution, which, when truncated to the first $L$ dimensions, degenerates to linear convolution $r_t = \sum_{j=0}^{t} v_j g_{t-j}$, automatically discarding dependencies on future tokens.
    - **Design Motivation**: Transform the "seemingly unsolvable" causality problem into a geometric arrangement of padding/truncation, essentially trading $2\times$ sequence length for a single forward pass causal convolution, eliminating the need to run FFT for each $t$ during training.

3. **No Positional Encoding + Hybrid SWA Local Compensation**:

    - **Function**: Completely removes explicit positional encodings like RoPE/ALiBi, while using SWA to preserve local resolution.
    - **Mechanism**: The FFT basis $e^{-i \frac{2\pi}{L} tj}$ inherently encodes sequence position, and downstream SWA layers also do not require positional encoding. SWA is implemented with FlashAttention, window size 256, keeping costs manageable. The MHF:SWA ratio is 2:1, balancing global long-range dependencies and local phrase-level patterns.
    - **Design Motivation**: Modern positional encodings (RoPE, YaRN) are increasingly complex yet still fail to solve the extrapolation problem fundamentally; making the model inherently position-aware at the architectural level is theoretically more suitable for arbitrary long contexts.

### Loss & Training
Standard next-token prediction cross-entropy loss is used, with no auxiliary losses outside the architecture. Training follows GPT-3 style hyperparameters (Tiny 63M → Large 724M), and all baselines use hardware-optimized kernels (Mamba uses mamba_ssm, Llama uses FlashAttention).

## Key Experimental Results

### Main Results
Nine zero-shot common-sense reasoning and LM benchmarks (LMB / Hellaswag / ARC-e/c / Wino / BoolQ / PIQA / SIQA), with a full sweep across four model sizes:

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
| Large | **Caracal (724M)** | 29.39 | 49.01 |

Caracal matches Llama / Mamba / Jamba in average accuracy across all sizes, slightly surpassing Llama at the Large scale (49.01 vs. 48.73).

### Ablation Study
Aligned with broader baselines at 345M parameters, 15B tokens, and 4096 context length:

| Model | LMB ppl↓ | Avg acc↑ |
|-------|----------|----------|
| Transformer++ | 41.08 | 42.92 |
| RetNet | 49.73 | 42.54 |
| GLA | 43.02 | 44.09 |
| Mamba | 40.21 | 43.59 |
| Gated DeltaNet | 30.94 | 45.42 |
| Moneta | 29.31 | 46.45 |
| Yaad | 29.11 | 45.94 |

Caracal ranks in the top tier alongside Mamba and DeltaNet, clearly outperforming earlier Transformer++/RetNet.

### Key Findings
- **Algorithmic "middle ground" replaces hardware tricks**: Trades $\mathcal{O}(L \log L)$ for SSM's $\mathcal{O}(L)$, maintaining performance while greatly reducing implementation complexity—everything uses standard FFT operators.
- **High LMB ppl (219.90) on Tiny** is a weakness of Caracal—dynamic gating is underfit at small model sizes; however, average accuracy remains tied for first, indicating ppl ≠ task performance.
- **No drop in performance after removing positional encoding** shows the implicit positional information in the FFT basis is sufficient, leaving room for long-context extrapolation (though the paper does not directly test extrapolation, which is a clear gap).
- **SWA is necessary**: Ablation shows pure MHF is weaker on ARC-c; adding SWA at a 2:1 ratio restores local capability.

## Highlights & Insights
- **Mathematically elegant causality trick**: The pad-2L → FFT → multiply → iFFT → truncate pipeline is a classic DSP technique, but this is the first complete demonstration in generative LM context, paired with data-dependent gating, solving a longstanding challenge for Fourier-based generative models.
- **Unified perspective of "content-adaptive convolution kernels"**: Attention, SSM, and FFT can all be viewed as $r_t = \sum_j w_{tj} v_j$ with different sources of weights—attention uses query/key, S4 is static, Mamba is input-dependent state, Caracal uses gate-generated content-aware filters. This framing clarifies the essential similarities and differences among the three architectures.
- **Hardware independence** is the true engineering value. Caracal can be deployed on any hardware supporting FFT (including TPU, dedicated NPUs), unlike Mamba, which is tied to NVIDIA GPUs.
- The overall approach ("frequency-domain multiplication + causal padding") is transferable to tasks requiring causality and long context, such as speech autoregressive modeling, long video generation, and protein generation.

## Limitations & Future Work
- **The authors acknowledge that theoretical $\mathcal{O}(L \log L)$ is slower than SSM's $\mathcal{O}(L)$**; Caracal is still at a disadvantage for extremely long contexts (100k+ tokens), and the paper does not report million-token experiments.
- **No explicit length extrapolation experiments**; the claim that "FFT basis inherently encodes position" is only theoretically justified, with no empirical zero-shot extension (e.g., 50k→200k) comparisons.
- **2L padding wastes half the computation**: Whether actual wall-clock throughput surpasses FlashAttention depends on the FFT implementation; the paper does not report real speed comparisons for short contexts (1k–4k).
- Future directions: (a) Use RFFT (real FFT) to further halve computation; (b) Explore more aggressive MHF:SWA ratios (e.g., 4:1) for ultra-long contexts; (c) Apply this approach to image autoregressive modeling for sub-quadratic autoregressive ViT.

## Related Work & Insights
- **vs Mamba/Mamba-2**: Both are attention replacements, but Caracal does not require hardware kernels and is more portable; performance is on par for small and medium models.
- **vs Hyena**: Hyena also uses FFT, but its filter is position-based (generated by MLP from $t$), not content-aware; Caracal's gate stream is dynamically generated from input, closer to Mamba's selectivity.
- **vs FNet / FNO / AFNO**: These are pure encoder models and non-causal, thus unsuitable for generation; Caracal is among the first strictly causal FFT replacements.
- **vs Monarch Mixer**: M2 uses GEMM to approximate convolution for hardware efficiency, while Caracal uses standard FFT for implementation simplicity; their trade-offs differ.
- **vs FlashButterfly / SPECTRE**: FlashButterfly uses a static global kernel with no extrapolation; SPECTRE uses a fixed sliding window, cutting off long-range dependencies; Caracal addresses both issues with dynamic filters.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of frequency-domain causality and content-aware gating is implemented in autoregressive LMs for the first time; while each component is not entirely new, the integration forms an elegant new architecture.
- Experimental Thoroughness: ⭐⭐⭐ Four model sizes and multiple baseline comparisons are solid, but lacks hard data for truly long contexts (≥32k) and training throughput.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper clearly traces the logic from attention/FFT first principles to the causality masking dilemma and the pad-truncate trick, making it highly suitable for teaching architecture concepts.
- Value: ⭐⭐⭐⭐ Provides a truly portable SSM alternative for non-NVIDIA hardware users, with strong potential for industrial adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Spectral Image Tokenizer](../../ICCV2025/image_generation/spectral_image_tokenizer.md)
- [\[CVPR 2026\] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](../../CVPR2026/image_generation/adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)
- [\[CVPR 2026\] MPDiT: Multi-Patch Global-to-Local Transformer Architecture for Efficient Flow Matching](../../CVPR2026/image_generation/mpdit_multi-patch_global-to-local_transformer_architecture_for_efficient_flow_ma.md)
- [\[CVPR 2026\] Causal Motion Diffusion Models for Autoregressive Motion Generation](../../CVPR2026/image_generation/causal_motion_diffusion_models_for_autoregressive_motion_generation.md)
- [\[CVPR 2026\] SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models](../../CVPR2026/image_generation/seacache_spectral-evolution-aware_cache_for_accelerating_diffusion_models.md)

</div>

<!-- RELATED:END -->
