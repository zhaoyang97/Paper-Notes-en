---
title: >-
  [Paper Note] Float8@2bits: Entropy Coding Enables Data-Free Model Compression
description: >-
  [ICML 2026][Model Compression][Post-Training Quantization] EntQuant preserves weights with Float8/Int8 precision but applies an additional $\ell_1$ regularization during the quantization phase to "align" weights towards…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Post-Training Quantization"
  - "Entropy Coding"
  - "ANS"
  - "Data-Free Compression"
  - "Extreme Low-bit"
date: 2026-05-08
content_hash: ea1367e5167f565b
---

# Float8@2bits: Entropy Coding Enables Data-Free Model Compression

**Conference**: ICML 2026  
**arXiv**: [2601.22787](https://arxiv.org/abs/2601.22787)  
**Code**: https://github.com/merantix-momentum/entquant  
**Area**: Model Compression  
**Keywords**: Post-Training Quantization, Entropy Coding, ANS, Data-Free Compression, Extreme Low-bit  

## TL;DR
EntQuant preserves weights with Float8/Int8 precision but applies an additional $\ell_1$ regularization during the quantization phase to "align" weights towards a low-entropy distribution. It then utilizes parallel ANS entropy coding on the GPU to losslessly compress weights to approximately 2 bits. This approach compresses a 70B LLM by over 8$\times$ without requiring any calibration data, completes in under 10 minutes without recovery training, and results in inference speeds only 1.5–2$\times$ slower than the baseline.

## Background & Motivation
**Background**: Current Post-Training Quantization (PTQ) for LLMs follows two main paths according to Nagel's four-level classification. One is the "data-heavy" route: Level 2 uses calibration sets for GPTQ/OmniQuant; Levels 3-4 perform recovery training or QAT (e.g., QuIP#, EfficientQAT, AQLM), achieving 2-bit compression but requiring 40-50 hours on an 8$\times$A100 node. The alternative is the "data-light" route: Level 1 methods like RTN/NF4/HQQ are entirely data-free and can compress any model in minutes, but suffer a total functional collapse when dropping below 4 bits.

**Limitations of Prior Work**: The extreme low-bit ($<$4 bit) range is the most commercially valuable, but currently, only data-heavy methods remain stable there. These methods have three hidden costs: (1) Training data for many high-quality instruction-tuned/reasoning models (e.g., LLaMA-3.3 Instruct, Mistral Large) is not public, forcing the use of generic proxies like C4 for calibration; (2) Scenarios governed by GDPR (e.g., medical, finance) prohibit the reuse of sensitive data; (3) Calibration itself can damage safety-tuning and reasoning alignment, causing performance drops on benchmarks like IFEval/GSM8K that far exceed what is suggested by perplexity alone.

**Key Challenge**: Existing quantization paradigms rigidly couple "compression rate" with "bit-width"—achieving 8$\times$ compression necessitates using only 4 discrete values to represent all weights. This "bit-width = expressivity" coupling is tolerable above 4 bits, but at 2 bits, it means complex Gaussian-long-tail weight distributions are compressed into just 4 bins, leading to unrecoverable loss of high-frequency outliers. Data-heavy methods essentially use extensive patches (codebooks, grouping, outlier separation, recovery FT) to compensate for this loss of expressivity.

**Goal**: Can 2-bit extreme compression be achieved under the Level 1 data-free constraint while preserving the reasoning and alignment capabilities of instruction-tuned models? This requires solving two sub-problems simultaneously: how to compress to 2 bits without losing expressivity, and how to ensure decompression does not become an inference bottleneck.

**Key Insight**: The authors note that the signal processing field separated "quantization" from "efficient representation" decades ago—JPEG first quantizes DCT coefficients and then applies Huffman/Arithmetic coding. The compression rate of the entire pipeline is determined by entropy coding rather than the quantization bit-width. Recent works like DFloat11 have demonstrated that the overhead of running ANS decoding on GPUs can be minimal. Combining these facts suggests a new design space: maintain high expressivity with Float8/Int8 weights, but actively reduce the entropy of the weight distribution during quantization, then use entropy coding to compress the storage.

**Core Idea**: Use the $\ell_1$ norm as a differentiable proxy for discrete entropy to optimize channel-wise scaling factors, clustering Float8 weights toward a low-entropy state. Use the nvCOMP ANS encoder to losslessly compress these low-entropy 8-bit symbols to any target bits-per-weight (bpw). During inference, weights are decoded on-the-fly at the transformer block level.

## Method

### Overall Architecture
EntQuant takes pre-trained weights $\mathbf{W} \in \mathbb{R}^{M \times N}$ and outputs a triplet (compressed bitstream $\mathbf{z}$, channel scaling vector $S$, ANS metadata $\mathcal{M}$). The pipeline has two stages: **offline compression** performs independent entropy-constrained quantization per layer, then concatenates the Float8 weights of an entire transformer block into a symbol stream for ANS; **online inference** maintains a block-sized decompression buffer in GPU VRAM. Before the forward pass of a block, nvCOMP decodes the weights in parallel back to Float8. After the forward pass using Float8 Marlin GEMM, the buffer is overwritten by the next block. The key design is decoupling "numerical precision" from "storage cost"—the former is determined by Float8 (retaining full dynamic range), while the latter is determined by the compressed bitstream length (achievable below 2 bpw).

### Key Designs

1. **Rate-Distortion Objective + $\ell_1$ Entropy Proxy**:
    - **Function**: Simplifies the combinatorial optimization problem of "minimizing quantized weight entropy + constraining reconstruction error" into a differentiable objective solvable via L-BFGS.
    - **Mechanism**: The ideal objective is $\min_{\mathbf{W}_q} \hat{H}(\mathbf{W}_q)$ s.t. $d(\mathbf{W},\hat{\mathbf{W}})<\epsilon$, where $\hat{H}$ is the empirical entropy $-\frac{1}{MN}\sum \log_2 \hat p(\mathbf{W}_q^{(i,j)})$, but discrete entropy is non-differentiable. The authors use Lagrangian relaxation: $\min_{\mathbf{W}_q} d(\mathbf{W},\hat{\mathbf{W}})+\lambda R(\mathbf{W}_q)$. The reconstruction loss $d$ uses outlier-robust relative $\ell_1$: $d=\|\mathbf{W}-\hat{\mathbf{W}}\|_1/\|\mathbf{W}\|_1$, and the entropy proxy $R(\mathbf{X})=\|\mathbf{X}\|_1$. Appendix B.2 provides a max-entropy bound proof: under a fixed $\ell_1$ budget, the maximum entropy distribution has finite support; thus, minimizing $\ell_1$ shrinks the support and increases peak density, effectively reducing $\hat H$. In practice, they only differentiate with respect to the channel-wise scale factors $S$ (using a straight-through estimator), converging in seconds per layer. The relationship between $\lambda$ and final bpw is empirically log-linear and consistent across models, allowing users to tune for a target bpw directly.
    - **Design Motivation**: To achieve extreme compression under Level 1 constraints, all dependencies on forward activations (like GPTQ’s Hessian) must be abandoned. $\ell_1$ serves as both an outlier-robust reconstruction metric and an effective entropy proxy, making optimization a lightweight task of tuning a single set of scales—completing 70B models in 10 minutes.

2. **Bit-width-Compression Decoupled Float8 Quantization**:
    - **Function**: Strips storage cost from the bit-width, allowing a Float8 base to reach below 2 bpw.
    - **Mechanism**: Traditional methods use $\mathbf{W}_q = \text{clamp}(\lfloor\mathbf{W}/s\rceil, -Q_{\max}, Q_{\max})$ and store it at the fixed bit-width; 4$\times$ compression mandates using 4 bits. EntQuant does the opposite: it uses $\gamma \in \{$ Float8, Int8 $\}$ to retain the dynamic range of all $\sim 2^8$ representable values. However, $\ell_1$ optimization reduces the number of unique values actually used (Table 1: at 2-bit equivalence, EntQuant still uses 34.61 unique values on average, far more than the 4 values in fixed 2-bit quantization). The quantized weights of a transformer block are flattened into a 1D symbol stream for the ANS encoder. The code length is given by the Shannon bound $\sim \hat H(\mathbf{W}_q)$, so the final bpw is determined by the optimized empirical entropy. It can be continuously tuned from 8 bits down to below 2 bits. Channel-wise scaling ($s_j$ per output channel) naturally allocates precision to critical channels while avoiding explicit outlier separation.
    - **Design Motivation**: This is the core paradigm shift—the assumption that "bit-width determines compression rate" is a holdover from the fixed-codebook era and is outdated given the availability of GPU-accelerated ANS. Retaining Float8 expressivity means inference can leverage existing Float8 Marlin GEMM kernels without needing custom low-bit kernels.

3. **Block-wise On-device ANS Decoding**:
    - **Function**: Promotes entropy coding from "offline storage optimization" to an "online inference component," enabling 2-bit models to run at speeds close to Float8.
    - **Mechanism**: Traditional entropy coding (Han et al. 2016) only compresses on disk; weights are decompressed to FP16/INT4 at load time, consuming full VRAM. EntQuant keeps the compressed bitstream $\mathbf{z}$ in VRAM. Each transformer block is assigned a decompression buffer "just large enough for one block of Float8 weights." Before entering a block's forward pass, nvCOMP’s parallel ANS decoder expands the block's q/k/v/o/MLP weights into the buffer (sharing memory via tensor views). This block-level granularity is ~50% faster than layer-level (as ANS achieves higher GPU utilization on larger chunks). The decompression cost correlates with the total weight volume rather than sequence length, meaning long contexts naturally amortize the overhead.
    - **Design Motivation**: If extreme compression slows inference by 5-10$\times$, it won't be deployed. The block-wise pipeline allows peak VRAM usage to reflect the compressed size (Table 2: a 70B model at 2.1 bpw uses 18.8 GiB compressed + 0.8 GiB buffer + 1.25 GiB KV cache, fitting into a single 32 GiB 5090), while remaining only 1.5-2$\times$ slower than BFloat16—on par with NF4 and faster than HQQ.

### Loss & Training
No training data is used. Each layer independently optimizes $\min_S d(\mathbf{W},\hat{\mathbf{W}})+\lambda R(\mathbf{W}_q)$ via L-BFGS. Quantization gradients are handled by a straight-through estimator, and scales are stored in BFloat16. Since the relationship between $\lambda$ and bpw is log-linear and consistent, $\lambda$ can be set via a lookup table. Total compression for a 70B model takes less than 10 minutes on an H100.

## Key Experimental Results

### Main Results
Covering 16 open-source LLMs (LLaMA-1/2/3.1/3.3 Base & Instruct, Qwen3, OLMo 3.1, Mistral Large 24.11) with over 480 experiments, comparing against data-free methods like HQQ and NF4 on C4/WikiText-2 perplexity and 8 zero-shot LM Eval tasks.

| Model | Method | bpw | C4 PPL ↓ | LM Eval Avg ↑ |
|------|------|-----|---------|---------------|
| LLaMA-2 70B | Base (BF16) | 16 | 5.52 | 72.3 |
| LLaMA-2 70B | HQQ g64 | 3 | 6.02 | 70.4 |
| LLaMA-2 70B | EntQuant | 3 | 5.74 | 71.1 |
| LLaMA-2 70B | HQQ g64 | 2 | 2.8e3 (Collapse) | 30.4 |
| LLaMA-2 70B | EntQuant | 2.1 | **6.47** | **67.9** |
| LLaMA-3.1 70B | HQQ g64 | 2 | 1.3e4 (Collapse) | 29.9 |
| LLaMA-3.1 70B | EntQuant | 2.1 | **9.92** | **68.6** |

2.1 bpw is the strongest range for EntQuant—where all data-free baselines collapse (PPL in the thousands), EntQuant retains 92-94% of the BF16 average accuracy. Compared to data-heavy methods (Table 4 (b)): at 2 bits, GPTQ drops 52.8% and OmniQuant drops 24.6% on LLaMA-2 70B; EntQuant at 2.1 bits drops only 5.8%. It is close to EfficientQAT (5.3% drop, 41 hours) and slightly behind QuIP# (2.6% drop, 50 hours), yet EntQuant takes $<$ 10 minutes and zero data.

### Ablation Study

| Configuration | Key Observation | Description |
|------|---------|------|
| Float8 base (Default) | C4 PPL 6.47 at 2.1 bpw | Recommended configuration |
| Int8 base | Poorer PPL in some models at 2.1 bpw | Sensitive to super weights |
| Int8 + Excl. super weights | Performance close to Float8 | Exclude $<$ 10 outliers to recover performance |
| Float8 + Long prefill (8192) | Gap with Float8 Marlin $<$ 10% | Decompression cost is block-level, decoupled from seq len |
| W8A8 (Quanto dynamic) | Slight drop on 70B | Lacks fused kernels, speed not tested |

### Key Findings
- After $\ell_1$ optimization, 2-bit EntQuant still utilizes 34.61 unique values on average (Table 1), twice that of fixed 4-bit quantization (16 values). This explains why entropy coding maintains expressivity while reaching lower bpw.
- The log-linear relationship between $\lambda$ and bpw is nearly identical across models (Figure A.1), meaning new models don't need re-tuning, just a lookup table.
- Block-wise decompression is ~50% faster than layer-wise; its cost is independent of prefill length, so longer contexts better amortize decompression overhead.
- LLaMA-2 70B at 2.1 bpw can run standalone on a 32 GiB 5090, turning "70B on consumer GPUs" from a privilege of heavy retraining methods into a 10-minute plug-and-play solution.
- On "hard" benchmarks like IFEval/GSM8K/GPQA, EntQuant 3-bit is almost lossless and 2-bit remains usable, whereas calibration methods often collapse on instruction-following.

## Highlights & Insights
- **Paradigm Redefinition**: Bringing the "Quantization + Entropy Coding" architecture from JPEG back to LLMs proves "bit-width = compression rate" is a design choice, not a physical constraint.
- **$\ell_1$ as an Entropy Proxy**: Solves the "non-differentiable discrete entropy" problem using a classic norm, backed by a max-entropy bound proof—simple, yet theoretically sound.
- **GPU ANS as Undervalued Infrastructure**: nvCOMP has existed in the NVIDIA stack for years, but only recent works have elevated it to a first-class citizen of LLM inference.
- **Transferability**: The $\ell_1$ + entropy coding paradigm can be applied to (a) activation quantization (KV cache), (b) gradient quantization (distributed training), (c) vision models, and (d) Diffusion UNet weights.

## Limitations & Future Work
- Inference is 1.5-2$\times$ slower than BF16; low-latency serving scenarios require further optimization of the ANS decoding kernel.
- The log-linear $\lambda$→bpw relationship is empirical; extreme weight distributions might require secondary tuning.
- Currently only validates weight-only (W8A16) and coarse-grained W8A8; entropy coding for activations and KV cache is unexplored.
- Decompression buffers provide the most memory benefit for small batches; in large-batch inference, the relative ratio of weights to KV cache decreases.

## Related Work & Insights
- **vs HQQ / NF4 (Level 1)**: Both are data-free, but HQQ/NF4 collapse below 4 bits. EntQuant dominates the 2-bit range by decoupling bit-width from compression.
- **vs GPTQ / OmniQuant (Level 2)**: These require calibration sets and 2-4 hours. EntQuant is data-free, takes 10 minutes, and outperforms them at 2 bits.
- **vs QuIP# / EfficientQAT / AQLM (Level 3-4)**: These require 40-50 hours of training. EntQuant is slightly behind in accuracy (2-3 points) but is 200$\times$ cheaper and preserves context/instruction alignment.
- **vs DFloat11**: DFloat11 losslessly compresses BF16 (~30% reduction); EntQuant lossily compresses to Float8/Int8 then entropy codes (~80% reduction).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Efficient Learned Image Compression without Entropy Coding](efficient_learned_image_compression_without_entropy_coding.md)
- [\[ICML 2026\] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training](decouple_searching_from_training_scaling_data_mixing_via_model_merging_for_large.md)
- [\[ICML 2026\] Selective Coupling of Decoupled Informative Regions: Masked Attention Alignment for Data-Free Quantization of Vision Transformers](selective_coupling_of_decoupled_informative_regions_masked_attention_alignment_f.md)
- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](entropy-aware_on-policy_distillation_of_language_models.md)
- [\[ICML 2026\] ProjQ: Project-and-Quantize for Adapter-Aware LLM Compression](projq_project-and-quantize_for_adapter-aware_llm_compression.md)

</div>

<!-- RELATED:END -->
