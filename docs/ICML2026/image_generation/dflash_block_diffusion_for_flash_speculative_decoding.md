---
title: >-
  [Paper Note] DFlash: Block Diffusion for Flash Speculative Decoding
description: >-
  [ICML 2026][Image Generation][Speculative Decoding] DFlash replaces the autoregressive drafter of EAGLE-3 with a lightweight "block diffusion" draft model. By injecting multi-layer hidden features from the target model a…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Speculative Decoding"
  - "Block Diffusion"
  - "Draft Model"
  - "KV Injection"
  - "Parallel Drafting"
date: 2026-05-08
content_hash: 65eac57d8609c897
---

# DFlash: Block Diffusion for Flash Speculative Decoding

**Conference**: ICML 2026  
**arXiv**: [2602.06036](https://arxiv.org/abs/2602.06036)  
**Code**: https://dflash.z-lab.ai (Available, including GitHub + HuggingFace)  
**Area**: LLM Efficiency / Speculative Decoding / Diffusion Language Models  
**Keywords**: Speculative Decoding, Block Diffusion, Draft Model, KV Injection, Parallel Drafting

## TL;DR
DFlash replaces the autoregressive drafter of EAGLE-3 with a lightweight "block diffusion" draft model. By injecting multi-layer hidden features from the target model as KV into every layer of the draft model, it enables parallel drafting of an entire block of tokens in a single forward pass. DFlash achieves up to 6× lossless acceleration end-to-end, approximately 2.5× faster than EAGLE-3.

## Background & Motivation

**Background**: Autoregressive LLM inference is constrained by "token-by-token serial generation," resulting in extremely low GPU utilization. Speculative decoding has become a mainstream acceleration scheme by "quickly guessing a segment with a small draft model and verifying in parallel with a large target model." EAGLE-3 is the current SOTA, but its drafter remains autoregressive—merely with fewer layers and shorter sequences.

**Limitations of Prior Work**: Autoregressive drafting faces two inherent issues: (1) Drafting time $T_{\text{draft}}=\gamma\cdot t_{\text{step}}$ grows linearly with the speculative budget $\gamma$, forcing the drafter to use extremely shallow (1-layer Transformer) structures; (2) Shallow models lack sufficient capacity, causing the acceptance length $\tau$ to saturate quickly. These factors clamp overall acceleration at a 2–3× ceiling.

**Key Challenge**: The "quality" and "latency" of the drafting phase are strongly coupled via $\gamma$. To increase $\tau$, one needs a larger $\gamma$, but a larger $\gamma$ causes the drafting latency to explode.

**Previous Diffusion Drafting Attempts**: DiffuSpec and SpecDiff-2 utilize 7B dLLMs as drafters; while drafting quality is high, the parameter count is too large, and the drafting latency consumes the speedup. PARD trains small models to mimic diffusion-style parallel generation, but acceptance length remains low due to limited capacity. Consequently, "diffusion drafting" sounds promising but falls into a "too large or too weak" dilemma in practice.

**Goal**: Develop a diffusion drafter that is both lightweight (5-layer Transformer), capable of achieving long acceptance lengths ($\tau\!\ge\!6$), and deeply conditioned on the target model.

**Key Insight**: The authors observe that the hidden features of the target model already "implicitly" encode information about multiple future tokens (echoing Samragh et al. 2025). Therefore, the drafter does not need to "infer from scratch" but rather acts as a lightweight "diffusion adapter" to translate the hidden features computed by the target model during prefill into future block tokens.

**Core Idea**: Use block diffusion for parallel drafting combined with injecting multi-layer hidden features of the target model as KV into each layer of the draft model. This allows "drafting time remaining nearly independent of $\gamma$" and "acceptance length scaling steadily with the number of draft layers" to occur simultaneously.

## Method

### Overall Architecture
DFlash modifies only the drafting side within the standard "draft → verify" speculative decoding cycle. Given a prompt, the target model $\mathcal{M}_t$ performs a standard prefill, producing the first bonus token while extracting hidden states from several uniformly sampled layers (defaulting to 5 layers, from the 2nd to the 3rd-to-last layer). These cross-layer hidden states are fused through a lightweight projection layer into a compact target context feature, which is then injected into the KV cache of every layer of the draft model $\mathcal{M}_d$ (defaulting to 5 layers, block size 16). $\mathcal{M}_d$ then predicts all tokens of the next block in parallel via block diffusion in a **single forward pass**, which are passed to $\mathcal{M}_t$ for parallel verification. Subsequent drafting rounds reuse the same target context feature (cached in KV).

### Key Designs

1. **Target Model Hidden Features as Persistent KV Injection into Every Draft Layer**:
    - **Function**: Continuously injects the "future token information implicit in the large model's mind" into every layer of the draft Transformer.
    - **Mechanism**: Extracts hidden states from 5 uniformly distributed layers of the target model, concatenates them, and fuses them into a target context feature through a projection layer. Unlike EAGLE-3, which concatenates these with draft token embeddings only at the input, DFlash **separately projects this feature into the Key/Value matrices of every layer** and writes them into the draft model's KV cache, reusing them across multiple drafting rounds. This way, every attention layer can directly access target features.
    - **Design Motivation**: Ablations show that EAGLE-3's "input fusion" causes the conditioning signal to dilute as the draft model deepens, preventing $\tau$ from increasing with more layers. KV injection allows $\tau$ to scale steadily with layer count, which is the fundamental prerequisite for DFlash to utilize 5-layer or even 8-layer drafters. Table 9 shows that for a 5-layer drafter, input fusion yields a GSM8K $\tau$ of 3.5, which increases to 4.2 with KV injection.

2. **Block Diffusion Parallel Drafting Replacing Autoregressive Drafting**:
    - **Function**: Decodes the mask tokens of an entire block in parallel in a single forward pass, decoupling drafting time from $\gamma$.
    - **Mechanism**: The draft model follows a block-diffusion style. Given an anchor within the block (the bonus token from the previous target model step), the remaining $\text{block\_size}-1$ positions are initialized as mask tokens. A single forward pass predicts tokens for all mask positions simultaneously. Drafting time is approximately $T_{\text{draft}}\!\approx\!t_{\text{parallel}}$, which is nearly independent of block size (Figure 3 shows a 5-layer DFlash drafting 16 tokens is faster than a 1-layer EAGLE-3 drafting 8 tokens).
    - **Design Motivation**: The linear growth of autoregressive drafting cost $T_{\text{draft}}=\gamma\cdot t_{\text{step}}$ locks drafters into shallow structures. By switching to parallel drafting, "using deeper/stronger draft models" and "using longer draft blocks" become compatible, pushing the Pareto frontier toward the top-right.

3. **Random Anchor Sampling + Early Position Weighted Loss during Training**:
    - **Function**: Aligns the training distribution with the inference behavior of "drafting at random positions with a bonus token as anchor" and ensures early positions, which facilitate acceptance length, are well-trained.
    - **Mechanism**: During training, instead of fixed block slicing as in standard block diffusion, several anchor tokens are randomly sampled from the response. Each anchor serves as the start of a block, with subsequent positions masked, and the draft model predicts the following $\text{block\_size}-1$ tokens. Multiple blocks are packed into a single sequence using Flex Attention and trained simultaneously with sparse attention masks (intra-block bidirectional + looking at target features; inter-block invisible). An exponentially decaying position weight $w_k=\exp(-\tfrac{k-1}{\gamma})$ is applied to the loss for position $k$ within the block, as errors in early positions invalidate the entire block.
    - **Design Motivation**: Random anchors expose the drafter to more diverse target context features. Table 13 reports this strategy significantly increases acceptance length and acceleration. Early position weighting directly addresses the bottleneck, as speculative decoding's acceptance length is determined by the "first rejected position."

### Loss & Training
The base objective is cross-entropy with the aforementioned $w_k$ position weights. The draft model's token embeddings and LM head are **shared and frozen** with the target model, training only the draft Transformer layers. This reinforces the role of the draft model as a lightweight diffusion adapter. Training uses approximately 800K samples (NVIDIA Nemotron Post-Training V2 + CodeAlpaca), with responses regenerated by the target model to align distributions. For long context, only 1.6K LongAlign samples are needed for 3 epochs of fine-tuning to extend from 4K to 32K.

## Key Experimental Results

### Main Results

| Model / Task | Method | Speedup | $\tau$ |
|---|---|---|---|
| Qwen2-7B GSM8K (T=0) | EAGLE-3 (tree=16) | 1.94× | 3.23 |
| Qwen2-7B GSM8K (T=0) | EAGLE-3 (tree=60) | 2.23× | 3.71 |
| Qwen2-7B GSM8K (T=0) | **DFlash (block=16)** | **5.15×** | **6.54** |
| Qwen2-7B HumanEval (T=0) | EAGLE-3 (tree=60) | 2.17× | 3.65 |
| Qwen2-7B HumanEval (T=0) | **DFlash (block=16)** | **5.14×** | **6.50** |
| Qwen2-1.5B 8-Task Avg (T=0) | EAGLE-3 (tree=16) | 1.81× | 3.05 |
| Qwen2-1.5B 8-Task Avg (T=0) | **DFlash (block=16)** | **4.91×** | **6.54** |
| Qwen2-7B Math500 SGLang(B200) C=1 | Baseline → DFlash | **5.1×** | 8.01 |
| Qwen2-Coder-32B HumanEval SGLang C=32 | DFlash | **3.1×** | 8.09 |

Key Points: Under an equitable drafting budget (EAGLE-3 tree=16 vs. DFlash block=16), DFlash nearly doubles $\tau$ and increases speedup by 2.4–2.7×. Even when EAGLE-3 is relaxed to tree=60 (maximizing verification cost), DFlash remains superior. On production-grade SGLang + FA4 backends with a single B200, it consistently achieves 4–5× acceleration.

### Ablation Study

| Configuration | Key Finding |
|---|---|
| Draft layers 3 / 5 / 8 (Table 6) | Speedup 4.69× / 4.71× / 4.64×; 8 layers has highest $\tau$ but slower drafting; 5 layers is the optimal balance. |
| Target hidden layers 3 / 5 (Table 7) | Math500 $\tau$ 5.38 → 5.64; extracting more target feature layers steadily increases $\tau$ at the cost of doubling training cache. |
| Train BS 16 / Test BS 8 (Table 8) | Speedup 3.87× vs 3.97× for Train=Test=8; models trained with large blocks can generalize downward, but not vice versa. |
| Input fusion vs KV Injection (Table 9, 5 layers) | GSM8K $\tau$ 3.5 → 4.2; KV injection is key to "deepening the drafter to increase $\tau$." |

### Key Findings
- What distinguishes DFlash from EAGLE-3 is not "diffusion" alone but the combination of **KV Injection + Block Parallelism**: Ablating diffusion for autoregressive + KV injection (DFlash-AR) still exceeds EAGLE-3-5L in $\tau$, but its speedup is far inferior to the full DFlash. This indicates KV injection contributes quality while block diffusion contributes speed; both are indispensable.
- The hardware-level fact that drafting time is nearly independent of block size (Figure 3) is the "enabler" for the entire paper—it allows the drafter to become both deeper ($\rightarrow$ high $\tau$) and wider ($\rightarrow$ long block), breaking the Pareto frontier of autoregressive drafters.
- Training on 4K base models + 3 epochs of LongAlign fine-tuning supports up to 32K (Table 4), suggesting that target hidden features already possess long-context representations, and the drafter only needs to learn short-range adaptation.

## Highlights & Insights
- **Redefining the Position of Diffusion Language Models**: Rather than competing head-on with autoregressive LLMs in end-to-end generation, it is better to treat dLLMs as "parallel accelerators dedicated to drafting." This reframing justifies "minimizing diffusion steps (ideally 1)" and "guaranteeing quality through verification," providing a solid foothold for the diffusion paradigm in LLM inference.
- **KV Injection is Transferable**: Scenarios where "small models depend on large model hidden states" (distillation drafting, parallel heads, early-exit fallback verification, etc.) can benefit from this—stuffing condition information into KV rather than input prevents signal dilution in deeper small models.
- **Specific Targeting of Acceptance Length with Position-Weighted Loss**: Identifying that "acceptance length is determined by the earliest error" and using exponentially decaying weights to attack this bottleneck is a clever and effective training technique. This idea is applicable to all parallel generation tasks where prefix correctness determines suffix validity.

## Limitations & Future Work
- The authors acknowledge a lack of direct code-level comparison with other diffusion drafting methods like DiffuSpec / SpecDiff-2 / TiDAR (citing a lack of open-source implementations). Thus, DFlash's SOTA status is primarily compared against EAGLE-3; horizontal positioning within diffusion drafters awaits third-party replication.
- Training costs are non-trivial: 800K samples + mandatory regeneration of responses by the target model for alignment, and hidden feature cache grows linearly with the number of extracted layers. Scaling to 70B+ target models will impose significantly higher storage and compute pressure than the 8–30B settings in the paper.
- Block size selection remains an offline decision. The paper notes that reducing block size is better for large batch/compute-bound scenarios but leaves this for future work; an ideal next step is an online scheduler that dynamically adjusts block size based on batch size and acceptance history.
- Speedup on open-domain dialogue tasks like MT-Bench / Alpaca is significantly lower than on Code/Math (Q3-8B MT-Bench only 2.75× vs HumanEval 5.14×). This suggests the core assumption that "target hidden contains future token info" is weaker in open generation, requiring more targeted training data or conditioning methods.

## Related Work & Insights
- **vs EAGLE-3**: Both utilize target hidden features, but EAGLE-3 concatenates them with token embeddings only at the input layer and remains autoregressive. DFlash treats features as KV injected into every layer ($\rightarrow$ deepening the drafter works) and replaces autoregressive with block diffusion ($\rightarrow$ drafting time doesn't scale with $\gamma$). Consequently, under equal tree=block=16, speedup increases by >2.4× and exceeds EAGLE-3 even with tree=60.
- **vs PARD**: PARD uses small autoregressive models to mimic diffusion parallel generation but remains limited by capacity. DFlash uses true diffusion parallelism + deep KV injection, raising the ceiling of diffusion drafting from 3× to 6×.
- **vs DiffuSpec / SpecDiff-2**: These methods use 7B-class dLLMs for drafting; $\tau$ is high, but drafting latency negates the speedup. DFlash uses a 5-layer (~30M) specialized diffusion adapter, shifting from "quality from self" to "quality from target hidden conditioning," turning the diffusion drafter from a "heavy asset" into a "light asset."

## Rating
- Novelty: ⭐⭐⭐⭐ Block diffusion for drafting is not entirely novel, but the "KV injection per layer + block parallel + position weighting" combination makes diffusion drafting production-ready with clear insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers Qwen2-1.5B/7B/Coder-32B, LLaMA-3.1-8B across Math/Code/Chat tasks, T=0/T=1, and Transformers/SGLang/vLLM backends, including long context and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ The narrative (autoregressive bottleneck $\rightarrow$ diffusion temptation $\rightarrow$ prior failure $\rightarrow$ DFlash solution) is very smooth, with well-integrated formulas and charts.
- Value: ⭐⭐⭐⭐⭐ Provides ~6× lossless acceleration and integration into SGLang. This has high practical value for reducing inference costs and offers a new answer for the role of diffusion models in LLM pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation](speculative_coupled_decoding_for_training-free_lossless_acceleration_of_autoregr.md)
- [\[AAAI 2026\] Annealed Relaxation of Speculative Decoding for Faster Autoregressive Image Generation](../../AAAI2026/image_generation/annealed_relaxation_of_speculative_decoding_for_faster_autor.md)
- [\[ICCV 2025\] Grouped Speculative Decoding for Autoregressive Image Generation](../../ICCV2025/image_generation/grouped_speculative_decoding_for_autoregressive_image_generation.md)
- [\[CVPR 2026\] SJD-PAC: Accelerating Speculative Jacobi Decoding via Proactive Drafting and Adaptive Continuation](../../CVPR2026/image_generation/sjd-pac_accelerating_speculative_jacobi_decoding_via_proactive_drafting_and_adap.md)
- [\[ICLR 2026\] FlowCast: Trajectory Forecasting for Scalable Zero-Cost Speculative Flow Matching](../../ICLR2026/image_generation/flowcast_trajectory_forecasting_for_scalable_zero-cost_speculative_flow_matching.md)

</div>

<!-- RELATED:END -->
