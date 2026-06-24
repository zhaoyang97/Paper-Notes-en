---
title: >-
  [Paper Note] HierarchicalPrune: Position-Aware Compression for Large-Scale Diffusion Models
description: >-
  [AAAI 2026][Image Generation][Diffusion Model Compression] This paper proposes HierarchicalPrune, which exploits the hierarchical functional differences among blocks in MMDiT-based diffusion models—early blocks establish semantic structure while late blocks refine texture details—and combines three techniques: Hierarchical Position Pruning (HPP), Positional Weight Preservation (PWP), and Sensitivity-Guided Distillation (SGDistill), together with INT4 quantisation. Applied to…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Diffusion Model Compression"
  - "Pruning"
  - "Knowledge Distillation"
  - "MMDiT"
  - "Quantisation"
date: 2026-05-08
content_hash: 4a0c8973105d5b2b
---

# HierarchicalPrune: Position-Aware Compression for Large-Scale Diffusion Models

**Conference**: AAAI 2026
**arXiv**: [2508.04663](https://arxiv.org/abs/2508.04663)  
**Code**: None  
**Area**: Image Generation / Model Compression
**Keywords**: Diffusion Model Compression, Pruning, Knowledge Distillation, MMDiT, Quantisation

## TL;DR
This paper proposes HierarchicalPrune, which exploits the hierarchical functional differences among blocks in MMDiT-based diffusion models—early blocks establish semantic structure while late blocks refine texture details—and combines three techniques: Hierarchical Position Pruning (HPP), Positional Weight Preservation (PWP), and Sensitivity-Guided Distillation (SGDistill), together with INT4 quantisation. Applied to SD3.5 Large Turbo (8B), the method compresses the model from 15.8 GB to 3.24 GB (79.5% memory reduction) with only a 4.8% degradation in image quality.

## Background & Motivation
State-of-the-art text-to-image diffusion models such as SD3.5 and FLUX have reached parameter scales of 8–11B. While their image quality far surpasses earlier models like SDXL and SD1.5, the enormous model size poses serious deployment challenges.

**Key Challenge**: Although quantitative metrics (e.g., GenEval) suggest that smaller models such as SANA-Sprint-1.6B (2B parameters) perform competitively, user evaluations on the Artificial Analysis leaderboard reveal a **perceptual quality gap between small and large models that quantitative metrics fail to capture**. Deploying large models in resource-constrained settings therefore remains an essential need.

**Limitations of Prior Work**:
1. **Reducing sampling steps and efficient operators**: improve speed only, without reducing memory.
2. **Depth pruning methods** (KOALA, BK-SDM): effective on small models (≤2.6B U-Net), but on large models such as SD3.5 (8B) and FLUX (11B), even 20–30% memory reduction causes **38–45% severe quality degradation**.
3. Existing methods employ **whole-block removal** strategies that ignore the differential importance of sub-components within each block.

**Key Findings of This Paper**: MMDiT-based diffusion models exhibit a **dual-level hierarchical structure**:
- **Inter-block hierarchy**: blocks at different positions are responsible for different aspects of the image—early blocks establish semantic structure, while late blocks handle texture refinement.
- **Intra-block hierarchy**: the importance of sub-components within each MMDiT block (Norm, Attention, MLP, etc.) varies by position and component type.

## Method

### Overall Architecture
HierarchicalPrune is a three-stage compression framework:
1. **Stage 1 (HPP)**: block-level and sub-component-level pruning based on hierarchical position awareness.
2. **Stage 2 (PWP + SGDistill)**: freezing critical early blocks combined with sensitivity-guided knowledge distillation to recover quality.
3. **Stage 3 (PTQ)**: INT4 weight quantisation for further compression.

### Key Designs

**Design 1: Hierarchical Position Pruning (HPP)**

HPP is based on the core insight that late MMDiT blocks contribute less to core visual structure. For each block and sub-component, the performance impact $\Delta P(i,c)$ is computed by removing it on a calibration set. A positional weighting function is then introduced to measure each block's pruning eligibility:

$$Score(i,c) = -|\Delta P(i,c)| \times W_{pos}(i)$$

$$W_{pos}(i) = e^{(i - |\mathcal{B}|) / |\mathcal{B}|}$$

where $i$ is the block index and $|\mathcal{B}|$ is the total number of blocks. $W_{pos}$ is an exponential decay function that assigns higher weights to later blocks (larger $i$), making them more likely to be pruned. Unlike KOALA's cosine-similarity ranking or BK-SDM's CLIP-score ranking, HPP directly leverages the joint information of performance degradation and position to perform fine-grained pruning at the sub-component level.

**Design 2: Positional Weight Preservation (PWP)**

During the distillation stage, PWP freezes the weights of unpruned early blocks and only allows later, less critical blocks to be updated. This simple strategy ensures that structurally important blocks—critical to image formation—remain intact:
- Under moderate compression (20% parameter reduction): HPP alone causes 79.4% quality degradation; adding PWP reduces this to only 2.5%.
- Weight preservation of early blocks is the key to maintaining quality.

**Design 3: Sensitivity-Guided Distillation (SGDistill)**

Under aggressive compression (≥30%), unacceptable quality degradation (31.9%) persists even with PWP. SGDistill is based on a counterintuitive yet effective principle: **blocks with higher importance are more sensitive to change, and forcing updates on them is harmful**.

The distillation loss consists of a feature distillation loss and a knowledge distillation loss:

$$\mathcal{L} = \mathcal{L}_{feat} + \mathcal{L}_{KD}$$

$$\mathcal{L}_{KD} = \mathbb{E}\left[\|v_{\boldsymbol{\theta'}}(x_t, t) - v_{\boldsymbol{\theta}}(x_t, t)\|^2\right]$$

$$\mathcal{L}_{feat} = \mathbb{E}\left[\sum_{i \in [0,|\mathcal{B}^*|-1]} \|f_{\boldsymbol{\theta'}}^{i'}(x_t, t) - f_{\boldsymbol{\theta}}^i(x_t, t)\|^2\right]$$

SGDistill scales the parameter updates of each block by the inverse of its sensitivity $\frac{1}{\Delta P(i,c)}$—assigning minimal or zero update weight to the most important blocks, and concentrating updates on less sensitive components. This reduces quality degradation from 31.9% to 10.1%.

### Loss & Training
- Distillation dataset: YE-POP (500K images)
- 4-bit weight quantisation via bitsandbytes (W4A16)
- Target models: SD3.5 Large Turbo (8B parameters) and FLUX.1-Schnell (12B parameters)
- Training cost: only 615–1,287 A100 GPU hours, compared to 140k–200k A100 GPU hours required to train SD1.4/SD2.1 from scratch
- Threshold $r_{thres} = 0.25$: SGDistill is enabled when the compression ratio exceeds this value

## Key Experimental Results

### Main Results

**Image Quality and Memory Compression (SD3.5 Large Turbo):**

| Method | Memory (GB) | Reduction | GenEval↑ | HPSv2↑ | Quality Degradation↓ |
|--------|-------------|-----------|---------|--------|----------------------|
| Original | 15.8 (100%) | - | 0.71 | 30.29 | - |
| KOALA | 12.6 (79.4%) | 20.6% | 0.37 | 19.99 | 41.2% |
| KOALA+Quant | 3.56 (22.5%) | 77.5% | 0.33 | 18.44 | 46.4% |
| BK-SDM | 12.6 (79.4%) | 20.6% | 0.38 | 21.21 | 38.2% |
| BK-SDM+Quant | 3.56 (22.5%) | 77.5% | 0.34 | 19.83 | 43.3% |
| **Ours (HPP+PWP+Q)** | **3.56 (22.5%)** | **77.5%** | **0.69** | **28.15** | **4.8%** |
| **Ours (All)** | **3.24 (20.5%)** | **79.5%** | **0.62** | **26.29** | **13.3%** |
| SANA-Sprint-1.6B | 3.14 (100%) | - | 0.77 | 29.61 | - |

**FLUX.1-Schnell Results**: Ours (All) at 4.44 GB (19.6%) achieves GenEval 0.64 and HPSv2 28.69, with only 3.2% quality degradation, whereas KOALA degrades by 28.7% at 15.9 GB (70.5%).

**User Study (85 participants)**: HierarchicalPrune exhibits only a 4.8% drop in text alignment and 5.3% in image quality; SANA-Sprint drops 14.2% and 11.1% respectively; BK-SDM/KOALA drop 44.0–52.2%.

**Latency Comparison (A6000 GPU):**

| Model | Method | Latency | Reduction |
|-------|--------|---------|-----------|
| SD3.5 Large Turbo | Original | 823 ms | - |
| SD3.5 Large Turbo | Ours (HPP+PWP+Q) | 593 ms | 27.9% |
| FLUX.1-Schnell | Original | 756 ms | - |
| FLUX.1-Schnell | Ours (All) | 469 ms | 38.0% |

### Ablation Study

**Contribution of Each Component (SD3.5 Large Turbo):**

| Compression Level | Method | GenEval↑ | HPSv2↑ | Quality Degradation↓ |
|-------------------|--------|---------|--------|----------------------|
| None (0%) | Original | 0.71 | 30.29 | - |
| Moderate (20%) | HPP only | 0.03 | 11.08 | 79.4% |
| Moderate (20%) | +PWP | 0.71 | 28.97 | 2.5% |
| Moderate (20%) | +Quant | 0.69 | 28.15 | 4.8% |
| Aggressive (30%) | HPP only | 0.0 | 7.00 | 88.4% |
| Aggressive (30%) | +PWP | 0.46 | 21.74 | 31.9% |
| Aggressive (30%) | +SGDistill | 0.64 | 27.29 | 10.1% |
| Aggressive (30%) | +Quant | 0.62 | 26.29 | 13.3% |

Key observations: HPP alone collapses under moderate compression (GenEval 0.03); PWP is the cornerstone of quality recovery. SGDistill reduces degradation from 31.9% to 10.1% under aggressive compression. Quantisation introduces an additional 2.4–3.5% quality loss.

### Key Findings
- Removing early MMDiT blocks causes drastic changes to image structure, while removing late blocks affects only fine stylistic details.
- The compressed model (3.24 GB) has comparable memory footprint to SANA-Sprint (3.14 GB) but significantly superior image quality.
- HierarchicalPrune preserves the original model's text rendering capability, which neither SANA-Sprint nor other compression methods can achieve.

## Highlights & Insights
- The **dual-level hierarchical insight** is the core contribution: systematically characterising both inter-block functional differences and intra-block sub-component differences in MMDiT, establishing a new paradigm for large-scale diffusion model compression.
- **Counterintuitive distillation strategy**: applying minimal updates to the most important blocks runs counter to the conventional practice of prioritising updates on critical layers, yet proves highly effective under aggressive compression.
- This is the first work to unify depth pruning, knowledge distillation, and INT4 quantisation into a single framework for 8B+ diffusion models.

## Limitations & Future Work
- Validation is limited to SD3.5 and FLUX under the MMDiT architecture; U-Net and other generative model architectures are not evaluated.
- The 13.3% quality degradation under aggressive compression (30%) may still be unacceptable in certain application scenarios.
- Distillation cost (615–1,287 A100 GPU hours) remains relatively high for academic teams.
- The user study involves only 85 participants, a relatively limited sample size.

## Related Work & Insights
- **vs. KOALA**: Both are depth pruning methods, but KOALA relies on block-level cosine-similarity ranking and whole-block removal, resulting in 41.2% degradation on SD3.5; HierarchicalPrune achieves only 4.8% degradation through position-aware, sub-component-level fine-grained pruning.
- **vs. BK-SDM**: BK-SDM's CLIP-score-based block pruning incurs 38.2% degradation, reflecting a lack of understanding of the hierarchical structure of MMDiT.
- **vs. SANA-Sprint-1.6B**: This compactly trained model from scratch suffers 11.1–14.2% perceptual quality degradation, whereas the compressed large model degrades by only 4.8–5.3%, demonstrating the viability of the "compress a large model rather than train a small one" paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐ — The dual-level hierarchical insight and counterintuitive distillation strategy demonstrate strong originality.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Quantitative and qualitative evaluation, an 85-person user study, multi-GPU latency benchmarks, and comprehensive ablations are all included.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-motivated arguments, and thorough experimental descriptions.
- Value: ⭐⭐⭐⭐⭐ — First to achieve effective compression of 8B+ diffusion models from 15.8 GB to 3.24 GB, with significant practical deployment implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Compositional amortized inference for large-scale hierarchical Bayesian models](../../ICLR2026/image_generation/compositional_amortized_inference_for_large-scale_hierarchical_bayesian_models.md)
- [\[AAAI 2026\] HACK: Head-Aware KV Cache Compression for Efficient Visual Autoregressive Modeling](head-aware_kv_cache_compression_for_efficient_visual_autoreg.md)
- [\[CVPR 2026\] CG-Floor: Centroid-Guided Diffusion for Large-Scale Floorplan Generation](../../CVPR2026/image_generation/cg-floor_centroid-guided_diffusion_for_large-scale_floorplan_generation.md)
- [\[ICLR 2026\] Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency](../../ICLR2026/image_generation/large_scale_diffusion_distillation_via_score-regularized_continuous-time_consist.md)
- [\[ICLR 2026\] MMaDA-Parallel: Multimodal Large Diffusion Language Models for Thinking-Aware Editing and Generation](../../ICLR2026/image_generation/mmada-parallel_multimodal_large_diffusion_language_models_for_thinking-aware_edi.md)

</div>

<!-- RELATED:END -->
