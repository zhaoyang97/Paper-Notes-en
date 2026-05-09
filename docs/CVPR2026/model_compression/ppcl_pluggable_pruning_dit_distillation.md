---
title: >-
  [Paper Note] PPCL: Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers
description: >-
  [CVPR 2026][Model Compression][diffusion transformer] This paper proposes PPCL, a structured pruning framework tailored for large-scale Multi-Modal Diffusion Transformers (MMDiT, 8–20B parameters). It trains linear probes (Linear Probe) to assess the substitutability of each layer, automatically localizes contiguous redundant layer intervals via first-order differences of CKA, and applies non-sequential alternating distillation for dual-axis pruning along depth and width. On Qwen-Image 20B, PPCL achieves 50% parameter reduction and 1.8× inference speedup with an average performance drop of only 2.61%.
tags:
  - CVPR 2026
  - Model Compression
  - diffusion transformer
  - structured pruning
  - contiguous layer redundancy
  - knowledge distillation
  - MMDiT
date: 2026-05-08
content_hash: c74be223a73ae217
---

# PPCL: Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers

**Conference**: CVPR 2026
**arXiv**: [2511.16156](https://arxiv.org/abs/2511.16156)
**Code**: [GitHub](https://github.com/OPPO-Mente-Lab/Qwen-Image-Pruning)
**Area**: Model Compression / Diffusion Models
**Keywords**: diffusion transformer, structured pruning, contiguous layer redundancy, knowledge distillation, MMDiT

## TL;DR

This paper proposes PPCL, a structured pruning framework tailored for large-scale Multi-Modal Diffusion Transformers (MMDiT, 8–20B parameters). It trains linear probes (Linear Probe) to assess the substitutability of each layer, automatically localizes contiguous redundant layer intervals via first-order differences of CKA, and applies non-sequential alternating distillation for dual-axis pruning along depth and width. On Qwen-Image 20B, PPCL achieves 50% parameter reduction and 1.8× inference speedup with an average performance drop of only 2.61%.

## Background & Motivation

**Background**: State-of-the-art text-to-image (T2I) diffusion models have fully transitioned from UNet architectures to Multi-Modal Diffusion Transformers (MMDiT). While SDXL has 2.6B parameters, FLUX.1 reaches 12B and Qwen-Image reaches 20B (60 MMDiT blocks), greatly improving generation quality at the cost of substantially increased inference overhead.

**Limitations of Prior Work**: (a) Existing structured pruning methods (e.g., TinyFusion, SnapFusion) primarily target UNet architectures and cannot be directly transferred to the dual-stream structure of MMDiT; (b) prior methods evaluate redundancy independently per layer (e.g., sensitivity analysis), overlooking functional coupling between adjacent layers in DiT; (c) in conventional sequential distillation, compression errors from early layers propagate and accumulate along the network, causing severe representational drift in the student model.

**Key Challenge**: Through experiments, the authors find that redundancy in DiT exhibits **depth-wise contiguity**—removing contiguous layers has less impact on performance than removing an equal number of non-contiguous layers. Existing pruning methods do not exploit this property.

**Goal**: To systematically identify contiguous redundant layer intervals in MMDiT and design a distillation scheme that avoids error accumulation, thereby preserving quality under high compression ratios.

**Key Insight**: Replace traditional layer importance estimation with layer *substitutability*—if the input-output mapping of a layer can be approximated by a linear transformation, that layer is functionally redundant with respect to its neighbors.

**Core Idea**: In MMDiT, redundant layers are distributed contiguously along the depth dimension. They can be automatically localized and removed in segments via linear probes combined with CKA difference analysis, while non-sequential distillation eliminates error accumulation.

## Method

### Overall Architecture

PPCL proceeds in two stages and three steps:

- **Stage 1 (Depth Pruning)**: (1) Train a linear probe for each MMDiT block to assess layer substitutability; (2) automatically delineate the set of contiguous redundant layer intervals $\mathcal{I}$ by detecting inflection points in the first-order difference of CKA; (3) train the student model via non-sequential distillation, optimizing each interval independently.
- **Stage 2 (Width Pruning)**: Identify redundancy in the text stream and FFN, replacing them with lightweight linear projectors for further parameter reduction.
- A brief full-parameter fine-tuning is performed at the end.

### Key Designs

**1. Contiguous Redundant Layer Detection via Linear Probes**

- **Function**: Automatically identify a set of non-overlapping contiguous redundant layer intervals $\mathcal{I} = \{[u_i, v_i]\}$.
- **Mechanism**: A residual-structured linear probe $l_i$ is constructed for each teacher layer $T_i$, initialized via least squares and trained with an alignment loss $\mathcal{L}_{fit}(i) = \|l_i(T_{i-1}^D) + T_{i-1}^D - T_i(T_{i-1}^D)\|_2^2$. After training, the CKA similarity between the proxy model output (with consecutive layers replaced by linear probes) and the teacher layer output is computed on a calibration set. The first-order difference $\Delta(u,k) = -(\text{cka}(u,k) - \text{cka}(u,k-1))$ is defined; when $\Delta$ first decreases then increases, the inflection point $v$ marks the right endpoint of the redundant interval.
- **Design Motivation**: (a) Each linear probe receives the actual input of its corresponding layer, ensuring independent and non-interfering evaluation; (b) a finite composition of linear transformations remains linear, guaranteeing transitivity of substitutability across contiguous layers; (c) using the inflection point of the CKA difference rather than a fixed threshold enables adaptive localization of redundant interval lengths.
- **Key Formula**: Closed-form initialization of the linear probe: $W_i^* = (T_i(T_{i-1}^D) - T_{i-1}^D)(T_{i-1}^D)^\top(T_{i-1}^D(T_{i-1}^D)^\top)^{-1}$

**2. Non-Sequential Depth Pruning Distillation**

- **Function**: For each detected redundant interval $[u,v]$, replace it with a single student layer to avoid error accumulation.
- **Mechanism**: Each interval is optimized independently—student layer $S^u$ receives the output of teacher layer $u-1$ as input and is aligned to the output of teacher layer $v$. Loss function: $\mathcal{L}_{depth}^{[u,v]} = \|\text{Norm}(S^u(T_{u-1}^D)) - \text{Norm}(T_v^D)\|_2^2$, where Norm denotes L2 normalization, emphasizing directional alignment.
- **Design Motivation**: In conventional sequential distillation, student layer $k$ receives the output of student layer $k-1$, causing early errors to accumulate layer by layer. The non-sequential scheme provides each interval directly with teacher inputs, severing the error propagation chain.
- **Plug-and-Play Property**: Since each interval is trained independently, selected intervals can be flexibly enabled or bypassed at inference time—e.g., after training a 10B model, replacing some student layers back with teacher layers yields 12B or 14B variants without retraining.

**3. Width Pruning: Text Stream and FFN Compression**

- **Function**: Further compress parameters within retained layers by replacing redundant text stream structures and FFNs.
- **Mechanism**: (a) **Text stream pruning**: CKA heatmaps reveal highly similar cross-layer representations in the text stream; the text stream of redundant layers (excluding QKV projections) is replaced by two lightweight linear projections $l_p^z$ and $l_p^h$. (b) **FFN pruning**: Layers where replacing the FFN with a linear projection yields minimal MSE are identified, and their FFNs are replaced with linear projections $l_q^{img}$ and $l_q^{txt}$.
- **Design Motivation**: Text stream tokens in MMDiT exhibit high similarity and small inter-layer variation, enabling significant compression; FFNs are substantially over-parameterized, with many layers performing nearly linear transformations.
- **Loss & Training**: The width distillation loss consists of two terms—a layer-level alignment loss $\mathcal{L}_{width}^j$ (consistent with the depth distillation format) and a linear projection alignment loss $\mathcal{L}_{linear}^j$ (constraining linear projection outputs to approximate the corresponding intermediate teacher representations).

### Loss & Training

- **Data**: 100K images sampled from LAION-2B-en, with detailed captions generated by Qwen2.5-VL.
- **Three training stages**: Depth pruning 6k steps → Width pruning 2k steps → Full-parameter fine-tuning 1k steps (8 × H20 GPUs).
- **Optimizer**: AdamW ($\beta_1$=0.9, $\beta_2$=0.95, weight decay=0.02), BF16 mixed precision + gradient checkpointing.

## Key Experimental Results

### Main Results: Comparison on FLUX.1-dev

| Method | Params (B) | Memory (%) | Latency (ms) | DPG↑ | GenEval↑ | B-VQA↑ | UniDet↑ | Avg. Drop (%)↓ |
|---|---|---|---|---|---|---|---|---|
| Base model | 12 | 100 | 715 | 83.8 | 0.665 | 0.640 | 0.426 | 0 |
| TinyFusion | 8 | 74.4 | 534 | 77.2 | 0.511 | 0.584 | 0.369 | 13.80 |
| HierarchicalPrune | 8 | 74.4 | 543 | 75.7 | 0.503 | 0.579 | 0.371 | 13.38 |
| Dense2MoE | 12 | 100 | 312 | 73.6 | 0.403 | 0.473 | 0.311 | 21.52 |
| FLUX.1 Lite | 8 | 78.8 | 572 | 82.1 | 0.623 | 0.547 | 0.379 | 6.09 |
| Chroma1-HD | 8.9 | 82.5 | 1714 | 84.0 | 0.593 | 0.621 | 0.339 | 1.02 |
| **PPCL(8B)** | **8** | **74.4** | **535** | **80.0** | **0.605** | **0.615** | **0.391** | **4.03** |
| **PPCL(6.5B)** | **6.5** | **69.2** | **428** | **81.2** | **0.593** | **0.581** | **0.398** | **0.07** |

### Main Results: Comparison on Qwen-Image

| Method | Params (B) | Memory (%) | Latency (ms) | DPG↑ | GenEval↑ | LongText-EN↑ | LongText-ZH↑ | Avg. Drop (%)↓ |
|---|---|---|---|---|---|---|---|---|
| Base model | 20 | 100 | 2625 | 88.9 | 0.870 | 0.943 | 0.946 | 0 |
| TinyFusion(14B) | 14 | 79.4 | 1789 | 80.7 | 0.739 | 0.859 | 0.857 | 8.75 |
| HierarchicalPrune(14B) | 14 | 79.4 | 1786 | 83.3 | 0.766 | 0.884 | 0.881 | 6.49 |
| **PPCL(14B)** | **14** | **79.4** | **1792** | **87.9** | **0.847** | **0.929** | **0.946** | **0.42** |
| **PPCL(12B)** | **12** | **71.4** | **1514** | **83.6** | **0.801** | **0.893** | **0.917** | **3.03** |
| **PPCL(10B+FT)** | **10** | **66.9** | **1462** | **86.7** | **0.828** | **0.902** | **0.931** | **3.29** |

### Ablation Study (Qwen-Image, all pruned to ~10–12B)

| Configuration | LongText↑ | DPG↑ | GenEval↑ | Avg. | Params (B) | Avg. Drop (%)↓ |
|---|---|---|---|---|---|---|
| Original (20B) | 0.942 | 0.885 | 0.854 | 0.894 | 20 | 0 |
| Baseline (CKA + sequential distillation) | 0.625 | 0.763 | 0.728 | 0.706 | 12 | 18.2 |
| +LP (linear probe) | 0.712 | 0.795 | 0.776 | 0.761 | 12 | 14.5 |
| +LP-a (CKA threshold instead of difference) | 0.664 | 0.778 | 0.712 | 0.718 | 12 | 19.7 |
| +LP-b (enlarged interval upper bound) | 0.678 | 0.769 | 0.731 | 0.726 | 12 | 18.8 |
| +DP (non-sequential distillation) | 0.905 | 0.836 | 0.801 | 0.848 | 12 | 5.22 |
| +WP-text (text stream pruning) | 0.915 | 0.846 | 0.819 | 0.860 | 11 | 3.79 |
| +WP-ffn (FFN pruning) | 0.906 | 0.835 | 0.809 | 0.850 | 10 | 4.91 |
| +Fine-tuning | 0.916 | 0.867 | 0.828 | 0.870 | 10 | 2.61 |

### Key Findings

- **Contiguous vs. non-contiguous removal**: On the 60-layer Qwen-Image model, removing 1–3 layers contiguously consistently yields better generation quality than removing an equal number of non-contiguous layers, validating the depth-wise contiguity hypothesis.
- **Large gain from non-sequential distillation**: The average performance drop decreases from 14.5% (+LP) to 5.22% (+DP), demonstrating that non-sequential distillation alone contributes approximately 9 percentage points of improvement.
- **Width pruning improves metrics while reducing parameters**: +WP-text raises average performance from 0.848 to 0.860 while reducing parameters by 1B, attributed to the additional trainable linear layers compensating for residual layer misalignment.
- **Plug-and-play property**: PPCL(14B) and PPCL(12B) are obtained directly from the 10B model by replacing some student layers with teacher layers, requiring no additional training.
- **Re-pruning an already-pruned model**: Re-pruning FLUX.1 Lite (8B) down to 6.5B results in an average performance drop of only 0.07%.

## Highlights & Insights

- **Contiguous redundancy is an intrinsic property of DiT**: Unlike CNNs where redundancy is scattered, adjacent layers in MMDiT undergo smooth transitions in representation space, forming functional units that can be removed as contiguous segments. This finding establishes a new paradigm for DiT pruning.
- **Linear probes as a substitutability metric**: Compared to directly removing layers to assess sensitivity, linear probes more stably quantify the degree of linear approximability between layers and require only a single training pass to handle all layers simultaneously.
- **Non-sequential distillation is the key**: Ablation results show that non-sequential distillation contributes more than the redundancy detection method itself (9pp vs. 3.7pp), demonstrating that severing the error accumulation chain is the central factor for preserving quality under high compression ratios.
- **Complementarity of dual-axis compression**: Depth pruning reduces the number of layers while width pruning reduces parameters within retained layers; the two are orthogonal and additive, jointly achieving 50% compression.

## Limitations & Future Work

- The inflection-point detection via first-order CKA differences lacks rigorous theoretical grounding; the authors acknowledge this is primarily a successful engineering heuristic.
- INT4 quantization performs poorly after PPCL pruning—pruning reduces network redundancy and thereby narrows the error tolerance margin for quantization; joint optimization of pruning and quantization warrants further investigation.
- Training still requires 8 × H20 GPUs, and scalability to larger models (e.g., 100B scale) remains to be verified.
- Details of the selection strategy for which layers belong to $\mathcal{R}_{txt}$ and $\mathcal{R}_{ffn}$ in width pruning are deferred to the appendix, with limited description in the main text.

## Related Work & Insights

- **vs. TinyFusion**: TinyFusion uses differentiable gating parameters to identify removable layers but evaluates redundancy independently per layer, ignoring contiguity, and applies standard sequential distillation leading to error accumulation. On FLUX.1 and Qwen-Image, PPCL achieves average drops of 4.03% / 0.42%, significantly outperforming TinyFusion's 13.80% / 8.75%.
- **vs. HierarchicalPrune**: HPP's layer importance estimation is relatively coarse, and generated results exhibit visual artifacts; PPCL shows a clear advantage at the same compression ratio (0.42% vs. 6.49%).
- **vs. Dense2MoE**: Dense2MoE replaces FFNs with MoE to reduce activation cost without reducing parameter count, and achieves an average drop of 21.52%, demonstrating that naive sub-structure replacement is inferior to systematic pruning combined with distillation.
- **vs. Chroma1-HD**: Chroma1-HD achieves the lowest average drop on FLUX.1 (1.02%) but increases inference latency by 2.4×, failing to meet acceleration requirements.

## Rating

- Novelty: ⭐⭐⭐⭐ The contiguous redundancy hypothesis is well validated experimentally; both the linear probe + CKA difference interval detection and non-sequential distillation represent effective innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on two mainstream MMDiT models (FLUX.1 and Qwen-Image); ablation studies carefully decompose the contribution of each component.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, with a complete logical chain from observation → hypothesis → design.
- Value: ⭐⭐⭐⭐⭐ Directly addresses deployment bottlenecks for 20B-scale DiT models; the engineering value of 50% compression and 1.8× speedup is substantial, further enhanced by the plug-and-play property.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers](binaryattention_one-bit_qk-attention_for_vision_and_diffusion_transformers.md)
- [\[CVPR 2026\] OPAD: Adversarial Concept Distillation for One-Step Diffusion Personalization](opad_adversarial_concept_distillation_for_one-step_diffusion_personalization.md)
- [\[CVPR 2026\] FAIR-Pruner: Leveraging Tolerance of Difference for Flexible Automatic Layer-Wise Neural Network Pruning](fair-pruner_leveraging_tolerance_of_difference_for_flexible_automatic_layer-wise.md)
- [\[CVPR 2026\] HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](hiap_a_multigranular_stochastic_autopruning_framew.md)
- [\[CVPR 2026\] Adversarial Concept Distillation for One-Step Diffusion Personalization](adversarial_concept_distillation_for_one-step_diffusion_personalization.md)

</div>

<!-- RELATED:END -->
