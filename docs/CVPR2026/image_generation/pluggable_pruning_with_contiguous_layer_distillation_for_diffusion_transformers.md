---
title: >-
  [Paper Note] Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers
description: >-
  [CVPR 2026][Image Generation][Diffusion Transformer Pruning] This paper proposes the PPCL framework, which employs linear probing and first-order CKA difference analysis to detect contiguous redundant layer intervals in…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion Transformer Pruning"
  - "MMDiT Compression"
  - "Contiguous Layer Distillation"
  - "Plug-and-Play Inference"
  - "Structured Pruning"
date: 2026-05-08
content_hash: 0bf858881d35c003
---

# Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers

**Conference**: CVPR 2026
**arXiv**: [2511.16156](https://arxiv.org/abs/2511.16156)  
**Code**: [https://github.com/OPPO-Mente-Lab/Qwen-Image-Pruning](https://github.com/OPPO-Mente-Lab/Qwen-Image-Pruning)  
**Area**: Diffusion Models / Model Compression
**Keywords**: Diffusion Transformer Pruning, MMDiT Compression, Contiguous Layer Distillation, Plug-and-Play Inference, Structured Pruning

## TL;DR

This paper proposes the PPCL framework, which employs linear probing and first-order CKA difference analysis to detect contiguous redundant layer intervals in MMDiT, combined with non-sequential distillation to enable depth pruning (plug-and-play) and width pruning (replacing text streams/FFNs with linear projections). The approach compresses Qwen-Image from 20B to 10B with only a 3.29% performance drop.

## Background & Motivation

**Background**: Diffusion Transformers (DiT) have become the dominant architecture for text-to-image generation. Models such as SD3.5, FLUX.1, and Qwen-Image substantially surpass the previous U-Net generation in image quality and text alignment. However, parameter counts have escalated from 2.6B (SDXL) to 20B (Qwen-Image), imposing prohibitive inference costs.

**Limitations of Prior Work**: Existing structured pruning methods suffer from three critical limitations: (a) incompatibility with the MMDiT (multimodal DiT) architecture; (b) poor flexibility in layer pruning, lacking support for plug-and-play configurations; and (c) insufficient understanding of inter-layer dependencies in deep diffusion models.

**Key Challenge**: Through extensive experiments on Qwen-Image (60-layer MMDiT), the authors identify two key phenomena: randomly removing 1–3 layers has negligible impact on generation quality (indicating high layer redundancy), and **contiguous removal** consistently outperforms **non-contiguous removal**. This suggests that redundancy exhibits depth-wise continuity, yet efficiently detecting such contiguous redundant intervals remains an open problem.

**Goal**: (a) How to maximally identify contiguous redundant layer subsets? (b) How to prevent error accumulation across layers during post-pruning distillation? (c) How to support plug-and-play deployment across different compression ratios without retraining?

**Key Insight**: The representational evolution of a teacher model does not proceed uniformly but rather in stages — within each stage, layer activations transition smoothly and can be approximated by linear functions. When the input-output mapping of a layer can be fitted by a linear probe, that layer is functionally redundant with respect to its neighbors.

**Core Idea**: Linear probes combined with first-order CKA difference analysis detect contiguous redundant layer intervals; non-sequential distillation breaks the error propagation chain; lightweight linear projections perform width pruning — together realizing plug-and-play DiT compression.

## Method

### Overall Architecture

PPCL operates in two stages: **depth pruning** and **width pruning**.

Depth pruning consists of three steps: (1) linear probe training — a linear probe is trained per MMDiT block to approximate its input-output mapping; (2) simulated pruning — first-order CKA difference trend analysis identifies the set of contiguous redundant layer intervals $\mathcal{I}$; (3) non-sequential layer-wise distillation — student layers directly receive the output of the preceding teacher interval as input and optimize each interval independently.

Width pruning targets the dual-stream structure of MMDiT, replacing redundant text streams and FFNs with lightweight linear projections. A brief full-parameter fine-tuning step follows.

### Key Designs

1. **Linear Probing Redundancy Detection (Linear Probing + CKA First-Order Difference)**

    - **Function**: Identify contiguous redundant layer intervals that can be replaced by linear functions.
    - **Mechanism**: A residual-structured linear probe $l_i$ is constructed for each teacher layer $T_i$. Its weight $W_i^*$ is initialized via least squares and then trained to fit the input-output mapping of $T_i$, with loss $\mathcal{L}_{fit}(i) = \|l_i(T_{i-1}^D) + T_{i-1}^D - T_i(T_{i-1}^D)\|_2^2$. After training, inference is performed on a calibration set to construct substitute models $T^{[u \to k]}$ (replacing $T_{u+1}, \dots, T_k$ with corresponding linear probes). The first-order difference of CKA similarity is computed as $\Delta(u,k) = -(\text{cka}(u,k) - \text{cka}(u,k-1))$. An inflection point where $\Delta$ first decreases then increases signals the end of a contiguous redundant interval.
    - **Design Motivation**: Linear probe training uses inputs consistent with actual layer inputs, ensuring each layer is modeled independently. A composition of finitely many linear transforms remains linear, guaranteeing transitivity of contiguous-layer replaceability. Compared to simple CKA thresholding or layer sensitivity ranking, first-order difference trend analysis more precisely localizes the boundaries of redundant intervals.

2. **Non-Sequential Depth-wise Distillation**

    - **Function**: Perform knowledge distillation independently for each detected redundant interval $[u, v]$.
    - **Mechanism**: A student layer $S^u_{init}$ is initialized with the weights of teacher layer $T_u$. The output of teacher layer $u-1$ is fed directly as the student input, and the student is trained to align its output with that of teacher layer $v$. The loss is $\mathcal{L}_{depth}^{[u,v]} = \|\text{Norm}(S^u_{init}(T_{u-1}^D)) - \text{Norm}(T_v^D)\|_2^2$, where Norm denotes L2 normalization to emphasize directional alignment. The total loss is the sum over all intervals.
    - **Design Motivation**: In conventional sequential distillation, errors in earlier layers propagate and amplify. The non-sequential design breaks this error propagation chain, allowing each interval to be optimized independently. More importantly, at inference time specific layers can be flexibly activated or skipped — enabling 12B and 14B variants to be derived directly from a single 10B model without retraining.

3. **Width-wise Pruning (Stream + FFN)**

    - **Function**: Further compress the width redundancy of MMDiT on top of depth pruning.
    - **Mechanism**: CKA heatmap analysis reveals highly similar cross-layer representations in the text stream, indicating significant redundancy. The method replaces the text stream (excluding QKV projections) in redundant layers with two lightweight linear projections $l_p^z$ and $l_p^h$. For FFN-redundant layers, the image-stream and text-stream FFNs are replaced by linear projections $l_q^{img}$ and $l_q^{txt}$, respectively. The distillation loss includes a layer-level output alignment loss $\mathcal{L}_{width}^j$ and a linear projection alignment loss $\mathcal{L}_{linear}^j$.
    - **Design Motivation**: Depth pruning reduces model depth while width redundancy persists. Text-stream tokens exhibit high similarity and minimal cross-layer variation, making them amenable to aggressive compression. FFNs are substantially over-parameterized and can be effectively approximated by linear projections. Dual-axis compression further reduces model size and mitigates distillation target drift.

### Loss & Training

- **Training proceeds in three stages**: depth distillation for 6k steps (8×H20 GPUs, micro-batch=2) → width distillation for 2k steps → full-parameter fine-tuning for 1k steps (micro-batch=4).
- Training data: 100k images sampled from LAION-2B-en, with captions generated by Qwen2.5-VL and training pairs generated by Qwen-Image.
- Optimizer: AdamW ($\beta_1$=0.9, $\beta_2$=0.95, weight decay=0.02), BF16 mixed precision with gradient checkpointing.

## Key Experimental Results

### Main Results

Comparison with multiple compression methods on FLUX.1-dev and Qwen-Image across DPG, GenEval, LongText-Bench, OneIG-Bench, and T2I-CompBench:

| Model | Method | Params (B) | Latency (ms) | Avg. Performance Drop (%) |
|-------|--------|------------|--------------|--------------------------|
| FLUX.1-dev | Original | 12 | 715 | 0 |
| FLUX.1-dev | TinyFusion | 8 | 534 | 13.80 |
| FLUX.1-dev | HierarchicalPrune | 8 | 543 | 13.38 |
| FLUX.1-dev | **PPCL (8B)** | 8 | 535 | **4.03** |
| FLUX.1 Lite | **PPCL (6.5B)** | 6.5 | 428 | **0.07** |
| Qwen-Image | Original | 20 | 2625 | 0 |
| Qwen-Image | TinyFusion | 14 | 1789 | 8.75 |
| Qwen-Image | HierarchicalPrune | 14 | 1786 | 6.49 |
| Qwen-Image | **PPCL (14B)** | 14 | 1792 | **0.42** |
| Qwen-Image | **PPCL (10B+FT)** | 10 | 1462 | **3.29** |

### Ablation Study

Incremental component addition on Qwen-Image (25 layers pruned, evaluated by average LongText/DPG/GenEval scores):

| Configuration | LongText | DPG | GenEval | Avg. | Params (B) | Drop (%) |
|---------------|----------|-----|---------|------|------------|----------|
| Original | 0.942 | 0.885 | 0.854 | 0.894 | 20 | 0 |
| Baseline (CKA sensitivity + sequential distillation) | 0.625 | 0.763 | 0.728 | 0.706 | 12 | 18.2 |
| +LP (linear probe detection) | 0.712 | 0.795 | 0.776 | 0.761 | 12 | 14.5 |
| +DP (non-sequential distillation) | 0.905 | 0.836 | 0.801 | 0.848 | 12 | 5.22 |
| +WP-text (text stream → linear) | 0.915 | 0.846 | 0.819 | 0.860 | 11 | 3.79 |
| +WP-ffn (FFN → linear) | 0.906 | 0.835 | 0.809 | 0.850 | 10 | 4.91 |
| +Fine-tuning | 0.916 | 0.867 | 0.828 | 0.870 | 10 | **2.61** |

### Key Findings

- **Contiguous vs. non-contiguous removal**: Experiments removing 1–3 layers from Qwen-Image consistently show that contiguous removal outperforms non-contiguous removal, validating the depth-wise continuity hypothesis of redundancy.
- **Non-sequential distillation is the dominant contributor**: From baseline to +DP, the average score improves from 0.706 to 0.848 (+14.2 percentage points), demonstrating that breaking the error propagation chain is the core factor.
- **Plug-and-play flexibility**: 12B (−3.03%) and 14B (−0.42%) variants are obtained directly from the trained 10B model by substituting student layers with teacher layers, requiring no additional training.
- **Effectiveness on already-compressed models**: Applying the method to FLUX.1 Lite (8B) and further pruning to 6.5B yields only a 0.07% performance drop.
- **50% compression ratio**: Qwen-Image 20B→10B achieves nearly 2× inference speedup and approximately 33% GPU memory reduction.

## Highlights & Insights

- **The discovery of contiguous redundancy** is a key observation — redundancy is not randomly distributed across layers but forms functionally coupled units of consecutive layers that can be replaced as a whole, which is more efficient than per-layer sensitivity analysis.
- **The linear probe + CKA first-order difference** detection strategy is highly lightweight: each probe comprises a single linear layer, training is independent and parallelizable, and detection requires only one calibration-set inference pass.
- **Non-sequential distillation** is an elegant design — independent optimization per interval naturally supports plug-and-play usage and multi-compression-ratio deployment, which is highly practical for production deployment.
- **Dual-axis compression** exploits the characteristics of MMDiT's dual-stream architecture (text-stream redundancy substantially exceeds image-stream redundancy), constituting an architecture-aware compression strategy.
- The overall training cost is low: 6k+2k+1k steps on 8 H20 GPUs, which is negligible compared to full retraining.

## Limitations & Future Work

- **Lack of theoretical guarantees for CKA first-order difference inflection point detection**: The authors acknowledge that this is a successful engineering heuristic without rigorous theoretical foundations.
- **Incompatibility with INT4 quantization**: Pruning reduces model redundancy, narrowing the error tolerance for quantization, resulting in degraded INT4 performance. Joint optimization of pruning and quantization warrants further exploration.
- Experiments are conducted solely on T2I tasks; extension to video generation (e.g., DiT-based video models) has not been explored.
- Linear probe detection depends on the calibration set; different calibration sets may yield different interval partitions, and robustness remains to be verified.

## Related Work & Insights

- **TinyFusion** (CVPR 2025): Employs differentiable gating parameters for layer selection and removal with standard distillation, but achieves limited compression ratios.
- **HierarchicalPrune**: Uses hierarchical positional pruning with positional weight preservation, but layer importance estimation is relatively coarse.
- **Dense2MoE**: Replaces FFNs with MoE to reduce activation cost, but total parameter count remains unchanged.
- **FLUX.1 Lite / Chroma1-HD**: Open-source compressed variants; the former achieves 20% speedup with quality degradation, while the latter preserves quality but paradoxically increases inference latency.
- **Core Insight**: **Structured pruning must be aligned with architectural characteristics** — the dual-stream design, residual connections, and inter-layer similarity patterns of MMDiT all provide informative compression signals.

## Rating

| Dimension | Score (1–10) | Notes |
|-----------|-------------|-------|
| Novelty | 7 | Contiguous redundant layer detection strategy is innovative; plug-and-play distillation design is practical |
| Technical Depth | 7 | Linear probe + CKA analysis is thorough, though some design choices lack theoretical justification |
| Experimental Thoroughness | 8 | Multi-model (FLUX.1/Qwen-Image), multi-benchmark evaluation with complete ablations |
| Value | 9 | Low training cost, high compression ratio, plug-and-play deployment — strong industrial applicability |
| Writing Quality | 7 | Structure is clear, but notation is dense and some descriptions could be more concise |
| **Overall** | **7.6** | An efficient compression framework for MMDiT with outstanding engineering practicality |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models](razor_ratio-aware_layer_editing_for_targeted_unlearning_in_vision_transformers_a.md)
- [\[CVPR 2026\] From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [\[CVPR 2026\] Learnability-Guided Diffusion for Dataset Distillation](learnability-guided_diffusion_for_dataset_distillation.md)
- [\[CVPR 2026\] PixelDiT: Pixel Diffusion Transformers for Image Generation](pixeldit_pixel_diffusion_transformers_for_image_generation.md)
- [\[CVPR 2026\] Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers](circuit_mechanisms_for_spatial_relation_generation_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
