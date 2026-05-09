---
title: >-
  [Paper Note] OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning
description: >-
  [ICCV 2025][LLM Evaluation][image difference captioning] This paper introduces OmniDiff, a fine-grained image difference captioning dataset comprising 324 diverse scenes (real-world and 3D synthetic), and proposes a plug-and-play Multi-scale Differential Perception (MDP) module integrated into an MLLM to build the M3Diff model, achieving state-of-the-art performance on OmniDiff and multiple public benchmarks.
tags:
  - ICCV 2025
  - LLM Evaluation
  - image difference captioning
  - multimodal large language models
  - multi-scale differential perception
  - change detection
  - benchmark
date: 2026-05-08
content_hash: ad44ce24174d22c0
---

# OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning

**Conference**: ICCV 2025
**arXiv**: [2503.11093](https://arxiv.org/abs/2503.11093)
**Code**: [yuan-liu-omnidiff.github.io](https://yuan-liu-omnidiff.github.io)
**Area**: Vision-Language Understanding
**Keywords**: image difference captioning, multimodal large language models, multi-scale differential perception, change detection, benchmark

## TL;DR

This paper introduces OmniDiff, a fine-grained image difference captioning dataset comprising 324 diverse scenes (real-world and 3D synthetic), and proposes a plug-and-play Multi-scale Differential Perception (MDP) module integrated into an MLLM to build the M3Diff model, achieving state-of-the-art performance on OmniDiff and multiple public benchmarks.

## Background & Motivation

### Problem Definition

Image Difference Captioning (IDC) aims to generate natural language descriptions that precisely characterize subtle differences between two similar images, requiring both accurate visual change localization and coherent semantic expression.

### Limitations of Prior Work

Existing datasets suffer from deficiencies in both **breadth** and **depth**:

**Insufficient breadth**: Prior datasets are confined to limited object changes within specific scenarios. Spot-the-Diff covers only fixed-viewpoint street surveillance; Birds-to-Words focuses exclusively on fine-grained differences between bird species; CLEVR-Change renders only simple tabletop scenes.

**Insufficient depth**: Descriptions in prior benchmarks are overly concise. IEdit has an average description length of only 8 words, which fails to capture the complexity of real-world changes.

**Key Challenge**: The absence of a unified benchmark that combines scene diversity with descriptive granularity limits the applicability of IDC models in complex, dynamic environments.

### Starting Point

The paper addresses the problem from both data and model perspectives simultaneously: constructing a high-quality dataset covering 12 change types with an average description length of 60 words, and designing a dedicated module to enhance fine-grained difference perception in MLLMs.

## Method

### Overall Architecture

M3Diff is built upon the LLaVA-OneVision-7B architecture. An MDP module is incorporated into the standard MLLM framework to perform feature-level differencing on image pairs. Adaptive cross-layer fusion generates coherent difference representations, which are then passed to the language decoder for caption generation.

### Key Designs

#### 1. **OmniDiff Dataset Construction**
- **Function**: Constructs a high-quality IDC dataset containing 324 scenes, covering both real-world and 3D synthetic environments.
- **Mechanism**:
    - Real data: on-site photography and web crawling, covering 224 real-world scenes.
    - Synthetic data: complex 3D scenes rendered with Blender (50 indoor + 50 outdoor scenes sourced from ArtStation).
    - Manual annotation: fine-grained difference descriptions averaging 60 words, encompassing 12 change types (viewpoint, lighting, addition, disappearance, replacement, size, color, orientation, pose, OCR, counting, etc.).
- **Design Motivation**: Unlike the CLEVR series restricted to simple tabletop environments, this work emphasizes constructing complex scenes close to real-world conditions, thereby imposing higher demands on models' 3D spatial perception.

#### 2. **Differential Perception Module**
- **Function**: Extracts differential features from image pairs via channel-wise subtraction and fuses them back into the original features through cross-attention.
- **Mechanism**:
  1. Channel-wise gated modulation of image pair features $\mathbf{F}_1^i, \mathbf{F}_2^i$: $\boldsymbol{\lambda}_k = \sigma(\mathbf{W}_m[\mathbf{F}_1^i \| \mathbf{F}_2^i])$
  2. Differential feature computation: $\Delta\mathbf{F}^i = \mathbf{W}_p[\hat{\mathbf{F}}_1^i \| \hat{\mathbf{F}}_2^i \| (\hat{\mathbf{F}}_1^i - \hat{\mathbf{F}}_2^i)]$
  3. Differential signals are fused back into the original features via self-attention and cross-attention.
- **Design Motivation**: Naive subtraction is fragile for misaligned image pairs; the gating mechanism and cross-attention enhance robustness against confounding factors such as viewpoint and lighting changes.

#### 3. **Multi-Scale Integration**
- **Function**: Fuses features from multiple layers (layers 17/20/23/26) of the visual encoder.
- **Mechanism**:
    - Per-layer fusion weights $\text{Score}^i$ are computed via average pooling and an MLP.
    - Final features are obtained by weighted summation: $\mathbf{F}_k' = \sum_i \text{Score}^i \odot \tilde{\mathbf{F}}_k^i$
- **Design Motivation**: Low-level features lack semantic consistency, while high-level features lose fine-grained perceptual detail; multi-layer fusion balances semantics and detail.

### Loss & Training

- A simple and efficient **single-stage fine-tuning** strategy is adopted, in contrast to prior multi-stage approaches.
- LoRA (rank=128, alpha=256) is applied to the LLM for parameter-efficient fine-tuning.
- The visual encoder, projector, and MDP module are fully fine-tuned.
- The fine-tuning dataset contains 896K QA pairs sourced from OmniDiff, Spot-the-Diff, IEdit, Birds-to-Words, CLEVR-Change, and CLEVR-DC.
- Training is conducted on 8×A100 (40G) GPUs with a global batch size of 256, taking 26 hours.

## Key Experimental Results

### Main Results (OmniDiff Benchmark)

| Method | Real BLEU-4 | Real CIDEr | Render BLEU-4 | Render CIDEr |
|--------|------------|------------|---------------|-------------|
| CARD (ACL'24) | 9.1 | 9.2 | 11.3 | 7.3 |
| GPT-4o (zero-shot) | 3.1 | 5.2 | 4.6 | 5.6 |
| Qwen-2.5-VL-7B (zero-shot) | 3.8 | 6.2 | 2.1 | 3.3 |
| FINER-MLLM (MM'24) | 8.9 | 11.7 | 13.6 | 14.0 |
| **M3Diff (Ours)** | **14.3** | **31.3** | **15.7** | **28.3** |

### Cross-Benchmark Performance (Spot-the-Diff / IEdit / CLEVR-DC)

| Benchmark | Metric | M3Diff | Prev. SOTA | Gain |
|-----------|--------|--------|------------|------|
| Spot-the-Diff | CIDEr | **71.1** | 61.8 (FINER-MLLM) | +15% |
| IEdit | CIDEr | **136.6** | 109.6 (OneDiff) | +25% |
| CLEVR-DC | CIDEr | **109.4** | 84.1 (DIRL) | +30% |

### Ablation Study

| Configuration | OmniDiff-Real CIDEr | IEdit CIDEr | Notes |
|---------------|---------------------|-------------|-------|
| w/o OmniDiff & MDP | 1.1 | 133.5 | Baseline |
| w/o OmniDiff | 1.9 | 132.8 | MDP only |
| w/o MDP | 35.3 | 135.2 | OmniDiff data only |
| M3Diff (full) | **31.3** | **136.6** | Both data and module contribute |

### Key Findings

1. Zero-shot MLLMs (GPT-4o, Qwen-2.5-VL) perform poorly on complex scene difference captioning, indicating that IDC requires dedicated differential perception capabilities.
2. The OmniDiff dataset is critical for complex scene captioning—without OmniDiff training data, the model nearly fails on this benchmark.
3. The MDP module and OmniDiff data are complementary: the data provides scene diversity while the module enhances differential perception.
4. M3Diff's single-stage fine-tuning strategy achieves state-of-the-art performance while avoiding complex training pipelines.

## Highlights & Insights

1. **Well-designed dataset**: 12 change types × real + synthetic scenes × 60-word average description length, addressing both breadth and depth deficiencies of existing IDC benchmarks.
2. **Plug-and-play design**: The MDP module can be incorporated into existing MLLMs via single-stage fine-tuning, with a clean engineering implementation.
3. **Multi-scale differential perception**: Explicitly modeling feature differences across multiple layers of the visual encoder proves more effective than relying solely on the final layer.
4. The 3D synthetic scenes leverage ArtStation assets to construct near-realistic complex environments, surpassing the simplistic tabletop settings of the CLEVR series.

## Limitations & Future Work

1. Although the dataset (15,598 pairs) offers advantages in description quality, its total scale is smaller than CLEVR-Change (79,606 pairs) and can be further expanded.
2. The current work only supports difference captioning for image pairs and does not extend to continuous change description in video sequences.
3. The ratio of 3D synthetic to real data is approximately 1:1.2, which may introduce domain shift.
4. The MDP module exhibits minor fluctuations on certain metrics (e.g., OmniDiff-Real CIDEr), necessitating more robust fusion strategies.

## Related Work & Insights

- FINER-MLLM and OneDiff are representative recent works applying MLLMs to IDC, but lack dedicated differential perception modules.
- The CLEVR series provides a controllable synthetic evaluation environment; this paper extends the synthetic setting from simple tabletop scenes to complex realistic environments.
- The multi-scale feature fusion paradigm can be transferred to other tasks requiring fine-grained visual comparison, such as VQA and image editing evaluation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dataset construction is thorough and the MDP module design is clear, though the technical approach is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers six benchmarks with comprehensive ablations and zero-shot MLLM comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — The organizational logic for both the dataset and method is clear.
- **Value**: ⭐⭐⭐⭐ — OmniDiff fills a gap in IDC benchmarks, and M3Diff provides a strong baseline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark](3dsrbench_a_comprehensive_3d_spatial_reasoning_benchmark.md)
- [\[ACL 2026\] Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation](../../ACL2026/llm_evaluation/rethinking_meeting_effectiveness_a_benchmark_and_framework_for_temporal_fine-gra.md)
- [\[ICCV 2025\] On the Robustness Tradeoff in Fine-Tuning](on_the_robustness_tradeoff_in_fine-tuning.md)
- [\[NeurIPS 2025\] RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases](../../NeurIPS2025/llm_evaluation/rdb2g-bench_a_comprehensive_benchmark_for_automatic_graph_modeling_of_relational.md)
- [\[ICLR 2026\] Enabling Fine-Grained Operating Points for Black-Box LLMs](../../ICLR2026/llm_evaluation/enabling_fine-grained_operating_points_for_black-box_llms.md)

</div>

<!-- RELATED:END -->
