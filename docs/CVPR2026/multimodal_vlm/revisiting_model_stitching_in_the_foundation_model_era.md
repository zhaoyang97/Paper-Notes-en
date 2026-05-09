---
title: >-
  [Paper Note] Revisiting Model Stitching In the Foundation Model Era
description: >-
  [CVPR 2026][Multimodal VLM][Model stitching] A two-stage stitching training method (Final Feature Matching + Task Loss Training) for heterogeneous Vision Foundation Models (VFMs) is proposed, demonstrating that heterogeneous VFMs can be reliably stitched and fused for complementary knowledge. A VFM Stitch Tree (VST) architecture is also designed to achieve controllable accuracy–efficiency trade-offs in multi-VFM systems.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Model stitching
  - vision foundation models
  - representation compatibility
  - feature matching
  - multi-VFM fusion
  - efficiency optimization
date: 2026-05-08
content_hash: af5fe93a1fb6b955
---

# Revisiting Model Stitching In the Foundation Model Era

**Conference**: CVPR 2026
**arXiv**: [2603.12433](https://arxiv.org/abs/2603.12433)
**Code**: To be confirmed
**Area**: Multimodal VLM
**Keywords**: Model stitching, vision foundation models, representation compatibility, feature matching, multi-VFM fusion, efficiency optimization

## TL;DR

A two-stage stitching training method (Final Feature Matching + Task Loss Training) for heterogeneous Vision Foundation Models (VFMs) is proposed, demonstrating that heterogeneous VFMs can be reliably stitched and fused for complementary knowledge. A VFM Stitch Tree (VST) architecture is also designed to achieve controllable accuracy–efficiency trade-offs in multi-VFM systems.

## Background & Motivation

1. **Proliferation of VFMs with unknown representation compatibility**: Contemporary VFMs such as CLIP, DINOv2, and SigLIP 2 differ substantially in training objectives (contrastive learning vs. self-supervised reconstruction), data sources (LAION vs. WebLI vs. LVD-142M), and modality combinations (vision-language vs. vision-only), yet whether their internal representations are mutually compatible remains unclear.

2. **Model stitching as a probe for representation compatibility**: Prior work has shown that small models trained on the same dataset (e.g., ResNet-18 on CIFAR-10) can be stitched with near-lossless accuracy even under different initializations or objectives, but whether this finding generalizes to heterogeneous VFMs has not been verified.

3. **Failure of existing stitching strategies on VFMs**: Conventional Layer Feature Matching (LFM)—which aligns features at the stitch point—and Task Loss Training (TLT)—which directly optimizes the downstream loss—perform poorly in the VFM setting, especially at shallow stitch points.

4. **Gradient propagation difficulty at shallow stitch points**: When the stitch point is shallow, all subsequent layers of the target model are frozen, so gradients must traverse a long chain of frozen layers to update the stitch layer, making optimization difficult.

5. **Efficiency bottleneck in multi-VFM systems**: Modern multimodal LLMs (e.g., MoF-LLaVA, Cambrian-1) deploy multiple VFMs to capture complementary visual information, incurring linearly growing compute and memory costs ($k$ VFMs = $k\times$ overhead).

6. **Demand for a transition from diagnostic tool to practical solution**: There is a need to elevate model stitching from a representation analysis tool to a practical approach for fusing complementary VFM capabilities.

## Method

### Overall Architecture

Given a source model $f_\theta$ and a target model $f_\phi$, stitching is performed at layer $n$: the first $n$ layers of the source model and the last $N-n$ layers of the target model are frozen, and only a lightweight stitch layer $S$ is trained. The stitched model is defined as $F(x) = T_\phi^N \circ S \circ R_\theta^n(x)$.

### Key Designs: Two-Stage Training

**Stage 1: Final Feature Matching (FFM)**

- Rather than matching intermediate features at the stitch point, the final-layer output features of the target model are matched.
- Loss function: $\mathcal{L}_{FFM} = \frac{1}{M}\sum_{i=1}^{M}\|T_\phi^N(S(R_\theta^n(x_i))) - T_\phi^N(R_\phi^n(x_i))\|_2^2$
- Key finding: Although FFM directly matches final-layer features, it also implicitly preserves local alignment at the stitch point (layer feature distance is comparable to that of LFM).
- This stage requires **no labels**; only unlabeled images are needed.

**Stage 2: Task Loss Training (TLT)**

- The stitch layer is initialized with the parameters obtained from Stage 1 FFM, avoiding the optimization difficulties caused by random initialization.
- Fine-tuning on the downstream task: $\mathcal{L}_{task} = \frac{1}{M}\sum_{i=1}^{M}\ell(F(x_i), y_i)$
- FFM initialization places the stitch layer in a more favorable loss landscape, and subsequent fine-tuning converts this favorable initialization into strong stitching accuracy.

### Stitch Layer Design

- **Linear**: Processes tokens independently; weakest expressive capacity.
- **MLP** (default): Two-layer perceptron with ReLU; best overall performance.
- **LoRA**: Applies LoRA to layer $n$ of the source model, allowing inter-token interactions; strongest expressive capacity in theory, yet underperforms MLP—indicating that moderate mismatch facilitates complementary information fusion.

### VFM Stitch Tree (VST)

- **Core Idea**: Multiple VFMs share shallow-layer computation, retaining only their respective specialized deep layers, connected via stitch layers.
- **Architecture**: A tree structure in which the trunk consists of the shallow layers of one VFM and branches consist of the deep layers of each VFM.
- Using Cambrian-1 (4 VFMs) as an example, stitching at layer 14 reduces GPU memory and compute by 54%.

## Key Experimental Results

### Two-Stage Training vs. Naive TLT (fMoW Classification, Accuracy %)

| Direction | Pretrain | L2 | L6 | L10 | L14 | L18 | L22 |
|-----------|----------|------|------|------|------|------|------|
| DINOv2→SigLIP2 | None | 25.1 | 39.4 | 52.6 | 62.3 | 68.6 | 68.6 |
| DINOv2→SigLIP2 | FFM | **51.7** | **55.8** | **59.3** | **68.0** | **72.0** | **71.8** |
| SigLIP2→DINOv2 | None | 38.7 | 56.7 | 58.3 | 64.4 | 70.4 | 70.1 |
| SigLIP2→DINOv2 | FFM | **53.8** | **53.8** | **61.9** | **69.6** | **70.4** | **72.2** |

FFM initialization yields gains of up to **+26.6%** at shallow stitch points (L2), with consistent improvements at deeper layers as well.

### Consistency Across Datasets and Tasks (Classification Accuracy % / Segmentation mIoU %)

| Direction | fMoW (L6/14/22) | iNaturalist (L6/14/22) | Aircraft (L6/14/22) | ADE20K (L14/22) |
|-----------|-----------------|------------------------|---------------------|-----------------|
| Self-Stitch DINOv2 | 41.5/59.7/69.9 | 56.9/81.5/91.2 | 37.8/79.3/91.2 | 35.4/50.9 |
| Self-Stitch SigLIP2 | 50.5/62.0/68.9 | 71.2/88.5/87.3 | 67.9/88.1/89.3 | 44.5/50.5 |
| DINOv2→SigLIP2 | **55.8/68.0/71.8** | **75.9/89.1/92.8** | **77.8/87.6/92.4** | **44.9/51.2** |
| SigLIP2→DINOv2 | **53.8/69.6/72.2** | **86.3/88.9/91.9** | **80.7/89.0/91.0** | **49.0/51.4** |

Cross-model stitching **consistently surpasses self-stitching baselines**, with classification gains of +0.7%–+5.5% and segmentation gains of +0.5–+0.7 mIoU.

### VST Accuracy–Efficiency Trade-off (MoF-LLaVA)

| Configuration | Additional Resources | Gain Recovery Ratio |
|---------------|----------------------|----------------------|
| Single-VFM baseline | 0% | 0% |
| VST-22 (1 specialized layer) | 4.3% | 45% |
| VST-14 (9 specialized layers) | 39% | 84% |
| Full dual-VFM | 100% | 100% |

### Ablation Study: Stitch Layer Type Comparison (fMoW Accuracy %)

| Method | L2 | L6 | L10 | L14 | L18 | L22 |
|--------|------|------|------|------|------|------|
| D→S Linear | 26.1 | 54.3 | 59.5 | 66.5 | 69.1 | 69.6 |
| D→S MLP | **51.7** | **55.8** | **59.3** | **68.0** | **72.0** | **71.8** |
| D→S LoRA | 49.1 | 49.4 | 57.4 | 61.7 | 67.7 | 67.3 |

MLP consistently outperforms both Linear and LoRA across all stitch points; LoRA, despite its greater expressive capacity, underperforms MLP.

## Highlights & Insights

- **Precise problem diagnosis**: The paper clearly analyzes why LFM fails at shallow stitch points (small errors amplified through frozen layers) and why TLT faces gradient propagation difficulties at shallow depths; the FFM solution is elegant and effective.
- **Well-designed self-stitching baseline**: Self-stitch control experiments eliminate "stitch layer capacity gains" as a confounding factor, establishing that complementary knowledge fusion is a genuine effect.
- **Complete loop from analysis to application**: Starting from representation analysis, the work identifies VFM stitchability, and subsequently designs the VST architecture to address the practical efficiency problem in multi-VFM systems.
- **Comprehensive and systematic experiments**: Covering 4 VFMs, 4 datasets, 2 task types (classification + segmentation), 3 stitch layer types, and 6 stitch depths.

## Limitations & Future Work

- **Weak source model constraint**: Stitching performance degrades when CLIP is used as the source model, suggesting that a weak encoder may discard critical information that the target model cannot recover.
- **Limited VST evaluation**: Validation is confined to the LLaVA framework and a small number of VQA benchmarks, without coverage of broader MLLM architectures or more extensive benchmark suites.
- **Unexplored stitch layer design space**: Only Linear, MLP, and LoRA are evaluated; more complex mechanisms such as cross-attention remain unexplored.
- **Restricted to ViT architectures**: All evaluated VFMs are Transformer-based; stitchability of CNN or hybrid architectures is not verified.
- **Static stitch points**: The stitch point is fixed; input-adaptive dynamic stitching strategies are not explored.

## Related Work & Insights

- **vs. Bansal et al. (2021)**: Prior work validated stitchability only on small models trained on the same dataset; this paper is the first to systematically extend the investigation to heterogeneous VFMs and demonstrate the failure of naive approaches.
- **vs. Collins et al. (2025)**: That work observed that TLT may create out-of-distribution representations; FFM initialization in this paper directly mitigates this issue by preserving representation fidelity before task adaptation.
- **vs. Smith et al. (2025)**: That work questioned whether stitching success merely reflects representation clustering rather than semantic similarity; the self-stitching control experiments in this paper directly address this concern.
- **vs. SN-Net (Pan et al.)**: SN-Net focuses on stitchability training across scales within the same model family, whereas this paper addresses post-hoc stitching of independently trained heterogeneous VFMs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The FFM training strategy and VST architecture are innovative, though the core ideas build upon the existing model stitching framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers multiple VFMs, datasets, tasks, stitch layer types, and stitch depths, with well-designed control experiments.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Logically coherent; the problem–analysis–solution–validation narrative flows smoothly, and the introduction of the self-stitching baseline is convincing.
- **Value**: ⭐⭐⭐⭐ — VST provides a practical solution for efficiency optimization in multi-VFM systems, though weak source model limitations and limited MLLM evaluation somewhat reduce its practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](anomalyvfm_--_transforming_vision_foundation_models_into_zero-shot_anomaly_detec.md)
- [\[CVPR 2026\] VL-RouterBench: A Benchmark for Vision-Language Model Routing](vl-routerbench_a_benchmark_for_vision-language_model_routing.md)
- [\[CVPR 2026\] UniGame: Turning a Unified Multimodal Model Into Its Own Adversary](unigame_turning_a_unified_multimodal_model_into_its_own_adversary.md)
- [\[CVPR 2026\] Medic-AD: Towards Medical Vision-Language Model's Clinical Intelligence](medic-ad_towards_medical_vision-language_models_clinical_intelligence.md)
- [\[CVPR 2026\] Adaptive Vision-Language Model Routing for Computer Use Agents](adaptive_visionlanguage_model_routing_for_computer.md)

</div>

<!-- RELATED:END -->
