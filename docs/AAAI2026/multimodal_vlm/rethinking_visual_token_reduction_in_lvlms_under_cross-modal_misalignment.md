---
title: >-
  [Paper Note] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment
description: >-
  [Multimodal VLM] This paper identifies three forms of cross-modal misalignment (causal, semantic, and spatial) in text-guided visual token importance estimation within LVLMs, and proposes VisionDrop—a training-free progressive token pruning framework that relies exclusively on visual self-attention. The framework performs multi-stage compression across both the visual encoder and LLM decoder, retaining over 91% of original performance while keeping only 5.6% of tokens.
tags:
  - Multimodal VLM
date: 2026-05-08
content_hash: 1bc4855b73f87907
---

# Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment

- **Conference**: AAAI 2026
- **arXiv**: [2506.22283](https://arxiv.org/abs/2506.22283)
- **Code**: [https://github.com/Ruixxxx/VisionDrop](https://github.com/Ruixxxx/VisionDrop)
- **Area**: Multimodal VLM
- **Keywords**: visual token compression, cross-modal alignment misalignment, training-free pruning, attention scoring, large vision-language models

## TL;DR

This paper identifies three forms of cross-modal misalignment (causal, semantic, and spatial) in text-guided visual token importance estimation within LVLMs, and proposes VisionDrop—a training-free progressive token pruning framework that relies exclusively on visual self-attention. The framework performs multi-stage compression across both the visual encoder and LLM decoder, retaining over 91% of original performance while keeping only 5.6% of tokens.

## Background & Motivation

- **Background**: Large Vision-Language Models (LVLMs) encode images as dense patch-level token sequences to capture fine-grained semantics; however, the number of visual tokens far exceeds that of text tokens (e.g., LLaVA-NeXT produces 2,880 tokens per image), leading to quadratic growth in attention computation and inference efficiency bottlenecks.
- **Limitations of Prior Work**: Existing visual token pruning methods operating inside the LLM (e.g., FastV, PyramidDrop) predominantly rely on **text-guided scoring strategies**, using the attention from text tokens to visual tokens as a proxy for importance. This implicitly assumes that visual and text modalities remain well-aligned within LLM layers—an assumption this paper demonstrates to be invalid.
- **Key Challenge**: The authors identify three types of cross-modal misalignment:
  1. **Causal Misalignment**: The causal attention in autoregressive LLMs causes the last text token to disproportionately attend to visual tokens near the end of the sequence, introducing positional bias.
  2. **Semantic Misalignment**: As tokens propagate through LLM layers, visual and text representations become deeply entangled, preventing text tokens from faithfully reflecting visual importance.
  3. **Spatial Misalignment**: Visual and text tokens are flattened into a single sequence with mixed positional encodings; text tokens lack spatial awareness, causing spatially relevant regions to be incorrectly discarded.
- **Goal**: Controlled experiments demonstrate that replacing the text-guided scoring in PyramidDrop with visual self-attention scoring consistently yields better performance on benchmarks such as GQA and MMBench, with the advantage growing as the compression ratio increases.

## Method

### 1. Progressive Dominant Token Selection

The LVLM architecture is partitioned into $N$ stages $\mathcal{S} = \{s_0, s_1, \ldots, s_N\}$, spanning both the visual encoder and LLM decoder. At the end of each stage $s_n$, pruning is performed according to a stage-specific retention ratio $\lambda_n$.

**Core Idea**: Token importance is assessed solely via visual-to-visual self-attention, without relying on any text signal.

For visual query tokens $\mathbf{x}_V^q \in \mathbb{R}^{L_1 \times D}$, the attention matrix is computed as:

$$\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{D}}\right)$$

The attention weights corresponding to visual key tokens $\mathbf{A}_{:,\mathcal{V}}$ are extracted and averaged over all visual queries to obtain importance scores:

$$\mathbf{S} = \frac{1}{L_1}\sum_{l=1}^{L_1}\mathbf{A}[l, \mathcal{V}]$$

Top-ranked tokens are selected based on threshold $\tau_n$ (determined by $\lambda_n$) and passed to the next stage:

$$\mathbf{x}_V^{(s_{n+1})} = \{x_V^i \in \mathbf{x}_V^{(s_n)} \mid S(i) \geq \tau_n\}$$

**Visual Query Selection Strategy**:
- In the LLM: image-to-image attention maps within the visual subspace are extracted.
- In the visual encoder: if a [CLS] token is present (e.g., CLIP), its attention over each patch serves as the importance score; otherwise (e.g., SigLIP), the mean-pooling strategy consistent with the LLM is applied.

### 2. Stage-wise Contextual Token Merging

Pruned non-dominant tokens may contain subtle yet useful visual cues. To prevent information loss, a lightweight merging step is performed at the end of each stage:

- Key embeddings from the attention module are reused to compute inter-token semantic similarity (dot product).
- In the LLM, image token key vectors are explicitly extracted to ensure modality-pure merging.
- Non-dominant tokens are divided into a candidate set and a reference set; each candidate token is paired with its most similar reference token and fused accordingly.
- The resulting contextual tokens are passed to the next stage alongside the dominant tokens.

**Implementation Details**: The model is divided into 5 stages: the first stage concludes at the visual encoder output, and the subsequent four stages correspond to layers 8, 16, 24, and the final decoding layer of the LLM. The number of tokens retained at the second stage is set to 1.5× the final target (for image understanding) or 3× (for video understanding).

## Key Experimental Results

### Table 1: Performance Comparison on LLaVA-1.5-7B at Different Retention Ratios

| Method | # Tokens | GQA | MMB | POPE | SQA | VQAv2 | Avg. |
|--------|----------|------|------|------|------|-------|------|
| Full (upper bound) | 576 | 61.92 | 66.31 | 86.81 | 69.51 | 78.53 | 100% |
| FastV | 192 | 52.62 | 57.74 | 75.59 | 68.07 | 70.51 | 88.45% |
| PyramidDrop | 192 | 57.27 | 63.51 | 82.40 | 69.56 | 75.57 | 96.11% |
| SparseVLM | 192 | 59.44 | 65.41 | 86.45 | 68.86 | 77.01 | 98.64% |
| **VisionDrop** | **192** | **59.99** | **65.19** | **87.23** | **69.06** | **77.28** | **98.76%** |
| VisionZip | 32 | 51.80 | 58.02 | 75.11 | 68.72 | 67.12 | 89.92% |
| **VisionDrop** | **32** | **52.79** | **60.31** | **77.19** | **69.41** | **68.55** | **91.46%** |

### Table 2: Efficiency Analysis on LLaVA-NeXT-7B

| Method | # Tokens | FLOPs (T) | Latency (ms) | Speedup |
|--------|----------|-----------|--------------|---------|
| LLaVA-1.5 baseline | 576 | 9.06 | 237 | 1.0× |
| VisionDrop | 64 | 2.11 | 117 | 2.0× |
| LLaVA-NeXT baseline | 2880 | 46.25 | 593 | 1.0× |
| VisionDrop | 320 | 7.70 | 216 | 2.7× |

VisionDrop achieves a 6.0× FLOPs reduction on LLaVA-NeXT while retaining 95.71% of original performance.

## Key Findings

1. **Text-guided scoring degrades severely under high compression**: Controlled experiments show that at 64 retained tokens, visual self-attention scoring comprehensively outperforms text-guided scoring, with the gap widening as compression increases.
2. **Causal attention induces clear positional bias**: Visualizations show that tokens retained after shallow-layer pruning consistently cluster at the bottom of the image (i.e., end of the sequence), independent of semantic content.
3. **Progressive pruning outperforms single-stage approaches**: Multi-stage pruning across both the encoder and LLM is more stable than pruning at either end alone.
4. **Effectiveness on video understanding**: On Video-LLaVA, retaining 12.5% of tokens still achieves the best average accuracy of 47.3%.
5. **Ablation study**: A 33.3% retention ratio in the visual encoder is optimal; performance remains stable across varying dominant-to-contextual token ratios.

## Highlights & Insights

- **Precise and significant problem identification**: The paper systematically characterizes three forms of cross-modal misalignment—causal, semantic, and spatial—with rigorous theoretical analysis and visualizations.
- **Elegant and simple solution**: Training-free, requiring no additional modules, and directly reusing existing attention maps for plug-and-play deployment.
- **Unified pipeline design**: This is the first work to treat the visual encoder and LLM as a unified system for progressive pruning.
- **Comprehensive experimental coverage**: Evaluated across 9 image benchmarks and 3 video benchmarks, at multiple compression ratios, against 5+ state-of-the-art methods.

## Limitations & Future Work

- Validation is limited to the LLaVA model family; generalizability to newer architectures such as Qwen-VL and InternVL has not been examined.
- Stage boundaries (layers 8/16/24) are manually specified, lacking an adaptive selection mechanism.
- The contextual token merging pairing strategy (nearest neighbor) is relatively simple and may not represent the optimal fusion approach.
- On certain tasks such as VizWiz (low-quality images), the method is occasionally outperformed by encoder-side approaches such as VisionZip.
- Joint application with other efficiency techniques (e.g., model distillation or quantization) has not been explored.

## Related Work & Insights

- **LLM-internal pruning**: FastV (ECCV 2024) ranks tokens by text attention during generation; PyramidDrop (CVPR 2025) performs progressive pruning guided by the last instruction token; VScan employs global-local scanning.
- **Visual encoder pruning**: VisionZip / VisPruner select dominant tokens via attention and merge by similarity; FlowCut leverages cross-layer information flow; CDPruner maximizes instruction-conditioned diversity.
- **Cross-modal guided pruning**: SparseVLM computes token importance via cross-attention with text guidance.
- **Modality alignment research**: Venhoff et al. (2025) find that joint self-attention introduces modality entanglement.

## Rating

⭐⭐⭐⭐ — The problem identification is precise and significant, with thorough analysis of three misalignment types supported by theory and visualization. The method is concise and effective, requiring no training. Experiments are comprehensive and rigorous. Minor shortcomings include limited model coverage and the lack of an adaptive stage partitioning mechanism.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Multi-Faceted Attack: Exposing Cross-Model Vulnerabilities in Defense-Equipped Vision-Language Models](multi-faceted_attack_exposing_cross-model_vulnerabilities_in_defense-equipped_vi.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[AAAI 2026\] ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models](recad_reinforcement_learning_enhanced_parametric_cad_model_generation_with_visio.md)
- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)

<!-- RELATED:END -->
