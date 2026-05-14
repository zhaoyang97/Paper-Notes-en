---
title: >-
  [Paper Note] Imagine How To Change: Explicit Procedure Modeling for Change Captioning
description: >-
  [ICLR 2026][LLM Pretraining][change captioning] ProCap reframes change captioning from static image-pair comparison to dynamic procedure modeling. In the first stage…
tags:
  - "ICLR 2026"
  - "LLM Pretraining"
  - "change captioning"
  - "procedure modeling"
  - "frame interpolation"
  - "masked reconstruction"
  - "learnable queries"
  - "vision-language"
date: 2026-05-08
content_hash: 8cf6c4938ac73dde
---

# Imagine How To Change: Explicit Procedure Modeling for Change Captioning

**Conference**: ICLR 2026
**arXiv**: [2603.05969](https://arxiv.org/abs/2603.05969)
**Code**: [GitHub](https://github.com/BlueberryOreo/ProCap)
**Area**: LLM Pre-training
**Keywords**: change captioning, procedure modeling, frame interpolation, masked reconstruction, learnable queries, vision-language

## TL;DR

ProCap reframes change captioning from static image-pair comparison to dynamic procedure modeling. In the first stage, a procedure encoder is trained via frame interpolation and masked reconstruction to capture spatiotemporal change dynamics; in the second stage, learnable process queries implicitly infer the change procedure, surpassing state-of-the-art methods on three benchmarks.

## Background & Motivation

**Change captioning** involves generating textual descriptions of the differences between two similar images, with applications in remote sensing, medical diagnosis, urban planning, and industrial quality control.

Fundamental limitations of existing methods:

**Static image-pair modeling**: only compares "before" and "after", ignoring the **dynamic process** of change.

**Absence of temporal cues**: cannot understand *how* a change occurred.

**Encoder limitations**: various difference extractors and alignment mechanisms remain purely spatial rather than spatiotemporal.

**Key insight**: A latent continuous transition exists between two images, containing rich spatiotemporal dynamics. For instance, object displacement can reveal motion trajectories through intermediate frames.

## Method

### Overall Architecture

ProCap consists of two stages:
- **Stage 1 — Explicit Procedure Modeling (EPM)**: learns spatiotemporal dynamics of the change process.
- **Stage 2 — Implicit Procedure Captioning (IPC)**: replaces explicit intermediate frames with learnable queries.

### Stage 1: Explicit Procedure Modeling

**Process generation module**: a pre-trained frame interpolation model recursively generates intermediate frames. The FI model predicts bidirectional optical flow, produces warped image pairs, and a Transformer generates soft masks and residuals to synthesize intermediate frames.

**Confidence-based frame sampling**: selects keyframes that are "semantically equidistant" — frames whose semantic distances to the start and end frames are equal receive the highest score. A squared-difference term penalizes frames biased toward either endpoint.

**Procedure modeling module**: Transformer encoder + image tokenizer. Input includes a visual stream (patch features), a text stream (caption tokens), and special tokens (frame consistency + cross-modal alignment).

**Multi-granularity masking** (one of four strategies selected randomly):
1. Whole-frame masking: forces reconstruction using the caption.
2. Random patch masking: encourages distributed representations.
3. Intra-block masking: learns local texture.
4. Extra-block masking: learns region-to-scene relationships.

### Loss & Training

$$L_{\text{PRO}} = L_{\text{msm}} + L_{\text{align}} + L_{\text{csy}}$$

- $L_{\text{msm}}$: cross-entropy over discrete token prediction at masked positions.
- $L_{\text{align}}$: contrastive loss distinguishing matched/unmatched caption–procedure pairs.
- $L_{\text{csy}}$: distinguishes normal from shuffled frame sequences.

### Stage 2: Implicit Procedure Captioning

Core idea: $k \times n_I$ learnable process queries replace explicit intermediate frames. Leveraging the spatiotemporal understanding acquired in Stage 1, the model implicitly infers the change procedure from an image pair. A Transformer decoder translates these queries into a caption. Frame interpolation is not required at inference time.

### Training Strategy

Stage 2 is trained end-to-end with an autoregressive loss. At inference, only $k \times n_I$ additional parameters are introduced (negligible overhead when $k=2$).

## Key Experimental Results

### Main Results

**Comparison with SOTA on three datasets** (Table 1, CIDEr):

| Method | CLEVR-Change | Spot-the-Diff | Image-Editing |
|--------|-------------|---------------|---------------|
| DUDA (2019) | 112.3 | 32.5 | 22.8 |
| SCORER+CBR (2023) | 126.8 | 38.9 | 33.4 |
| MCT-CCDiff (2025) | 131.7 | 41.7 | 38.3 |
| FINER (LLM, 2024) | 137.2 | 61.8 | 50.5 |
| LLaVA-1.5+RP (LLM) | — | 43.2 | 60.9 |
| **ProCap (Ours)** | **135.6** | **42.7** | **40.6** |

ProCap achieves comprehensive superiority among non-LLM methods and substantially narrows the gap with LLM-based approaches.

### Ablation Study

**Component ablation** (CLEVR-Change CIDEr):

| EPM | IPC | k | CIDEr |
|-----|-----|---|-------|
| N | N | 0 | 108.4 |
| Y | N | 0 | 112.7 |
| N | Y | 1 | 106.2 |
| Y | Y | 1 | **128.5** |

Combining both components yields a CIDEr gain of +20.1 (108.4 → 128.5).

**Query length $k$**:

| k | TPS | CIDEr |
|---|-----|-------|
| 1 | 766 | 128.5 |
| 2 | 699 | **135.6** |
| 4 | 461 | 128.7 |
| 7 | 271 | 130.5 |

$k=2$ achieves the best performance with reasonable efficiency.

**Loss ablation** (CLEVR / StD CIDEr):

| msm | align | csy | CLEVR | StD |
|-----|-------|-----|-------|-----|
| Y | N | N | 127.5 | 29.7 |
| Y | N | Y | 128.6 | 36.3 |
| Y | Y | Y | **135.6** | **42.7** |

The full combination improves CIDEr on Spot-the-Diff by 13.0 over $L_{\text{msm}}$ alone.

### Key Findings

1. **Procedure modeling substantially outperforms static comparison.**
2. **Pre-training and queries are synergistic**: pre-training provides spatiotemporal understanding; queries enable efficient inference.
3. **Lightweight yet powerful**: a non-LLM model approaches or matches LLM-based methods.
4. **Cross-domain generalization**: strong performance across synthetic, natural, and open-domain scenarios.

## Highlights & Insights

1. **Paradigm shift**: from "static spatial comparison" to "dynamic spatiotemporal procedure modeling."
2. **Elegant two-stage design**: explicit frames during training, implicit queries during inference — balancing representation quality and efficiency.
3. **Creative confidence-based sampling**: selecting "semantically equidistant" frames focuses learning on critical transition moments.
4. **Multi-granularity masking**: enables understanding at scales ranging from frame level to patch level.
5. **Competitive without LLMs**: demonstrates that architectural innovation, rather than scale alone, can yield substantial gains.

## Limitations & Future Work

1. **Dependence on frame interpolation quality**: the quality of the FI model directly constrains the upper bound of Stage 1.
2. **Assumption of interpolable changes**: abrupt appearance or disappearance of objects cannot be modeled via optical flow.
3. **No LLM decoder**: integrating a large language model decoder may yield further improvements.
4. **Restricted to image pairs**: the framework has not been extended to video change captioning.
5. **Confidence-based sampling requires a predefined similarity function.**

## Related Work & Insights

- **DUDA** [Park et al., 2019]: foundational framework — ProCap fundamentally extends the modeling paradigm.
- **FINER** [Zhang et al., 2024]: LLM-augmented change captioning — ProCap achieves comparable performance without an LLM.
- **VideoMAE** [Han et al., 2022]: video masked autoencoding — inspires the procedure modeling design.
- **VQGAN** [Esser et al., 2021]: image tokenizer — used as the reconstruction target.
- **RIFE** [Lu et al., 2022]: frame interpolation — used for explicit process generation.

## Rating

| Dimension | Score |
|-----------|-------|
| Theoretical Depth | ⭐⭐⭐ |
| Novelty | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Optimal Online Change Detection via Random Fourier Features](../../NeurIPS2025/llm_pretraining/optimal_online_change_detection_via_random_fourier_features.md)
- [\[ICLR 2026\] RECON: Robust symmetry discovery via Explicit Canonical Orientation Normalization](recon_robust_symmetry_discovery_via_explicit_canonical_orientation_normalization.md)
- [\[ICLR 2026\] TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling](taste_text-aligned_speech_tokenization_and_embedding_for_spoken_language_modelin.md)
- [\[NeurIPS 2025\] How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?](../../NeurIPS2025/llm_pretraining/how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train.md)
- [\[ICLR 2026\] MoMa: A Simple Modular Deep Learning Framework for Material Property Prediction](moma_a_modular_deep_learning_framework_for_material_property_prediction.md)

</div>

<!-- RELATED:END -->
