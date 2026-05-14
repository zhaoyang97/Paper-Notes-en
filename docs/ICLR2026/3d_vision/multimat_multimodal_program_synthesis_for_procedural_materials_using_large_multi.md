---
title: >-
  [Paper Note] MultiMat: Multimodal Program Synthesis for Procedural Materials using Large Multimodal Models
description: >-
  [ICLR 2026][3D Vision][Procedural Materials] This paper presents MultiMat, the first framework to apply large multimodal models (LMMs) to procedural material node graph synthesis. By incorporating intermediate visual ren…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "Procedural Materials"
  - "Node Graph"
  - "Multimodal Generation"
  - "Constrained Tree Search"
  - "Substance Designer"
date: 2026-05-08
content_hash: 476f72d10cdbbed3
---

# MultiMat: Multimodal Program Synthesis for Procedural Materials using Large Multimodal Models

**Conference**: ICLR 2026
**arXiv**: [2509.22151](https://arxiv.org/abs/2509.22151)
**Code**: None
**Area**: 3D Vision / Program Synthesis
**Keywords**: Procedural Materials, Node Graph, Multimodal Generation, Constrained Tree Search, Substance Designer

## TL;DR

This paper presents MultiMat, the first framework to apply large multimodal models (LMMs) to procedural material node graph synthesis. By incorporating intermediate visual rendering feedback of partially generated nodes into the autoregressive generation process (via two conditioning modes: mixed and graph), and pairing this with an incremental constrained tree search for on-the-fly validation and backtracking, MultiMat is trained on 6,878 production-grade Substance Designer materials and substantially outperforms text-only baselines in both unconditional and conditional generation.

## Background & Motivation

Procedural materials (e.g., Adobe Substance Designer) define PBR materials through directed acyclic graphs (DAGs), offering resolution independence, parametric controllability, and non-destructive editing—properties widely exploited in games, film, and VR/AR production. However, manually constructing node graphs requires professional expertise, posing a steep barrier for non-specialist users. Recent neural program synthesis methods (MatFormer, VLMaterial) have attempted to automate this process, but suffer from three critical issues:

1. **Text-only modeling disregards the visual nature of the task**: Existing methods serialize node graphs into plain-text programs, discarding the inherently visual-spatial intuition of node graphs.
2. **Difficulty reasoning without visual feedback**: Models must infer complex spatial relationships and visual effects from text alone; reasoning difficulty escalates sharply as material complexity increases.
3. **No guarantee of structural correctness**: Validation occurs only after a complete program is generated, resulting in large volumes of invalid outputs (invalid connections, type mismatches) that degrade inference efficiency.

MultiMat's core mechanism is to **emulate the workflow of a human material artist**—rendering intermediate states after each node is generated and feeding them back to the model, forming a visual-text multimodal feedback loop, while leveraging topological ordering to enable incremental per-node validation.

## Method

### Overall Architecture

MultiMat is built upon QWen2.5VL (7B), a vision-language model. The core pipeline follows **node-by-node autoregressive generation with intermediate rendering feedback**:

1. Given a partially generated material graph $G_t = \{v_1, v_2, \ldots, v_t\}$ with nodes arranged in topological order,
2. The current graph state $G_t$ and intermediate rendering output $I_t$ are fed to the LMM in multimodal form,
3. The model generates the next node definition $v_{t+1}$ (including node type, parameters, and connections),
4. A transpiler compiles $v_{t+1}$ into SBS format, which the material engine executes and renders,
5. If successful, the graph is updated to $G_{t+1}$ with output $I_{t+1}$ and generation continues; if not, backtracking is triggered.

Training uses a standard cross-entropy loss:

$$\mathcal{L} = -\sum_{t=1}^{T}\sum_{s=1}^{S}\log p(v_{t,s} \mid v_{t,<s}, G_t, I_t, x; \theta)$$

where $v_{t,s}$ is the $s$-th token of node $v_t$ in the intermediate text format, and $x$ is the input condition (empty for unconditional generation).

### Key Designs

**(1) Dual Conditioning Strategy**

Two complementary multimodal program representations are proposed:

| Conditioning Mode | Input Form | Image Cost per Node | Characteristics |
|:--|:--|:--|:--|
| Mixed Conditioning | Text node definitions + interleaved 140×140 rendered images (25 patches) per node | 25 patches/node | Retains full textual structure; parameters are implicitly encoded via images |
| Graph Conditioning | Full graph visualization (embedded intermediate visual output), up to 6,144 tokens | One global image | Closer to human visual editing experience; no explicit text node definitions provided |

Experiments show that Graph Conditioning achieves the best visual quality (lowest KID), while Mixed Conditioning yields the lowest error rate (lowest NER).

**(2) Incremental Constrained Tree Search**

Topological ordering enables validity checking via the transpiler and material engine immediately after each node is generated. Upon detecting an invalid node, an adaptive backtracking strategy is applied:

- At the $i$-th backtracking step, the most recent $2^{(i-1)}$ nodes are discarded,
- The entire generation process forms a search tree $\mathcal{T}$ containing valid (✓) and invalid (✗) nodes,
- Compared to conventional "generate-then-validate" approaches, this significantly improves inference efficiency (disabling tree search in VLMaterial degrades NER from 14.8% to 34.0%).

**(3) Automatic Error Repair**

Two common error patterns are identified and automatically corrected:

- **Parameter deletion**: Extraneous parameters unsupported by the node type are removed (only ~1% of MultiMat nodes require this repair),
- **Type conversion insertion**: A grayscale conversion node is automatically inserted when a color output connects to a grayscale input; a gradient map node is inserted for grayscale-to-color connections.

### Loss & Training

**Training Configuration**:

| Setting | Value |
|:--|:--|
| Base Model | QWen2.5VL 7B (multimodal) / QWen3 8B (text-only baseline) |
| Maximum Sequence Length | 8,192 tokens |
| Training Epochs | 5 |
| Optimizer | AdamW |
| Learning Rate | $5 \times 10^{-5}$ |
| Batch Size | 128 |
| Inference Temperature | 0.8 |
| Top-p | 0.95 |
| Hardware | 8 × A100 80GB |

**Dataset Construction**: 6,878 production-grade materials are collected from Adobe Substance 3D Assets, constituting the largest dataset to date (MatFormer: 2,820; VLMaterial: 3,663). A bidirectional transpiler converts SBS format into a compact YAML format, CompactSBS (reducing average length by 80%+), supporting the full feature set (including pixel processors and function graphs) and up to 128 nodes.

**Parameter Optimization for Conditional Generation**: The DiffMat differentiable renderer is used to gradient-optimize generated graphs, aligning output materials more closely with input images. The optimized variant is denoted MultiMat+.

## Key Experimental Results

### Main Results — Unconditional Generation

| Model | KID ↓ | ROUGE-L ↓ | NER ↓ |
|:--|:--|:--|:--|
| VLMaterial (SBS) | 14.155 | 3.641 | 14.846 |
| MultiMat (Mixed) | 6.752 | 2.195 | **8.923** |
| MultiMat (Graph) | **2.365** | **1.915** | 15.024 |

- MultiMat (Graph) reduces KID by **11.8 percentage points** over VLMaterial, substantially surpassing text-only methods in visual quality.
- ROUGE-L remains below 4% across all models, indicating no significant memorization; MultiMat variants exhibit lower reproduction rates.
- MultiMat (Mixed) achieves the lowest error rate (NER 8.9%); errors in the Graph variant are primarily attributable to OCR-style misreading of node names.

### Main Results — Conditional Generation (Inverse Material Synthesis)

| Model | DSim ↑ | CLIP ↑ | Style ↓ | KID ↓ |
|:--|:--|:--|:--|:--|
| VLMaterial (SBS) | 31.344 | 65.678 | 3.211 | 14.976 |
| MultiMat (Mixed) | 34.922 | 66.737 | 3.199 | 3.675 |
| MultiMat (Graph) | **36.609** | **67.907** | **3.178** | **2.801** |
| VLMaterial+ (SBS) | 31.348 | 65.867 | 3.126 | 27.862 |
| MultiMat+ (Mixed) | 40.258 | 69.687 | 3.093 | 17.792 |
| MultiMat+ (Graph) | **40.367** | **70.114** | **3.046** | **14.886** |

- Perceptual similarity rankings consistently follow Graph > Mixed > VLMaterial, mirroring unconditional generation trends.
- Parameter optimization (+) yields approximately 6–8% perceptual improvement for MultiMat, whereas VLMaterial+ improves by only ~1% (its generated outputs deviate too far from the target to benefit substantially from optimization).
- Human evaluation (8 experts, 33 challenging test samples) further confirms that MultiMat+ (Graph) is most preferred, while VLMaterial+ is least preferred.

### Ablation Study — Automatic Repair Analysis

| Model | Parameter Deletion ↓ | Type Conversion ↓ |
|:--|:--|:--|
| VLMaterial (SBS) | 2.71% | 12.26% |
| MultiMat (Mixed) | **1.18%** | 3.51% |
| MultiMat (Graph) | 1.10% | 6.49% |

MultiMat variants require substantially fewer repairs than VLMaterial, demonstrating that multimodal feedback genuinely aids the model in understanding graph structure.

## Highlights & Insights

- ⭐⭐⭐ **Multimodal Program Synthesis Paradigm**: This is the first work to incorporate intermediate visual rendering feedback into procedural material generation, emulating the visual editing workflow of human artists.
- ⭐⭐⭐ **Incremental Constrained Tree Search**: Topological ordering enables per-node validation and adaptive backtracking, transforming the inference process into an efficient tree search.
- ⭐⭐ **Full Feature Set Support**: A bidirectional SBS↔CompactSBS transpiler is developed, providing the first support for the complete Substance Designer feature set (including pixel processors and function graphs) while reducing program length by 80%+.
- ⭐⭐ **Largest Production-Grade Dataset**: 6,878 legitimately licensed production materials are collected, representing an 88% increase over the previously largest dataset.

## Limitations & Future Work

1. **Training inefficiency**: MultiMat must adapt visual context for each node individually, resulting in training times far exceeding text-only methods (days vs. hours), though the absolute cost remains manageable given the relatively small dataset size.
2. **OCR errors in Graph Conditioning**: Reading node names and function types from graph visualizations is prone to OCR-style errors, leading to elevated NER (~15%).
3. **Limited data scale**: Only 6,878 materials constrain the model's generalization; future work may employ self-training techniques to generate synthetic training data using the unconditional model.
4. **Single-tool dependency**: The current system supports only Substance Designer; future work plans to develop a unified model spanning multiple node graph systems.
5. **Remaining gap in conditional generation**: Even after parameter optimization, reconstruction quality for complex materials exhibits notable shortcomings (see failure cases in the paper).

## Personal Reflections

MultiMat's central contribution lies in surfacing an important insight: **procedural materials are fundamentally visual-spatial programs and should be treated visually rather than forcibly reduced to text**. This principle has broad implications: any program synthesis task with visual intermediate representations—such as vector graphics, UI layout, or data visualization—may benefit from analogous multimodal feedback mechanisms.

The incremental tree search is another elegant design—topological ordering transforms "post-hoc validation" into "on-the-fly validation," a paradigm transferable to any sequential generation task with verifiable intermediate states. The exponential backtracking strategy $2^{(i-1)}$ also merits attention, as it balances exploration efficiency against backtracking depth.

Regarding limitations, training efficiency is an inherent cost of multimodal program synthesis—the overhead of rendering intermediate states at each step is unavoidable. Practical deployment may necessitate lightweight intermediate representations (e.g., low-resolution thumbnails or feature summaries) to reduce computational cost. Furthermore, while 6,878 materials represents the current state of the art, the dataset remains extremely sparse compared to general-purpose visual datasets; exploring pretrain-finetune paradigms or cross-domain transfer learning may be necessary.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation](../../ICCV2025/3d_vision/robotron-mani_all-in-one_multimodal_large_model_for_robotic_manipulation.md)
- [\[AAAI 2026\] Point Cloud Quantization through Multimodal Prompting for 3D Understanding](../../AAAI2026/3d_vision/point_cloud_quantization_through_multimodal_prompting_for_3d_understanding.md)
- [\[CVPR 2026\] Adapting Point Cloud Analysis via Multimodal Bayesian Distribution Learning](../../CVPR2026/3d_vision/adapting_point_cloud_analysis_via_multimodal_bayesian_distribution_learning.md)
- [\[AAAI 2026\] Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](../../AAAI2026/3d_vision/rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)
- [\[ICCV 2025\] EgoM2P: Egocentric Multimodal Multitask Pretraining](../../ICCV2025/3d_vision/egom2p_egocentric_multimodal_multitask_pretraining.md)

</div>

<!-- RELATED:END -->
