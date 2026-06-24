---
title: >-
  [Paper Note] MoMa: Modulating Mamba for Adapting Image Foundation Models to Video Recognition
description: >-
  [ICML2025][Video Understanding][Mamba] The MoMa framework is proposed, which injects the linear complexity SSM of Mamba into a frozen CLIP Transformer via a scale-bias sequence modulation operation (SeqMod) to achieve efficient global spatiotemporal dynamic modeling, reaching SOTA performance on multiple video recognition benchmarks with lower computational cost.
tags:
  - "ICML2025"
  - "Video Understanding"
  - "Mamba"
  - "PEFT"
  - "Image Foundation Model Adaptation"
  - "State Space Models"
  - "Spatiotemporal Modeling"
date: 2026-05-08
content_hash: c59b1d3c87cd8ce6
---

# MoMa: Modulating Mamba for Adapting Image Foundation Models to Video Recognition

**Conference**: ICML2025  
**arXiv**: [2506.23283](https://arxiv.org/abs/2506.23283)  
**Authors**: Yuhuan Yang, Chaofan Ma, Zhenjie Mao, Jiangchao Yao, Ya Zhang, Yanfeng Wang
**Institution**: Shanghai Jiao Tong University
**Code**: To be confirmed  
**Area**: Video Understanding  
**Keywords**: Mamba, PEFT, Video Understanding, Image Foundation Model Adaptation, State Space Models, Spatiotemporal Modeling

## TL;DR

The MoMa framework is proposed, which injects the linear complexity SSM of Mamba into a frozen CLIP Transformer via a scale-bias sequence modulation operation (SeqMod) to achieve efficient global spatiotemporal dynamic modeling, reaching SOTA performance on multiple video recognition benchmarks with lower computational cost.

## Background & Motivation

- **Core Problem**: How to efficiently adapt Image Foundation Models (IFMs) to video understanding tasks? Existing PEFT methods (such as AIM, DualPath, DiST) process spatial and temporal information separately, failing to explicitly capture complete spatiotemporal dynamic relationships.
- **Limitations of Prior Work**: Computing full attention over the entire spatiotemporal sequence has a complexity of $O((HWT)^2)$, which is unscalable. Directly inserting lightweight Mamba modules into the frozen Transformer disrupts pre-trained features (experiments confirm poor performance).
- **Key Insight**: The SSM of Mamba has linear sequence modeling complexity, making it suitable for processing long spatiotemporal sequences in videos. However, a fusion method that "does not disrupt pre-trained weights" is required, thereby balancing the powerful representations of IFMs with the efficient sequence modeling of Mamba.

## Method

### Overall Architecture: Divide-and-Modulate

MoMa introduces two stages in each Transformer layer of CLIP:

1. **Divide Stage**: Reduces attention computation overhead
2. **Modulate Stage**: Injects global spatiotemporal dynamic information through Mamba SSM

The modified forward process is:

$$\mathbf{V}^{i+1} = \text{FFN}(\text{Modulate}(\text{Divide}(\mathbf{V}^i)))$$

where $\mathbf{V}^i \in \mathbb{R}^{HWT \times C}$ represents the video features of the $i$-th layer.

### Divide Stage

Each frame is divided into non-overlapping windows of size $w \times w$, and self-attention is performed independently within each window:

$$\mathbf{V}^i \rightarrow [\mathbf{s}_1^i, \mathbf{s}_2^i, \ldots, \mathbf{s}_N^i], \quad \mathbf{s}_n^i \in \mathbb{R}^{w^2 \times C}, \quad N = \frac{HWT}{w^2}$$

$$\mathbf{s}_n^{i\prime} = \text{Attention}(\mathbf{s}_n^i)$$

The complexity is reduced from frame-by-frame attention $O((HW)^2 T)$ to $O(w^2 \cdot HWT)$, achieving linear time complexity. In the experiments, $w=8$.

### Modulate Stage and SeqMod Operation

**SSM Forward Layer**: The Divide output $\mathbf{x}^i$ is passed through a modified Mamba SSM layer. The output project channels are doubled and split along the channel dimension into two sequences:

$$\mathbf{y}_1^i, \mathbf{y}_2^i = \text{SSM}(\mathbf{x}^i)$$

The internal SSM adopts multiple bidirectional scans (spatial + temporal dimensions), referencing the design of VideoMamba.

**SeqMod Operation**: Inspired by adaptive normalization (AdaN/FiLM), but expanding the scalar scale/bias to tensor sequences of equal length to the input, achieving fine-grained sequence-to-sequence modulation:

$$\text{SeqMod}(\mathbf{x}, \mathbf{y}_1, \mathbf{y}_2) = \underbrace{\mathbf{y}_1}_{\text{scale}} \odot \mathbf{x} + \underbrace{\mathbf{y}_2}_{\text{bias}} + \underbrace{\mathbf{x}}_{\text{skip}}$$

- $\mathbf{y}_1$ serves as the sequence-level scale, and $\mathbf{y}_2$ serves as the sequence-level bias, where $\odot$ denotes element-wise multiplication.
- Skip connections are retained to ensure that the original CLIP features are not disrupted.
- Key difference from AdaN: AdaN uses global scalar modulation, whereas SeqMod uses tensor modulation of the same length as the sequence, preserving fine-grained spatiotemporal information.

The final output is fed into the frozen FFN layer of CLIP:

$$\mathbf{V}^{i+1} = \text{FFN}(\text{SeqMod}(\mathbf{x}^i, \mathbf{y}_1^i, \mathbf{y}_2^i))$$

### Loss & Training

- Average pooling is performed on the output of the final layer to obtain the video representation: $\hat{\mathbf{y}}_o = \text{Average}(\mathbf{V}^L)$
- **Freeze all CLIP parameters**, training only the newly introduced SSM layers.
- Loss function = classification loss + CLIP distillation loss (to preserve zero-shot understanding capability and prevent excessive feature space shift).
- End-to-end training.

## Training and Experimental Settings

| Configuration | Value |
|---|---|
| Base Model | CLIP ViT-B/16 / ViT-L/14 |
| Window size $w$ | 8 |
| SSM hidden state | 16 |
| SSM hidden dimension | 384 |
| Activation function | GELU |
| Optimizer | AdamW, lr=3e-4, wd=0.05 |
| Hardware | 8x Tesla V100, fp16 |
| Training duration | ~12h (K400, 30 epochs) |
| Trainable parameters | 11M (ViT-B/16) / 39M (ViT-L/14) |

## Main Results

### Kinetics-400

| Method | Backbone | GFLOPs | Trainable Params | Top-1 |
|---|---|---|---|---|
| AIM | ViT-B/16 | 1214 | 11M | 84.5 |
| DiST | ViT-B/16 | 986 | 26M | 84.4 |
| **MoMa** | **ViT-B/16** | **902** | **11M** | **84.8** |
| AIM | ViT-L/14 | 5604 | 38M | 87.3 |
| DiST | ViT-L/14 | 4534 | 40M | 87.6 |
| **MoMa** | **ViT-L/14** | **4152** | **39M** | **87.8** |

### Something-Something V2 (Strong Temporal Modeling Requirement)

| Method | Backbone | GFLOPs | Top-1 |
|---|---|---|---|
| AIM | ViT-B/16 | 2496 | 69.1 |
| DiST | ViT-B/16 | 1972 | 70.9 |
| **MoMa** | **ViT-B/16** | **1804** | **71.5** |
| DiST | ViT-L/14 | 9068 | 73.1 |
| **MoMa** | **ViT-L/14** | **8304** | **73.8** |

### Long Video Recognition (Breakfast / COIN)

| Method | Breakfast | COIN |
|---|---|---|
| VideoMamba (64 frames) | 95.8 | 89.5 |
| **MoMa (64 frames)** | **96.9** | **90.0** |

### Zero-Shot Transfer (HMDB51 / UCF101)

Evaluated directly after training on K400, the performance outperforms DiST, indicating that the CLIP distillation loss effectively preserves zero-shot generalization capabilities.

## Highlights & Insights

1. **Clever Fusion Strategy Design**: Instead of directly replacing or concatenating Transformer features with Mamba features, they are injected via scale-bias modulation, maximizing the stability of the pre-trained weights. This design is inspired by FiLM/AdaIN/DiT but innovatively extends scalar modulation to sequence-level modulation.
2. **Significant Efficiency Advantages**: Reduces FLOPs by 25.6% compared to AIM and by 8.5% compared to DiST on K400, while requiring fewer trainable parameters (only 11M for ViT-B/16).
3. **Complementarity of Windowed Attention and SSM**: The Divide stage captures short-range spatial dependencies using windowed local attention, while the Modulate stage captures global long-range spatiotemporal dependencies using SSM, establishing a clear division of labor.
4. **Outstanding Performance in Long-Video Scenarios**: Not only surpasses traditional methods on the Breakfast and COIN datasets but also outperforms VideoMamba, which is specifically designed for long videos.

## Limitations & Future Work

1. **The 8-frame setting of ViT-L/14 on SSv2 is lower than DiST** (72.2 vs 73.1). It requires 32 frames to surpass it, indicating that Mamba's advantage diminishes when the number of frames is limited.
2. **The 8-frame setting of ViT-L/14 on K400 is also lower than competitors** (86.7 vs 87.3), showing that MoMa has a heavier reliance on the number of frames.
3. **The window size $w$ is a fixed hyperparameter**, without exploring adaptive or multi-scale window strategies.
4. **Only validated on classification tasks**, without involving more complex video understanding tasks such as video QA or video captioning.
5. **The specific design of bidirectional multiple scans** (number of scans, scanning order) and its impact on performance is not fully analyzed.
6. **The setting of the weight for the CLIP distillation loss** lacks detailed discussion.

## Key Reproducibility Factors

- Based on pre-trained CLIP models, publicly available.
- Hyperparameters are fully disclosed (lr, wd, SSM parameters, window size).
- 8x V100 + fp16, moderate hardware threshold.
- Uses ActionCLIP prompt templates.
- Training on K400 for 30 epochs takes about 12 hours, ensuring controllable reproduction costs.
- Code is not yet public, but the method description is detailed enough (SSM layer structure and SeqMod formula are explicit).

## Related Work & Insights

- **AIM / DiST / EVL / ST-Adapter**: Belong to the same PEFT paradigm for adapting CLIP to video understanding. The core difference of MoMa lies in using SSM to replace extra attention/encoder modules for temporal modeling.
- **VideoMamba**: A pure Mamba video model trained from scratch, whereas MoMa embeds Mamba into a pre-trained Transformer.
- **FiLM / AdaIN / DiT**: The source of AdaN techniques, which inspired the design of SeqMod.
- **Jamba / MambaVision**: Pioneers of hybrid Mamba-Attention architectures, but both are trained from scratch, whereas MoMa faces different constraints.

## Rating
- Novelty: 4/5 - The SeqMod operation generalizes AdaN to sequence-level modulation for Mamba-Transformer fusion, which is an innovative concept.
- Experimental Thoroughness: 4/5 - Covers K400, SSv2, long video, and zero-shot transfer, with ablation studies analyzing multiple fusion methods.
- Writing Quality: 4/5 - Clear structure, complete mathematical derivations, and sufficient motivation.
- Value: 4/5 - Provides a generic, non-disruptive fusion paradigm for embedding Mamba into pre-trained Transformers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PAVE: Patching and Adapting Video Large Language Models](../../CVPR2025/video_understanding/pave_patching_and_adapting_video_large_language_models.md)
- [\[CVPR 2025\] Efficient Transfer Learning for Video-language Foundation Models](../../CVPR2025/video_understanding/efficient_transfer_learning_for_video-language_foundation_models.md)
- [\[CVPR 2026\] Gamba: Mamba-based Graph Convolutional Network with Dynamic Graph Topology Learning for Action Recognition](../../CVPR2026/video_understanding/gamba_mamba-based_graph_convolutional_network_with_dynamic_graph_topology_learni.md)
- [\[NeurIPS 2025\] MimeQA: Towards Socially-Intelligent Nonverbal Foundation Models](../../NeurIPS2025/video_understanding/mimeqa_towards_socially-intelligent_nonverbal_foundation_models.md)
- [\[CVPR 2026\] UniVBench: Towards Unified Evaluation for Video Foundation Models](../../CVPR2026/video_understanding/univbench_towards_unified_evaluation_for_video_foundation_models.md)

</div>

<!-- RELATED:END -->
