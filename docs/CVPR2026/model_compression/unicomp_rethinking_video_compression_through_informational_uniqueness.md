---
title: >-
  [Paper Note] UniComp: Rethinking Video Compression Through Informational Uniqueness
description: >-
  [CVPR 2026][Model Compression][visual token compression] This paper proposes UniComp, a video token compression framework grounded in informational uniqueness rather than attention scores. Through three modules—Frame Gro…
tags:
  - "CVPR 2026"
  - "Model Compression"
  - "visual token compression"
  - "informational uniqueness"
  - "video understanding"
  - "MLLM efficiency"
  - "plug-and-play"
date: 2026-05-08
content_hash: b66936232e66de1a
---

# UniComp: Rethinking Video Compression Through Informational Uniqueness

**Conference**: CVPR 2026
**arXiv**: [2512.03575](https://arxiv.org/abs/2512.03575)  
**Code**: [TimeMarker-LLM/UniComp](https://github.com/TimeMarker-LLM/UniComp)  
**Area**: Model Compression
**Keywords**: visual token compression, informational uniqueness, video understanding, MLLM efficiency, plug-and-play

## TL;DR

This paper proposes UniComp, a video token compression framework grounded in informational uniqueness rather than attention scores. Through three modules—Frame Group Fusion, Token Allocation, and Spatial Dynamic Compression—UniComp maximally preserves unique information across temporal, spatial, and global dimensions, surpassing the uncompressed baseline even when retaining only 10% of tokens.

## Background & Motivation

**Background**: Multimodal large language models face severe computational bottlenecks when processing video—32 frames can generate thousands of visual tokens. Existing compression methods such as VisionZip and HoliTom rely primarily on attention scores for importance estimation and token selection.

**Limitations of Prior Work**: Attention-based methods suffer from three issues: (1) saliency bias causes selected tokens to be highly redundant with each other; (2) fine-grained details tend to be overlooked; (3) information loss becomes severe under aggressive compression ratios. Moreover, FastVid and HoliTom require tuning 5+ hyperparameters, while DyCoke and similar methods require modifications to the internal attention layers of the LLM, making cross-architecture transfer difficult.

**Key Challenge**: High attention score does not imply informational uniqueness. Highly attended tokens may be mutually similar, and retaining them does not maximize information fidelity. The essence of compression should be preserving irreplaceable information, not the most salient.

**Goal**: Given a limited computational budget, select a token subset that best represents the overall visual information, such that the information of discarded tokens can be reconstructed from the retained ones.

**Key Insight**: The problem is formulated from an information-theoretic perspective, modeling compression as minimizing the conditional entropy $H(\mathcal{X}|\mathcal{S})$, and deriving a theoretical connection between reconstruction error upper bounds and token uniqueness.

**Core Idea**: Replace attention scores with "informational uniqueness" measured by cosine distance as the token importance criterion, and achieve information-optimal compression via greedy selection combined with neighborhood fusion.

## Method

### Overall Architecture

UniComp consists of three cascaded modules: (1) Frame Group Fusion (FGF), which adaptively merges semantically similar frames along the temporal dimension; (2) Token Allocation (TA), which globally distributes the token budget based on frame-level uniqueness; and (3) Spatial Dynamic Compression (SDC), which greedily selects and fuses tokens within each frame based on token-level uniqueness. The input is the visual tokens from the ViT encoder output, and the compressed token sequence is passed directly to the LLM.

### Key Designs

1. **Frame Group Fusion (FGF)**:

    - **Function**: Adaptively merge temporally redundant frames.
    - **Mechanism**: A global feature is obtained via average pooling for each frame. The frame sequence is scanned sequentially; if the uniqueness $u(f_t, f_r) < U_f$ between the current frame and the group's reference frame falls below threshold, the frame is assigned to the current group; otherwise, a new group is created. Each group is fused into a representative feature via mean pooling.
    - **Design Motivation**: Consecutive frames in static scenes are aggressively merged, while semantically abrupt transitions are preserved at fine granularity, enabling adaptive temporal compression.

2. **Token Allocation (TA)**:

    - **Function**: Dynamically allocate the token budget per frame based on frame-level uniqueness.
    - **Mechanism**: The uniqueness of each fused frame is computed as $U_t = 1 - \frac{1}{K_f}\sum_s \cos(f_t, f_s)$. After mean normalization, the scores are amplified by $\sqrt{K_f}$ to accentuate differences, and softmax yields the allocation ratio $K_t = \lfloor \frac{e^{U_t}}{\sum e^{U_s}} \cdot \text{TOKEN}_{max} \rfloor$.
    - **Design Motivation**: Frames with higher uniqueness are more critical for video understanding and should receive a larger token budget.

3. **Spatial Dynamic Compression (SDC)**:

    - **Function**: Greedily select the most representative tokens within each frame based on token-level uniqueness.
    - **Mechanism**: The intra-frame token uniqueness matrix is computed, and tokens are greedily selected in descending order: the most unique token is selected first, tokens with a uniqueness gap $< U_c$ are marked as redundant, and neighborhood fusion is applied to merge them. This is theoretically equivalent to minimizing the reconstruction error upper bound $\mathcal{E}(\mathcal{S}) \leq 2\sum_j \min_{i \in \mathcal{S}} u_{ij}$.
    - **Design Motivation**: Fusing rather than discarding redundant tokens preserves aggregated information.

### Loss & Training

UniComp is a training-free, plug-and-play method. Only 2 hyperparameters are required: the frame group fusion threshold $U_f$ and the spatial compression threshold $U_c$, whose default values transfer across different ViT and LLM architectures. Uniqueness is computed using the Key features from the last attention layer of the ViT.

## Key Experimental Results

### Main Results (32-frame input, LLaVA-OneVision-7B)

| Method | Retention Ratio | LongVideoBench | EgoSchema | MLVU | VideoMME | Avg. | vs. Baseline |
|--------|----------------|---------------|-----------|------|---------|------|-------------|
| Vanilla | 100% | 56.3 | 60.4 | 64.7 | 58.4 | 59.95 | 100% |
| VisionZip | 25% | 56.5 | 60.3 | 64.8 | 58.2 | 59.95 | 100% |
| HoliTom | 25% | 56.7 | 61.2 | 64.7 | 58.6 | 60.30 | 100.6% |
| **UniComp** | **25%** | **57.6** | **61.6** | **65.0** | **58.9** | **60.78** | **101.4%** |
| VisionZip | 10% | 49.3 | 58.0 | 59.7 | 53.4 | 55.10 | 91.9% |

### Ablation Study

| Configuration | LongVideoBench | VideoMME | Note |
|--------------|---------------|---------|------|
| Full UniComp | 57.6 | 58.9 | Complete model |
| w/o FGF | 56.8 | 58.2 | Removing FGF: −0.8 |
| w/o TA | 57.0 | 58.5 | Removing TA: −0.6 |
| w/o SDC fusion | 56.5 | 57.8 | Removing neighborhood fusion: −1.1 |

### Key Findings

- UniComp surpasses the uncompressed baseline at 25% retention (101.4%), suggesting that compression removes redundant information that interferes with the LLM.
- At 10% retention, UniComp maintains approximately 100% of baseline performance, while VisionZip drops to 91.9%.
- As a plug-and-play method, UniComp is effective across three architectures: LLaVA-OV, LLaVA-Video, and Eagle2.5.

## Highlights & Insights

- **Informational Uniqueness vs. Attention**: The perspective shift is compelling—highly attended tokens may be mutually redundant, while tokens with high uniqueness guarantee diverse information coverage. The visualizations clearly illustrate the difference between the two criteria.
- **Theory-Practice Loop**: The greedy algorithm is derived from minimizing conditional entropy, establishing a theoretical connection between reconstruction error and the uniqueness upper bound. This theory-driven design is elegant and principled.
- **Compression Surpassing Baseline**: This finding implies that LLMs are disturbed by redundant visual tokens when processing too many inputs, and that selective filtering is beneficial.

## Limitations & Future Work

- Uniqueness is measured by cosine distance, which may misidentify tokens that are directionally similar yet semantically distinct.
- Frame group fusion relies on sequential scanning, which may not handle flashbacks or non-linear narratives appropriately.
- The two hyperparameters may require fine-tuning in extreme scenarios.

## Related Work & Insights

- **vs. VisionZip**: Selects tokens based on attention; drops to 91.9% at 10% retention, while UniComp maintains ~100%—uniqueness demonstrates a clear advantage under aggressive compression.
- **vs. HoliTom/DyCoke**: These methods require modifications to the LLM's internal structure; UniComp operates after the ViT output, making it more architecturally general.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The informational uniqueness perspective is an entirely new theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-ratio, multi-benchmark evaluation with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear and the motivation is compelling.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play with compression exceeding baseline; strong practical and academic value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Generative Video Compression with One-Dimensional Latent Representation](generative_video_compression_with_one-dimensional_latent_representation.md)
- [\[ICLR 2026\] Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation](../../ICLR2026/model_compression/taming_momentum_rethinking_optimizer_states_through_low-rank_approximation.md)
- [\[CVPR 2026\] PriVi: Towards a General-Purpose Video Model for Primate Behavior in the Wild](privi_towards_a_general-purpose_video_model_for_primate_behavior_in_the_wild.md)
- [\[ICLR 2026\] Cut Less, Fold More: Model Compression through the Lens of Projection Geometry](../../ICLR2026/model_compression/cut_less_fold_more_model_compression_through_the_lens_of_projection_geometry.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)

</div>

<!-- RELATED:END -->
