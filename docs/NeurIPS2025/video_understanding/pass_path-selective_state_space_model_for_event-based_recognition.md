---
title: >-
  [Paper Note] PASS: Path-Selective State Space Model for Event-Based Recognition
description: >-
  [NeurIPS 2025][Video Understanding][Event Camera] PASS proposes the Path-selective Event Aggregation and Scan (PEAS) module and the Multi-faceted Selection Guiding (MSG) loss, leveraging the linear complexity and frequency generalization capability of SSMs to perform event-based recognition across a broad distribution of event lengths from $10^6$ to $10^9$, while limiting performance degradation to only 8.62% under varying inference frequencies (compared to 20.69% for the bas…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "Event Camera"
  - "State Space Model"
  - "Frequency Generalization"
  - "Long-Sequence Modeling"
  - "Mamba"
date: 2026-05-08
content_hash: 2e3b89975cd73165
---

# PASS: Path-Selective State Space Model for Event-Based Recognition

**Conference**: NeurIPS 2025
**arXiv**: [2409.16953](https://arxiv.org/abs/2409.16953)  
**Code**: [GitHub](https://github.com/jiazhou-garland/PASS_Homepage)  
**Area**: Video Understanding / Event Camera
**Keywords**: Event Camera, State Space Model, Frequency Generalization, Long-Sequence Modeling, Mamba

## TL;DR

PASS proposes the Path-selective Event Aggregation and Scan (PEAS) module and the Multi-faceted Selection Guiding (MSG) loss, leveraging the linear complexity and frequency generalization capability of SSMs to perform event-based recognition across a broad distribution of event lengths from $10^6$ to $10^9$, while limiting performance degradation to only 8.62% under varying inference frequencies (compared to 20.69% for the baseline).

## Background & Motivation

Event cameras are bio-inspired sensors that asynchronously capture brightness changes, offering high temporal resolution, high dynamic range, and low latency. However, existing event-based recognition methods face two critical challenges:

**Limited event-length distribution**: Existing datasets cover event lengths only in the $10^6$–$10^7$ range, whereas high-speed or long-duration event streams require handling a much wider range ($10^6$–$10^9$); the quadratic complexity of Transformers creates a computational bottleneck at large event volumes.

**Poor inference-frequency generalization**: Although event cameras have a natural advantage for high-speed dynamic scenes, model performance degrades significantly (up to −20.69%) when the inference sampling frequency deviates from the training frequency, preventing full exploitation of the high temporal resolution.

Existing model architectures each have their own drawbacks:
- **Step-by-step** structures: support parallel processing but incur high attention complexity.
- **Recurrent** structures: cannot be parallelized and tend to forget early information.

**Core Idea**: Exploit the linear complexity and input-frequency generalization properties of SSMs (Mamba), combined with an adaptive event-frame selection mechanism, to handle broad event distributions and generalize across different inference frequencies.

## Method

### Overall Architecture

Event stream → Fixed-event-length sampling + frame aggregation → PEAS module (selective scan encoding into fixed-dimension features) → MSG loss-guided optimization → SSM spatio-temporal modeling module → Classification prediction

### Key Designs

1. **Event Sampling and Frame Aggregation**:

    - Samples are taken at fixed time windows $1/f$ (where $f$ is the sampling frequency), with a fixed number $G$ of events per sample.
    - This yields $P = Tf$ event groups, converted into event-frame representations $F \in \mathbb{R}^{P \times H \times W \times 3}$.
    - Fixed-event-count aggregation is more robust than fixed-time-window aggregation.

2. **PEAS Module (Path-selective Event Aggregation and Scan)**:

    - **Selection mask prediction**: A two-layer 3D convolution followed by an activation function generates a selection mask $M \in \mathbb{R}^{K \times P}$ from event frames $F$ ($K$ = number of selected frames, $P$ = total frames).
    - **Differentiable selection**: Gumbel Softmax is used during training for differentiable frame selection; standard Softmax is used at inference.
    - **Matrix-multiplication selection**: Einsum multiplies the mask with original frames to obtain selected frames $F' \in \mathbb{R}^{K \times H \times W \times 3}$.
    - **Bidirectional event scanning**: Selected frames are unrolled into a 1D sequence in spatio-temporal order (following VideoMamba's spatio-temporal scanning), concatenated left-to-right and top-to-bottom.
    - Core value: adaptively compresses variable-length event streams ($10^6$–$10^9$) into fixed-dimension features in an end-to-end learnable manner.

3. **MSG Loss (Multi-faceted Selection Guiding)**:

    - **WEIE Loss (Within-Frame Event Information Entropy)**:
        - Computes the grayscale histogram entropy of each selected frame.
        - Maximizing this loss encourages selection of information-rich frames and reduces the randomness of selecting empty (padded) frames.
        - $\mathcal{L}_{WEIE} = -\sum_{k=1}^{K}\sum_{i=1}^{N}P_i^k \log P_i^k / K$
    - **IEMI Loss (Inter-frame Event Mutual Information)**:
        - Computes the joint-distribution mutual information between adjacent selected frames (including spatial position information).
        - Minimizing this loss reduces redundancy among selected frames, ensuring each frame carries unique information.
    - Total objective: $\mathcal{L}_{total} = \mathcal{L}_{IEMI} - \mathcal{L}_{WEIE} + \mathcal{L}_{CLS}$

4. **Event Spatio-Temporal Modeling Module**:

    - 3D convolution ($1\times16\times16$) for patch embedding.
    - Concatenation of a learnable CLS token + spatial positional embeddings + temporal embeddings.
    - Input to $L$ stacked B-Mamba blocks (bidirectional Mamba).
    - CLS token extracted, passed through layer normalization and a linear classification head for final prediction.
    - Initialized with VideoMamba pretrained weights.

### Loss & Training

- Total loss = IEMI (minimized) − WEIE (maximized) + cross-entropy classification loss.
- Model scales: Tiny (7M), Small (25M), Middle (74M).
- The number of selected frames $K$ is a hyperparameter, with different values per dataset (1/2/8/16/32).
- Self-constructed datasets: ArDVS100 (100-class action conversion, event lengths 1s–256s), TemArDVS100 (fine-grained temporal annotations), Real-ArDVS10 (10-class real-world dataset).

## Key Experimental Results

### Main Results

| Dataset | Event Scale | Metric | Ours | Prev. SOTA | Gain |
|---------|-------------|--------|------|------------|------|
| N-Caltech101 | ~$10^6$ | Top-1 | **94.60%** | EventDance: 92.35% | +2.25% |
| N-Imagenet | ~$10^6$ | Top-1 | **61.32%** | MEM: 57.89% | +3.43% |
| PAF | ~$10^7$ | Top-1 | **98.28%** | ExACT: 94.83% | +3.45% |
| SeAct | ~$10^7$ | Top-1 | **66.38%** | ExACT: 66.07% | +0.38% |
| HARDVS | ~$10^7$ | Top-1 | **98.41%** | S5-ViT: 95.98% | +8.31% |
| ArDVS100 | ~$10^9$ | Top-1 | **97.35%** | S5-ViT: 93.39% | +3.96% |
| TemArDVS100 | ~$10^9$ | Top-1 | **89.00%** | S5-ViT: 79.62% | +9.38% |
| Real-ArDVS10 | ~$10^9$ | Top-1 | **100%** | S5-ViT: 93.33% | +6.67% |

### Ablation Study

| Configuration | PAF Top-1 | ArDVS100 Top-1 | Notes |
|---------------|-----------|----------------|-------|
| No sampling | 92.90% | 92.31% | All frames used directly |
| Random sampling | 92.98% | 92.23% | Random selection of $K$ frames |
| PEAS | 93.33% | 92.84% | +0.35% / +0.61% |
| PEAS + MSG | **94.83%** | **93.85%** | +1.85% / +1.62% |

| Frequency Generalization | Performance Drop: Train 60 Hz → Infer 100 Hz |
|--------------------------|----------------------------------------------|
| Time-window baseline | −20.69% |
| Event-count baseline | ~−15% |
| PASS | **−8.62%** |

### Key Findings

- Although PEAS compresses the number of frames, the selected frames retain task-critical information (outperforming the no-sampling baseline by +0.43%).
- The two components of the MSG loss are complementary: IEMI reduces redundancy (+0.77%), while WEIE further reduces selection randomness (an additional +1.08%).
- PASS maintains strong performance (97.35%) on $10^9$-scale events, where baseline methods struggle with such long sequences.
- Frequency generalization is a core advantage of PASS: regardless of whether training is conducted at 20 Hz, 60 Hz, or 100 Hz, cross-frequency inference performance degrades by at most 8.62%.

## Highlights & Insights

- **Natural fit between SSM and event cameras**: The linear complexity and frequency generalization of SSMs align perfectly with the high temporal resolution characteristics of event streams.
- **Information-theoretic frame selection**: Using information entropy and mutual information as selection guidance signals is more principled than heuristic rules.
- **End-to-end selection via Gumbel Softmax**: Differentiable frame selection enables end-to-end training of PEAS, avoiding the complexity of two-stage training pipelines.
- **Self-constructed long-sequence datasets**: ArDVS100 and TemArDVS100 fill the gap in $10^9$-scale event recognition benchmarks.
- **Practical significance of frequency generalization**: In real-world deployment, inference frequencies often differ from training frequencies; the strong generalization of PASS substantially reduces deployment difficulty.

## Limitations & Future Work

- Larger-scale VideoMamba models exhibit overfitting, necessitating better regularization strategies.
- The number of selected frames $K$ is set manually and cannot be determined adaptively.
- Event-frame representation is only one of many event representations; comparisons with voxel grids, time surfaces, and other representations are insufficient.
- The self-constructed datasets are synthesized by concatenation, which may introduce distribution gaps relative to real-world continuous long-duration event streams.

## Related Work & Insights

- **vs. ExACT**: ExACT brute-forces the problem with a 471M-parameter model, whereas PASS achieves superior results more efficiently with 74M parameters.
- **vs. S5-ViT**: S5-ViT is the first to introduce SSMs into event-based detection but addresses frequency generalization via a low-pass constraint loss; PASS tackles frequency generalization more fundamentally through frame selection.
- **vs. VideoMamba**: PASS builds upon VideoMamba but introduces the PEAS module and MSG loss tailored to event-stream characteristics, rather than directly applying the original framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The motivation for applying SSMs to event-based recognition is natural; PEAS + MSG exhibits originality, though it does not represent a breakthrough contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Five public datasets plus three self-constructed datasets, comprehensive frequency generalization experiments, and thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear and figures are abundant, though some notation in the equations is slightly ambiguous.
- **Value**: ⭐⭐⭐⭐ — Provides an efficient long-sequence modeling solution for event-based recognition; the frequency generalization property offers strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] VideoMamba: Spatio-Temporal Selective State Space Model](../../ECCV2024/video_understanding/videomamba_spatio-temporal_selective_state_space_model.md)
- [\[ECCV 2024\] VideoMamba: State Space Model for Efficient Video Understanding](../../ECCV2024/video_understanding/videomamba_state_space_model_for_efficient_video_understanding.md)
- [\[CVPR 2025\] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](../../CVPR2025/video_understanding/mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)
- [\[CVPR 2025\] GG-SSMs: Graph-Generating State Space Models](../../CVPR2025/video_understanding/gg-ssms_graph-generating_state_space_models.md)
- [\[NeurIPS 2025\] Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity](lattice_boltzmann_model_for_learning_real-world_pixel_dynamicity.md)

</div>

<!-- RELATED:END -->
