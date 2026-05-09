---
title: >-
  [Paper Note] PRISM: Video Dataset Condensation with Progressive Refinement and Insertion for Sparse Motion
description: >-
  [CVPR 2026][LLM Evaluation][video dataset condensation] This paper proposes PRISM, a holistic video dataset condensation method that begins from only two temporal anchors (first and last frames), adaptively inserts keyframes by detecting gradient direction conflicts, and achieves state-of-the-art storage efficiency while preserving content–motion coupling integrity — reaching 17.9% accuracy with 20 MB on miniUCF 1VPC, a 5× storage reduction over prior methods (94 MB).
tags:
  - CVPR 2026
  - LLM Evaluation
  - video dataset condensation
  - keyframe insertion
  - gradient guidance
  - spatiotemporal coupling
  - storage efficiency
date: 2026-05-08
content_hash: f9587a19f949211f
---

# PRISM: Video Dataset Condensation with Progressive Refinement and Insertion for Sparse Motion

**Conference**: CVPR 2026
**arXiv**: [2505.22564](https://arxiv.org/abs/2505.22564)
**Code**: None (no public code mentioned)
**Area**: LLM Evaluation
**Keywords**: video dataset condensation, keyframe insertion, gradient guidance, spatiotemporal coupling, storage efficiency

## TL;DR

This paper proposes PRISM, a holistic video dataset condensation method that begins from only two temporal anchors (first and last frames), adaptively inserts keyframes by detecting gradient direction conflicts, and achieves state-of-the-art storage efficiency while preserving content–motion coupling integrity — reaching 17.9% accuracy with 20 MB on miniUCF 1VPC, a 5× storage reduction over prior methods (94 MB).

## Background & Motivation

1. **State of the Field**: Dataset distillation/condensation aims to synthesize a compact set far smaller than the original dataset such that models trained on it approach the performance of models trained on the full data. This field is well-studied in the image domain (DC, DSA, DM, MTT, etc.) but nearly unexplored for video — with only one prior work, Wang et al.

2. **Limitations of Prior Work**: The sole prior work decomposes video into "static content" (frozen pretrained images) and "dynamic motion" (auxiliary signals) optimized in two separate stages. This decomposition strategy is fundamentally flawed — in real-world actions, content and motion are **inseparable**. For instance, a frame of two hands coming together in a clapping action is visually identical to a frame of two hands beginning to separate, yet the two belong to entirely different motion trajectories.

3. **Root Cause**: Video data contains substantial temporal redundancy (adjacent frames are highly similar), yet prior methods use a fixed number of frames (e.g., 16) per synthetic video, wasting storage on simple motions and potentially underrepresenting complex ones.

4. **Paper Goals**: (a) Design a holistic video condensation method that preserves content–motion coupling integrity; (b) adaptively allocate representational capacity — using more frames only where needed, while simple motions are adequately represented via linear interpolation.

5. **Starting Point**: The authors build on a key assumption — simple or low-speed motions can be effectively approximated by linear interpolation. Thus, it suffices to identify frames where linear interpolation **fails** (i.e., nonlinear spatiotemporal transition points) and promote them to keyframes. These frames are identified via gradient direction conflicts: if the gradient of an intermediate frame is in the opposite direction to those of its two neighboring keyframes, optimizing the keyframes cannot reduce the loss at that intermediate frame.

6. **Core Idea**: Starting from the minimal set of temporal anchors (first and last frames), PRISM adaptively inserts keyframes during training by detecting negative cosine similarity between intermediate-frame gradients and those of adjacent keyframes, realizing an "allocate frames on demand" paradigm for video condensation.

## Method

### Overall Architecture

PRISM takes a large-scale video dataset as input and produces a compact set of synthetic videos. Each synthetic video is parameterized by a sparse keyframe set $\mathcal{K}$, initially containing only the first and last frames. Intermediate frames are generated via linear interpolation during training; the complete video is fed into a 3D CNN to extract features, which are then matched against the distribution of real videos. When gradient conflicts are detected, a new keyframe is inserted at that position. Training proceeds in three phases: warm-up (stabilizing anchors, 20% of iterations) → progressive insertion (60%) → cool-down (fully optimizing selected frames, 20%). All keyframes are initialized from Gaussian noise, reflecting the holistic synthesis philosophy.

### Key Designs

1. **Temporal Frame Interpolation Parameterization**

   - **Function**: Represent complete video sequences using sparse keyframes.
   - **Mechanism**: Each synthetic video is defined by a keyframe set $\mathcal{K}_c^j = \{s_{c,k_1}, ..., s_{c,k_n}\}$, with $n=2$ initially (first and last frames). Non-keyframes are generated via linear interpolation: $s_{c,t} = \alpha_t s_{c,k_i} + (1-\alpha_t) s_{c,k_{i+1}}$, where $\alpha_t = (k_{i+1}-t)/(k_{i+1}-k_i)$. Only keyframes are trainable parameters; intermediate frames are updated indirectly through interpolation.
   - **Design Motivation**: This parameterization ensures holistic spatiotemporal coupling — optimizing keyframes simultaneously affects all intermediate frames, so the video is optimized as a unified whole from the outset, rather than treating content and motion separately.

2. **Gradient-Guided Frame Insertion**

   - **Function**: Adaptively identify temporal positions requiring additional keyframes.
   - **Mechanism**: For each candidate intermediate frame $s_{c,t}$, the cosine similarities $\cos_i^t$ and $\cos_{i+1}^t$ between its gradient $\nabla \mathcal{L}(s_{c,t})$ and those of its two neighboring keyframes are computed. When both fall below threshold $\epsilon$ (default 0, i.e., negative), the frame is in a "gradient conflict" state — optimizing the endpoint keyframes increases rather than decreases the loss at this intermediate frame. Per the theoretical proof in Lemma 1, no convex combination update of the endpoints can reduce the intermediate frame's loss; it must be promoted to a keyframe for direct optimization. All frames satisfying this condition are inserted simultaneously at each iteration.
   - **Design Motivation**: This direction-based selection criterion is more reliable than L2 distance — it captures semantic-level motion transitions rather than mere pixel differences. Ablation experiments confirm that cosine similarity significantly outperforms L2 distance (7.5% vs. 6.0%).

3. **Warm-up and Cool-down Phases**

   - **Function**: Ensure stability in the frame insertion dynamics.
   - **Mechanism**: Frame insertion is disabled during the first 20% of iterations (warm-up), allowing the endpoint frames to stabilize before serving as reference anchors for subsequent insertions; insertion is also disabled during the last 20% of iterations (cool-down), ensuring that already-selected frames receive sufficient optimization. Progressive frame insertion is performed during the middle 60%.
   - **Design Motivation**: Premature insertion leads to incorrect frame selection due to noisy gradients; frames inserted too late receive insufficient training. The two phases address "when to start" and "when to stop" insertion, respectively. Ablation results show that removing warm-up reduces accuracy by more than 0.7%, and removing cool-down by more than 1.0%.

### Loss & Training

The optimization objective is distribution matching:

$$\min_{\mathcal{K}} \sum_c \left\|\frac{1}{|\mathcal{B}_c^{real}|}\sum_x f_\theta(x) - \frac{1}{|\mathcal{B}_c^{syn}|}\sum_s f_\theta(s)\right\|^2$$

where $f_\theta$ is miniC3D (4-layer Conv3D). SGD with momentum 0.95 and $\epsilon=0$ is used. Synthetic videos are initialized from Gaussian noise. Videos are sampled at 16 frames with stride 4 at resolution 112×112.

## Key Experimental Results

### Main Results

| Dataset | VPC | PRISM | Wang et al. | DM | Herding | PRISM Storage | Prev. Best Storage |
|---------|-----|-------|-------------|-----|---------|---------------|--------------------|
| miniUCF | 10 | 31.0 | - | 30.0 | **33.7** | **324 MB** | 1150 MB |
| miniUCF | 5 | **28.0** | 27.2 | 25.7 | 26.3 | **133 MB** | 455 MB |
| miniUCF | 1 | **17.9** | 17.5 | 15.3 | 13.2 | **20 MB** | 94 MB |
| HMDB51 | 10 | **12.8** | - | 12.1 | 10.8 | **287 MB** | 1150 MB |
| HMDB51 | 5 | **10.5** | 8.2 | 8.0 | 9.0 | **137 MB** | 455 MB |
| HMDB51 | 1 | **7.5** | 6.0 | 6.1 | 3.0 | **22 MB** | 94 MB |

PRISM achieves comprehensive superiority at VPC 5/1 with storage only 1/3 to 1/5 of prior methods.

### Ablation Study

| Ablation | miniUCF (1VPC) | HMDB51 (1VPC) | Notes |
|----------|----------------|----------------|-------|
| Full PRISM | 17.9 | 7.5 | Baseline |
| No frame insertion | 15.8 | 6.1 | Largest drop (−2.1); insertion is critical |
| Random position insertion | 16.8 | 6.8 | Gradient-guided vs. random: +1.1 |
| L2 distance instead of cosine | 15.7 | 6.0 | Direction vs. distance: +2.2 |
| No warm-up | 16.1 | 6.8 | Unstable early insertion |
| No cool-down | 16.9 | 6.3 | Late-inserted frames undertrained |
| Initial 2 frames (default) | 17.9 | 7.5 | Optimal |
| Initial 8 frames | 15.3 | 5.6 | More initial frames perform worse |

### Cross-Architecture Generalization

| Evaluation Model | PRISM | Wang et al. | DM |
|------------------|-------|-------------|-----|
| ConvNet3D | **17.9** | 17.5 | 15.3 |
| CNN+GRU | **18.9** | 12.0 | 9.9 |
| CNN+LSTM | **18.2** | 10.3 | 9.2 |

PRISM generalizes strongly to architectures not used during training — outperforming the nearest baseline by 6.9% on CNN+GRU and 7.9% on CNN+LSTM.

### Key Findings

- **"Starting from fewer frames" outperforms "starting from more"** — 2 initial frames is optimal; 8 initial frames yields the worst results. Excess initial frames generate conflicting gradient signals that interfere with early learning of motion trajectories.
- **Storage does not scale linearly** — owing to adaptive insertion, scaling VPC from 1 to 10 increases storage by only ~15× (rather than 10×), as more videos share similar motion complexity.
- **Cross-architecture generalization is an unexpected benefit of PRISM** — sparse keyframes reduce overfitting to the inductive biases of the training backbone.
- **Action retrieval validates the semantic quality of PRISM** — R@1 on HMDB51 improves from 22.6% to 38.0%, indicating that adaptively inserted frames genuinely capture semantically important spatiotemporal cues.

## Highlights & Insights

- **The "start from noise, add frames on demand" philosophy** fundamentally challenges the dense optimization paradigm. This is analogous to progressive network expansion in NAS, but applied along the temporal dimension.
- **Lemma 1 provides a theoretical foundation** — gradient conflict implies that endpoint updates necessarily increase the loss at intermediate frames, making frame insertion theoretically grounded rather than merely heuristic.
- **Holistic vs. decomposed representations** — the authors use the "clapping action" example to precisely illustrate why decomposing content and motion fails, a perspective generalizable to representation learning in other spatiotemporal tasks.
- **Initialization from Gaussian noise** challenges the common assumption that synthetic data should be initialized from real data — demonstrating that initialization source is secondary when the optimization objective is well-designed.

## Limitations & Future Work

- **Limited handling of extreme abrupt motions** — the linear interpolation assumption may not hold for very rapid scene changes, potentially requiring nonlinear interpolation schemes.
- **Instability for long sequences (>16 frames)** — current validation is limited to 8/16 frames; longer videos may require segment-wise processing.
- **Evaluated only on action recognition and retrieval** — effectiveness on tasks requiring finer spatiotemporal detail (e.g., video generation) remains unknown.
- **Backbone fixed to miniC3D** — applicability to larger 3D backbones (e.g., Video Swin, TimeSformer) is unexplored.
- **Reduced advantage on large-scale datasets** — on Kinetics-400 at VPC 5, DM (9.1%) outperforms PRISM (8.1%), likely due to the low-resolution setting (8 frames + 64×64) limiting the potential of adaptive insertion.

## Related Work & Insights

- **vs. Wang et al. (the only prior work)**: A two-stage decomposition method relying on frozen pretrained static images with motion as an auxiliary signal. PRISM optimizes holistically from scratch, avoiding the content–motion decoupling problem inherent to decomposition.
- **vs. DM (image condensation applied directly to video)**: DM initializes from real frames and optimizes them as independent images. Under extremely low-data settings (Kinetics-400 VPC 5), DM outperforms PRISM, possibly because real-data initialization is more stable under high compression.
- **Relation to image-domain dataset distillation**: PRISM's distribution matching objective is inherited from DM, but the innovation lies in sparse temporal representation and adaptive insertion. Future work could combine gradient matching or trajectory matching objectives with PRISM's frame insertion mechanism.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First holistic video condensation method; gradient-guided frame insertion is theoretically grounded and empirically effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 4 datasets, multiple VPC settings, cross-architecture evaluation, storage analysis, and comprehensive ablations; experiments with larger backbone models are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear; theory and experiments are well-integrated; figures and tables are carefully designed.
- **Value**: ⭐⭐⭐⭐ — Establishes a new paradigm for video condensation; 5× storage savings carry practical significance.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline](pioneering_perceptual_video_fluency_assessment_a_novel_task_with_benchmark_datas.md)
- [\[CVPR 2026\] SparseCam4D: Spatio-Temporally Consistent 4D Reconstruction from Sparse Cameras](sparsecam4d_spatio-temporally_consistent_4d_reconstruction_from_sparse_cameras.md)
- [\[CVPR 2026\] TacSIm: A Dataset and Benchmark for Football Tactical Style Imitation](tacsim_a_dataset_and_benchmark_for_football_tactical_style_imitation.md)
- [\[CVPR 2026\] VGA-Bench: A Unified Benchmark for Video Aesthetics and Generation Quality Evaluation](vga_bench_unified_benchmark_for_video_aesthetics_and_generation_quality.md)
- [\[CVPR 2026\] Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](weakly_supervised_video_anomaly_detection_with_anomaly-connected_components_and_.md)

<!-- RELATED:END -->
