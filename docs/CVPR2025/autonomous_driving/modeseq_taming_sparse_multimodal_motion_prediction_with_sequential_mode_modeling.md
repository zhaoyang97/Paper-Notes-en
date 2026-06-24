---
title: >-
  [Paper Note] ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling
description: >-
  [CVPR 2025][Autonomous Driving][Multimodal Motion Prediction] Proposes ModeSeq—a novel paradigm that models trajectory modes as a sequence. By decoding multimodal trajectories progressively (instead of one-step parallel decoding), it explicitly captures inter-mode correlations. Combined with the Early-Match-Take-All (EMTA) training strategy, it significantly improves trajectory diversity and confidence calibration in sparse multimodal motion prediction…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Multimodal Motion Prediction"
  - "Sequential Mode Modeling"
  - "Sparse Prediction"
  - "Winner-Take-All"
  - "Trajectory Diversity"
date: 2026-05-08
content_hash: 33ab654698159db9
---

# ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling

**Conference**: CVPR 2025  
**arXiv**: [2411.11911](https://arxiv.org/abs/2411.11911)  
**Code**: None  
**Area**: Autonomous Driving / Motion Prediction  
**Keywords**: Multimodal Motion Prediction, Sequential Mode Modeling, Sparse Prediction, Winner-Take-All, Trajectory Diversity

## TL;DR
Proposes ModeSeq—a novel paradigm that models trajectory modes as a sequence. By decoding multimodal trajectories progressively (instead of one-step parallel decoding), it explicitly captures inter-mode correlations. Combined with the Early-Match-Take-All (EMTA) training strategy, it significantly improves trajectory diversity and confidence calibration in sparse multimodal motion prediction, without relying on dense mode predictions or heuristic post-processing.

## Background & Motivation

**Background**: Motion prediction in autonomous driving requires predicting multiple possible future trajectories along with their confidence scores for each traffic participant. Since only a single actual trajectory can be observed in the real world (lacking multimodal ground-truth), mainstream methods employ the Winner-Take-All (WTA) strategy for training—supervising only the predicted mode closest to the GT.

**Limitations of Prior Work**: WTA training easily leads to mode collapse, where multiple predicted trajectories heavily overlap and their confidence scores are hard to distinguish. To alleviate this, some methods generate a large number of candidate trajectories (dense mode prediction) and then select representative modes via post-processing like NMS. However, the hyperparameters of post-processing are difficult to tune, fail to generalize across different scenes, and dense generation followed by filtering impairs trajectory precision.

**Key Challenge**: Existing methods adopt "parallel mode modeling," where all trajectories are decoded at once, making the modes completely independent. Consequently, the model cannot utilize the information of decoded modes to infer what the next mode should be. Trajectory diversity relies solely on parameter differences or anchor differences, which is inherently uncontrolled.

**Goal**: To directly generate a small set of diverse, high-quality, and well-calibrated representative trajectories in an end-to-end manner without relying on dense predictions and post-processing.

**Key Insight**: Since parallel decoding ignores inter-mode relationships, the modes can be modeled as a sequence—decoding one trajectory at each step, conditioned on the previously decoded modes. This forces the model to "see what has already been predicted," thereby avoiding redundancy and improving coverage.

**Core Idea**: To progressively decode trajectory modes in a sequential manner, transforming the unordered multimodal prediction problem into an ordered conditional generation problem.

## Method

### Overall Architecture
ModeSeq adopts an encoder-decoder architecture. The encoder (based on QCNet) processes the map and historical trajectories to generate scene embeddings. The decoder consists of multiple ModeSeq Layers, with each layer progressively outputting $K$ mode embeddings in a recurrent manner. Each mode embedding predicts a trajectory and its corresponding confidence score via an MLP head. The sequential order is coordinated across multiple layers through Mode Rearrangement, and the entire framework is trained with the EMTA strategy.

### Key Designs

1. **Single-layer ModeSeq (Memory Transformer + Context Transformer)**:

    - **Function**: Progressively decodes $K$ mode embeddings in a recurrent manner, with each step conditioned on the previously decoded modes.
    - **Mechanism**: During the $k$-th decoding step, it first utilizes a Memory Transformer with the current mode embedding $\mathbf{m}_k^{(\ell-1)}$ as a query to perform cross-attention over the sequence of decoded modes $\Omega_{k-1}^{(\ell)}$, gaining an "awareness of previous modes." Then, a Context Transformer is used to incorporate the scene embedding $\Psi$ (map, historical trajectories, neighboring agents) to generate the final mode embedding $\mathbf{m}_k^{(\ell)}$, which is appended to the sequence for subsequent steps. The Context Transformer is decomposed into mode-time, mode-map, and mode-agent sub-attention modules to reduce complexity.
    - **Design Motivation**: Unlike DETR-like decoders that only perform self-attention among modes (weak correlation), sequential decoding establishes a causal conditional dependency chain, naturally enforcing repulsion between modes. Moreover, since parameters are shared across steps, the number of modes can be dynamically adjusted during inference (simply by increasing or decreasing decoding steps), which is infeasible for parallel methods.

2. **Multi-layer Iterative Refinement + Mode Rearrangement**:

    - **Function**: Progressively optimizes trajectory quality and mode sequence through multiple rounds of decoding.
    - **Mechanism**: Six ModeSeq Layers are stacked. After each layer decodes $K$ modes, the mode embedding sequence is rearranged in descending order of confidence before being input to the next layer. This ensures high-confidence modes are decoded earlier and low-confidence modes later, establishing a monotonically decreasing confidence order. Training losses are applied to the outputs of each layer.
    - **Design Motivation**: The first few steps in single-layer decoding might produce low-quality modes, disrupting the learning of subsequent modes. Rearrangement ensures high-quality modes are prioritized at the beginning for refined training, creating synergy with the EMTA training strategy.

3. **Early-Match-Take-All (EMTA) Training Strategy**:

    - **Function**: Replaces the WTA strategy to encourage the model to decode a GT-matching trajectory as early as possible, thereby releasing subsequent slots for other potential modes.
    - **Mechanism**: Among the $K$ predictions, all trajectories matching the GT are identified (based on a displacement threshold). The **earliest matching trajectory** (rather than the best match) is labeled as the positive sample, while the remaining ones (including subsequent matches) are labeled as negative. If no match is found, it falls back to the WTA strategy to select the one with the minimum error. Trajectory regression uses Laplace NLL loss, and confidence utilizes Binary Focal Loss.
    - **Design Motivation**: Traditional WTA only optimizes the best match, leading to multiple high-confidence modes in the same region. By labeling the early match as positive and subsequent duplicate matches as negative, EMTA forces the model to free up redundant modes to cover other unexplored futures, thus improving coverage and confidence discriminability.

### Loss & Training
Each layer is supervised with the EMTA loss (Laplace NLL for trajectory regression + Binary Focal Loss for confidence). Training is conducted on WOMD using AdamW for 30 epochs with an initial learning rate of $5 \times 10^{-4}$ and cosine annealing.

## Key Experimental Results

### Main Results

| Dataset | Method | Soft mAP6↑ | mAP6↑ | MR6↓ | minADE6↓ | minFDE6↓ |
|---|---|---|---|---|---|---|
| WOMD Val | QCNet | 0.4508 | 0.4452 | 0.1254 | 0.5122 | 1.0225 |
| WOMD Val | **ModeSeq** | **0.4562** | **0.4507** | **0.1206** | 0.5237 | 1.0681 |
| Argoverse2 | QCNet | - | - | 0.16 | 0.65 | 1.29 |
| Argoverse2 | **ModeSeq** | - | - | **0.14** | **0.63** | **1.26** |

ModeSeq comprehensively outperforms QCNet in mode coverage (MR) and confidence score (mAP), with only minor degradation in trajectory accuracy. ModeSeq ranked first among non-LiDAR entries in the 2024 Waymo Open Motion Prediction Challenge.

### Ablation Study

| Decoder | Training Strategy | Soft mAP6↑ | MR6↓ | minADE6↓ |
|---|---|---|---|---|
| DETR w/ Refinement | WTA | 0.4096 | 0.1536 | 0.5660 |
| ModeSeq | WTA | 0.4138 | 0.1502 | 0.5563 |
| ModeSeq | EMTA | **0.4231** | **0.1457** | **0.5700** |

### Key Findings
- Using only sequential mode modeling (without changing the training strategy) already improves mAP by 0.4%+ and reduces MR by 0.3%+, indicating the inherent value of conditional dependencies between modes.
- EMTA further boosts mAP (+0.9%) and reduces MR (-0.45%) on top of sequential mode modeling, with only a minor reduction in trajectory accuracy of 0.014m (minADE).
- Mode rearrangement is crucial for EMTA: Soft mAP of 0.4231 with rearrangement vs. 0.4112 without rearrangement.
- ModeSeq naturally possesses mode extrapolation capabilities: trained with 6 modes, it can still generate reasonable and diverse trajectories when decoding 24 modes during inference.
- The mAP6 of a 3-mode ModeSeq even outperforms a 6-mode QCNet, demonstrating that sequential modeling can cover more behaviors with fewer modes.

## Highlights & Insights
- **The Paradigm Shift from "Parallel to Sequential"**: Transforming an unordered multimodal prediction problem into an ordered sequential generation problem is a highly elegant concept. Similar to the evolution from set prediction to autoregressive generation in NLP, it introduces a new design dimension for motion prediction.
- **The Design Philosophy of EMTA**: Prioritizing the "earliest match" over the "best match". This seemingly counter-intuitive design cleverly leverages the temporal structure of the sequence to enforce diversity.
- **Mode Extrapolation Capability**: Parameter sharing allows the number of modes in inference to differ from that during training. This "elastic prediction" capability is highly practical in highly uncertain scenarios, which cannot be achieved by other fixed-anchor methods.

## Limitations & Future Work
- Inference latency is roughly double that of QCNet (6-mode: 128ms vs 69ms) because sequential decoding cannot be fully parallelized.
- minADE/minFDE degrades slightly (approx. 0.01m/0.05m), indicating a minor quality trade-off for diversity.
- The order of modes in the sequence affects the results, and whether the optimal sorting strategy (descending by confidence) remains optimal in all scenarios is not fully verified.
- Lacks comparison with diffusion-based motion prediction methods (e.g., MotionDiffuser).

## Related Work & Insights
- **vs QCNet**: Under the same encoder, ModeSeq outperforms in coverage and confidence through sequential decoding, proving the bottleneck lies at the decoder rather than the encoder side.
- **vs MTR/MTR++**: The MTR series relies on dense anchors and post-processing to archive diversity, whereas ModeSeq achieves this end-to-end without any anchors or post-processing.
- **vs FJMP/M2I**: These methods perform serialization along the agent dimension (predicting influencers first, then the influenced). This work is the first to serialize along the mode dimension, offering a brand-new disassembly perspective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Sequential mode modeling is a brand-new paradigm, with the designs of EMTA and mode rearrangement working in close coordination.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across both WOMD and Argoverse2 benchmarks with very thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, smooth derivation of motivations, and meticulously designed figures.
- Value: ⭐⭐⭐⭐⭐ Deeply influential for the motion prediction field, showing both theoretical elegance and practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PAR: Poly-Autoregressive Prediction for Modeling Interactions](poly-autoregressive_prediction_for_modeling_interactions.md)
- [\[CVPR 2025\] Generating Multimodal Driving Scenes via Next-Scene Prediction](generating_multimodal_driving_scenes_via_next-scene_prediction.md)
- [\[CVPR 2025\] SDGOcc: Semantic and Depth-Guided BEV Transformation for 3D Multimodal Occupancy Prediction](sdgocc_semantic_and_depth-guided_birds-eye_view_transformation_for_3d_multimodal.md)
- [\[CVPR 2025\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)
- [\[ICCV 2025\] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting](../../ICCV2025/autonomous_driving/emd_explicit_motion_modeling_for_high-quality_street_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
