---
title: >-
  [Paper Note] Local All-Pair Correspondence for Point Tracking
description: >-
  [ECCV 2024][Video Understanding][Point Tracking] This paper proposes LocoTrack, which achieves all-pair correspondence matching for any points in a video via a local 4D correlation volume. Combined with a lightweight correlation encoder and a length-generalizable Transformer, it obtains state-of-the-art accuracy across all TAP-Vid benchmarks while executing nearly 6 times faster than SOTA methods.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Point Tracking"
  - "4D Correlation"
  - "All-Pair Correspondence"
  - "TAP-Vid"
  - "Efficient Transformer"
date: 2026-05-08
content_hash: 579716d442ff3914
---

# Local All-Pair Correspondence for Point Tracking

**Conference**: ECCV 2024  
**arXiv**: [2407.15420](https://arxiv.org/abs/2407.15420)  
**Code**: [https://github.com/KU-CVLAB/LocoTrack](https://github.com/KU-CVLAB/LocoTrack)  
**Area**: Video Understanding / Point Tracking  
**Keywords**: Point Tracking, 4D Correlation, All-Pair Correspondence, TAP-Vid, Efficient Transformer

## TL;DR
This paper proposes LocoTrack, which achieves all-pair correspondence matching for any points in a video via a local 4D correlation volume. Combined with a lightweight correlation encoder and a length-generalizable Transformer, it obtains state-of-the-art accuracy across all TAP-Vid benchmarks while executing nearly 6 times faster than SOTA methods.

## Background & Motivation

**Background**: Tracking Any Point (TAP) is an emerging foundational visual task. Given a video and a query point, the objective is to locate the corresponding position and predict the occlusion state of this point across all frames. Representative methods such as TAPIR and CoTracker typically employ a local 2D correlation map—computing point-to-region similarity matching between the feature of the query point and a local region in the target frame to locate the corresponding point.

**Limitations of Prior Work**: Local 2D correlation is essentially a "one-to-many" matching approach, where a single query point is matched against all pixels within a region. This leads to severe matching ambiguity in homogeneous regions (e.g., white walls) or repetitive textures (e.g., fences), resulting in multiple response peaks on the similarity map and making it impossible to determine the correct correspondence. Existing methods attempt to alleviate ambiguity through temporal context (MLP-Mixers, 1D convolutions, Transformers), but spatial matching ambiguity remains a performance bottleneck in heavily occluded or complex scenarios.

**Key Challenge**: Dense correspondence methods (such as RAFT in optical flow estimation) utilize a global 4D correlation volume (similarity between all pixel pairs) to effectively resolve ambiguities via bidirectional matching consistency and matching smoothness. However, the computational complexity is $O(H^2W^2)$, which is infeasible at high resolutions. How can the disambiguating advantages of 4D correlation be introduced into point tracking while maintaining computational efficiency?

**Goal**: (1) How to exploit the bidirectional matching and smoothness priors of 4D correlation to resolve matching ambiguity in point tracking? (2) How to efficiently process high-dimensional 4D correlation volumes? (3) How to design a temporal modeling architecture capable of processing videos of arbitrary length?

**Key Insight**: The authors observe that a global 4D correlation is unnecessary; calculating all-pair correlation only within the local neighborhoods of the query point and the predicted location provides sufficient information for disambiguation. This "local 4D correlation" simplifies the problem from infeasible global computation to local computation, drastically reducing complexity.

**Core Idea**: Transition point tracking from point-to-region 2D correlation to region-to-region local 4D correlation, leveraging the bidirectional consistency and matching smoothness priors of all-pair matching to eliminate ambiguity while maintaining efficient computation.

## Method

### Overall Architecture
LocoTrack adopts a two-stage architecture: the **Track Initialization Stage** uses a global 2D correlation map to determine initial correspondences for each frame; the **Track Refinement Stage** iteratively refines the trajectories using a local 4D correlation volume and a Transformer. The inputs are the video $\mathcal{V}$ and the query point $q=(x_q, y_q, t_q)$, and the outputs are the trajectories $\mathcal{T}$ and occlusion probabilities $\mathcal{O}$ across all frames. Feature extraction is performed via ResNet18 with Instance Normalization, generating multi-scale feature pyramids.

### Key Designs

1. **Local 4D Correlation Volume**:

    - **Function**: Establish all-pair dense correspondences within local neighborhoods of the query point and the target location, providing bidirectional matching and smoothness priors for disambiguation.
    - **Mechanism**: In the $k$-th iteration of the refinement stage, a target region of radius $r_p$ is cropped around the current predicted position $\mathcal{T}^k_t$, and a query region of radius $r_q$ is cropped around the query point. The cosine similarity of all pixel pairs between the two regions is computed to form a 4D tensor $L_t \in \mathbb{R}^{h_p \times w_p \times h_q \times w_q}$, where $h_p = w_p = 2r_p+1$ and $h_q = w_q = 2r_q+1$ (in experiments, $r_p = r_q = 3$, yielding a $7 \times 7 \times 7 \times 7$ 4D volume). Compared to 2D correlation, which only contains unidirectional "query point $\to$ target region" information, 4D correlation provides bidirectional all-pair matching information.
    - **Design Motivation**: 2D correlation produces multiple response peaks under repetitive textures, failing to resolve ambiguity. 4D correlation provides two key priors: (1) **Bidirectional consistency**—if A corresponds to B, then B should also correspond to A; (2) **Matching smoothness**—correspondences of adjacent points should be spatially continuous. Both priors have been shown in dense correspondence literature to effectively resolve matching ambiguity.

2. **Lightweight 4D Correlation Encoder**:

    - **Function**: Compress the high-dimensional 4D correlation volume into a compact feature embedding while preserving disambiguation information.
    - **Mechanism**: Directly processing 4D tensors incurs prohibitive computational and parameter overhead. The encoder factorizes the 4D correlation into two symmetric branches: one branch treats query dimensions as spatial dimensions and flattens target dimensions into channel dimensions; the other branch functions vice versa. Each branch uses stacked strided 2D convolutions + Group Normalization + ReLU to progressively reduce spatial dimensions, culminating in global average pooling to obtain a compact vector. The outputs of both branches are concatenated to yield the correlation embedding $E_t^k$. This factorization strategy enables highly efficient processing of 4D data using only standard 2D convolutions.
    - **Design Motivation**: Directly processing 4D tensors requires 4D convolutions, demanding huge parameter and computational budgets. Inspired by Cost Aggregation concepts in dense correspondence literature, this work splits 4D processing into two sets of 2D processing via dimension factorization, substantially lowering complexity while retaining bidirectional matching profiles.

3. **Length-Generalizable Transformer**:

    - **Function**: Integrate temporal context and enable the model to process videos of arbitrary lengths without sliding-window inference.
    - **Mechanism**: This component stacks a 3-layer Transformer to apply self-attention over the correlation embedding sequence. The key innovation lies in its positional encoding: instead of using sinusoidal positional encodings (which suffer from performance degradation when the sequence length deviates from the training duration), it employs relative position biases (ALiBi-style). To enable the bias to distinguish left/right directions (whether token A is before or after B), the attention heads are split into two groups—one encoding the relative positions of left-side tokens only, and the other encoding right-side tokens. The bias function is defined as $b(t_1, t_2; h) = -s_h |t_1 - t_2|$, where the scaling factor $s_h$ is configured differently across attention heads.
    - **Design Motivation**: MLP-Mixers cannot handle variable-length sequences, necessitating sliding-window inference, while 1D convolutions require deep stacking to achieve sufficient receptive fields. Transformers yield global receptive fields in a single layer. Equipped with relative position biases, they generalize seamlessly to sequence lengths unseen during training, entirely removing the need for sliding-window inference and its associated computational overhead.

### Loss & Training
Huber loss is used to supervise trajectory position accuracy, and cross-entropy loss is used for occlusion classification. Predictions from multiple refinement iterations contribute to the loss function with exponentially decaying weights (giving higher weight to later iterations). The model is trained on the synthetic Panning MOVi-E dataset: the initialization stage is first trained for 100K steps, followed by 300K steps for the refinement stage. The optimization uses AdamW (lr=1e-3) with a cosine decay scheduler.

## Key Experimental Results

### Main Results (Strided Query Mode, 256×256)

| Dataset | Metric (AJ↑) | LocoTrack-S | LocoTrack-B | TAPIR | CoTracker | Gain |
|--------|----------|-------------|-------------|-------|-----------|------|
| TAP-Vid-DAVIS | AJ | 66.9 | 67.8 | 61.3 | 65.9* | +1.9 vs CoTracker |
| TAP-Vid-Kinetics | AJ | 59.6 | 59.5 | 57.2 | - | +2.3 vs TAPIR |
| TAP-Vid-RGB-Stacking | AJ | 77.4 | 77.1 | 62.7 | - | +14.4 vs TAPIR |

*CoTracker results under 384×512 resolution

| Model | Throughput (points/s)↑ | Parameters (M) | FLOPs per point (G) |
|------|-------------------|----------|-------------|
| LocoTrack-S | 7,244 | 8.2 | 1.08 |
| LocoTrack-B | 4,359 | 11.5 | 2.10 |
| TAPIR | 2,097 | 29.3 | 5.12 |
| CoTracker | 1,147 | 45.5 | 4.65 |

### Ablation Study

| Configuration | AJ↑ | δ_avg↑ | OA↑ | Description |
|------|-----|--------|-----|------|
| 2D Correlation (no neighborhood) | 65.0 | 77.2 | 89.0 | Baseline 2D method |
| Random sampled neighborhood | 65.7 | 77.8 | 88.9 | Random points underperform dense grid |
| Horizontal line neighborhood | 66.5 | 78.4 | 89.4 | Limited improvement with 1D neighborhood |
| Regular grid (r=2) | 67.2 | 79.1 | 89.5 | Smaller 4D region |
| Regular grid (r=3, Ours) | 67.8 | 79.6 | 89.9 | Optimal configuration |
| Sinusoidal positional encoding | 61.9 | 73.9 | 83.5 | Poor variable-length generalization |
| Relative position bias (Ours) | 67.8 | 79.6 | 89.9 | Supports arbitrary length |

### Key Findings
- **4D vs 2D Correlation**: Upgrading from 2D to 4D yields a +2.8 AJ improvement, verifying the disambiguation advantage of all-pair matching.
- **Importance of Dense Sampling**: Random and linear neighborhoods underperform the regular grid, indicating that dense and spatially continuous all-pair matching is required to fully exploit smoothness priors.
- **Crucial Role of Positional Encodings**: Sinusoidal encoding drops AJ to 61.9 during variable-length inference, whereas relative position bias maintains 67.8 (a 5.9 gap), underscoring that length generalization is key.
- **TAPIR Beaten in a Single Iteration**: LocoTrack outperforms TAPIR (which uses 4 iterations) with only 1 refinement iteration while running 9 times faster.
- **Significant Efficiency Advantage**: LocoTrack-S has only 8.2M parameters (1/5.5 of CoTracker) and delivers a throughput of 7,244 points/s (6.3x of CoTracker).

## Highlights & Insights
- **Paradigm Shift from 2D to 4D**: Elevating point tracking from "point-to-region" 2D matching to "region-to-region" 4D matching represents more than a dimensional upgrade. It introduces two powerful priors—bidirectional consistency and matching smoothness. This approach can be extended to other matching tasks requiring disambiguation.
- **Dimension Factorization Encoder**: Factorizing 4D operations into two symmetric 2D operations is both elegant and highly efficient. Future work can leverage this factorization strategy to handle other high-dimensional tensors.
- **Practical Value of Length Generalization**: Eradicating sliding-window inference via relative position biases enables the model to directly process videos of any length, which is crucial for practical applications. The left-right grouping design resolves direction-awareness issues.

## Limitations & Future Work
- Performance counterintuitively degrades on certain datasets at higher resolutions (384×512). The authors attribute this to the shrinking effective receptive field of local correlation at higher resolutions—adaptive adjustment of the local region size could be considered.
- The local region sizes of $r_p = r_q = 3$ are fixed, which might fail to cover correct correspondences in large-displacement scenarios. Designing multi-scale 4D correlations is a potential remedy.
- Training relies solely on synthetic data, which may restrict generalization to real-world scenarios. Fine-tuning on real data could offer further improvements.
- Occlusion prediction accuracy (OA) shows limited improvement (+0.9), suggesting that 4D correlation contributes less to occlusion estimation than to spatial precision.

## Related Work & Insights
- **vs TAPIR**: TAPIR uses 2D correlation combined with 1D convolution for temporal modeling, requiring multiple iterations to converge. LocoTrack utilizes 4D correlation to provide stronger spatial priors, outperforming TAPIR in a single iteration.
- **vs CoTracker**: CoTracker leverages spatial context by tracking auxiliary support points, introducing substantial computational overhead (45.5M vs. 8.2M parameters). LocoTrack secures stronger spatial contexts via 4D correlation without tracking additional support points.
- **vs RAFT (Optical Flow)**: RAFT employs a global 4D correlation for optical flow estimation. LocoTrack introduces this concept into point tracking, localized to ensure feasibility for long videos.
- 4D correlation has also shown major benefits in tasks like few-shot segmentation (HSNet) and video object segmentation (XMem), suggesting wide potential for exploitation in diverse visual tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Successfully introduces 4D correlation from dense correspondence into point tracking, with clever localization and factorization encoder designs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Thoroughly evaluated across all TAP-Vid benchmarks; ablation studies cover every component of the core design.
- Writing Quality: ⭐⭐⭐⭐⭐ Logically coherent, progressing step-by-step from 2D/4D comparisons to localization motivations, and onto the encoder and Transformer designs.
- Value: ⭐⭐⭐⭐⭐ Comprehensively outperforms SOTA in both accuracy and efficiency, possesses strong practicality, and holds profound implications for future point tracking research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Matching Every Pair to Track Every Point: PairFormer for All-Pairs Tracking and Video Trajectory Fields](../../CVPR2026/video_understanding/matching_every_pair_to_track_every_point_pairformer_for_all-pairs_tracking_and_v.md)
- [\[ECCV 2024\] DINO-Tracker: Taming DINO for Self-Supervised Point Tracking in a Single Video](dino-tracker_taming_dino_for_self-supervised_point_tracking_in_a_single_video.md)
- [\[ECCV 2024\] Self-Supervised Any-Point Tracking by Contrastive Random Walks](self-supervised_any-point_tracking_by_contrastive_random_walks.md)
- [\[CVPR 2026\] Generative Point Tracking and Forecasting](../../CVPR2026/video_understanding/generative_point_tracking_and_forecasting.md)
- [\[ICLR 2026\] Point Prompting: Counterfactual Tracking with Video Diffusion Models](../../ICLR2026/video_understanding/point_prompting_counterfactual_tracking_with_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
