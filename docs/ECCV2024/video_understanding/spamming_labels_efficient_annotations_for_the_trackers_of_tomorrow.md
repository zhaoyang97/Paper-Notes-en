---
title: >-
  [Paper Note] SPAMming Labels: Efficient Annotations for the Trackers of Tomorrow
description: >-
  [ECCV 2024][Video Understanding][Multiple Object Tracking] Propopses the SPAM video annotation engine, which combines synthetic data pre-training, pseudo-label self-training, and graph-hierarchy active learning, generating Multiple Object Tracking (MOT) annotations close to ground-truth (GT) quality with only 3-20% of the manual annotation effort.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Multiple Object Tracking"
  - "Video Annotation Engine"
  - "Pseudo-labels"
  - "Active Learning"
  - "Graph Neural Networks"
date: 2026-05-08
content_hash: 74296930446bf2f0
---

# SPAMming Labels: Efficient Annotations for the Trackers of Tomorrow

**Conference**: ECCV 2024  
**arXiv**: [2404.11426](https://arxiv.org/abs/2404.11426)  
**Code**: [https://research.nvidia.com/labs/dvl/projects/spam](https://research.nvidia.com/labs/dvl/projects/spam)  
**Area**: Video Understanding (Multiple Object Tracking / Efficient Annotation)  
**Keywords**: Multiple Object Tracking, Video Annotation Engine, Pseudo-labels, Active Learning, Graph Neural Networks

## TL;DR

Propopses the SPAM video annotation engine, which combines synthetic data pre-training, pseudo-label self-training, and graph-hierarchy active learning, generating Multiple Object Tracking (MOT) annotations close to ground-truth (GT) quality with only 3-20% of the manual annotation effort.

## Background & Motivation

Multiple Object Tracking (MOT) is a core task in video understanding, but high-quality trajectory annotation is extremely expensive:
- **High Annotation Cost**: Each frame requires detection, localization (bounding box with two clicks), and cross-frame identity association (one click), which is highly time-intensive.
- **Small Scale of Existing Datasets**: MOT17 contains only 14 video sequences, and MOT20 has only 8, which is far smaller than the large-scale datasets in the image domain.
- **Limitations of Prior Work**:
    - Most methods ignore dense temporal dependencies in videos (e.g., only selecting keyframes for annotation).
    - Or are limited to single-object scenarios.
    - There is no unified solution to handle both detection and association annotations concurrently.

**Key Insight**:
1. Association in most tracking scenarios is "easy" — pre-trained models can generate high-quality pseudo-labels at zero cost.
2. Trajectory annotations exhibit spatiotemporal dependencies — annotating a single trajectory cascades to influence neighboring trajectories, suggesting annotation should be trajectory-centric rather than frame-centric.

## Method

### Overall Architecture

SPAM = **S**ynthetic pre-training + **P**seudo-labeling + **A**ctive learning + graph-based **M**odel

Pipeline:
1. Pre-train the detector, ReID network, and GNN hierarchy on synthetic data (MOTSynth).
2. Generate pseudo-labels on the target real-world dataset using the pre-trained model, followed by self-training fine-tuning.
3. Annotate the majority of the data using pseudo-labels from the updated model; use active learning to select hard and uncertain samples for manual annotation.
4. Output the final high-quality annotations to train downstream trackers.

### Key Designs

1. **Hierarchical Graph Model (Hierarchical GNN + GNN_node)**:

    - Hierarchical Graph Neural Network based on SUSHI: Divides videos into subsequences to construct subgraphs, merging short tracklets into long trajectories layer-by-layer.
    - Nodes = detection candidates, Edges = association hypotheses.
    - **Novelty: Newly introduced GNN_node layer for detection filtering**:
        - Uses a detector with a low confidence threshold to obtain an over-complete set of candidates (high recall → many false positives).
        - GNN_node utilizes spatiotemporal consistency to classify nodes on the graph as valid/invalid detections.
        - Experiments demonstrate that adding low-confidence bounding boxes without GNN_node results in a dramatic drop in performance (MOTA falls from 64.4 to 60.6), whereas adding GNN_node improves it to 65.4.

2. **Synthetic Pre-training + Domain Gap Analysis**:

    - Comprehensively analyzes the synthetic-to-real domain gap of the three major tracking components (detection, association, ReID).
    - Conclusion: **Detection is most affected by the domain gap** (a gap of 9.9 HOTA points), ReID is almost unaffected, and association has a moderate gap (2.1 HOTA points).
    - Consequently, the annotation effort should focus on detection and association, while ReID can directly leverage models trained on synthetic data.

3. **Uncertainty-Based Active Learning (Graph-Hierarchical Annotation)**:

    - For each node $v$, compute the uncertainty: $\text{uncert}(v) = \max_{u \in N_v} H(\hat{y}_{(v,u)})$
    - $H$ is the binary cross-entropy uncertainty.
    - Nodes with high uncertainty are handed over to humans for manual annotation, while others use model pseudo-labels.
    - **Hierarchical Annotation**: Allocates the annotation budget $B$ across different hierarchy levels $B_1, ..., B_L$.
    - Deep nodes represent entire trajectories; annotating them once resolves identity associations for multiple detections, leading to highly efficient budget usage.
    - Annotation action types: (i) accept/reject detection (1 click), (ii) refine box (2 clicks), (iii) cross-frame association (1 click).

### Loss & Training

- The GNN model is trained end-to-end with edge classification and node classification.
- Synthetic pre-training → pseudo-label self-training (zero manual annotation cost) → active learning to annotate hard samples.
- Pseudo-label self-training yields a 4-6 HOTA point improvement (with zero manual cost).

## Key Experimental Results

### Main Results

Test set results with SPAM configured as a tracker (compared with SOTA trackers):

| Method | MOT17 HOTA↑ | MOT17 IDF1↑ | MOT20 HOTA↑ | DanceTrack HOTA↑ |
|------|------------|-------------|------------|-----------------|
| ByteTrack | 62.8 | 77.1 | 60.4 | 47.7 |
| GHOST | 62.8 | 77.1 | 61.2 | 56.7 |
| SUSHI | 66.5 | 83.1 | 64.3 | 63.3 |
| **SPAM** | **67.5** | **84.6** | **65.8** | **64.0** |

Downstream trackers trained with SPAM labels vs. GT labels (MOT17 validation set):

| Tracker | Label Source | Annotation Vol. | HOTA↑ | MOTA↑ |
|--------|---------|--------|-------|-------|
| ByteTrack | GT | 100% | 52.6 | 60.4 |
| ByteTrack | SPAM | 3.3% | 52.5 | 61.8 |
| GHOST | GT | 100% | 49.5 | 58.0 |
| GHOST | SPAM | 3.3% | 51.3 | 61.9 |

**Achieving or even exceeding the level of training with ground-truth (GT) labels using only 3.3% of manual annotation!**

### Ablation Study

| Configuration | HOTA↑ | MOTA↑ | IDF1↑ | Description |
|------|-------|-------|-------|------|
| High-confidence boxes only (w/o GNN_node) | 59.9 | 64.4 | 74.7 | Baseline |
| Low-confidence boxes added (w/o GNN_node) | 58.5 | 60.6 | 71.4 | Increased false positives, performance drop |
| Low-confidence boxes added + GNN_node | 60.4 | 65.4 | 75.1 | GNN_node effectively filters false positives |

Effect of pseudo-label self-training (SPAM model itself, without manual annotation):

| Dataset | HOTA w/o Pseudo-labels | HOTA w/ Pseudo-labels | Gain |
|--------|-------------|-------------|------|
| MOT17 | 60.0 | 63.8 | +3.8 |
| MOT20 | 52.2 | 58.7 | +6.5 |
| DanceTrack | 41.8 | 48.1 | +6.3 |

### Key Findings

- **Synthetic pre-training suffices for most simple scenarios**: ReID only needs training on synthetic data, while detection and association require fine-tuning on real data.
- **Striking effects of pseudo-label self-training**: Without any manual annotation, simply generating pseudo-labels from the synthetically pre-trained model and performing self-training improves performance by 4-6 HOTA points.
- **Graph-hierarchical active learning significantly outperforms frame-level annotation**: Comparative experiments show that uncertainty sampling at the node level is much better than image-level sampling.
- **Hierarchical annotation is more efficient**: Deep nodes represent long trajectories, and a single annotation resolves multiple points of uncertainty.

## Highlights & Insights

- **The core concept of SPAM is highly practical**: 3% annotation effort yields ≈ 100% of the training performance, which is of great significance for constructing large-scale tracking datasets.
- **A unified graph framework for both detection and association annotations**: GNN_node + edge classification jointly handle two types of annotation challenges within a single graph structure.
- **Domain gap analysis provides guidance on annotation priorities**: Detection > Association > ReID. This conclusion has direct guiding value for data collection in the tracking community.
- The self-training loop (synthetic pre-training → pseudo-labeling → re-training) establishes a robust baseline requiring no manual annotation.

## Limitations & Future Work

- The annotator itself does not generate new detections — if the detector misses targets entirely, it can only make up for it via low confidence thresholds, which does not guarantee full recovery.
- For extremely crowded scenarios (e.g., MOT20), false positive filtering by GNN_node might be insufficient.
- Re-training iterations after annotations have not been explored — can multi-round self-training continue to deliver improvements?
- Currently, only ByteTrack and GHOST have been validated as downstream trackers; evaluation with more trackers would be more convincing.

## Related Work & Insights

- SUSHI is the direct predecessor of the GNN hierarchical architecture in this work; SPAM builds on it by incorporating GNN_node and the annotation engine.
- Shares a common underlying philosophy with efficient annotation methods in the image domain (such as DINO self-training) but is the first to systematically apply it to video tracking.
- Highly inspiring for building future video understanding datasets: shifting from "annotating every frame" to "selectively annotating hard examples."

## Rating

- Novelty: ⭐⭐⭐⭐ (The system integration scheme is novel; individual components represent clever combinations of existing technologies.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three datasets MOT17/20/DanceTrack + complete ablation + domain gap analysis + downstream validation)
- Writing Quality: ⭐⭐⭐⭐ (The system description is clear, and the experiments are reasonably organized.)
- Value: ⭐⭐⭐⭐⭐ (Highly practical value for expanding tracking datasets; the finding regarding the 3% annotation effort is extremely appealing.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] OneTrack: Demystifying the Conflict Between Detection and Tracking in End-to-End 3D Trackers](onetrack_demystifying_the_conflict_between_detection_and_tracking_in_end-to-end_.md)
- [\[ECCV 2024\] VideoMamba: State Space Model for Efficient Video Understanding](videomamba_state_space_model_for_efficient_video_understanding.md)
- [\[ICCV 2025\] XTrack: Multimodal Training Boosts RGB-X Video Object Trackers](../../ICCV2025/video_understanding/xtrack_multimodal_training_boosts_rgb-x_video_object_trackers.md)
- [\[ECCV 2024\] SAFNet: Selective Alignment Fusion Network for Efficient HDR Imaging](safnet_selective_alignment_fusion_network_for_efficient_hdr_imaging.md)
- [\[ECCV 2024\] SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow](sea-raft_simple_efficient_accurate_raft_for_optical_flow.md)

</div>

<!-- RELATED:END -->
