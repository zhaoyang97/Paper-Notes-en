---
title: >-
  [Paper Note] VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity
description: >-
  [NeurIPS 2025][Video Understanding][Video Anomaly Detection] This paper proposes VADTree, a training-free video anomaly detection framework that leverages a pretrained Generic Event Boundary Detection (GEBD) model to construct a Hierarchical Granularity-aware Tree (HGTree), enabling adaptive sampling and multi-granularity reasoning for anomalous events of varying temporal spans. VADTree achieves state-of-the-art performance among training-free methods on three benchmarks—UCF-…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "Video Anomaly Detection"
  - "Training-Free"
  - "Hierarchical Granularity Tree"
  - "Generic Event Boundary Detection"
  - "Multi-Granularity Reasoning"
date: 2026-05-08
content_hash: f51fe937379ea392
---

# VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity

**Conference**: NeurIPS 2025
**arXiv**: [2510.22693](https://arxiv.org/abs/2510.22693)  
**Code**: [GitHub](https://github.com/wenlongli10/VADTree)  
**Area**: Interpretability
**Keywords**: Video Anomaly Detection, Training-Free, Hierarchical Granularity Tree, Generic Event Boundary Detection, Multi-Granularity Reasoning

## TL;DR

This paper proposes VADTree, a training-free video anomaly detection framework that leverages a pretrained Generic Event Boundary Detection (GEBD) model to construct a Hierarchical Granularity-aware Tree (HGTree), enabling adaptive sampling and multi-granularity reasoning for anomalous events of varying temporal spans. VADTree achieves state-of-the-art performance among training-free methods on three benchmarks—UCF-Crime, XD-Violence, and MSAD—and even surpasses certain weakly supervised approaches.

## Background & Motivation

Video Anomaly Detection (VAD) aims to localize anomalous events in videos and has broad applications in autonomous driving and industrial manufacturing. Conventional approaches (fully supervised, weakly supervised, and unsupervised) rely on training data and lack interpretability. Recent training-free methods exploit the knowledge of VLMs and LLMs for explainable anomaly detection, yet they universally adopt **fixed-length temporal window** sampling strategies. **Key Challenge**: Fixed temporal windows cannot accommodate the highly variable durations of real-world anomalous events—ranging from traffic accidents lasting a few seconds to burglaries spanning several minutes. Windows that are too short lose long-range context, while windows that are too long introduce irrelevant semantic noise. The core idea of this paper is to exploit pretrained GEBD knowledge to construct a hierarchical event tree structure, enabling adaptive multi-granularity anomaly detection.

## Method

### Overall Architecture

VADTree consists of three main modules:

- **Input**: A long video sequence $V = \{I_t\}_{t=1}^{T}$
- **Output**: Frame-level anomaly scores
- **Pipeline**: ① GEBD model generates a boundary confidence sequence → ② Construct a granularity-aware binary tree and partition into layers → ③ Generate descriptions and anomaly scores per node → ④ Intra-cluster refinement + inter-cluster fusion → ⑤ Frame-level anomaly scores

### Key Designs

1. **Hierarchical Granularity-aware Tree (HGTree) Construction**:

    - **Boundary Confidence Sequence Generation**: The long video is segmented into short clips via a sliding window and fed into a pretrained EfficientGEBD model. Boundary confidences from the central region of each window are retained and concatenated into a global sequence $C$; local maxima are extracted to yield the candidate boundary set $\hat{C}$.
    - **Generic Event Node Initialization**: Starting from the entire video as the root node, the tree $\mathcal{T}$ is built by recursively splitting at the highest-confidence boundary. Each node $\mathcal{N}_i = (\hat{c}_l^{(i)}, \hat{c}_r^{(i)}, V_{l:r}^{(i)})$ records the left and right boundary confidences along with the corresponding video segment.
    - **Adaptive Node Layering**: K-Means ($K=2$) clusters the boundary confidences into two groups: coarse-grained (high-confidence boundaries) and fine-grained (low-confidence boundaries). Redundant ancestor nodes are removed (RemoveDup) and indivisible leaf nodes are completed (Complete), yielding $\mathcal{T}' = \{\mathcal{S}'_{coarse}, \mathcal{S}'_{fine}\}$, where both clusters individually cover the complete video.

2. **Prior-Infused Node Anomaly Scoring**:

    - An LLM generates three-dimensional prior knowledge: scene prior $b_{scene}$, object prior $b_{obj}$, and action prior $b_{act}$, while explicitly excluding micro-expression and audio cues imperceptible to VLMs.
    - The priors are injected into VLM prompts to generate content descriptions for sampled frames of each node: $d_u^g = f_{VLM}(V_u^g, B \circ P_d)$.
    - An LLM performs anomaly scoring (discrete values in $[0,1]$) based on the descriptions: $a_u^g = f_{LLM}(d_u^g, P_s)$.

3. **Intra-cluster Node Refinement**:

    - Independent per-node scoring lacks long-range context and is prone to local false positives.
    - Node features are extracted via the ImageBind visual encoder, and cosine similarities are computed.
    - For each node, the top-$K$ most similar nodes are identified, and anomaly scores are refined via softmax-weighted averaging: $\hat{a}_u^g = \sum_{i=1}^{K} a_{\kappa_u^{(i)}} \cdot \frac{\exp(\text{sim}(u, \kappa_u^{(i)})/\tau)}{\sum_j^K \exp(\text{sim}(u, \kappa_u^{(j)})/\tau)}$.

4. **Inter-cluster Node Correlation**:

    - For each coarse-grained parent node, the variance of its child nodes' anomaly scores (cohesion weight $w_i$) is computed.
    - Low variance → semantically consistent children → parent node dominates fusion; high variance → conflicting children → fine-grained nodes take precedence.
    - Final frame-level score: $\bar{a}_{n_{ij}} = \frac{1}{2}(1 - \beta\hat{w}_i)\hat{a}_{n_i} + \frac{1}{2}(1 + \beta\hat{w}_i)\hat{a}_{n_{ij}}$.

### Loss & Training

The proposed method is entirely training-free and involves no parameter updates. At inference time, LLaVA-Video-7B is used as the VLM (64-frame input), DeepSeek-R1-Distill-Qwen-14B as the LLM (with chain-of-thought reasoning enabled), and ImageBind as the visual encoder. Key hyperparameters: $\gamma_{min} = 0.4$ (boundary confidence threshold), $\beta \in [-1, 1]$ (fusion control coefficient).

## Key Experimental Results

### Main Results

| Dataset | Metric | VADTree | Prev. SOTA (Training-Free) | Gain | Best Weakly Supervised |
|--------|------|---------|-------------------|------|-----------|
| UCF-Crime | AUC (%) | **84.74** | SUVAD 83.90 | +0.84 | GS-MoE 91.58 |
| XD-Violence | AUC (%) | **90.44** | EventVAD 87.51 | +2.93 | GS-MoE 94.52 |
| XD-Violence | AP (%) | **67.82** | SUVAD 70.10 | −2.28 | π-VAD 85.37 |
| MSAD | AUC (%) | **89.32** | -- | -- | π-VAD 88.68 |

### Ablation Study

| Configuration | AUC (%) | Note |
|------|--------|------|
| HGTree Fine Cluster (baseline) | 71.57 | Fine-grained cluster only + simple scoring |
| + Prior-Infused Node Scoring | 75.67 | Prior knowledge improves by 4.1% |
| + Intra-cluster Node Refinement | 83.05 | Intra-cluster refinement improves by 7.4% (largest gain) |
| + Inter-cluster Node Correlation | 84.74 | Inter-cluster fusion improves by 1.7% |
| $\gamma_{min}=0.3$ (single cluster) | 80.89 | Over-segmentation |
| $\gamma_{min}=0.4$ (single cluster) | 82.81 | Best single-cluster result |
| $\gamma_{min}=0.4$ (Coarse+Fine) | 84.74 | Hierarchical structure adds additional +1.9% |
| K-Medoids replacing K-Means | 85.24 | More robust clustering +0.5% |

### Key Findings

1. **Intra-cluster node refinement contributes the largest gain** (+7.4%), underscoring the importance of leveraging semantically similar nodes to suppress VLM/LLM hallucinations and noise.
2. **Hierarchical structure outperforms single-cluster** (+1.9%), validating the necessity of coarse-fine cooperative reasoning.
3. **HGTree achieves substantially higher mIoU than fixed windows**: mIoU on XD-Violence improves from 0.44 to 0.64, demonstrating the particular importance of adaptive sampling for long anomalous events.
4. **VADTree surpasses all weakly supervised methods on MSAD** (AUC 89.32% vs. π-VAD 88.68%), indicating strong generalization capability of the framework.
5. Performance varies only marginally across different VLM/LLM combinations (83.56%–84.74%), demonstrating robustness to model selection.

## Highlights & Insights

- Introducing pretrained GEBD knowledge into VAD is a natural and effective design choice—event boundaries inherently correspond to the onset and termination of anomalous events.
- The hierarchical tree structure is elegantly designed: the coarse-grained cluster captures global context while the fine-grained cluster enables precise localization, with the two dynamically fused via variance-based weighting.
- The intra-cluster node refinement is analogous to message passing on a graph, effectively exploiting semantic relationships among nodes.
- Surpassing weakly supervised methods without any training (on MSAD) demonstrates the significant potential of combining large pretrained models with well-structured reasoning.

## Limitations & Future Work

- The quality of the GEBD model directly governs the quality of the tree structure; inaccurate event boundary detection propagates errors to subsequent reasoning.
- Inference overhead is substantial: VLM and LLM must be invoked independently for each node, incurring high deployment costs.
- Only two levels of granularity (coarse and fine) are supported; three or more levels may be better suited for extremely long videos.
- VADTree does not surpass SUVAD on the AP metric for XD-Violence, indicating room for improvement in precise temporal localization.

## Related Work & Insights

- The design of HGTree shares conceptual similarities with VideoTree (ECCV 2024), both adopting event-based hierarchical video representations.
- The intra-cluster refinement mechanism is generalizable to other scenarios that require suppression of VLM hallucinations.
- The GEBD+VAD combination can be extended to event-aware tasks such as video summarization and video question answering.
- The variance-driven coarse-fine fusion strategy constitutes a general-purpose hierarchical decision fusion method.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing GEBD into VAD for event tree construction is a novel design, though hierarchical representation itself is not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, extensive ablations, multiple model combinations, and qualitative analysis are all provided.
- **Writing Quality**: ⭐⭐⭐⭐ Formulations and pipelines are clear, though notation is somewhat dense.
- **Value**: ⭐⭐⭐⭐ A significant advance in training-free VAD; surpassing weakly supervised methods on MSAD is a noteworthy contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection](monitor_exploiting_large_language_models_with_instruction_for_online_video_anoma.md)
- [\[ICLR 2026\] HiTeA: Hierarchical Temporal Alignment for Training-Free Long-Video Temporal Grounding](../../ICLR2026/video_understanding/hitea_hierarchical_temporal_alignment_for_training-free_long-video_temporal_grou.md)
- [\[NeurIPS 2025\] PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer](panda_towards_generalist_video_anomaly_detection_via_agentic_ai_engineer.md)
- [\[CVPR 2025\] VideoGEM: Training-Free Action Grounding in Videos](../../CVPR2025/video_understanding/videogem_training-free_action_grounding_in_videos.md)
- [\[AAAI 2026\] HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection](../../AAAI2026/video_understanding/headhunt-vad_hunting_robust_anomaly-sensitive_heads_in_mllm_.md)

</div>

<!-- RELATED:END -->
