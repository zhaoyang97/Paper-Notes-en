---
title: >-
  [Paper Note] Fine-Grained Representation for Lane Topology Reasoning
description: >-
  [AAAI 2026][Autonomous Driving][Lane topology reasoning] This paper proposes TopoFG, a framework that replaces conventional single-query lane modeling with fine-grained queries (each lane represented by multiple spatially-aware queries), combined with hierarchical prior extraction, region-focused decoding, and boundary-point-based robust topology reasoning, achieving new state-of-the-art results of 48.0% OLS (subset\_A) and 45.4% OLS (subset\_B) on OpenLane-V2.
tags:
  - AAAI 2026
  - Autonomous Driving
  - Lane topology reasoning
  - fine-grained queries
  - BEV perception
  - boundary point topology
  - denoising training
date: 2026-05-08
content_hash: 2bb0f1ce812aac38
---

# Fine-Grained Representation for Lane Topology Reasoning

**Conference**: AAAI 2026
**arXiv**: [2511.12590](https://arxiv.org/abs/2511.12590)
**Authors**: Guoqing Xu, Yiheng Li, Yang Yang (Beijing Institute of Technology)
**Code**: [GitHub](https://github.com/GXmmm18/TopoFG)
**Area**: Autonomous Driving
**Keywords**: Lane topology reasoning, fine-grained queries, BEV perception, boundary point topology, denoising training

## TL;DR

This paper proposes TopoFG, a framework that replaces conventional single-query lane modeling with fine-grained queries (each lane represented by multiple spatially-aware queries), combined with hierarchical prior extraction, region-focused decoding, and boundary-point-based robust topology reasoning, achieving new state-of-the-art results of 48.0% OLS (subset\_A) and 45.4% OLS (subset\_B) on OpenLane-V2.

## Background & Motivation

### State of the Field
Lane topology reasoning is a core task in autonomous driving perception, requiring simultaneous detection of lane centerlines and traffic elements, as well as inference of their topological relationships (lane-lane connectivity and lane-traffic sign association). Accurate topology modeling directly affects navigation and control decisions.

### Limitations of Prior Work
- **Insufficient expressiveness of single-query modeling**: Methods such as TopoNet and TopoLogic represent an entire lane with a single query vector, making it difficult to capture complex shapes and local geometric variations.
- **Unreliable instance-level topology reasoning**: Connectivity is predicted by computing the overall feature similarity between two lanes; however, two connected lanes may only meet locally at their endpoints, making overall similarity uninformative.
- **Typical failure scenario**: When the endpoint of lane $a$ is followed by two geometrically similar parallel lanes $b$ and $c$, instance-level features may render $b$ and $c$ nearly indistinguishable, causing the model to incorrectly predict a connection $a \to c$.

### Core Idea
Replace single-query representations with fine-grained query sequences per lane, enabling the model to capture local geometric details. Topology reasoning is then focused on boundary point (start/end) features rather than holistic instance features, improving connection prediction accuracy.

## Method

### Overall Architecture
TopoFG consists of three core modules: the Hierarchical Prior Extractor (HPE), the Region-Focused Decoder (RFD), and the Robust Boundary point Topology Reasoning module (RBTR).

Input multi-view images → CNN backbone (ResNet-50) + FPN for multi-scale features → Deformable attention for BEV feature generation → HPE → RFD → RBTR → Output lane lines and topological relationships.

### Module 1: Hierarchical Prior Extractor (HPE)

Extracts two complementary types of prior information:

**Global Spatial Prior**:
1. MaskFormer predicts lane masks $\boldsymbol{M}$ from BEV features.
2. A weight vector $\boldsymbol{A}$ is computed using threshold $\tau$, with high-confidence regions emphasized by scaling factor $\alpha$.
3. Sine-cosine encoding generates BEV grid positional embeddings $\boldsymbol{P}$; a weighted summation yields the spatial prior $\boldsymbol{Q}^{\text{pos}}$.

**Local Sequence Prior**:
1. Learnable queries $\boldsymbol{Q}'$ are initialized to represent local points on each lane.
2. Keypoints on each lane are assigned ordered indices $I=\{1,...,k\}$.
3. Positional encoding followed by linear projection transforms these into ordered embeddings $\boldsymbol{Q}^{\text{seq}}$ that preserve local geometric structure.

### Module 2: Region-Focused Decoder (RFD)

**Fine-Grained Query Initialization**:
Spatial and sequence priors are fused to produce fine-grained queries: $\boldsymbol{Q}_{i,t}^F = \boldsymbol{Q}_i^{pos} + \mathcal{F}(\boldsymbol{Q}_t^{seq})$, where $i$ is the lane instance index and $t$ is the keypoint index.

**Two-Stage Self-Attention**:
1. Inter-instance self-attention: captures interactions between different lane instances.
2. Intra-instance self-attention: refines the point-level structure within a single lane.

**Region-Guided Cross-Attention**:
- Reference point sampling guided by lane masks (rather than random initialization) constrains attention to lane-relevant regions.
- Deformable attention enables efficient interaction between BEV features and queries.

### Module 3: Robust Boundary Point Topology Reasoning (RBTR)

**Boundary Point Topology Reasoning**:
- From the fine-grained query sequence of each lane, only the first and last queries are retained as boundary point features: $f_i^{\text{start}} = Q_{i,1}^F$, $f_i^{\text{end}} = Q_{i,k}^F$.
- For any lane pair $(i,j)$, the endpoint feature of lane $i$ and the start-point feature of lane $j$ are concatenated and fed into a shared MLP to predict connection probability.
- Euclidean distances between boundary points are also computed to form a geometric topology matrix.
- Final topology = similarity topology + geometric topology.

**Denoising Training Strategy**:
- Hungarian matching causes inconsistent supervision matrices across epochs, undermining training stability for topology learning.
- Noisy queries are generated from each GT instance, forming $N_{gt} \times G$ denoising queries ($G=5$ groups).
- The original adjacency matrix is expanded into a block-diagonal form as a fixed supervision signal.
- During inference, only vanilla queries are used; denoising queries are discarded.

### Loss & Training
Following the design of TopoLogic, each fine-grained query is responsible for predicting the coordinate of a single keypoint on the lane.

## Key Experimental Results

### Main Results: OpenLane-V2 Comparison

| Method | Conference | Dataset | OLS↑ | DET_l↑ | DET_t↑ | TOP_ll↑ | TOP_lt↑ |
|--------|------------|---------|------|--------|--------|---------|---------|
| STSU | ICCV2021 | subset_A | 29.3 | 12.7 | 43.0 | 2.9 | 19.8 |
| TopoNet | Arxiv2023 | subset_A | 39.8 | 28.6 | 48.6 | 10.9 | 23.8 |
| TopoMLP | ICLR2024 | subset_A | 44.1 | 28.5 | 49.5 | 21.7 | 26.9 |
| TopoLogic | NeurIPS2024 | subset_A | 44.1 | 29.9 | 47.2 | 23.9 | 25.4 |
| **TopoFG** | **AAAI2026** | **subset_A** | **48.0(+3.9)** | **33.8(+3.9)** | 47.2 | **30.8(+6.9)** | **30.9(+4.0)** |
| TopoNet | Arxiv2023 | subset_B | 36.8 | 24.3 | 55.0 | 6.7 | 16.7 |
| TopoLogic | NeurIPS2024 | subset_B | 42.3 | 25.9 | 54.7 | 21.6 | 17.9 |
| **TopoFG** | **AAAI2026** | **subset_B** | **45.4(+3.1)** | **30.0(+4.1)** | 53.0 | **27.2(+5.6)** | **21.7(+3.8)** |

TopoFG surpasses the second-best method by 3.9 and 3.1 OLS points on subset\_A and subset\_B, respectively. The topology reasoning metric TOP\_ll shows the most notable improvement (+6.9/+5.6).

### Ablation Study: Contribution of Each Module

| Configuration | OLS↑ | DET_l↑ | DET_t↑ | TOP_ll↑ | TOP_lt↑ |
|---------------|------|--------|--------|---------|---------|
| Baseline (TopoLogic) | 44.1 | 29.9 | 47.2 | 23.9 | 25.4 |
| + HPE | 45.4 | 31.3 | 47.5 | 26.1 | 26.6 |
| + HPE + RFD | 45.8 | 31.8 | 47.2 | 26.8 | 27.7 |
| + HPE + RFD + RBTR (Full) | **48.0** | **33.8** | 47.2 | **30.8** | **30.9** |

All three modules contribute consistent performance gains when stacked incrementally. RBTR contributes the largest improvement (OLS +2.2), confirming that boundary-point topology reasoning and the denoising strategy are critical.

### Additional Ablation: Sub-module Analysis

- **HPE sub-modules**: Local sequence prior and global spatial prior are each individually effective; their combination achieves 45.4% OLS.
- **RFD sub-modules**: The combination of fine-grained query initialization and sampled reference points yields the best result (45.8% OLS).
- **RBTR sub-modules**: Boundary point topology reasoning (BTR) raises OLS to 46.6%; adding denoising training (DTR) further improves it to 47.3%; combining both reaches 48.0%.

## Highlights & Insights

- **New paradigm for fine-grained query modeling**: Each lane is represented by $k=11$ spatially-aware queries rather than a single holistic query, fundamentally enhancing expressiveness for complex lane structures.
- **Intuitive rationale for boundary point topology reasoning**: Lane connectivity is inherently determined by endpoints; reasoning over start/end-point features is more principled than overall similarity, validated by a 6.9% gain in TOP\_ll.
- **Denoising training addresses supervision instability**: Label inconsistency caused by Hungarian matching is a practical challenge in training; block-diagonal denoising supervision provides a stable learning signal.
- **Complementary hierarchical prior design**: Global spatial priors provide mask-level localization, while local sequence priors preserve the ordered structure of lane keypoints; their combination outperforms either alone.
- **Code is open-sourced**, and experiments follow the standard OpenLane-V2 v2.1.0 evaluation protocol, ensuring reproducibility.

## Limitations & Future Work

- **No improvement on DET_t**: Traffic element detection (DET_t) remains at 47.2% on subset\_A; fine-grained lane modeling does not benefit traffic element perception, suggesting a dedicated enhancement module may be needed.
- **Computational overhead not discussed**: Expanding each lane from 1 query to 11 significantly increases the self-attention and cross-attention cost in the decoder; inference speed and FLOPs comparisons are not reported.
- **Evaluated only on OpenLane-V2**: Generalization to other topology reasoning benchmarks or real deployment scenarios is not verified.
- **Lightweight backbone**: Only ResNet-50 is used; the impact of stronger backbones (e.g., Swin Transformer) or higher input resolution is not explored.
- **Denoising queries discarded at inference**: Denoising queries are used only during training and fully discarded at inference, leaving potential room for further utilization.
- **Single-frame inference**: Temporal information is not exploited, which may limit performance in continuous scenarios compared to methods with temporal modeling such as BEVFormer v2.

## Related Work & Insights

- **TopoNet (ICLR2023)**: Models lane and traffic element graphs with GNNs, but instance-level queries limit geometric expressiveness, achieving only 39.8% OLS.
- **TopoMLP (ICLR2024)**: Uses a lightweight MLP to predict topological relationships, achieving 44.1% OLS, but lacks fine-grained modeling of intra-lane structure.
- **TopoLogic (NeurIPS2024)**: Employs an interpretable topology reasoning strategy based on spatial positional relationships between lanes, achieving 44.1% OLS; used as the baseline in this work, which outperforms it on all key metrics.
- **LaneSegNet**: Models lanes as semantically rich lane segments with a Lane Attention mechanism, but topology reasoning remains at the segment-level.
- **Topo2Seq**: Achieves 33.6% mAP on lane segment detection, while TopoFG surpasses it with 34.4% and superior topology metrics.
- **Mask2Map**: Generates rasterized maps before vectorization; TopoFG draws on its multi-scale BEV feature design but adopts an end-to-end framework.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of fine-grained queries, boundary point topology, and denoising training is novel, though each individual component draws inspiration from prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Main experiments, multiple ablation studies, and qualitative visualizations are all provided, but efficiency analysis and cross-dataset validation are absent.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, intuitive figures, and well-articulated motivation.
- Value: ⭐⭐⭐⭐ — Substantially advances the state of the art on the important lane topology reasoning task with broadly applicable methodology.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions](../../ICCV2025/autonomous_driving/seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)
- [\[CVPR 2026\] Plant Taxonomy Meets Plant Counting: A Fine-Grained, Taxonomic Dataset for Counting Hundreds of Plant Species](../../CVPR2026/autonomous_driving/plant_taxonomy_meets_plant_counting_a_fine-grained_taxonomic_dataset_for_countin.md)
- [\[AAAI 2026\] TawPipe: Topology-Aware Weight Pipeline Parallelism for Accelerating Long-Context Large Models Training](tawpipe_topology-aware_weight_pipeline_parallelism_for_accelerating_long-context.md)
- [\[AAAI 2026\] Dual-branch Spatial-Temporal Self-supervised Representation for Enhanced Road Network Learning](dual-branch_spatial-temporal_self-supervised_representation_for_enhanced_road_ne.md)
- [\[AAAI 2026\] WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving](worldrft_latent_world_model_planning_with_reinforcement_fine-tuning_for_autonomo.md)

<!-- RELATED:END -->
