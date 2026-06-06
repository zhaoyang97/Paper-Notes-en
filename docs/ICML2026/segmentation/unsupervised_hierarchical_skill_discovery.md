---
title: >-
  [Paper Note] Unsupervised Hierarchical Skill Discovery
description: >-
  [ICML 2026][Segmentation][Skill Discovery] HiSD starts from unlabeled observation trajectories—performing skill segmentation via optimal transport and discovering multi-level skill hierarchies using Sequitur grammar indu…
tags:
  - "ICML 2026"
  - "Segmentation"
  - "Skill Discovery"
  - "Hierarchical Structure"
  - "Unsupervised Learning"
  - "Grammar Induction"
  - "Minecraft"
date: 2026-05-08
content_hash: a7d178a56e7ebf1e
---

# Unsupervised Hierarchical Skill Discovery

**Conference**: ICML 2026  
**arXiv**: [2601.23156](https://arxiv.org/abs/2601.23156)  
**Code**: https://github.com/dmhHarvey/hisd  
**Area**: Reinforcement Learning  
**Keywords**: Skill Discovery, Hierarchical Structure, Unsupervised Learning, Grammar Induction, Minecraft

## TL;DR
HiSD starts from unlabeled observation trajectories—performing skill segmentation via optimal transport and discovering multi-level skill hierarchies using Sequitur grammar induction, requiring no action labels or reward signals.

## Background & Motivation

**Background**: Human planning is inherently hierarchical, involving reasoning over goals and subtasks. The reinforcement learning community has also found that hierarchical decomposition significantly enhances learning efficiency and policy reuse in high-dimensional environments like Minecraft. However, manually defined hierarchies (such as HTN) require substantial human effort and domain knowledge.

**Limitations of Prior Work**: Existing skill discovery methods heavily rely on action labels, reward signals, or online interactions. Methods like CompILE and OMPN require known skill sequences and complete state-action trajectories rather than learning solely from observational data. These methods typically produce flat segmentations rather than deep compositional hierarchies.

**Key Challenge**: Skill discovery and hierarchical structure learning are often coupled. If observation features are the sole input, the discovery process cannot benefit from action/reward information guidance.

**Goal**: Design a completely unsupervised framework—extracting reusable multi-level skill hierarchies purely from observational data, capable of handling high-dimensional real-world environments.

**Key Insight**: Decouple skill discovery into two stages—(1) Skill segmentation: using optimal transport to find visually consistent behavioral units; (2) Hierarchy induction: using Sequitur grammar compression to discover reusable subroutines.

**Core Idea**: Through a two-stage pipeline of optimal transport + grammar induction, a skill system that is both segmented into semantic units and organized into hierarchical structures is discovered unsupervised from pure observation trajectories.

## Method

### Overall Architecture
The HiSD pipeline consists of four steps—(1) inputting raw observation trajectories and feature vectors; (2) performing frame-level skill segmentation using ASOT optimal transport to obtain discrete skill label sequences; (3) constructing a global corpus (concatenating multiple trajectories) and using Sequitur grammar induction to discover reusable mid-level subroutines; (4) generating the final hierarchical parse tree.

### Key Designs

1.  **Adaptive Soft Optimal Transport (ASOT) Segmentation**:
    - **Function**: Discretizes continuous observation sequences into semantically consistent skill units.
    - **Mechanism**: A cost matrix $C_{tk}$ is constructed to measure the visual discrepancy between observation $X_t$ and the $k$-th skill prototype. The optimal assignment $\Gamma^*$ is solved by minimizing the weighted objective $\langle C,\Gamma\rangle + \alpha\mathcal{R}_{\text{temp}}(\Gamma) + \lambda D_{\text{KL}}(\Gamma^\top\mathbf{1}_n \| q)$. $\mathcal{R}_{\text{temp}}$ uses Gromov-Wasserstein distance to constrain temporal consistency between adjacent frames. Finally, the assignment is hardened to obtain discrete labels $z_t = \arg\max_k \Gamma^*_{tk}$.
    - **Design Motivation**: The temporal regularization term $\mathcal{R}_{\text{temp}}$ directly controls the minimum segment length (via the radius parameter $r$), penalizing different skill assignments for adjacent frames within $nr$ steps, which automatically generates coherent skill segments.

2.  **Global Corpus Construction + Sequitur Induction**:
    - **Function**: Automatically discovers mid-level subroutines and their hierarchical relationships that can be reused across different trajectories from multiple skill sequences.
    - **Mechanism**: Adjacent frames with identical labels are collapsed into a single symbol. All trajectories are separated by boundary tokens $\phi$ and concatenated into a global corpus $\mathcal{S}_{\text{corp}} = S^{(1)} \oplus \phi \oplus S^{(2)} \oplus \cdots$. The Sequitur algorithm is run, maintaining two invariants—digram uniqueness and rule utility. Boundary tokens $\phi$ are explicitly forbidden from being included in any generated rules to prevent cross-trajectory stitching.
    - **Design Motivation**: Sequitur naturally supports recursive structures and arbitrary depths, being a linear-time algorithm. it can simultaneously learn multi-level and cross-trajectory reusable subroutines without pre-specifying hierarchy depth or segment numbers.

3.  **Decoupled Feature Pipeline**:
    - **Function**: Enables HiSD to scale to arbitrary observation types (pixels, video, etc.) without modifying the core algorithm.
    - **Mechanism**: Raw pixels are mapped to a fixed-dimensional feature space using pre-trained feature extractors (e.g., PCA, CLIP) prior to ASOT.
    - **Design Motivation**: Decoupling learning from feature representation allows the method to handle both fully observable environments (Craftax+PCA) and partially observable real environments (Minecraft+MineCLIP).

## Key Experimental Results

### Main Results

Segmentation performance on Wood-Stone Random and Stone Pickaxe Static tasks in Craftax:

| Task | Method | mIoU Full | F1 Full | Description |
|------|------|-----------|---------|------|
| WS Random | HiSD | 58% (±16) | 74% (±11) | No actions/rewards |
| WS Random | CompILE | 74% (±4) | 94% (±3) | Actions + segments known |
| WS Random | OMPN | 72% (±11) | 91% (±9) | Actions + depth known |
| Stone Pickaxe Static | HiSD | 65% (±17) | 82% (±13) | Unsupervised |
| Stone Pickaxe Static | CompILE | 40% (±18) | 67% (±20) | Supervised but unstable |
| Stone Pickaxe Static | OMPN | 26% (±4) | 56% (±4) | Supervised but failed |

### Ablation Study

Hierarchy quality metrics on the Minecraft 44-skill dataset:

| Configuration | Unique Trees | Avg Depth | Avg Size | Mean Branching |
|------|--------------|-----------|----------|-----------------|
| HiSD (Full) | 12 | 3.2 | 47 | 2.8 |
| OMPN (Supervised) | 156 | 2.1 | 51 | 3.9 |
| CompILE (flat) | 500 | 1.0 | 44 | 5.2 |
| Ground Truth Grammar | 1 | 3.5 | 48 | 2.7 |

### Key Findings
- **Stone Pickaxe Static**: CompILE/OMPN F1 dropped from 94% to 56-67%, while HiSD remained stable (82%) as it does not rely on actions or order.
- **Hierarchy Reusability**: HiSD's 12 trees are close to the ground truth of 1, whereas OMPN produced 156 trees and CompILE produced 500.
- **Depth Matching**: HiSD's average depth of 3.2 (ground truth 3.5) significantly outperforms OMPN's 2.1 and CompILE's 1.0.
- **Downstream RL Acceleration**: HiSD's hierarchy accelerates training by 1.5-2.3 times compared to flat policies.

## Highlights & Insights
- **True Unsupervised Paradigm**: Uses only observation features, completely removing dependence on action labels, rewards, or online interactions.
- **Two-Stage Decoupled Design**: Separates segmentation from hierarchy induction, simplifying the problem and making the segmentation algorithm pluggable.
- **Elegance of Grammar Induction**: Sequitur's two invariants naturally induce sparse, reusable hierarchies.
- **Cross-Trajectory Generalization**: Boundary token constraints ensure discovered subroutines represent true cross-trajectory patterns.

## Limitations & Future Work
- **Feature Dependence**: Method performance relies heavily on feature quality (e.g., PCA/MineCLIP).
- **K Sensitivity**: Underestimating the number of skills $K$ leads to the merging of distinct skills.
- **Long Sequence Scalability**: The computational and memory cost of Sequitur on extremely long sequences (>10k frames) is not fully discussed.
- **Improvements**: Integration with clustering or deep generative models to automatically estimate $K$; online/streaming versions of Sequitur; explicit constraints on grammar sparsity and hierarchy quality.

## Related Work & Insights
- **vs CompILE**: CompILE requires action supervision and known segment counts; HiSD is entirely unsupervised.
- **vs OMPN**: OMPN requires actions and known hierarchy depth; HiSD only requires observations and the number of skills $K$.
- **vs Pure Clustering (DEC, VQ-VAE)**: Lacking temporal constraints often results in frequent switching; HiSD enforces temporal smoothness via Gromov-Wasserstein.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to deeply integrate fully unsupervised skill discovery with grammar induction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Dual environments (Craftax + Minecraft), multiple metrics, and downstream RL validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logic and intuitive flowcharts.
- **Value**: ⭐⭐⭐⭐ Significant insights for learning from unlabeled demonstrations, hierarchical RL, and video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Open-World Skill Discovery from Unsegmented Demonstration Videos](../../ICCV2025/segmentation/open-world_skill_discovery_from_unsegmented_demonstration_videos.md)
- [\[ICCV 2025\] Ensemble Foreground Management for Unsupervised Object Discovery](../../ICCV2025/segmentation/ensemble_foreground_management_for_unsupervised_object_discovery.md)
- [\[ICML 2026\] Geometry-Preserving Unsupervised Alignment for Heterogeneous Foundation Models](geometry-preserving_unsupervised_alignment_for_heterogeneous_foundation_models.md)
- [\[CVPR 2026\] GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation](../../CVPR2026/segmentation/geoguide_hierarchical_geometric_guidance_for_open-vocabulary_3d_semantic_segment.md)
- [\[AAAI 2026\] Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation](../../AAAI2026/segmentation/bridging_granularity_gaps_hierarchical_semantic_learning_for_cross-domain_few-sh.md)

</div>

<!-- RELATED:END -->
