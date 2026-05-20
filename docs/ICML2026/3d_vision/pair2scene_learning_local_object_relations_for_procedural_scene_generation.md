---
title: >-
  [Paper Note] Pair2Scene: Learning Local Object Relations for Procedural Scene Generation
description: >-
  [ICML 2026][3D Vision][3D scene generation] Pair2Scene shifts 3D indoor scene generation from "directly fitting the global joint distribution" to "learning pairwise local object relations (support + function) and recursi…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "3D scene generation"
  - "local object relations"
  - "support relations"
  - "functional relations"
  - "MoL distribution"
  - "rejection sampling"
date: 2026-05-08
content_hash: 7c04ede8c8b82399
---

# Pair2Scene: Learning Local Object Relations for Procedural Scene Generation

**Conference**: ICML 2026  
**arXiv**: [2604.11808](https://arxiv.org/abs/2604.11808)  
**Code**: None (Project Page only)  
**Area**: 3D Scene Generation / Procedural Generation  
**Keywords**: 3D scene generation, local object relations, support relations, functional relations, MoL distribution, rejection sampling  

## TL;DR
Pair2Scene shifts 3D indoor scene generation from "directly fitting the global joint distribution" to "learning pairwise local object relations (support + function) and recursively assembling them via a scene hierarchy tree." With point cloud geometric encoding, Mixture-of-Logistics probabilistic heads, and collision-aware rejection sampling, it can generate complex scenes with object counts rising from about 4 to about 14 using only 3D-Front training data. Both FID and user studies outperform baselines such as ATISS, DiffuScene, and LayoutVLM.

## Background & Motivation

**Background**: High-fidelity 3D indoor scene generation mainly follows two lines: (i) **Learning-based** (ATISS, DiffuScene, LayoutVLM, FactoredScenes) fit the joint distribution of scenes end-to-end on a single dataset; (ii) **LLM/VLM-based** (GALA3D, I-Design, HoloDeck, HSM) use language models' commonsense for overall layout reasoning.

**Limitations of Prior Work**: Learning-based methods are severely limited by training set capacity—3D-Front averages only 4.07 furniture items per scene, so the learned distribution never reaches the density of "real apartments with dozens of items." As object count increases, modeling all pairwise global dependencies grows with $O(N^2)$ complexity, making it unlearnable. LLM/VLM-based methods are semantically rich but poor at spatial reasoning, often producing physically implausible layouts such as interpenetration or floating objects.

**Key Challenge**: The "global joint distribution" assumption posits that every object's position depends on all others in the scene. However, the authors observe that **real object placement is almost always influenced only by a few nearby support/functional partners**; most global dependencies are redundant. Forcing global modeling is akin to fitting a super high-dimensional manifold with scarce data, inevitably leading to underfitting.

**Goal**: (a) Reconstruct the problem from a **local relation** perspective, allowing "relation sample count" to accumulate across datasets, no longer limited by single-scene capacity; (b) Physically ensure support stability and semantically ensure functional plausibility; (c) Enable generated scene complexity to exceed the training distribution.

**Key Insight**: Decompose scenes into relation quadruples $\mathcal{T}_i = \langle\mathcal{O}_{dep,i}, \mathcal{O}_{sup,i}, \{\mathcal{O}_{fnc,i}\}_{opt}\rangle$ (dependent object + required support anchor + optional functional anchor), learn the conditional density "given anchor geometry and position, the distribution of the dependent object's position," and assemble global scenes from local rules using a hierarchical tree and rejection sampling.

**Core Idea**: Replace global joint distribution modeling with "local relation learning + procedural hierarchical assembly."

## Method

### Overall Architecture
Pair2Scene consists of three main modules: (1) **Data Construction Pipeline**—extracts about 140k relation quadruples from heterogeneous sources (3D-Front, MesaTask, InternScenes) using physics simulation, geometric heuristics, and LLM distillation, forming the 3D-Pairs dataset; (2) **Pair2Scene Model**—uses Point-MAE to encode each object's point cloud geometry $z^{geo}$, MLP to encode anchor object OBB $B$ as spatial embedding $e^{bbox}$, fuses them with cascaded Transformer blocks (relation self-attention + geometric cross-attention), and finally uses an MLP to output Mixture-of-Logistics parameters $\Theta$ for the dependent object's 12D OBB, yielding a multimodal conditional density $P(B_{dep}\mid\Theta)$; (3) **Procedural Assembly**—automatically constructs support tree $\mathbb{T}_s$ and functional tree $\mathbb{T}_f$ from text or room type, traverses the relation sequence via BFS (support) + DFS (function), samples positions from the model distribution at each step, rejects and resamples on collision, and finally applies minor gravity simulation for adjustment.

### Key Designs

1. **Support/Functional Relations + Mixture-of-Logistics Multimodal Distribution**:

    - **Function**: Formalizes the core conditional density of scene generation as a multimodal distribution "given anchor information, predict dependent object OBB," avoiding unimodal regression's inability to express natural multi-solutions (e.g., "a chair can be placed on any side of a table").
    - **Mechanism**: Support relations $R_s$ are gravity-dominated (e.g., computer-on-table), functional relations $R_f$ are semantically driven (e.g., keyboard-mouse). The model predicts $K$ Logistic components for $B_{dep}\in\mathbb{R}^{12}$ (center + size + 6D rotation): $P(B_{dep}\mid\Theta) = \sum_{k=1}^K \pi_k\prod_{d=1}^{12} L(B_{dep,d}\mid\mu_{k,d}, s_{k,d})$. Training objective is NLL plus entropy regularization: $\mathcal{L}_{total} = \mathcal{L}_{nll} + \lambda\mathcal{L}_{ent}$, where $\mathcal{L}_{ent} = \sum_k \hat\pi_k\log\hat\pi_k$ encourages high mixture entropy and prevents mode collapse.
    - **Design Motivation**: Explicitly separating support (physical) and function (semantic) aligns with human intuition for furniture arrangement; MoL is chosen over Gaussian mixture because Logistic distributions have closed-form CDFs, efficient sampling, and have proven effective for multimodal structured distributions in PixelRNN/PixelCNN++.

2. **Geometry + Relation Dual-Attention Layout Predictor**:

    - **Function**: Enables the model to perceive both object geometry (non-planar support surfaces, irregular orientations) and relation topology (which is anchor, which is dependent).
    - **Mechanism**: Each role $m\in\{dep, sup, fnc\}$ is represented by a learnable query token $x_m$; anchor object position embedding $e_m^{bbox} = \mathrm{MLP}_{pos}(B_m)$ is added only to the self-attention key/value (the dependent object only queries its own geometry, not its bbox since it's unknown). Relational Self-Attention is $X = \mathrm{SelfAttn}(X, X+E^{bbox}, X+E^{bbox})$, allowing dep to attend to the spatial presence of sup/fnc; Geometry-Aware Cross-Attention is $x_m = \mathrm{CrossAttn}(x_m, z_m^{geo}, z_m^{geo})$, where each role token interacts only with its own point cloud features, avoiding geometric information leakage. Finally, $x_{dep}$ passes through an MLP head to output $\Theta$.
    - **Design Motivation**: Relying solely on semantic class (e.g., "table") for support surface judgment fails—many tables have non-flat tops, chairs have curved backs; using point clouds and Point-MAE pretraining allows the model to "see" true object shapes. Anchor tokens receive position embeddings while dep tokens do not, structurally ensuring "the position to be predicted is dep's, without ground-truth leakage."

3. **Hierarchical Tree Assembly + Rejection Sampling to Upgrade Local Rules to Global**:

    - **Function**: Assembles globally consistent, collision-free, physically plausible scenes without learning the global distribution.
    - **Mechanism**: Represents the scene as a support tree $\mathbb{T}_s$ (rooted at the floor) plus, for each non-leaf node, a functional tree $\mathbb{T}_f$ (semantic dependencies among objects sharing a support surface). During generation, BFS traverses $\mathbb{T}_s$ (ensuring support surfaces are placed first), then for each node, DFS traverses $\mathbb{T}_f$, yielding a relation sequence $\mathcal{S} = \{\mathcal{T}_1, \ldots, \mathcal{T}_N\}$. At each step, sample candidate positions from the local distribution $p_{\text{local}}(x)$; define the feasible set $\mathcal{F}$ as "no collision with placed objects or scene boundaries." The target global distribution is $p_{\text{global}}(x) = p_{\text{local}}(x)/Z$ if $x\in\mathcal{F}$, otherwise 0, approximated via rejection sampling. After successful sampling, a short gravity simulation is applied. Tree construction supports both "statistical synthesis" (expanding via frequency/co-occurrence statistics) and "LLM guidance" (using LLM to convert text descriptions into hierarchical trees).
    - **Design Motivation**: Rejection sampling naturally upgrades "local conditional density" to "global collision-constrained distribution" without retraining; BFS+DFS traversal enforces causal order—any dep's anchors exist before its prediction, avoiding "chicken-and-egg" issues. LLM is used only for tree structure generation (its strength in natural language), not direct coordinate prediction (its weakness), achieving a division of labor between LLM and geometric model.

### Loss & Training

The training objective is $\mathcal{L}_{total} = \mathcal{L}_{nll} + \lambda\mathcal{L}_{ent}$, with NLL fitting the MoL distribution and entropy regularization preventing mode collapse. Point-MAE is pretrained on a 3D asset library aggregated by the paper and used as the geometric encoder. Data comes from 3D-Pairs, about 140k relation quadruples extracted from 3D-Front (furniture), MesaTask (desktop), and InternScenes Real-to-Sim subset (open scenes).

## Key Experimental Results

### Main Results

Two evaluation settings: (A) **3D-Front only**—trained only on 3D-Front, compared with ATISS / DiffuScene / LayoutVLM / FactoredScenes; (B) **multi-source**—trained on all 3D-Pairs, compared with procedural/LLM-based systems (Holodeck, Infinigen-Indoors, LayoutVLM, FactoredScenes).

| Method (3D-Front only) | FID ↓ | KID×1e-3 ↓ | Avg. Object Count |
|---|---|---|---|
| ATISS | 71.24 | 42.18 | 7.65 |
| DiffuScene | 67.45 | 31.72 | 6.75 |
| LayoutVLM | 120.87 | 138.54 | -- |
| FactoredScenes | 104.12 | 129.45 | 8.53 |
| **Ours-Fit** | **65.92** | **22.14** | 6.98 |
| **Ours-Beyond** | 75.88 | 69.05 | **14.15** |

A user study with 22 participants on the 3D-Front setting shows Ours-Beyond achieves SA 5.23 / PP 5.00 / SC 5.23 / MQ 5.12 / CFS 4.46, nearly all top scores; in the multi-source setting, Ours achieves SA 4.55 / PP 4.32 / SC 4.73, CFS 4.20, far surpassing the second-best LayoutVLM at 1.72.

### Ablation Study

| Variant | FID ↓ | KID×1e-3 ↓ | Notes |
|---|---|---|---|
| w/o relation (no explicit support/functional split) | 92.34 | 82.74 | Relation decomposition necessary |
| w/o pretrain (no Point-MAE pretraining) | 81.14 | 73.91 | Geometric prior critical |
| Full model Ours-Fit | 65.92 | 22.14 | All designs enabled |

### Key Findings
- Ours-Fit achieves a KID of only 22.14, far surpassing the second-best DiffuScene at 31.72, indicating it outperforms all baselines within the dataset distribution; Ours-Beyond increases object count from 6.98 to 14.15, proving it can exceed the density limit of the training distribution.
- In the user study, LayoutVLM scores high on Scene Complexity but extremely low on Physical Plausibility (2.14), confirming the LLM/VLM line's "rich but chaotic" issue; Pair2Scene scores high on both SC and PP, a structural advantage.
- Relation decomposition (w/o relation ablation) has the largest impact, indicating that "support/functional" is the core inductive bias of this method, not just an engineering wrapper.

## Highlights & Insights
- The observation that "global joint distribution is redundant, object placement is mainly governed by local dependencies" directly challenges the mainstream modeling assumption in recent years and is experimentally shown to enable more scalable local learning.
- The three data sources (curated furniture, desktops, real-to-sim open scenes) are highly heterogeneous; the authors use "relation quadruples" as a unified interface, essentially designing a cross-dataset extensible protocol, which is methodologically significant for the scene dataset ecosystem.
- Using LLM for "generating hierarchical trees" rather than "directly generating coordinates" exemplifies the new LLM-as-controller, geometric model-as-executor paradigm.

## Limitations & Future Work
- Relation quadruples are limited to "single sup + single opt fnc," restricting expressiveness for truly complex multi-party dependencies (e.g., triangular table-chair geometric constraints).
- Rejection sampling becomes less efficient in high-density scenes and does not consider global aesthetics (symmetry, style consistency); future work could combine with global priors.
- Statistical synthesis for tree construction still relies on dataset statistics; whether it can generate room types never seen in the data (e.g., circular studies) is unclear; LLM-guided mode is affected by LLM commonsense blind spots.
- Code is not released, making reproduction difficult.

## Related Work & Insights
- **vs ATISS / DiffuScene**: These treat scenes as sequences and fit the global distribution with Transformer/Diffusion, limited by dataset size; Pair2Scene's local learning + procedural assembly enables sample accumulation across datasets.
- **vs HoloDeck / GALA3D / HSM**: LLM/VLM-based methods rely on commonsense for overall layout but lack spatial precision; Pair2Scene lets LLM generate only the hierarchy, with the geometric model handling precise layout, greatly improving physical plausibility.
- **vs Infinigen-Indoors**: Purely procedural generation depends on hand-crafted rules; Pair2Scene learns the rules, allowing rule quantity and diversity to grow with data.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The shift away from global distribution + relation quadruple protocol are both original contributions
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual settings + 22-person user study + key ablations all included
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical definitions, intuitive pipeline diagrams, smooth narrative logic
- Value: ⭐⭐⭐⭐⭐ Simultaneously addresses data scarcity and global complexity explosion, highly significant for downstream 3D scene generation applications

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Global-Local Tree Search in VLMs for 3D Indoor Scene Generation](../../CVPR2025/multimodal_vlm/global-local_tree_search_in_vlms_for_3d_indoor_scene_generation.md)
- [\[ICML 2026\] R$^3$L: Reasoning 3D Layouts from Relative Spatial Relations](r3l_reasoning_3d_layouts_from_relative_spatial_relations.md)
- [\[NeurIPS 2025\] GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity](../../NeurIPS2025/multimodal_vlm/glsim_detecting_object_hallucinations_in_lvlms_via_globalloc.md)
- [\[ICML 2026\] Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models](revis_sparse_latent_steering_to_mitigate_object_hallucination_in_large_vision-la.md)
- [\[CVPR 2026\] HOG-Layout: Hierarchical 3D Scene Generation, Optimization and Editing via Vision-Language Models](../../CVPR2026/multimodal_vlm/hog_layout_hierarchical_3d_scene_generation_optimization_and_editing.md)

</div>

<!-- RELATED:END -->
