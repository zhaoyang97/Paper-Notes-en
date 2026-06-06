---
title: >-
  [Paper Note] Pair2Scene: Learning Local Object Relations for Procedural Scene Generation
description: >-
  [ICML 2026][Multimodal VLM][3D scene generation] Pair2Scene transforms 3D indoor scene generation from "directly fitting a global joint distribution" into "learning one-to-one local object relations (support + functional…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
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
Pair2Scene transforms 3D indoor scene generation from "directly fitting a global joint distribution" into "learning one-to-one local object relations (support + functional) and recursively assembling them via a scene hierarchy tree." Combined with point cloud geometric encoding, Mixture-of-Logistics probability heads, and collision-aware rejection sampling, it enables the complexity of generated scenes to leap from ~4 to ~14 objects when trained only on 3D-Front, outperforming baselines like ATISS, DiffuScene, and LayoutVLM in FID and user studies.

## Background & Motivation

**Background**: High-fidelity 3D indoor scene generation followed two main paths: (i) **Learning-based** (ATISS, DiffuScene, LayoutVLM, FactoredScenes) end-to-end fitting of the joint distribution of scenes on a single dataset; (ii) **LLM/VLM-based** (GALA3D, I-Design, HoloDeck, HSM) using commonsense knowledge from language models for global layout reasoning.

**Limitations of Prior Work**: Learning-based methods are severely restricted by training set capacity—3D-Front averages only 4.07 furniture items per scene, so learned distributions never reach the density of "dozens of items in real apartments." When the number of objects increases, modeling global pairwise dependencies triggers an $O(N^2)$ complexity surge, making learning infeasible. LLM/VLM methods are semantically rich but lack spatial reasoning, often resulting in physical implausibility like collisions and floating objects.

**Key Challenge**: The "global joint distribution" assumes the position of every object depends on all other objects in the scene. However, the authors observe that **the placement of real objects is almost exclusively influenced by a few neighboring support or functional partners**; global dependencies are largely redundant. Forcing the model to fit an extremely high-dimensional manifold under data scarcity inevitably leads to underfitting.

**Goal**: (a) Reconstruct the problem through a **local relations** perspective so that "relation samples" can be accumulated across multiple datasets without being limited by single-scene capacity; (b) physically guarantee the stability of support relations and semantically ensure the rationality of functional relations; (c) allow the generated complexity to exceed the training distribution.

**Key Insight**: Decompose scenes into relation quadruplets $\mathcal{T}_i = \langle\mathcal{O}_{dep,i}, \mathcal{O}_{sup,i}, \{\mathcal{O}_{fnc,i}\}_{opt}\rangle$ (dependent object + required support anchor + optional functional anchor). Learn the conditional density of "the dependent object's position distribution given the anchor's geometry and position," then assemble local rules into global scenes using hierarchy trees and rejection sampling.

**Core Idea**: Replace global joint distribution modeling with "local relation learning + procedural hierarchical assembly."

## Method

### Overall Architecture
Pair2Scene works through three collaborative modules: (1) **Data pipeline**—extracts ~140k relation quadruplets from heterogeneous sources (3D-Front, MesaTask, InternScenes) via physical simulation, geometric heuristics, and LLM distillation to form the 3D-Pairs dataset; (2) **Pair2Scene model**—uses Point-MAE to encode geometric features $z^{geo}$ of object point clouds and an MLP to encode spatial embeddings $e^{bbox}$ of anchor object OBBs $B$. These are fused via cascaded Transformer blocks (relational self-attention + geometric cross-attention). Finally, an MLP outputs Mixture-of-Logistics (MoL) distribution parameters $\Theta$ to provide a multi-modal conditional density $P(B_{dep}\mid\Theta)$ for the 12-dimensional OBB of the dependent object; (3) **Procedural assembly**—automatically constructs support trees $\mathbb{T}_s$ and functional trees $\mathbb{T}_f$ based on text or room types. It follows a BFS(support) + DFS(functional) hybrid traversal to obtain relation sequences, samples positions from the model distribution at each step, performs rejection sampling for collisions, and applies minor gravity simulation for alignment.

### Key Designs

1. **Support/Functional Relations + Mixture-of-Logistics Multi-modal Distribution**:

    - **Function**: Formalizes the core conditional density of scene generation as a multi-modal distribution predicting "dependent object OBB given anchor information," avoiding the inability of unimodal regression to express natural multi-solution scenarios (e.g., "a chair can be placed on any side of a table").
    - **Mechanism**: Support relations $R_s$ are gravity-dominated (computer on table), while functional relations $R_f$ are semantic-neighbor-dominated (keyboard and mouse). The model predicts $K$ Logistic components for $B_{dep}\in\mathbb{R}^{12}$ (center + size + 6D rotation): $P(B_{dep}\mid\Theta) = \sum_{k=1}^K \pi_k\prod_{d=1}^{12} L(B_{dep,d}\mid\mu_{k,d}, s_{k,d})$. The training target is NLL plus entropy regularization: $\mathcal{L}_{total} = \mathcal{L}_{nll} + \lambda\mathcal{L}_{ent}$, where $\mathcal{L}_{ent} = \sum_k \hat\pi_k\log\hat\pi_k$ encourages high entropy in mixture coefficients to prevent mode collapse.
    - **Design Motivation**: Explicitly separating support (physical) and functional (semantic) relations aligns with human intuition of "furniture arrangement." MoL is chosen over Gaussian mixtures because the Logistic distribution has a closed-form CDF, high sampling efficiency, and has been proven in PixelRNN/PixelCNN++ to effectively represent multi-modal structured distributions.

2. **Geometry + Relation Dual-Attention Layout Predictor**:

    - **Function**: Enables the model to simultaneously perceive object geometry (non-planar support surfaces, irregular orientations) and relational topology (which is the anchor, which is the dependent).
    - **Mechanism**: Each role $m\in\{dep, sup, fnc\}$ is represented by a learnable query token $x_m$. Anchor position embeddings $e_m^{bbox} = \mathrm{MLP}_{pos}(B_m)$ are only added to self-attention keys/values (the dependent object searches for its own geometry but not its bbox, as it is unknown). Relational Self-Attention is defined as $X = \mathrm{SelfAttn}(X, X+E^{bbox}, X+E^{bbox})$, allowing $dep$ to attend to the spatial presence of $sup/fnc$. Geometry-Aware Cross-Attention is $x_m = \mathrm{CrossAttn}(x_m, z_m^{geo}, z_m^{geo})$; each role token only interacts with its own point cloud features to prevent geometric information crosstalk. Finally, $x_{dep}$ passes through an MLP head to output $\Theta$.
    - **Design Motivation**: Support surface judgment based solely on semantic categories (e.g., "table") fails because many tables have non-flat tops or chairs have curved backs. Using point clouds with Point-MAE pre-training allows the model to "see" the actual shape. Adding position embeddings to anchor tokens but not the $dep$ token structurally ensures that the model predicts the $dep$ position without leaking ground-truth.

3. **Hierarchical Tree Assembly + Rejection Sampling for Global Scaling**:

    - **Function**: Assembles globally consistent, collision-free, and physically reasonable scenes without learning a global distribution.
    - **Mechanism**: Represents the scene as a support tree $\mathbb{T}_s$ (root is the floor) with a functional tree $\mathbb{T}_f$ attached to each non-leaf node (semantic dependencies between objects sharing a support surface). Generation follows a BFS through $\mathbb{T}_s$ (ensuring support surfaces are placed first) and then a DFS through $\mathbb{T}_f$ for each node, yielding a relation sequence $\mathcal{S} = \{\mathcal{T}_1, \ldots, \mathcal{T}_N\}$. At each step, a candidate position is sampled from the local distribution $p_{\text{local}}(x)$. The feasible set $\mathcal{F}$ is defined as "no collision with existing objects or scene boundaries." The target global distribution $p_{\text{global}}(x)$ is $p_{\text{local}}(x)/Z$ when $x\in\mathcal{F}$ and 0 otherwise, approximated via rejection sampling. A brief gravity simulation follows successful sampling. Tree construction supports both "statistical synthesis" (procedural expansion based on frequency/co-occurrence) and "LLM-guided" (converting text descriptions to hierarchy trees) modes.
    - **Design Motivation**: Rejection sampling naturally upgrades "local conditional density" to a "global collision-constrained distribution" without retraining. BFS+DFS traversal enforces a causal sequence—any $dep$ has its anchors already in place during prediction, avoiding "chicken-and-egg" problems. LLMs are used only for generating tree structures (a natural language strength) rather than directly predicting coordinates (a weakness), achieving a functional division between LLM and geometric models.

### Loss & Training
The training target is $\mathcal{L}_{total} = \mathcal{L}_{nll} + \lambda\mathcal{L}_{ent}$, where NLL fits the MoL distribution and entropy regularization prevents mode collapse. Point-MAE is pre-trained on aggregated 3D asset libraries as the geometric encoder. Data comes from 3D-Pairs with ~140k relation quadruplets extracted from 3D-Front (furniture), MesaTask (tabletop), and InternScenes Real-to-Sim subsets (open scenes).

## Key Experimental Results

### Main Results
Two evaluation settings: (A) **3D-Front only**—trained only on 3D-Front, compared against ATISS / DiffuScene / LayoutVLM / FactoredScenes; (B) **multi-source**—trained on the full 3D-Pairs, compared with procedural / LLM-based systems (Holodeck, Infinigen-Indoors, LayoutVLM, FactoredScenes).

| Method (3D-Front only) | FID ↓ | KID×1e-3 ↓ | Avg. Objects |
|---|---|---|---|
| ATISS | 71.24 | 42.18 | 7.65 |
| DiffuScene | 67.45 | 31.72 | 6.75 |
| LayoutVLM | 120.87 | 138.54 | -- |
| FactoredScenes | 104.12 | 129.45 | 8.53 |
| **Ours-Fit** | **65.92** | **22.14** | 6.98 |
| **Ours-Beyond** | 75.88 | 69.05 | **14.15** |

In a 22-person user study on the 3D-Front setting, Ours-Beyond scored SA 5.23 / PP 5.00 / SC 5.23 / MQ 5.12 / CFS 4.46, leading in almost all categories. In the multi-source setting, Ours achieved SA 4.55 / PP 4.32 / SC 4.73, with its CFS of 4.20 far exceeding the second-place LayoutVLM (1.72).

### Ablation Study

| Variant | FID ↓ | KID×1e-3 ↓ | Description |
|---|---|---|---|
| w/o relation | 92.34 | 82.74 | Relational decomposition is necessary |
| w/o pretrain | 81.14 | 73.91 | Geometric priors are critical |
| Full Model (**Ours-Fit**) | 65.92 | 22.14 | Complete design |

### Key Findings
- The KID of Ours-Fit is only 22.14, significantly better than DiffuScene's 31.72, showing it surpasses all baselines within the dataset distribution. Ours-Beyond pushes the object count from 6.98 to 14.15, proving it can escape the density cap of the training distribution.
- In user studies, LayoutVLM scored high on Scene Complexity but extremely poor on Physical Plausibility (2.14), confirming the "rich but chaotic" pain point of LLM/VLM methods. Pair2Scene scores high on both SC and PP, representing a structural advantage.
- Relational decomposition (w/o relation ablation) had the largest impact, meaning "support/functional" is the core inductive bias of this method rather than just engineering wrapping.

## Highlights & Insights
- The observation that "global joint distribution is redundant and object placement is primarily driven by local dependencies" directly challenges the mainstream modeling assumptions of recent years and proves that it can be converted into more scalable local learning.
- The three data sources (curated furniture, tabletop, real-to-sim open scenes) are highly heterogeneous. The authors used the "relation quadruplet" as a unified interface, essentially designing a scalable protocol across datasets, which has methodological significance for the scene dataset ecosystem.
- The division of labor where LLM is used for "generating hierarchy trees" rather than "directly generating coordinates" is an elegant example of the LLM-as-controller and geometric-model-as-executor paradigm.

## Limitations & Future Work
- The relation quadruplet is limited to "single sup + single opt fnc," which restricts expressiveness for complex multi-party dependencies (e.g., triangular table-chair geometric constraints).
- Rejection sampling efficiency decreases in high-density scenes, and global aesthetics (symmetry, style consistency) are not considered; these could be integrated with global priors in the future.
- Tree construction via statistical synthesis still relies on dataset statistics; it remains unclear if it can generate room types never seen in the dataset (e.g., circular studies). LLM-guided mode is affected by LLM commonsense blind spots.
- The code has not been released, making the reproduction threshold high.

## Related Work & Insights
- **vs ATISS / DiffuScene**: They treat scenes as sequences and use Transformer/Diffusion to fit global distributions, limited by dataset scale. Pair2Scene uses local learning + procedural assembly to accumulate samples across datasets.
- **vs HoloDeck / GALA3D / HSM**: LLM/VLM methods rely on commonsense for layout but lack spatial precision. Pair2Scene lets the LLM create hierarchies while the geometric model handles precise layout, significantly improving physical feasibility.
- **vs Infinigen-Indoors**: Purely procedural generation relies on manual rules. Pair2Scene learns these rules, allowing the number and diversity of rules to grow with data.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The perspective shift of "rejecting global distribution" + relation quadruplet protocol are original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual settings + 22-person user study + critical ablations are complete.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical definitions, intuitive pipeline diagrams, and smooth narrative logic.
- Value: ⭐⭐⭐⭐⭐ Simultaneously addresses data scarcity and global complexity explosion, with significant implications for downstream applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] R$^3$L: Reasoning 3D Layouts from Relative Spatial Relations](r3l_reasoning_3d_layouts_from_relative_spatial_relations.md)
- [\[CVPR 2026\] HOG-Layout: Hierarchical 3D Scene Generation, Optimization and Editing via Vision-Language Models](../../CVPR2026/multimodal_vlm/hog_layout_hierarchical_3d_scene_generation_optimization_and_editing.md)
- [\[NeurIPS 2025\] GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity](../../NeurIPS2025/multimodal_vlm/glsim_detecting_object_hallucinations_in_lvlms_via_globalloc.md)
- [\[ICLR 2026\] Procedural Mistake Detection via Action Effect Modeling](../../ICLR2026/multimodal_vlm/procedural_mistake_detection_via_action_effect_modeling.md)
- [\[ICCV 2025\] Global and Local Entailment Learning for Natural World Imagery](../../ICCV2025/multimodal_vlm/global_and_local_entailment_learning_for_natural_world_imagery.md)

</div>

<!-- RELATED:END -->
