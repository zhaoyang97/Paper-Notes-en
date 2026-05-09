---
title: >-
  [Paper Note] Co-Layout: LLM-driven Co-optimization for Interior Layout
description: >-
  [AAAI 2026][Optimization][Interior layout] This paper proposes Co-Layout, a framework that leverages LLMs to extract structured constraints from natural language descriptions, then jointly optimizes room layout and furniture placement via a grid-based integer programming (IP) formulation augmented with a coarse-to-fine solving strategy, substantially outperforming existing two-stage approaches.
tags:
  - AAAI 2026
  - Optimization
  - Interior layout
  - integer programming
  - large language models
  - co-optimization
  - coarse-to-fine strategy
date: 2026-05-08
content_hash: 90482c16b793187f
---

# Co-Layout: LLM-driven Co-optimization for Interior Layout

**Conference**: AAAI 2026
**arXiv**: [2511.12474](https://arxiv.org/abs/2511.12474)
**Code**: None
**Project Page**: [https://xccelephant.github.io/paper/co-layout/](https://xccelephant.github.io/paper/co-layout/)
**Area**: Optimization / Interior Design
**Keywords**: Interior layout, integer programming, large language models, co-optimization, coarse-to-fine strategy

## TL;DR
This paper proposes Co-Layout, a framework that leverages LLMs to extract structured constraints from natural language descriptions, then jointly optimizes room layout and furniture placement via a grid-based integer programming (IP) formulation augmented with a coarse-to-fine solving strategy, substantially outperforming existing two-stage approaches.

## Background & Motivation

### Problem Setting
Interior design is a highly expertise-intensive task: designers must translate clients' vague requirements into concrete room plans and furniture arrangements, involving complex trade-offs among numerous design variables and constraints. Automated design tools can not only enhance designer productivity but also inspire creative exploration.

### Limitations of Prior Work
Existing methods universally decompose interior design into **two independent stages**: first generating a room layout, then placing furniture within each room. Representative works include HouseGAN++, Holodeck, and AnyHome. In practice, however, furniture determines room functionality, while room size and shape in turn constrain furniture dimensions and positions—the two are inherently coupled. Decoupled processing introduces several failure modes:

- **Disconnected corridors**: furniture blocks passageways, making certain rooms unreachable
- **Inaccessible rooms**: occupants must pass through a bedroom to reach the living room, and similar illogical paths
- **Wasted space**: irregularly shaped or overly narrow rooms that cannot accommodate furniture reasonably

### Root Cause
The authors observe that LLMs excel at interpreting high-level design requirements, translating natural language descriptions into detailed room and furniture lists along with relational constraints. However, LLMs are poor at generating **precise coordinates**—directly prompting LLMs to output coordinates produces numerous geometric conflicts (overlaps, out-of-bound placements, etc.). The paper's core idea is therefore to **delegate semantic constraint extraction to LLMs and precise geometric solving to integer programming**, achieving a principled division of labor between semantic understanding and exact optimization.

## Method

### Overall Architecture
Co-Layout consists of two stages:

1. **LLM Pre-processing Stage**: A carefully prompted LLM workflow receives natural language user descriptions (e.g., "design a 100 m² two-bedroom apartment with an open kitchen") and outputs a structured scene graph, including a room list $R = \{r_k\}_{k=1}^{N}$, a furniture list $S = \{s_{k,l}\}$, and various constraints (adjacency, area, relative position, etc.).

2. **Integer Programming Optimization Stage**: The LLM-generated constraints are encoded into a grid-based integer programming model that jointly solves for room partitioning and furniture placement, accelerated by a coarse-to-fine strategy.

The resulting layouts can be imported into Blender and rendered as 3D scenes using the 3D-FUTURE and Imaginarium asset libraries.

### Grid-based Representation
Inspired by the classical architectural theory of the Modulor, the space is discretized into a 2D grid $\mathcal{G} = \{(i,j): 0 \leq i < W, 0 \leq j < L\}$. Each grid cell $(i,j)$ is assigned a set of binary variables:

- $x_{i,j}^k \in \{0,1\}$: whether the cell belongs to room $k$
- $p_{i,j} \in \{0,1\}$: whether the cell belongs to a corridor
- $f_{i,j}^{k,l} \in \{0,1\}$: whether the cell is occupied by the $l$-th piece of furniture in room $k$

The advantage of this representation lies in its ability to naturally model high-level topological constraints such as corridor connectivity and room accessibility—compared to rasterized image representations and vector representations—while aligning with the modular design philosophy of the Modulor.

### Key Constraint Designs

**(1) Spatial Exclusivity Constraint**: Each grid cell must be uniquely assigned to exactly one room or corridor, with no overlaps:
$$p_{i,j} + \sum_{k=1}^{N} x_{i,j}^k = 1, \quad \forall (i,j) \in \mathcal{G}'$$

**(2) Corridor Connectivity Constraint**: This is one of the paper's core technical contributions. A **flow-based formulation** is employed to ensure that corridors and open rooms (e.g., certain living room configurations) form a fully connected network. Specifically, the entrance cell serves as the source node, injecting a total flow equal to the number of all corridor/open-room cells; each corridor cell consumes exactly one unit of flow. If the corridor is disconnected, the flow conservation constraints will be violated and such solutions are automatically excluded. Flow can only propagate between adjacent corridor or open-room cells, with big-M constraints governing flow direction.

**(3) Room Accessibility Constraint**: Each room must contain at least one "access point"—a cell within the room that is adjacent to a corridor cell and unoccupied by furniture. This ensures all rooms are reachable from the corridor, preventing inaccessible "dead rooms."

**(4) Room Adjacency Constraint**: For room pairs designated by the LLM as needing to be adjacent (e.g., master bedroom and master bathroom), at least one shared boundary cell is required.

**(5) Room Bounding Box Constraint**: The big-M method is used to define the axis-aligned bounding box (AABB) for each room, providing the basis for subsequent rectangularity objective terms.

**(6) Furniture Constraints**: These include containment constraints (furniture must lie within its assigned room), mutual exclusion constraints (at most one piece of furniture per cell), and shape/size constraints (furniture area equals the product of its AABB dimensions). Furniture orientation is controlled by two binary variables $\sigma_{k,l}$ and $\mu_{k,l}$, restricting placement to four axis-aligned orientations.

### Objective Function
The total objective is a weighted sum of multiple penalty terms: $E = \sum_s \omega_s \cdot E_s(\mathbf{x}, \mathbf{p}, \mathbf{f})$, organized into three levels:

**Geometric Quality Objectives**:
- **Room Rectangularity** $E_{\text{rect}}$: penalizes the difference between each room's bounding box area and its actual occupied cell count, encouraging regular rectangular rooms
- **Room Perimeter** $E_{\text{perim}}$: penalizes exposed boundary length, encouraging compact room shapes

**Functional Objectives**:
- **Area Target** $E_{\text{area}}$: penalizes deviation from the LLM-specified target area
- **Aspect Ratio Control** $E_{\text{aspect}}$: penalizes excessive disparity between room width and length

**Furniture Placement Objectives**:
- **Spatial Relations** $E_{\text{rel}}$: relative positional constraints between furniture items (e.g., nightstand beside the bed)
- **Distribution Balance** $E_{\text{bal}}$: distance between the furniture centroid and the geometric center of the room

### Coarse-to-Fine Optimization Strategy
The computational complexity of the joint optimization grows exponentially with grid resolution. The coarse-to-fine strategy proceeds in three steps:

1. **Coarse Stage**: The grid is downsampled (e.g., $12{\times}10 \rightarrow 6{\times}5$), reducing the variable count by 75%. Only room layout is optimized at this stage—furniture is ignored—enabling fast exploration of high-level spatial organization.
2. **Coarse-to-Fine Mapping**: The coarse solution is mapped back to the fine grid, with each fine cell inheriting the room assignment of its corresponding coarse cell.
3. **Fine Stage**: Full optimization (rooms + furniture) is performed at the original resolution. The coarse solution serves as a warm-start initialization, and an additional consistency penalty term $E_{\text{ref}}$ encourages the fine solution to maintain room assignments consistent with the coarse solution.

## Key Experimental Results

### Table 1: Quantitative Evaluation (Physical Plausibility + Visual Quality + Text-Image Alignment)

| Method | OOR↓ | OOB↓ | IQA↑ | IAA↑ | CLIP↑ |
|--------|------|------|------|------|-------|
| Holodeck | 0.82 | 2.33 | 4.03 | 3.32 | 25.15 |
| AnyHome | 0.00 | 0.04 | 4.10 | 3.32 | 25.75 |
| **Co-Layout (Ours)** | **0.00** | **0.00** | **4.17** | **3.35** | **26.50** |

Co-Layout leads on all metrics, most notably eliminating object overlap (OOR=0) and out-of-bound placements (OOB=0) entirely, demonstrating the inherent geometric correctness advantage of the integer programming approach.

### Table 2: User Study (71 participants, 64 valid responses, 5-point scale)

| Method | Semantic Alignment↑ | Layout Rationality↑ | Path Clarity↑ | MRR↑ |
|--------|---------------------|---------------------|---------------|------|
| Holodeck | 3.43 | 3.12 | 3.06 | 0.59 |
| AnyHome | 3.07 | 2.59 | 2.80 | 0.45 |
| **Co-Layout (Ours)** | **3.77** | **3.23** | **3.41** | **0.80** |

The user study further validates Co-Layout's comprehensive superiority in subjective evaluation. Its MRR of 0.80 substantially exceeds Holodeck (0.59) and AnyHome (0.45), indicating that users strongly prefer the jointly optimized results.

## Highlights & Insights

1. The **LLM + IP division-of-labor paradigm** is particularly elegant: LLMs excel at interpreting ambiguous natural language requirements and converting them into structured constraints, while IP excels at solving precise geometric configurations under those constraints. The two are complementary, each offsetting the other's weaknesses.
2. **Flow-based corridor connectivity** is a key technical contribution. Prior work rarely models corridor connectivity explicitly, resulting in logically incoherent floor plans. The flow conservation constraints in this paper guarantee connectivity directly from a graph-theoretic perspective, which is more reliable than heuristic post-processing.
3. **Joint optimization vs. two-stage separation**: The experiments clearly demonstrate the value of co-optimization—problems such as bedrooms blocking living room entrances and completely inaccessible rooms, common in baseline methods, are entirely eliminated in Co-Layout.
4. The **coarse-to-fine strategy** is an important engineering contribution. Ablation studies show that the speedup becomes increasingly significant as grid resolution grows, enabling the method to run on an ordinary laptop.

## Limitations & Future Work

1. **Limited furniture scope**: The current framework supports only floor-standing furniture and does not handle wall-mounted items (e.g., sconces, paintings) or surface-placed objects (e.g., desk lamps, decorative items), limiting the completeness of generated scenes.
2. **LLM constraint conflicts**: LLMs occasionally produce mutually contradictory constraints (e.g., requiring a room to be small while also fitting a large number of furniture pieces). No automatic detection or repair mechanism currently exists, which may degrade solution quality or yield infeasible problems.
3. **Single-story buildings only**: The framework does not account for inter-floor constraints in multi-story buildings (e.g., staircase alignment, vertical utility shaft continuity), precluding direct application to complex multi-story residential or commercial building design.
4. **Small evaluation scale**: Quantitative experiments compare only 5 examples, providing insufficient statistical significance; the user study, while comprising 64 valid responses, also covers a limited range of scenarios.
5. **Door and window positions excluded from optimization**: Door and window locations are determined via post-processing rather than being incorporated into the joint optimization, potentially yielding suboptimal placements.

## Related Work & Insights

- **Holodeck** (Yang et al., 2024) and **AnyHome** (Fu et al., 2024) are important two-stage baselines. While capable of generating rich room and furniture configurations, their layouts are highly stochastic and lack deep understanding of privacy, connectivity, and functional logic.
- **LayoutGPT** (Feng et al., 2023) and **I-Design** (Çelen et al., 2024) explore the application of LLMs to furniture layout, but are limited to the furniture placement stage and do not address room planning.
- Grid-based representations can be traced back to the facility layout optimization literature (Drira et al., 2007); this paper integrates them with the architectural Modulor concept in the LLM era, representing a valuable cross-disciplinary synthesis.
- The paradigm of **"LLM-extracted semantic constraints + optimization solver for plan generation"** is broadly generalizable and can be extended to other combinatorial optimization problems requiring simultaneous consideration of high-level semantics and precise geometry, such as urban planning, factory layout, and chip floorplanning.

## Rating

- Novelty: ⭐⭐⭐⭐ — The LLM+IP joint framework and flow-based connectivity constraints represent clear innovations
- Technical Depth: ⭐⭐⭐⭐ — Constraint modeling and coarse-to-fine strategy design are rigorous
- Experimental Thoroughness: ⭐⭐⭐ — Comparative experiments cover only 5 samples, limiting statistical power
- Practical Value: ⭐⭐⭐⭐ — Targets real-world interior design scenarios and runs on a laptop
- Overall Rating: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment](cost-minimized_label-flipping_poisoning_attack_to_llm_alignment.md)
- [\[ICLR 2026\] Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning](../../ICLR2026/optimization/scaf-grpo_scaffolded_group_relative_policy_optimization_for_enhancing_llm_reason.md)
- [\[NeurIPS 2025\] MeCeFO: Enhancing LLM Training Robustness via Fault-Tolerant Optimization](../../NeurIPS2025/optimization/mecefo_enhancing_llm_training_robustness_via_fault-tolerant_optimization.md)
- [\[AAAI 2026\] Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation](bridging_synthetic_and_real_routing_problems_via_llm-guided_instance_generation_.md)
- [\[AAAI 2026\] MOTIF: Multi-strategy Optimization via Turn-based Interactive Framework](motif_multi-strategy_optimization_via_turn-based_interactive_framework.md)

<!-- RELATED:END -->
