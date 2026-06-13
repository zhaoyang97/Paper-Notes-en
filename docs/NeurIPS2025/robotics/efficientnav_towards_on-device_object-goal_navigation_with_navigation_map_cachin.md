---
title: >-
  [Paper Note] EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval
description: >-
  [NeurIPS 2025][Robotics][object-goal navigation] Through discrete memory caching (group-independent KV cache computation with selective loading), attention-driven clustering (LLM shallow-layer attention guiding grouping)…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "object-goal navigation"
  - "on-device LLM"
  - "KV cache optimization"
  - "zero-shot navigation"
  - "edge deployment"
date: 2026-05-08
content_hash: fc9b9050f568d7aa
---

# EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval

**Conference**: NeurIPS 2025
**arXiv**: [2510.18546](https://arxiv.org/abs/2510.18546)  
**Code**: [GitHub](https://github.com/PKU-SEC-Lab/EfficientNav)  
**Area**: Robotics / Navigation / On-Device Deployment
**Keywords**: object-goal navigation, on-device LLM, KV cache optimization, zero-shot navigation, edge deployment

## TL;DR
Through discrete memory caching (group-independent KV cache computation with selective loading), attention-driven clustering (LLM shallow-layer attention guiding grouping), and semantics-aware retrieval (CLIP + knapsack problem adapted to varying memory budgets), EfficientNav is the first system to achieve zero-shot ObjNav on Jetson Orin using LLaMA-3.2-11b, surpassing the GPT-4 baseline by 11.1% SR while reducing real-time latency by 6.7×.

## Background & Motivation
**Background**: LLM-driven zero-shot object-goal navigation (ObjNav) has become mainstream — LLMs serve as planners, leveraging semantic and spatial information from graph-based navigation maps to select sub-goals. However, these systems heavily rely on cloud-based large models such as GPT-4.

**Cloud Deployment Issues**: Communication latency disrupts real-time performance, privacy risks are significant, and computational costs are prohibitive — motivating a shift from cloud to on-device deployment.

**On-Device Memory Bottleneck**: The Jetson Orin has only 32 GB DRAM. Model weights combined with navigation map KV caches cause memory overflow as exploration steps increase. Recomputing KV caches introduces substantial prefilling latency.

**Small Model Capability Bottleneck**: Directly replacing GPT-4 with LLaMA-11b leads to a significant drop in success rate. Experiments reveal that as navigation steps increase, the small model's attention scores on the correct sub-goal continuously decline — it cannot focus on key information within complex, redundant maps.

**Key Observations**: Navigation maps contain substantial redundancy; different targets require attention to different map subsets; semantically related objects (e.g., oven and pot) should be grouped together.

**Core Idea**: A three-tier optimization: (1) group-wise KV cache computation enabling selective loading (decoupling cache from sequence order); (2) LLM attention-guided grouping to ensure semantic coherence; (3) CLIP-driven semantic retrieval combined with a knapsack formulation to adapt to device memory budgets.

## Method

### Overall Architecture
RGB-D → Grounding DINO detection → Graph navigation map → {Discrete caching + Attention clustering + Semantic retrieval} → LLM planner selects sub-goal → Low-level controller navigation

### Key Designs

1. **Discrete Memory Caching**:

    - Function: Objects in the map are partitioned into groups, each group's KV cache is computed independently, and only selected groups are loaded during planning.
    - Mechanism: Group-independent computation decouples KV caches from group order, enabling direct reuse of cached KV caches when different group subsets are selected. Newly detected objects are appended to the end of their group (correctness guaranteed by causal attention) without invalidating existing caches. Selected groups' caches are loaded at once, avoiding frequent cache transfers during decoding.
    - Design Motivation: The full KV cache cannot fit in device memory; recomputation is too slow; frequent cache transfers incur high latency — group-wise computation combined with selective loading addresses all three issues simultaneously.

2. **Attention-based Memory Clustering**:

    - Function: Shallow-layer attention weights from the LLM itself guide object grouping.
    - Mechanism: Newly detected objects are passed through the first 1/10 of LLM layers, and their average attention with existing groups is computed. If it exceeds a threshold, the object is assigned to an existing group; otherwise, a new group is created. Semantically related objects (e.g., oven/pot) are clustered together, and intra-group cross-attention preserves semantic relationships.
    - Design Motivation: Uniform partitioning ignores semantic relationships among objects; coarse-grained grouping yields large KV caches with fewer selectable groups, risking the loss of critical information. Attention-driven adaptive grouping balances accuracy and granularity.

3. **Semantics-aware Memory Retrieval**:

    - Function: Dynamically selects relevant groups based on the current navigation goal, pruning redundant information to help the small model focus.
    - Mechanism: CLIP (only ~100M parameters) encodes each group's object information and the target; cosine similarity serves as the probability of a group containing the sub-goal. The selection is formulated as a knapsack problem: $\max \sum_i (P_i - \text{threshold}) \cdot x_i$ subject to $\sum_i M_i \cdot x_i \leq M$, adaptable to varying device memory budgets.
    - Design Motivation: Small models cannot independently focus on relevant information in complex maps — active pruning of redundancy provides structural guidance. CLIP is lightweight with negligible latency; the knapsack formulation makes the approach portable across arbitrary devices.

### System Collaboration
A small–large model collaboration paradigm: CLIP handles simple group selection (~100M parameters, low latency), while the LLM handles complex sub-goal planning — task division is matched to model capability.

## Key Experimental Results

### Main Results (HM3D Benchmark)

| Method | Zero-shot | LLM | On-device | SR↑ | SPL↑ |
|--------|-----------|-----|-----------|-----|------|
| InstructNav | ✓ | GPT-4V | ✗ | ~38 | ~18 |
| MapGPT | ✓ | GPT-4 | ✗ | ~40 | ~20 |
| **EfficientNav** | **✓** | **LLaMA-11b** | **✓** | **+11.1%** | **+Gain** |

### Latency Comparison

| Metric | GPT-4 Baseline | EfficientNav | Speedup |
|--------|---------------|-------------|---------|
| Real-time latency | High | Low | **6.7×** |
| End-to-end latency | High | Low | **4.7×** |

### Key Findings
- **Small model + information pruning > large model**: LLaMA-11b with semantic retrieval outperforms the GPT-4 baseline by 11.1% SR — demonstrating that "less is more," as carefully selected context is more effective than accumulating redundant information.
- **Three modules are complementary and each is necessary**: Removing any single module results in decreased SR or increased latency.
- **Attention clustering provides adaptive granularity**: Compared to fixed partitioning, attention-based clustering better preserves intra-group semantic coherence.
- **Knapsack formulation adapts to diverse devices**: By adjusting memory budget $M$, the approach transfers seamlessly to different hardware platforms.
- **High cache reuse rate between adjacent steps**: Map changes between consecutive planning steps are minimal, and selected groups largely overlap, yielding high cache hit rates and further reducing loading latency.

## Highlights & Insights
- **A complete on-device system solution**: Beyond model optimization, EfficientNav addresses KV cache management, memory allocation, and information retrieval, serving as a reference design for on-device LLM systems.
- **"Pruning redundancy" is more effective than "scaling capacity"**: Counterintuitively, helping a small model focus outperforms switching to a larger model — a broadly applicable insight for all long-context LLM applications.
- **Elegant knapsack formulation**: Unifying information selection and memory constraints into a classical combinatorial optimization problem is both theoretically principled and practically effective.

## Limitations & Future Work
- **CLIP retrieval quality ceiling**: CLIP's semantic understanding is weaker than an LLM's, potentially leading to incorrect group selection in complex scenes.
- **Loss of cross-group attention after partitioning**: Although clustering mitigates this, some information loss remains.
- **Only indoor scenarios validated**: Performance in outdoor large-scale navigation environments has not been tested.
- **Fixed thresholds**: Thresholds for attention clustering and semantic retrieval require manual tuning.
- **Future directions**: A small LLM could replace CLIP for group selection; lightweight inter-group attention or summary tokens could be introduced.

## Related Work & Insights
- **vs. InstructNav/MapGPT**: These methods rely on cloud-based GPT-4 inference; EfficientNav is the first to achieve zero-shot ObjNav on-device with higher SR.
- **vs. General KV cache compression**: General methods (quantization/distillation/pruning) do not exploit the semantic structure of navigation maps; EfficientNav leverages map semantics for targeted optimization.
- **Inspiration**: The three-tier optimization paradigm (group-wise caching + semantic clustering + adaptive retrieval) generalizes to all on-device LLM scenarios requiring long context and incremental updates.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined three-tier KV cache optimization design is clever; the knapsack formulation is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete comparisons on HM3D with latency analysis and ablation studies, though additional scenario validation is lacking.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear; system design is well-structured.
- Value: ⭐⭐⭐⭐⭐ The first on-device LLM ObjNav system, serving as a benchmark for on-device AI agents.

**Approach**:
- A lightweight CLIP model (~100M parameters) encodes group information and target objects, computing semantic similarity scores.
- Group selection is formulated as a **knapsack problem**: maximize the total relevance of selected groups to the target, subject to device memory budget constraints.
- $\max \sum(P_i - \text{threshold}) \cdot x_i$, s.t. $\sum M_i \cdot x_i \leq M$
- $P_i$ is the probability that group $i$ contains the sub-goal; $M_i$ is the KV cache size of group $i$; $M$ is the memory budget.
- CLIP inference latency is far lower than that of the LLM, making group selection overhead negligible.
- Minimal map changes between adjacent planning steps → high cache hit rates → further reduced loading overhead.

## Experimental Setup and Results

### Evaluation Environment
- Dataset: HM3D (Habitat-Matterport 3D)
- Platform: Habitat simulator with RGB-D navigation
- Hardware: NVIDIA A6000 GPU + Jetson AGX Orin
- Models: LLaVA-7b/13b/34b, LLaMA-3.2-11b

### Main Comparison Results

| Method | LLM | SR | SPL | Real-time Latency | End-to-end Latency |
|--------|-----|----|-----|------------------|--------------------|
| LFG (SOTA) | GPT-4 | 68.9 | 36.0 | 5.80s | 59.34s |
| EfficientNav-11b | LLaMA-11b | 74.2 | 39.5 | 0.35s | 12.70s |
| EfficientNav-34b | LLaVA-34b | **80.0** | **41.5** | 0.87s | 12.51s |

- vs. GPT-4 baseline: SR **+11.1%**, real-time latency reduced **6.7×**, end-to-end latency reduced **4.7×**
- vs. native LLaVA-34b planner: SR **+37.3%**, SPL **+20.5%**
- vs. vLLM acceleration: real-time latency still **5.1–6.5× lower** (vLLM cannot resolve the prefilling recomputation bottleneck)

### Ablation Study (LLaVA-34b)

| Configuration | SR | SPL | Real-time Latency | End-to-end Latency |
|---------------|----|-----|------------------|--------------------|
| Original planner | 42.7 | 21.0 | 5.63s | 55.32s |
| + Discrete caching | 43.1 | 21.0 | 2.42s | 36.94s |
| + Attention clustering | 63.3 | 34.2 | 2.32s | 32.58s |
| + Semantic retrieval | **80.0** | **41.5** | **0.87s** | **12.51s** |

- Attention clustering contributes the largest accuracy gain (+20.2% SR); semantic retrieval contributes the largest latency reduction.

### Latency Characteristics
- Conventional approaches exhibit linearly growing prompt length and latency as navigation steps increase.
- EfficientNav latency **stabilizes after a few steps** — semantic retrieval bounds the amount of information fed to the LLM.
- Cache hit rate rapidly approaches a high level as memory budget increases, further reducing loading overhead.

## Highlights & Deep Insights
1. **Counter-intuitive result: small model > large model**: By pruning information to help a small LLM focus on critical content, the system outperforms GPT-4 processing the full redundant context — more information is not always better.
2. **Knapsack formulation for group selection**: Elegantly transforms memory-constrained information selection into a classical optimization problem, adaptable to diverse device memory budgets.
3. **Generality of KV cache discretization**: The design is not limited to navigation; it is applicable to any on-device LLM scenario involving long context with incremental updates (e.g., dialogue history management, streaming document understanding).
4. **Systems thinking**: Rather than single-point optimization, the three modules collaborate — caching addresses latency, clustering ensures quality, and retrieval controls scale.

## Limitations & Future Work
- The coordination complexity of three modules makes implementation and hyperparameter tuning challenging (thresholds, grouping granularity, etc.).
- Validation is limited to indoor navigation (HM3D); generalization to outdoor or dynamic obstacle environments is unknown.
- Discrete caching sacrifices cross-group attention, introducing theoretical information loss (though experiments suggest the impact is manageable).
- Performance depends on Grounding DINO detection quality; detection errors propagate through all subsequent stages.

## Rating
- Novelty: ⭐⭐⭐⭐ The three-tier combination of KV cache discretization + attention clustering + semantic retrieval is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete comparison on HM3D with ablation, latency analysis, and cache hit rate analysis.
- System Completeness: ⭐⭐⭐⭐⭐ End-to-end system design spanning memory management, information retrieval, and planning improvement.
- Writing Quality: ⭐⭐⭐⭐ System design rationale is clear; challenge–solution correspondence is explicit.
- Value: ⭐⭐⭐⭐⭐ Significant practical value for on-device embodied AI deployment; the KV cache management scheme is broadly generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GoViG: Goal-Conditioned Visual Navigation Instruction Generation via Multimodal Reasoning](../../ACL2026/robotics/govig_goal-conditioned_visual_navigation_instruction_generation_via_multimodal_r.md)
- [\[NeurIPS 2025\] DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation](dynanav_dynamic_feature_and_layer_selection_for_efficient_visual_navigation.md)
- [\[NeurIPS 2025\] VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching](vla-cache_efficient_vision-language-action_manipulation_via_adaptive_token_cachi.md)
- [\[NeurIPS 2025\] RDD: Retrieval-Based Demonstration Decomposer for Planner Alignment in Long-Horizon Tasks](rdd_retrieval-based_demonstration_decomposer_for_planner_alignment_in_long-horiz.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](learning_interactive_world_model_for_object-centric_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
