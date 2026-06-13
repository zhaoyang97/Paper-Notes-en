---
title: >-
  [Paper Note] How LLMs Learn to Reason: A Complex Network Perspective
description: >-
  [ICLR 2026][Reinforcement Learning][RLVR] This paper proposes a "sparse concept network" theory from a complex network perspective to provide a unified explanation of four puzzling phenomena in RLVR training (V-shaped re…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "RLVR"
  - "concept network"
  - "sparse graph"
  - "catastrophic forgetting"
  - "annealing algorithm"
date: 2026-05-08
content_hash: 9809b0a189b81cf8
---

# How LLMs Learn to Reason: A Complex Network Perspective

**Conference**: ICLR 2026
**arXiv**: [2509.23629](https://arxiv.org/abs/2509.23629)  
**Code**: [https://anonymous.4open.science/r/CoNet-83A4](https://anonymous.4open.science/r/CoNet-83A4)  
**Area**: LLM Reasoning / Reinforcement Learning
**Keywords**: RLVR, concept network, sparse graph, catastrophic forgetting, annealing algorithm

## TL;DR
This paper proposes a "sparse concept network" theory from a complex network perspective to provide a unified explanation of four puzzling phenomena in RLVR training (V-shaped response length, two-stage learning curve, catastrophic forgetting, and policy collapse). It reveals that all four phenomena originate from the topological self-organization of sparse reasoning graphs with average degree approximately 2, and derives the Annealed-RLVR algorithm, which surpasses standard RLVR on mathematical reasoning benchmarks.

## Background & Motivation

**Background**: RLVR (Reinforcement Learning with Verifiable Rewards) is used to train the reasoning capabilities of LLMs, with representative work such as DeepSeek-R1. However, RLVR training exhibits four puzzling phenomena: (1) a two-stage learning curve featuring rapid improvement followed by a prolonged plateau; (2) a V-shaped trajectory in which the length of correct answers first decreases then increases; (3) catastrophic forgetting after SFT; and (4) policy diversity collapse.

**Limitations of Prior Work**: Existing explanations are isolated from one another—the plateau is attributed to entropy exhaustion, the V-shape to the emergence of self-reflection after redundant reasoning paths are pruned, and catastrophic forgetting is treated as an objective mismatch problem. No unified framework connects all four phenomena to a common underlying mechanism.

**Key Challenge**: Constructing microscopic reasoning graphs directly from the high-dimensional latent space of LLMs is extremely difficult, impeding direct investigation of the structural origins of RLVR dynamics.

**Goal**: To provide a unified physical framework that traces all four RLVR phenomena back to a common topological self-organization process.

**Key Insight**: Inspired by the renormalization group, the analysis operates at the semantic level on a coarse-grained "concept network"—a sparse network with average degree approximately 2—rather than analyzing the full reasoning graph at the token level. A simplified Concept Network Model (CoNet) serves as a computational microscope for validation.

**Core Idea**: The paper proposes and validates the central hypothesis that the concept network formed after RLVR training is a sparse network with average degree $\langle k \rangle \approx 2$. This predominantly tree-like topology is efficient yet fragile, and unifies the explanation of the V-shaped curve (paths necessarily lengthen when transitioning from local skill-island optimization to global network integration), catastrophic forgetting (cutting critical "backbone" edges renders subtrees unreachable), and policy collapse (sharp leaf-node learning transitions accumulate to freeze global exploration).

## Method

### Overall Architecture
(1) Reproduce the four RLVR phenomena on DeepSeek-R1-Distill-Qwen-1.5B; (2) replicate these phenomena with the minimal CoNet model, which abstracts reasoning as a graph traversal problem; (3) leverage CoNet's transparency to analyze microscopic topological mechanisms; (4) design the Annealed-RLVR intervention algorithm based on these findings.

### Key Designs

1. **Sparse Concept Network Hypothesis ($\langle k \rangle \approx 2$)**:

    - **Function**: Provides a unified explanation of the four macroscopic RLVR phenomena.
    - **Mechanism**: The descending segment of the V-shaped curve corresponds to parallel local optimization of independent "skill islands" (pruning redundant paths), while the ascending segment corresponds to the merging of skill islands into a global concept network, during which the sparse structure forces paths to lengthen (average geodesic distance increases with network growth). Catastrophic forgetting: $\langle k \rangle = 2$ implies backbone edges are the sole connections; SFT overwriting the weights at these critical branch points severs entire subtrees. Policy collapse: sharp leaf-node learning transitions (from exploration to exploitation) accumulate to cause global diversity loss.
    - **Design Motivation**: In complex systems, emergent behavior is typically governed by large-scale organization rather than microscopic details; sparse graphs provide the most parsimonious explanatory framework.

2. **CoNet Computational Microscope**:

    - **Function**: Provides a tractable minimal model for theoretical validation.
    - **Mechanism**: CoNet maps the LLM's "semantic states" to abstract nodes in a fixed random graph, and "logical transitions" to learnable probabilistic edges. The learning process reduces to a graph traversal problem, yet remarkably reproduces macroscopic behaviors of LLMs such as the V-shaped curve and the two-stage learning dynamics.
    - **Design Motivation**: Direct analysis of LLMs is infeasible; CoNet serves as a renormalized proxy that makes microscopic analysis tractable.

3. **Annealed-RLVR Algorithm**:

    - **Function**: Overcomes topological bottlenecks in RLVR to improve reasoning performance.
    - **Mechanism**: At the "maximum frustration state" (when skill-island competition is most intense, corresponding to the bottom of the V-shaped curve), a brief SFT "heating" phase is precisely inserted—applying SFT only to problems with very low accuracy (<0.1) that nonetheless have correct solutions—followed by resumption of RLVR "cooling." Analogous to simulated annealing: heating breaks local optima, and cooling guides the system toward a superior global configuration.
    - **Design Motivation**: The maximum frustration state coincides with the peak of exploration diversity (Figure 6b), making it the optimal moment for SFT intervention—skill islands have not yet consolidated into a global network and are therefore most robust to perturbation.

### Loss & Training
GRPO (Group Relative Policy Optimization) is used as the RLVR algorithm. Annealed-RLVR triggers the SFT heating phase (tens of steps) upon detecting the knee point of the reward curve and the bottom of the V-shaped curve, after which standard RLVR resumes.

## Key Experimental Results

### Main Results

| Method | Training Set (512 problems) | Minerva (OOD) | AIME 2024/2025 (OOD) |
|---|---|---|---|
| Standard RLVR | Baseline | Baseline | Baseline |
| **Annealed-RLVR** | **Superior** | **Superior** | **Superior** |

### Ablation Study

| Configuration | Performance | Notes |
|---|---|---|
| RLVR without intervention | Baseline; late-stage policy collapse | Inherent limitation of the standard approach |
| SFT intervention at incorrect timing | Catastrophic forgetting | Timing is critical |
| SFT intervention at maximum frustration state | **Optimal** | Best timing as predicted by theory |

### Key Findings
- CoNet (minimal model) and the 1.5B LLM exhibit remarkably consistent macroscopic dynamics, supporting the claim that emergent behavior is independent of microscopic details.
- The average degree of the concept network consistently stabilizes at approximately 2, directly validating the central hypothesis.
- Rapid recovery following catastrophic forgetting confirms the "topologically local damage" interpretation—knowledge is not erased but rendered unreachable.
- Annealed-RLVR outperforms standard RLVR on both in-distribution and OOD benchmarks.

## Highlights & Insights
- **A Unified Theory from a Physics Perspective**: A single concise topological hypothesis ($\langle k \rangle \approx 2$) unifies the explanation of four independent phenomena, demonstrating the power of interdisciplinary thinking.
- **Maximum Frustration State = Optimal Exploration Moment**: The paper reveals that the moment of apparently worst performance coincides with the peak of exploration diversity—an insight with direct implications for all practitioners using RLVR.
- **From Explanation to Prescription**: Beyond proposing a theoretical framework, the work directly derives a verifiable optimization algorithm, completing a coherent theory-to-practice loop.

## Limitations & Future Work
- CoNet is a highly simplified model with a substantial gap in scale and mechanism relative to real LLMs.
- The central hypothesis ($\langle k \rangle \approx 2$) lacks validation through direct extraction of reasoning graphs from within an LLM.
- Validation is limited to a 1.5B-parameter model; applicability to larger-scale models remains to be confirmed.
- Detection of the annealing trigger (V-shape bottom / reward knee point) may not always be unambiguous in practice.

## Related Work & Insights
- **vs. DeepScaleR/DeepSeek-R1**: This work provides a theoretical explanation of RLVR training dynamics rather than a new training method.
- **vs. Learning Curve Analysis Work**: Macroscopic phenomena are traced to topological structure rather than statistical or optimization perspectives.
- The sparse network self-organization framework may generalize to understanding other emergent capabilities, such as in-context learning and tool use.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Unifying RLVR phenomena through complex network theory represents an entirely novel perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ CoNet validation is thorough; LLM validation covers multiple benchmarks but is limited in model scale.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Exemplary narrative; the logical chain from phenomena to theory to algorithm is complete and elegant.
- **Value**: ⭐⭐⭐⭐⭐ Has far-reaching implications for understanding the mechanisms of reasoning acquisition in LLMs; Annealed-RLVR offers practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ExGRPO: Learning to Reason from Experience](exgrpo_learning_to_reason_from_experience.md)
- [\[CVPR 2026\] Reinforce to Learn, Elect to Reason: A Dual Paradigm for Video Reasoning](../../CVPR2026/reinforcement_learning/reinforce_to_learn_elect_to_reason_a_dual_paradigm_for_video_reasoning.md)
- [\[ICLR 2026\] How Far Can Unsupervised RLVR Scale LLM Training?](how_far_can_unsupervised_rlvr_scale_llm_training.md)
- [\[ICLR 2026\] On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification](on_the_generalization_of_sft_a_reinforcement_learning_perspective_with_reward_re.md)
- [\[ICLR 2026\] The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective](the_sample_complexity_of_online_reinforcement_learning_a_multi-model_perspective.md)

</div>

<!-- RELATED:END -->
