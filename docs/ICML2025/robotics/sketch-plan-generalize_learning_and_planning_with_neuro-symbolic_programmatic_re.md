---
title: >-
  [Paper Note] Sketch-Plan-Generalize: Learning and Planning with Neuro-Symbolic Programmatic Representations for Inductive Spatial Concepts
description: >-
  [ICML2025][Robotics][Neuro-Symbolic] Proposes SPG (Sketch-Plan-Generalize), a neuro-symbolic agent framework that decomposes inductive concept learning into a three-stage pipeline: concept signature inference (Sketch), MCTS-based grounded action sequence search (Plan), and LLM-driven programmatic inductive generalization (Generalize). It significantly outperforms pure LLM and pure neural methods in learning composable, generalizable spatial abstract concepts from a few demons…
tags:
  - "ICML2025"
  - "Robotics"
  - "Neuro-Symbolic"
  - "Program Synthesis"
  - "MCTS"
  - "Inductive Generalization"
  - "Concept Learning"
  - "Embodied AI"
date: 2026-05-08
content_hash: ad801127b115af3d
---

# Sketch-Plan-Generalize: Learning and Planning with Neuro-Symbolic Programmatic Representations for Inductive Spatial Concepts

**Conference**: ICML2025  
**arXiv**: [2404.07774](https://arxiv.org/abs/2404.07774)  
**Area**: Neuro-Symbolic Learning / Embodied Concept Learning  
**Keywords**: Neuro-Symbolic, Program Synthesis, MCTS, Inductive Generalization, Concept Learning, Embodied AI

## TL;DR

Proposes SPG (Sketch-Plan-Generalize), a neuro-symbolic agent framework that decomposes inductive concept learning into a three-stage pipeline: concept signature inference (Sketch), MCTS-based grounded action sequence search (Plan), and LLM-driven programmatic inductive generalization (Generalize). It significantly outperforms pure LLM and pure neural methods in learning composable, generalizable spatial abstract concepts from a few demonstrations.

## Background & Motivation

Effective human-robot collaboration requires robots to quickly learn personalized concept representations from extremely few human demonstrations. An ideal concept representation needs to satisfy four key properties:

**Demonstration-based grounding**: Learning directly from human demonstrations to reflect human intent, rather than solely relying on prior world knowledge.

**Inductive generalization**: Inferring a general program to build a tower of arbitrary height from just a few towers of specific heights.

**Hierarchical compositionality**: Complex concepts can be represented as compositions of simpler concepts (e.g., staircase = a sequence of towers with increasing heights).

**Constraint modifiability**: Learnable concept representations can flexibly adapt to additional constraints in new instructions (e.g., color constraints).

**Limitations of Prior Work**:

- **Pure LLM methods** (Singh et al., 2022; Liang et al., 2023): Rely on prior world knowledge to generate plans, failing to learn novel concepts that are linguistically unknown from demonstrations, and showing poor inductive generalization.
- **Pure neural methods** (Liu et al., 2023): Learn from demonstrations but lack symbolic inductive modeling, which limits their generalization to the training distribution and lacks modular reuse capabilities.
- **Neuro-symbolic methods** (Grand et al., 2023; Ellis et al., 2021): Search for generalizable programs in program spaces, but enumerative search is computationally intractable in complex program spaces (such as Python programs) and lacks effective guidance from human demonstrations.

**Key Insight**: Decomposing inductive concept learning into three complementary stages—coarse-grained signature inference, reward-based grounded planning search, and abstracting general programs from multiple concrete plans—can bypass the search intractability in massive program spaces.

## Method

### Overall Architecture

SPG decomposes the concept learning task into a three-stage pipeline:

$$\text{Demonstrations} \xrightarrow{\text{Sketch}} \text{Signature} \xrightarrow{\text{Plan}} \text{Grounded Plans} \xrightarrow{\text{Generalize}} \text{Inductive Program}$$

The design philosophy of the overall framework is "concrete first, abstract later": it first obtains grounded plans for multiple concrete instances via search, and then leverages the code generation capabilities of LLMs to distill them into inductive general programs.

### Stage 1: Sketch (Concept Signature Inference)

Given a demonstration of a new concept with linguistic annotations, an LLM is utilized to infer the **coarse-grained function signature** of the concept:

- **Input**: Linguistic description + visual demonstration
- **Output**: Function name, parameter list, and parameter types
- **Function**: Defining the boundary of the search space, constraining the infinite program space into a finite family of signatures

For example, for the "tower" concept, the Sketch stage might output the signature `def build_tower(height: int, color: str) -> ActionSequence`, specifying the dimensions that need to be parameterized.

The key to this stage is that the LLM does not need to write the complete implementation—it only needs to "sketch" the concept's interface specification, which drastically reduces the requirements for the accuracy of LLM code generation.

### Stage 2: Plan (MCTS-Guided Grounded Planning)

Using the function signature as a framework, Monte Carlo Tree Search (MCTS) is used to search the grounded action sequence space:

- **Search Space**: Given available atomic actions (such as placing a block, moving, rotating), they are combined to form action sequences.
- **Reward Signal**: Measuring the match between the constructed result and the target structure in the human demonstration.
- **Search Policy**: MCTS utilizes UCB (Upper Confidence Bound) to balance exploration and exploitation.

$$\text{UCB}(s, a) = \bar{Q}(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

where $\bar{Q}(s, a)$ is the average reward of action $a$ in state $s$, $N(s)$ is the visit count of state, and $c$ is the exploration coefficient.

For each demonstration instance, the Plan stage outputs a concrete grounded action sequence. Multiple demonstrations with different parameter configurations yield multiple grounded plans, providing an "inductive basis" for subsequent generalization.

**Key Designs**: The reward function calculates structural similarity based on neural representations. It integrates a visual encoder to encode 3D structures into vectors, measuring the distance to the target demonstration. This enables the system to guide discrete action search using continuous similarity signals without requiring exact symbolic matching.

### Stage 3: Generalize (Programmatic Inductive Generalization)

Leveraging the code generation capabilities of LLMs, grounded plans from multiple demonstrations are **distilled** into inductive general programs:

- **Input**: Multiple grounded plans + function signature + demonstration parameters
- **Processing**: The LLM compares concrete plans under different parameters, identifying invariant patterns and parameterized dimensions.
- **Output**: A parameterizable inductive program (e.g., `for i in range(height): place_block(x, y+i)`)

This program features:
- **Inductive Generalization**: It can handle parameter values unseen during training (e.g., generalizing from towers of heights 3 and 5 to height 100).
- **Modular Reuse**: Learned concepts can be invoked as subprograms by more complex concepts.
- **Continual Learning**: New concepts are incrementally added to the concept library, supporting hierarchical composition.

### Loss & Training

SPG adopts a **training-free compositional strategy** without relying on end-to-end gradient optimization:

1. **Concept Detection**: While executing instructions, the agent detects whether it encounters an unknown concept (one without a corresponding program in the library).
2. **Active Request for Demonstrations**: Upon identifying an unknown concept, the agent actively requests 2-3 demonstrations with different parameter configurations from the human.
3. **Three-Stage Learning**: Successively executing Sketch $\rightarrow$ Plan $\rightarrow$ Generalize.
4. **Concept Library Update**: Saving newly learned programs into the concept library to be directly invoked or combined in the future.

This design endows the system with **continual concept learning** capabilities—the library continuously expands, and new concepts can be built upon existing ones. For example, after learning "tower" and "row," it can learn "staircase" (a tower sequence of increasing heights) and "pyramid" (a stack of rows of decreasing lengths) through composition.

### Instruction Following and Constraint Reasoning

The learned programmatic representations support the flexible application of constraints in new instructions:

- **Discrete Constraints**: e.g., "Build a staircase, and no green blocks can be placed directly adjacent to blue blocks."
- **Mechanism**: The LLM compiles constraints into conditional statements in the program, inserting them at suitable locations within the learned program.
- **Advantage**: Programmatic representations naturally support conditional branches and constraint injection, whereas pure neural representations struggle to enforce hard constraints precisely.

## Key Experimental Results

### Main Results

| Method | Simple Concepts (Tower, Row) | Complex Concepts (Staircase, Pyramid) | Compositional Concepts (Boundary, L-shape) | Average Generalization Rate |
|------|----------------------|------------------------------|------------------------------|-----------|
| LLM-only (GPT-4) | High | Low | Very Low | ~40% |
| Pure Neural (Struct-Net) | Medium | Low | Low | ~35% |
| DreamCoder (Enumerative Search) | Medium | Medium | Low | ~45% |
| **SPG (Ours)** | **High** | **High** | **Medium-High** | **~75%** |

SPG significantly outperforms all baseline methods in inductive generalization performance, with the most notable advantage in complex concepts requiring hierarchical composition.

### Ablation Study

| Configuration | Sketch | Plan | Generalize | Generalization Accuracy | Delta |
|------|--------|------|------------|-----------|------|
| Full SPG | ✓ | ✓ | ✓ | ~75% | — |
| w/o Sketch (Direct Search) | ✗ | ✓ | ✓ | ~55% | -20% |
| w/o MCTS (Direct LLM Plan Generation) | ✓ | ✗ | ✓ | ~50% | -25% |
| w/o Generalize (Using Grounded Plans) | ✓ | ✓ | ✗ | ~30% | -45% |
| End-to-End LLM Code Generation | ✗ | ✗ | ✗ | ~40% | -35% |

The ablation study clearly demonstrates that: the Generalize stage contributes the most (reflecting the core value of inductive generalization), followed by the Plan stage (MCTS search provides high-quality grounded foundations), and the Sketch stage is also indispensable (constraining the search space and improving search efficiency).

### Instruction Following and Constraint Satisfaction

| Task Type | SPG | LLM-only | Pure Neural |
|----------|-----|----------|--------|
| Unconstrained Construction | High | Medium | Medium |
| Color Constraints | High | Medium-Low | Low |
| Spatial Relation Constraints | High | Low | Very Low |
| Compositional Constraints | Medium-High | Very Low | Very Low |

Programmatic representations demonstrate significant advantages in constraint satisfaction tasks because constraints can be compiled directly into program logic.

## Highlights & Insights

1. **Clever decomposition of "concrete first, abstract later"**: Avoiding direct search in massive program spaces, it instead uses MCTS to obtain concrete plans in small action spaces and then leverages LLMs for inductive abstraction. This divide-and-conquer strategy dramatically reduces the problem complexity.
2. **An exemplar of neuro-symbolic complementarity**: Neural components provide continuous matching reward signals to guide search, while symbolic components offer composable, generalizable programmatic structures—both are indispensable.
3. **Continual concept learning**: The concept library design supports incremental knowledge accumulation and hierarchical reuse, closely mimicking the cognitive process of humans learning new concepts.
4. **LLM as an "induction engine"**: Rather than requiring the LLM to generate complete programs in a single step, the framework leverages its pattern recognition capabilities to extract commonalities from multiple concrete instances. This reduces dependency on the accuracy of LLM code generation.
5. **Constraint compositionality**: Programmatic representations naturally support constraint injection, resolving the pain point of pure neural methods struggling with hard constraints.

## Limitations & Future Work

1. **Demonstration requirements**: Each new concept requires 2-3 demonstrations with different parameter configurations, which may increase user burden in real-world human-robot interaction.
2. **MCTS search overhead**: For complex structures with long action sequences, MCTS search time can increase significantly, limiting real-time deployment.
3. **Limitation in concept space**: Currently primarily validated on 3D block-building tasks; transferability to more general spatial concepts (e.g., deformable objects, continuous shapes) remains to be verified.
4. **LLM induction robustness**: The Generalize stage relies on LLM code generation. For highly non-linear or irregular concept patterns, LLMs may struggle to perform accurate induction.
5. **Risk of error propagation**: In the three-stage pipeline, signature errors in the Sketch stage can causally cascade and affect all subsequent stages.
6. **Neglecting physical constraints**: The current framework primarily handles geometric/spatial relationships without incorporating real-world physical constraints such as mechanical stability.

## Related Work & Insights

- **LLMs as Planners** (SayCan, Code-as-Policies): Leveraging LLM prior knowledge to generate action plans, but lacking the ability to learn new concepts from demonstrations.
- **DreamCoder** (Ellis et al., 2021): Synthesizing programs in DSLs via enumerative search and compression, but computationally intractable in complex program spaces.
- **LILO** (Grand et al., 2023): Combining LLM-guided program search and library learning, which is the closest baseline to SPG.
- **Struct-Net** (Liu et al., 2023): A pure neural structure prediction method, learning from demonstrations but with limited generalization.
- **Program Inductive Synthesis** (Gulwani et al., 2017): A classic paradigm for synthesizing programs from input-output examples, which SPG extends to the spatial concept domain.

The unique contribution of SPG lies in seamlessly integrating LLM code generation, MCTS search, and neural visual representations, creating a unified framework capable of both learning from demonstrations and generalizing through induction.

## Rating

- Novelty: ⭐⭐⭐⭐ (The three-stage decomposition paradigm is novel, cleverly combining LLMs and MCTS)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Generalization experiments on diverse concept types + detailed ablation + instruction following evaluation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, well-defined problem formulations, and intuitive illustrations)
- Value: ⭐⭐⭐⭐ (Provides a viable neuro-symbolic solution for few-shot concept learning, with practical significance in its continual learning design)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Reliable Code-as-Policies: A Neuro-Symbolic Framework for Embodied Task Planning](../../NeurIPS2025/robotics/towards_reliable_code-as-policies_a_neuro-symbolic_framework_for_embodied_task_p.md)
- [\[NeurIPS 2025\] MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents](../../NeurIPS2025/robotics/mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)
- [\[ICML 2025\] Efficient Robotic Policy Learning via Latent Space Backward Planning](efficient_robotic_policy_learning_via_latent_space_backward_planning.md)
- [\[NeurIPS 2025\] Learning Spatial-Aware Manipulation Ordering](../../NeurIPS2025/robotics/learning_spatial-aware_manipulation_ordering.md)
- [\[ICCV 2025\] SITE: towards Spatial Intelligence Thorough Evaluation](../../ICCV2025/robotics/site_towards_spatial_intelligence_thorough_evaluation.md)

</div>

<!-- RELATED:END -->
