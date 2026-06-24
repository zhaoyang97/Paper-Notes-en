---
title: >-
  [Paper Note] CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World
description: >-
  [ACL 2025][Multimodal VLM][Instruction Following] This paper proposes CrafText, a multimodal instruction-following benchmark based on the Craftax open-world environment. It contains 3,924 instructions with 3,423 unique words, covering four task categories: localization, conditional, building, and achievement. It also introduces a dual-evaluation protocol designed to test the language and goal generalization capabilities of agents.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Instruction Following"
  - "Multimodal Benchmark"
  - "Reinforcement Learning"
  - "Open World"
  - "Language Grounding"
date: 2026-05-08
content_hash: 2f5ebc5669309ab8
---

# CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World

**Conference**: ACL 2025  
**arXiv**: [2505.11962](https://arxiv.org/abs/2505.11962)  
**Code**: [GitHub](https://anonymous.4open.science/r/CrafText-D217/)  
**Area**: Multimodal VLMs  
**Keywords**: Instruction Following, Multimodal Benchmark, Reinforcement Learning, Open World, Language Grounding

## TL;DR

This paper proposes CrafText, a multimodal instruction-following benchmark based on the Craftax open-world environment. It contains 3,924 instructions with 3,423 unique words, covering four task categories: localization, conditional, building, and achievement. It also introduces a dual-evaluation protocol designed to test the language and goal generalization capabilities of agents.

## Background & Motivation

Instruction following in the real world faces two core challenges: (1) decision-making in dynamically changing environments, where the environment is unpredictable and states evolve independently of agent behavior; and (2) generalizing across various tasks and instruction expressions, where agents must correctly interpret diversely phrased instructions and ground them to observations.

Limitations of Prior Work:
- Most environments are static (e.g., Alfred, Touchdown), lacking environmental dynamics.
- Instructions are typically generated procedurally via templates, resulting in limited vocabularies (e.g., BabyAI, HomeGrid).
- Even environments with rich vocabularies (e.g., Alfred) lack diverse object interactions.
- No existing environments offer a dual-evaluation protocol to simultaneously test "language generalization" and "goal generalization."

CrafText aims to fill this gap by establishing a comprehensive benchmark that incorporates environmental dynamics, linguistic diversity, rich interactions, and a dual-evaluation scheme.

## Method

### Overall Architecture

CrafText is built upon Craftax (a Minecraft-like open-world RL environment) and extends it with a natural language instruction interface. The overall framework comprises three components: dataset design, the instruction generation pipeline, and environmental extension.

### Key Designs

1. **Hierarchical Dataset Structure**: A three-tier structure of "Scenario $\rightarrow$ Goal $\rightarrow$ Instruction" is adopted. Scenarios define abstract task classes (e.g., "build a square"), goals parameterize them into specific instances (e.g., "build a 2×2 wooden square"), and instructions represent multiple natural language expressions of the goal (approx. 5–6 formulations per goal).

2. **Four Task Categories**:

    - **Building**: Requires agents to build specific structures, demanding that they remember the starting point and potentially leave to collect additional resources.
    - **Conditional**: Tests instruction understanding, such as "Collect two stones and then craft a sword" vs. "Before crafting a sword, collect two stones."
    - **Localization**: Evaluates spatial instruction comprehension, including compass directions (South, East, West, North) and relative directions (right, above).
    - **Achievement**: Executes standard in-game achievements and their combinations, such as collecting wood and mining diamonds.

3. **Three Difficulty Levels**: Structured based on the sequence length of prerequisite actions required to complete the task:

    - Easy: Achievement scenarios, completing game achievements and their combinations.
    - Medium: All scenario types, but with shorter action sequences ($<10$ steps).
    - Hard: Complex goals or long action sequences.

4. **Instruction Generation Pipeline**: Combines procedural goal generation with GPT-4 language generation. First, expert-defined scenario checking functions and parameter ranges are used to enumerate combinations and generate a large set of goal templates. Then, GPT-4 is leveraged to generate diverse natural language instructions and paraphrases for each goal, ensuring linguistic complexity and diversity.

5. **Dual Evaluation Protocol**:

    - **Paraphrased Test Set**: Uses the same goals as the training set but with rephrased instructions to test language generalization capabilities.
    - **New Objects Test Set**: Introduces combinations of objects unseen during training (though all individual objects have appeared in the training set) to test goal-level generalization.

6. **JAX-Accelerated Environment**: All evaluation checking functions are implemented in JAX, supporting JIT compilation and GPU acceleration to enable highly parallelized, large-scale training.

### Reward System

- A reward of 1 is granted upon instruction completion.
- Achievement discovery rewards provided by the Craftax environment are scaled by 1/50.
- Scenario checking functions are executed at each step to verify the completion status.

## Key Experimental Results

### Main Results (Medium tasks, 50 seeds)

| Algorithm | Conditional | Building | Localization | Achievement | Total |
|---|---|---|---|---|---|
| PPO-T | 0.15 | 0.25 | 0.33 | 0.55 | 0.40 |
| PPO-T+ | 0.17 | 0.24 | 0.30 | 0.70 | **0.45** |
| Dynalang | 0.00 | 0.12 | 0.15 | 0.17 | 0.15 |
| FiLM | 0.07 | 0.38 | 0.29 | 0.76 | 0.43 |

### Generalization Experiments

| Test Set | PPO-T | PPO-T+ | Dynalang | FiLM |
|--------|-------|--------|----------|------|
| Train | 0.40 | 0.45 | 0.15 | 0.43 |
| Paraphrased | 0.36 | 0.35 | 0.05 | 0.35 |
| New Objects | 0.22 | **0.28** | 0.10 | 0.26 |

### Key Findings

- **Dynalang performs far below expectations**: Despite its outstanding performance in the Crafter environment, Dynalang only achieves a 0.15 success rate on CrafText. This indicates that the combination of complex linguistic instructions and dynamic environments dramatically increases the learning difficulty.
- **All baseline methods yield low success rates**: Even the best method, PPO-T+, only achieves 0.45, confirming the high difficulty of the CrafText benchmark.
- **Paraphrasing leads to a significant performance drop**: PPO-T+ drops from 0.45 to 0.35, demonstrating that current methods lack robustness against linguistic variations.
- **PPO-T+ (with planning) performs best on generalizing to new objects**: A success rate of 0.28 shows that decomposing instructions into structured plans aids goal-level generalization.
- **FiLM performs best on building tasks** (0.38), indicating that its feature-level modulation mechanism is more flexible in handling vision-language interactions.
- **Conditional tasks remain extremely challenging for all methods**: The highest success rate is only 0.17–0.20, showing that conditional logical reasoning is a major bottleneck for existing approaches.

## Highlights & Insights

- **Comprehensiveness**: CrafText concurrently addresses environmental dynamics, linguistic diversity, rich interactions, GPU acceleration, and a dual-evaluation protocol, making it uniquely comprehensive compared to existing benchmarks.
- **Uncovering Core Bottlenecks**: Experiments clearly demonstrate that methods performing well in static environments (such as Dynalang) fail completely when faced with dynamic and linguistically complex conditions.
- **JAX Implementation**: Support for large-scale parallel training resolves the practical efficiency bottlenecks associated with RL training.
- **Value of Planning Augmentation**: The simple yet effective GPT-4 planning step in PPO-T+ suggests that leveraging LLMs for task decomposition is a promising future direction.

## Limitations & Future Work

- All instructions in the dataset are AI-generated, lacking human-authored instructions, which may fail to capture the subtle nuances of human language.
- The environment lacks real-world interaction components, such as instruction negotiation, clarification, and dynamic dialogues.
- Success rates of current baseline methods are generally low; stronger methods are needed to fully evaluate the discriminative power of the benchmark.
- Though built on Craftax, it remains a 2D pixel-based environment, leaving a gap to the 3D physical world.
- Language representation is limited to DistilBERT and T5 embeddings, leaving the investigation of more powerful VLMs as policy networks to future study.

## Related Work & Insights

- Compared with template-based instruction environments like BabyAI and HomeGrid, CrafText provides richer vocabularies and higher linguistic complexity.
- Compared with MineDojo, CrafText offers precise objective verification functions and a dual-evaluation protocol.
- Insight: Integrating LLMs for instruction preprocessing/planning (e.g., PPO-T+) is a highly promising direction to advance instruction-following capabilities.
- The challenges posed by environmental dynamics for instruction following remain under-investigated, representing an important open problem.

## Rating

- Novelty: ⭐⭐⭐⭐ The first instruction-following benchmark to satisfy multiple key attributes simultaneously; the dual-evaluation protocol is novel.
- Experimental Thoroughness: ⭐⭐⭐ Baseline evaluations are limited (4 methods); lacks comparisons with VLM-based approaches and more diverse RL algorithms.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with complete comparative tables, though some environment descriptions could be more concise.
- Value: ⭐⭐⭐⭐ Fills a crucial gap for dynamic-environment, complex-instruction-following benchmarks, offering high value to the RL and NLP communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MM-IFEngine: Towards Multimodal Instruction Following](../../ICCV2025/multimodal_vlm/mm-ifengine_towards_multimodal_instruction_following.md)
- [\[ICLR 2026\] MCIF: Multimodal Crosslingual Instruction-Following Benchmark from Scientific Talks](../../ICLR2026/multimodal_vlm/mcif_multimodal_crosslingual_instruction-following_benchmark_from_scientific_tal.md)
- [\[CVPR 2025\] OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation](../../CVPR2025/multimodal_vlm/opening_a_comprehensive_benchmark_for_judging_open-ended_interleaved_image-text_.md)
- [\[ACL 2025\] REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark](real-mm-rag_a_real-world_multi-modal_retrieval_benchmark.md)
- [\[ACL 2025\] MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark](mmmu_pro_robust_benchmark.md)

</div>

<!-- RELATED:END -->
