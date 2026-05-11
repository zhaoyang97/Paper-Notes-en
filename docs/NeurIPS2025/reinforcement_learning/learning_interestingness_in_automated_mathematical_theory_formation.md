---
title: >-
  [Paper Note] Learning Interestingness in Automated Mathematical Theory Formation
description: >-
  [NeurIPS 2025][Reinforcement Learning][automated theory formation] This paper proposes Fermat—a reinforcement learning environment that models mathematical theory formation as an MDP—and EvoAbstract…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "automated theory formation"
  - "interestingness learning"
  - "evolutionary program synthesis"
  - "LLM-driven search"
date: 2026-05-08
content_hash: 13cdff3607b49172
---

# Learning Interestingness in Automated Mathematical Theory Formation

**Conference**: NeurIPS 2025
**arXiv**: [2511.14778](https://arxiv.org/abs/2511.14778)
**Code**: [https://github.com/trishullab/Fermat](https://github.com/trishullab/Fermat)
**Area**: Reinforcement Learning / Automated Mathematical Discovery
**Keywords**: automated theory formation, interestingness learning, reinforcement learning, evolutionary program synthesis, LLM-driven search

## TL;DR

This paper proposes Fermat—a reinforcement learning environment that models mathematical theory formation as an MDP—and EvoAbstract, an LLM-driven evolutionary algorithm with abstraction learning, for automatically synthesizing interestingness metrics for mathematical objects. The approach substantially outperforms hard-coded baselines in elementary number theory and finite fields.

## Background & Motivation

Automated mathematical discovery has long been a grand challenge for AI, from the Logic Theorist of the 1950s to the recent AlphaProof. However, existing work has primarily focused on **solving predefined problems** (theorem proving), whereas the essence of mathematical research is an **open-ended exploration** process: defining new concepts, studying their properties, formulating conjectures, and proving or refuting them.

Two core challenges:

**Lack of a complete theory formation framework**: Existing systems either perform only theorem proving or only conjecture generation; no unified framework supports the full pipeline of defining new concepts, proposing conjectures, and proving theorems.

**Search guidance problem**: The search space for mathematical exploration suffers from combinatorial explosion, with most paths leading to trivial or uninteresting mathematics. Human mathematicians rely on intuition to judge "what is worth studying"—i.e., **interestingness**—but prior systems (e.g., HR) rely on hard-coded interestingness metrics.

**Key Insight**: This paper frames the discovery of interestingness metrics as a **learning problem**—using evolutionary program synthesis to automatically search for optimal interestingness functions that guide an agent toward meaningful mathematical theories.

## Method

### Overall Architecture

**Fermat Environment**: Models mathematical theory formation as an MDP $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R})$:
- **State $\mathcal{S}$**: A knowledge graph $G=(V,E)$, where nodes are definitions, conjectures, and theorems, and edges represent construction dependencies.
- **Actions $\mathcal{A}$**: Definition-producing actions (Exists, Specialize, Compose, etc., 9 types), conjecture-producing actions (Implication, Equivalence, etc., 4 types), and proof actions (prove/disprove).
- **Extrinsic reward $\mathcal{R}_\mathcal{E}$**: A reward of 1 for discovering entities belonging to a predefined set of ground-truth mathematical entities $\mathcal{E}$.

**EvoAbstract**: Learns the optimal interestingness function $\mathcal{I}^* = \arg\max_\mathcal{I} \mathbb{E}_{\tau \sim \pi_\mathcal{I}}[\sum_t \gamma^t \mathcal{R}_\mathcal{E}]$.

### Key Designs

1. **Mathematical Entity Representation**: Each entity $m$ comprises three components:

    - Symbolic definition $m_{sym}$ (formal expression in the Fermat DSL)
    - Computational interpretation $m_{comp}$ (executable Python function)
    - Cached instances $\mathcal{X}(m) = (\mathcal{X}^+(m), \mathcal{X}^-(m))$ (positive and negative example sets)

2. **EvoAbstract Evolutionary Search**:

    - **EvolutionStep**: Uses an LLM $\mathcal{L}_{var}$ (GPT-4o-mini) as a mutation operator; takes high-scoring parent programs and templates as input to generate new interestingness function candidates.
    - **Island Model**: $k=4$ parallel populations maintain diversity.
    - **PolicyEvaluationStep**: Evaluates candidate programs via episodic rollouts in Fermat (64 episodes, 60-second timeout), using cumulative extrinsic reward as fitness.

3. **Abstraction Learning (Core Innovation of EvoAbstract)**:

    - **AbstractionStep**: Performed every 8 rounds; an LLM $\mathcal{L}_{abs}$ analyzes high-scoring programs to identify reusable subroutines (abstractions), which are added to an abstraction library $\text{Lib}_i$.
    - In subsequent evolution, $\mathcal{L}_{var}$ can utilize components from the abstraction library to construct more complex solutions.
    - Effect: Promotes modular, compositional construction and steers the search toward more promising regions of the program space.
    - Example: EvoAbstract autonomously discovers abstractions such as `compute_example_balance` (analogous to applicability) and `calculate_rule_diversity_score` (rule diversity).

4. **Production Rules**:

    - Definition production: Exists (existential quantification), Specialize (variable specialization), Compose (composition), ForAll (universal quantification), Match (equality matching), Negate (negation), Size (counting), etc.
    - Conjecture production: Implication, Equivalence, Nonexistence, Exclusivity.
    - Backend prover: Z3 SMT Solver.

### Loss & Training

The interestingness function serves as an intrinsic reward $\mathcal{R}_\mathcal{I}(S,a,S') = \mathcal{I}(m_{new}, S')$, and the policy $\pi_\mathcal{I}$ guides action selection based on interestingness scores. The optimization objective is to maximize cumulative extrinsic reward. Each configuration is run 4 times and results are averaged.

## Key Experimental Results

### Main Results (Cumulative Extrinsic Reward, Higher is Better)

| Method | succ_zero_eq | arithmetic_base | ff_27 |
|--------|------|----------|------|
| Random | 4.68 | 4.44 | 2.33 |
| Comprehensibility (HR best) | 8.23 | 8.55 | 5.38 |
| Equal Weight (HR combined) | 6.57 | 5.93 | 3.93 |
| GPT-4o (one-shot) | 5.26 | 6.46 | 2.36 |
| FunSearch | 10.23 | 22.41 | **11.34** |
| **EvoAbstract** | 9.62 | **23.98** | 9.82 |

EvoAbstract and FunSearch significantly outperform all baselines across all starting configurations, with EvoAbstract achieving the best performance on arithmetic_base (23.98 vs. 22.41 entities discovered).

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| FunSearch (w/o abstraction) | 22.41 (arith) | Evolutionary search is already effective |
| EvoAbstract (w/ abstraction) | **23.98** (arith) | Abstraction learning provides additional gains |
| GPT-4o (one-shot, best) | 9.45 (arith) | Limited effectiveness from single-shot sampling |
| HR Applicability | 5.71 (arith) | Single hand-crafted metric is insufficient |

### Key Findings
- **Interestingness functions generated directly by GPT-4o perform poorly**: They are even outperformed by the hand-crafted comprehensibility metric, as LLMs tend to reward construction depth and connectivity, causing the agent to focus on early but irrelevant entities.
- **Evolutionary search is highly effective**: Metrics discovered by FunSearch/EvoAbstract substantially outperform all static baselines.
- **Abstraction learning is beneficial but double-edged**: It yields gains on arithmetic_base, but may produce an "abstraction lock-in" effect on ff_27 and succ_zero_eq, limiting exploratory diversity in later stages.
- **Rediscovery of foundational mathematical concepts**: Starting from succ_zero_eq, the system rediscovers addition, multiplication, and divisibility; starting from arithmetic_base, it rediscovers exponentiation and the concept of prime numbers.
- Programs generated by EvoAbstract are more modular and readable, employing multiple semantically meaningful abstraction functions.

## Highlights & Insights
- The formalization of **mathematical theory formation as an RL problem** is highly elegant, transforming open-ended mathematical discovery into a quantitatively evaluable framework.
- **Interestingness learning** addresses the central challenge of automated mathematical discovery—it is not sufficient to prove theorems; one must also know what is worth proving.
- The abstraction learning mechanism in EvoAbstract parallels the human mathematical practice of extracting "general concepts"—distilling reusable patterns from concrete successful experiences.
- The design combining production rules with a symbolic prover ensures rigorous guarantees for all discovered results.

## Limitations & Future Work
- Policy templates constrain the action space exposure, limiting the complexity of mathematical objects that can be discovered.
- **Bottleneck entity problem**: When key concepts (e.g., prime numbers) are discovered, the knowledge graph has already grown too large, impeding subsequent valuable operations.
- The absence of entity equivalence checking leads to representational redundancy.
- The Z3 prover becomes computationally infeasible as the theory grows.
- The lack of integration with interactive theorem provers (e.g., Lean) constrains the mathematical domains that can be explored.

## Related Work & Insights
- **vs. HR [Colton 2000]**: HR relies on hard-coded interestingness metrics, whereas Fermat + EvoAbstract automatically learns such metrics.
- **vs. FunSearch [2024]**: EvoAbstract extends FunSearch with abstraction learning, yielding more modular programs.
- **vs. AlphaProof**: AlphaProof focuses on competition-level theorem proving, whereas Fermat targets open-ended theoretical exploration.
- **vs. Minimo [2024]**: Minimo is limited to conjecture-proving games over initial axiomatic definitions; Fermat supports the generation of new definitions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formalizing mathematical theory formation as RL is novel; the idea of interestingness learning is original and distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple starting configurations, extensive baseline comparisons, and qualitative analysis, though scale is limited by Z3 performance.
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear, though the technical details of production rules are dense.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for automated mathematical discovery; the Fermat framework has broad value for follow-up research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning](discover_automated_curricula_for_sparse-reward_reinforcement_learning.md)
- [\[NeurIPS 2025\] A Theory of Multi-Agent Generative Flow Networks](a_theory_of_multi-agent_generative_flow_networks.md)
- [\[NeurIPS 2025\] The Physical Basis of Prediction: World Model Formation in Neural Organoids via an LLM-Generated Curriculum](the_physical_basis_of_prediction_world_model_formation_in_neural_organoids_via_a.md)
- [\[ICLR 2026\] Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning](../../ICLR2026/reinforcement_learning/unveiling_the_cognitive_compass_theory-of-mind-guided_multimodal_emotion_reasoni.md)
- [\[AAAI 2026\] MathSmith: Towards Extremely Hard Mathematical Reasoning by Forging Synthetic Problems with a Reinforced Policy](../../AAAI2026/reinforcement_learning/mathsmith_towards_extremely_hard_mathematical_reasoning_by_forging_synthetic_pro.md)

</div>

<!-- RELATED:END -->
