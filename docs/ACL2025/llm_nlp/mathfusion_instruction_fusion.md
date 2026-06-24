---
title: >-
  [Paper Note] MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion
description: >-
  [ACL 2025][LLM (Other)][Mathematical Reasoning] Proposed the MathFusion framework, which synthesizes mathematical problems pairwise into new challenges using three problem fusion strategies (sequential, parallel, and conditional fusion). With only 45K additional synthesized data, it yields an average improvement of 18 percentage points in mathematical reasoning across multiple benchmarks.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Mathematical Reasoning"
  - "Data Augmentation"
  - "Instruction Fusion"
  - "SFT"
  - "Math Problem Synthesis"
date: 2026-05-08
content_hash: f9a020183bcc3f18
---

# MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion

**Conference**: ACL 2025  
**arXiv**: [2503.16212](https://arxiv.org/abs/2503.16212)  
**Code**: [QizhiPei/MathFusion](https://github.com/QizhiPei/MathFusion)  
**Area**: LLM/NLP  
**Keywords**: Mathematical Reasoning, Data Augmentation, Instruction Fusion, SFT, Math Problem Synthesis  

## TL;DR

Proposed the MathFusion framework, which synthesizes mathematical problems pairwise into new challenges using three problem fusion strategies (sequential, parallel, and conditional fusion). With only 45K additional synthesized data, it yields an average improvement of 18 percentage points in mathematical reasoning across multiple benchmarks.

## Background & Motivation

**Core Problem:** The mathematical reasoning capability of LLMs heavily relies on high-quality training data. Prior data augmentation methods primarily modify problems at an individual instance level (e.g., rewriting, scaling difficulty), failing to capture the relational structures of mathematical knowledge.

**Limitations of Prior Work:**
- Methods such as MetaMath and WizardMath focus on single-problem instance-level augmentation (rewriting, scaling difficulty, reverse reasoning), ignoring intrinsic correlations between problems.
- Real-world mathematical problems often consist of interdependent sub-problems that form a complex dependency graph, which prior methods fail to model.
- Combinatorial augmentation methods (e.g., Mosaic-IT, KPMath) exist but are not optimized for the logical consistency of mathematical problems.

**Design Motivation:** Humans develop mathematical competence through systematic exposure to interconnected concepts. Strategically fusing complementary mathematical instructions can activate deeper reasoning processes, achieving cross-problem knowledge integration.

## Method

### Overall Architecture

MathFusion selects problem pairs $(P_A, P_B)$ from an original mathematical dataset, synthesizes new problems $P_F$ through three fusion strategies, and generates solutions using GPT-4o-mini. This pipeline constructs the MathFusionQA dataset (60K samples) used for SFT fine-tuning.

### Key Designs

1. **Problem Pair Construction:** For each problem $P_A$, semantic similarity is calculated using OpenAI embedding (text-embedding-3-large). The most similar problem $P_B$ is selected to form a problem pair, ensuring proximity in category and context.

2. **Sequential Fusion:** $P_F^{seq} = P_B(P_A)$, where the answer to $P_A$ is used as the input condition for $P_B$ to establish a dependency chain for solving. For example: the answer for the number of people a boat can carry $\rightarrow$ becomes the input representing passengers on a bus.

3. **Parallel Fusion:** $P_F^{para} = \Phi(P_A', P_B')$, which integrates two analogous problems into a new problem, encapsulating their shared mathematical essence, with potential modifications to the original input conditions.

### Conditional Fusion

$P_F^{cond} = \Gamma(P_A, P_B)$, which integrates two problems into a realistic scenario where the final answer is determined by comparing or selecting from the results of $P_A$ and $P_B$, thereby enhancing conditional reasoning capabilities.

### Loss & Training

Standard SFT auto-regressive cross-entropy loss: $\mathcal{L} = -\sum_t \log P(y_t | y_{<t}, x)$.

## Key Experimental Results

### Main Results: Mathematical Reasoning Performance of Different Base Models

| Model | #Samples | MATH | GSM8K | College | DM | Olympiad | Theorem | AVG |
|------|------|------|------|------|------|------|------|------|
| DSMath-7B-Standard | 15K | 30.6 | 66.3 | 22.7 | 28.6 | 5.6 | 11.0 | 27.5 |
| DSMath-7B-DART-Math† | 60K | 51.4 | 82.9 | 39.1 | 62.8 | 21.0 | 27.4 | 47.4 |
| **MathFusion-DSMath-7B** | **60K** | **53.4** | 77.9 | 39.8 | **65.8** | **23.3** | 24.6 | **47.5** |
| Llama3-8B-Standard | 15K | 17.5 | 65.4 | 12.9 | 21.6 | 4.7 | 10.9 | 22.2 |
| Llama3-8B-DART-Math† | 60K | 34.1 | 77.2 | 23.4 | 36.0 | 8.7 | 18.2 | 32.9 |
| **MathFusion-Llama3-8B** | **60K** | **41.6** | **79.8** | **24.3** | **39.2** | **13.6** | 18.1 | **36.1** |
| Mistral-7B-Standard | 15K | 12.4 | 60.3 | 8.4 | 17.0 | 2.2 | 7.6 | 18.0 |
| Mistral-7B-DART-Math† | 60K | 34.1 | 77.2 | 23.4 | 36.0 | 8.7 | 18.2 | 32.9 |
| **MathFusion-Mistral-7B** | **60K** | **41.6** | **79.8** | **24.3** | **39.2** | **13.6** | 18.1 | **36.1** |

### Ablation Study: Comparison of Three Fusion Strategies (Llama3-8B)

| Fusion Strategy | #Samples | MATH | GSM8K | AVG |
|------|------|------|------|------|
| Standard (baseline) | 15K | 17.5 | 65.4 | 22.2 |
| Sequential Fusion | 30K | **38.8** | **77.9** | **35.6** |
| Parallel Fusion | 30K | 38.1 | 75.4 | 35.3 |
| Conditional Fusion | 30K | 34.7 | 76.9 | 31.3 |
| **Combined Strategies** | **60K** | **41.6** | **79.8** | **36.1** |

### Key Findings

1. MathFusion, using only 60K data, outperforms DART-Math which utilizes 590K data (achieving AVG 36.1 on Llama3-8B with less than 1/10 of the data scale).
2. Sequential Fusion performs best among the three strategies, modeling dependency links between solving steps, which matches the chain-like style of mathematical reasoning.
3. Combining all three fusion strategies outperforms any single strategy, showing that different fusion perspectives deliver complementary reasoning capabilities.
4. Integrating MathFusion with DART-Math yields further gains, proving their mutual complementarity.
5. Ingesting broader fusion neighborhoods (scaling from top-1 to top-4 nearest neighbors, 195K samples) boosts the AVG performance of DSMath-7B to 49.9.

## Highlights

- **Innovative Cross-Problem Data Augmentation Paradigm**: First to systematically introduce the relational structure of mathematical problems into data augmentation, moving beyond traditional single-issue modifications.
- **Extremely High Data Efficiency**: Employs just 45K additional synthesized data to achieve comparable or superior performance at less than 1/10 the dataset volume of DART-Math.
- **Intuitively Designed Fusion Strategies**: Sequential models dependency chains, Parallel models concept core similarities, and Conditional models conditional choices, covering diverse reasoning scenarios.
- **Strong Generalizability**: Achieves remarkable and consistent performance improvements across three distinct architectures: DeepSeekMath-7B, Mistral-7B, and Llama3-8B.

## Limitations

- Problem fusion can occasionally generate incomplete or logically flawed questions (as noted in the paper's appendix).
- The quality of solutions generated by GPT-4o-mini is hard to strictly guarantee, which might introduce wrong derivations.
- Building problem pairs relies heavily on the OpenAI embedding API, adding external dependency and monetary cost.
- Validations are only focused on 7-8B scale models; performance on much larger models remains to be verified.
- The design of the fusion strategies is heuristic and lacks a rigorous theoretical framework explaining why these specific modes best enhance reasoning.

## Related Work

- **Single-Problem Data Augmentation**: MetaMath (rewriting + reverse reasoning), WizardMath (difficulty scaling), DART-Math (rejection sampling for hard prompts), RefAug (reflection-driven augmentation).
- **Combinatorial Data Augmentation**: Mixup (linear interpolation), Mosaic-IT (instruction concatenation), Instruct-SkillMix (skill combination), KPMath-Plus (keypoint combination).
- **Mathematical LLMs**: DeepSeekMath (continued pre-training), Mistral, Llama3.

## Rating

| Dimension | Score |
|------|------|
| Novelty | 8/10 |
| Effectiveness | 8/10 |
| Experimental Thoroughness | 9/10 |
| Writing Quality | 8/10 |
| Overall Score | 8/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving](arithmattack_evaluating_robustness_of_llms_to_noisy_context_in_math_problem_solv.md)
- [\[ACL 2025\] Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)
- [\[ACL 2025\] Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning](veracity_bias_llm_hidden_beliefs.md)
- [\[ACL 2025\] Just a Scratch: Enhancing LLM Capabilities for Self-Harm Detection through Intent Refinement](just_a_scratch_enhancing_llm_capabilities_for_self-harm_detection_through_intent.md)
- [\[ACL 2025\] Semantic Exploration with Adaptive Gating for Efficient Problem Solving with Language Models](semantic_exploration_adaptive_gating.md)

</div>

<!-- RELATED:END -->
