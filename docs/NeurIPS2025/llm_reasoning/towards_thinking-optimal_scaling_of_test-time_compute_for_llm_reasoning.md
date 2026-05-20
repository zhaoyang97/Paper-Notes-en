---
title: >-
  [Paper Note] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning
description: >-
  [NeurIPS 2025][LLM Reasoning][Test-time compute] This paper demonstrates that excessively extending CoT length degrades LLM reasoning performance, and proposes Thinking-Optimal Scaling (TOPS)…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "Test-time compute"
  - "Chain-of-Thought"
  - "reasoning scaling"
  - "overthinking"
  - "self-improvement"
  - "optimal reasoning effort"
date: 2026-05-08
content_hash: 2a4721ada8bcb7c9
---

# Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2502.18080](https://arxiv.org/abs/2502.18080)  
**Code**: [RUCBM/TOPS](https://github.com/RUCBM/TOPS)  
**Area**: LLM Reasoning
**Keywords**: Test-time compute, Chain-of-Thought, reasoning scaling, overthinking, self-improvement, optimal reasoning effort

## TL;DR

This paper demonstrates that excessively extending CoT length degrades LLM reasoning performance, and proposes Thinking-Optimal Scaling (TOPS), a strategy that trains models to select the shortest correct response for each problem via self-improvement, outperforming existing distillation methods in both accuracy and efficiency.

## Background & Motivation

**Rise of System-2 Thinking**: Reasoning models exemplified by OpenAI o1 achieve significant gains on complex tasks by extending CoT to enable search, reflection, and backtracking.

**Subsequent Work Pursues Longer CoT**: Models such as QwQ-32B-Preview and DeepSeek-R1 further scale reasoning token counts via distillation or RL, aiming for improved performance.

**Efficiency Concerns around Overthinking**: Concurrent work has noted that o1-like models generate excessive redundant tokens for simple problems, yet focuses solely on efficiency without examining the impact on accuracy.

**Core Concern**: Does aggressively pursuing longer CoT actually *reduce* reasoning accuracy in certain domains? This represents a deeper question than efficiency alone.

**Preliminary Observation**: Comparing QwQ-32B-Preview with Qwen2.5-32B-Instruct reveals that the former uses substantially more tokens with only marginal performance gains, suggesting longer CoT is not universally beneficial.

**Research Goal**: Systematically investigate how CoT length scaling affects reasoning performance, and design a thinking-optimal scaling strategy that allows models to adaptively determine the required reasoning depth for each problem.

## Method

### Overall Architecture: TOPS (Thinking-OPtimal Scaling)

TOPS comprises three stages, with the core idea of identifying the "shortest correct response" for each problem as the training target.

#### Stage 1: Format Imitation

- A **tag model** is trained on a small seed dataset (~1.3K problems, 3 responses per problem at different reasoning depths, ~3.9K samples total).
- Three system prompts control Low/Medium/High reasoning effort, prompting QwQ-32B-Preview to generate correct CoTs of varying lengths.
- Responses for each problem are re-ranked by actual length, requiring adjacent length differences > 300 tokens to ensure genuinely distinct reasoning depths.
- The base model is fine-tuned on seed data to learn to adopt different reasoning depths according to different system prompts.

#### Stage 2: Reasoning Effort-Conditioned Generation

- The tag model generates one response each at Low/Medium/High reasoning effort for an additional 50K math problems.
- For each problem, the **shortest correct response** among the three is selected as the thinking-optimal response.
- Combined with low-effort responses from the seed data, this yields a thinking-optimal dataset of ~26K samples.

#### Stage 3: Self-Improvement

- The base model undergoes SFT on the thinking-optimal dataset (learning rate $1 \times 10^{-5}$, batch size 96, 2 epochs).
- The resulting TOPS model adaptively allocates fewer tokens to simple problems and more tokens to difficult ones.

### Key Designs

- **Distinct from fixed-length distillation**: STILL-2 and Sky-T1 distill directly from the original length distribution of o1-like models, whereas TOPS obtains a superior length distribution through multi-depth generation followed by shortest-correct selection.
- **Distinct from random selection**: The ablation baseline Qwen2.5-32B-Random, which selects a random correct response, consistently underperforms shortest-correct selection across all benchmarks.

### Loss & Training: Iterative Self-Improvement

- **Iter-SFT**: On an additional 4,500 MATH problems and AIME 1983–2023, 8 responses are sampled from the TOPS model; the shortest correct response is selected for continued SFT.
- **Iter-DPO**: Preference pairs are constructed with the shortest correct response as *chosen* and the longest incorrect response as *rejected* (to improve reasoning ability), along with the shortest incorrect response that is shorter than the shortest correct response as an additional *rejected* candidate (to prevent over-simplification). DPO optimization is then applied.

## Key Experimental Results

### Main Results (Qwen2.5-32B Series)

| Model | GSM8K Acc | GSM8K #Tokens | MATH500 Acc | MATH500 #Tokens | AIME2024 Acc | AIME2024 #Tokens |
|---|---|---|---|---|---|---|
| Qwen2.5-32B-Instruct (T=0) | 95.91 | 295 | 84.20 | 577 | 16.67 | 1407 |
| QwQ-32B-Preview | 95.23 | 761 | 92.02 | 2416 | **45.33** | 7637 |
| STILL-2-32B | 95.47 | 571 | 91.40 | 2005 | **45.33** | 6656 |
| Sky-T1-32B-Preview | 94.82 | 696 | 89.48 | 2022 | 35.33 | 5351 |
| **Qwen2.5-32B-TOPS** | **95.82** | **412** | 91.48 | 1883 | 43.33 | 7260 |
| **TOPS-Iter-DPO** | 95.80 | 385 | **91.60** | 1732 | **46.00** | 6427 |

### Ablation Study

| Analysis Dimension | Key Findings |
|---|---|
| Reasoning effort vs. difficulty | Low effort is optimal for simple tasks (GSM8K); High effort is superior for difficult tasks (AIME2024) |
| Negative mechanism of long CoT | Both the count and proportion of erroneous reasoning steps increase significantly in longer CoTs |
| Loss masking validation | Masking loss on incorrect steps outperforms computing loss over all steps, confirming that erroneous steps are harmful |
| Answer consistency | Under optimal reasoning effort, fewer distinct answers appear across multiple samples, indicating greater model stability |
| TOPS vs. Random selection | TOPS (shortest correct) outperforms Random (arbitrary correct) on every benchmark |
| Iterative DPO | Simultaneously improves accuracy and efficiency; achieves 46.00% on AIME2024 (surpassing QwQ-32B-Preview) |

### Key Findings

- Only 1.3K seed samples combined with self-improvement suffice to match or exceed STILL-2, which relies on 3.9K high-quality distilled samples.
- TOPS uses only 412 tokens on GSM8K (vs. 761 for QwQ), effectively mitigating overthinking.
- The method generalizes to LLaMA3.1-8B-Instruct, demonstrating cross-architecture transferability.

## Highlights & Insights

- **Core Insight: Longer ≠ Better.** This work is the first to systematically demonstrate that excessively long CoTs introduce more erroneous reasoning steps and can reduce accuracy, overturning the naive intuition that longer reasoning is always preferable.
- **Elegant Design via Shortest Correct Response**: No additional reward model or complex search is required; thinking-optimal distributions are obtained automatically through multi-depth sampling followed by shortest-correct selection.
- **Self-Improvement Loop**: With minimal seed data (1.3K), the model generates and filters large-scale training data itself, enabling a low-cost upgrade from System-1 to efficient System-2 reasoning.
- **Answer Consistency as a Signal**: Optimal reasoning depth is associated with the most concentrated answer distribution under repeated sampling, providing an indirect signal for assessing reasoning depth appropriateness without ground-truth labels.
- **Bidirectional Preference Pairs in Iterative DPO**: The design simultaneously guards against both overthinking and underthinking, offering a more principled formulation than standard DPO.

## Limitations & Future Work

1. **Domain Scope**: Analysis and experiments are primarily conducted on mathematical reasoning, where precise correctness verification is feasible; whether similar overthinking phenomena exist in code generation, scientific reasoning, or open-ended tasks remains to be explored.
2. **SFT-Only Setting**: TOPS has not been validated in RL training regimes (e.g., GRPO, PPO), where over-rewarding long correct responses may give rise to analogous issues.
3. **Discrete Reasoning Effort Levels**: Only three reasoning depths (Low/Medium/High) are used; finer-grained continuous control may yield further improvements.
4. **Single Sample per Effort Level**: Only one response is sampled per reasoning effort level; sampling multiple responses and selecting the shortest correct one could potentially improve performance.
5. **Reliance on Teacher Model for Seed Data**: Seed data is still generated by QwQ-32B-Preview; exploring teacher-free pure RL self-evolution is an important future direction.
6. **No Integration with PRM/ORM**: Incorporating process reward models during shortest-correct-response selection could enable more refined quality–length trade-offs.

## Related Work & Insights

- **Relation to STILL-2 / Sky-T1**: These methods distill directly from o1-like responses, inheriting the teacher model's length distribution; TOPS breaks this constraint through adaptive selection.
- **Complementarity with Overthinking Research**: Chen et al. (2024) address efficiency concerns; TOPS further reveals accuracy degradation and provides a concrete solution.
- **Connection to RL-Based Scaling**: The authors note that assigning identical rewards to all correct responses in RL (e.g., 1.0) suffers from a similar problem; shorter correct responses should receive higher preference.
- **Implications for DeepSeek-R1 and Subsequent Work**: Introducing length penalties or differentiated rewards based on reasoning step quality during RL training may represent a more promising direction.
- **Complementarity with Process Reward Models**: TOPS's selection of the shortest correct response serves as a coarse-grained proxy for process quality, and is complementary to the fine-grained step-level evaluation provided by PRMs.

## Rating

- ⭐ Novelty: 4/5 — First systematic analysis of the negative effects of excessive CoT length scaling; the TOPS method is simple yet grounded in a sharp insight.
- ⭐ Experimental Thoroughness: 4/5 — Covers multiple base models, benchmarks of varying difficulty, thorough ablations, and mechanistic analysis, though domain coverage is somewhat narrow.
- ⭐ Writing Quality: 4/5 — Logically structured, with a coherent narrative progressing from empirical observation to causal analysis to method design.
- ⭐ Value: 4.5/5 — Offers an important critical perspective on test-time scaling and a practical training strategy for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)
- [\[NeurIPS 2025\] Provable Scaling Laws for the Test-Time Compute of Large Language Models](provable_scaling_laws_for_the_testtime_compute_of_large_lang.md)
- [\[NeurIPS 2025\] Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)
- [\[NeurIPS 2025\] Atom of Thoughts for Markov LLM Test-Time Scaling](atom_of_thoughts_for_markov_llm_testtime_scaling.md)
- [\[NeurIPS 2025\] LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)

</div>

<!-- RELATED:END -->
