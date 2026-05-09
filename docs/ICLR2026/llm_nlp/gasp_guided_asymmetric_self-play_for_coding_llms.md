---
title: >-
  [Paper Note] GASP: Guided Asymmetric Self-Play For Coding LLMs
description: >-
  [ICLR 2026][LLM/NLP][Asymmetric self-play] GASP introduces "goalposts" (hard target problems) into asymmetric self-play to guide the teacher in generating targeted training problems. Through a lemma (simplified variant) → lift (harder variant) curriculum structure, the framework progressively approaches difficult targets, surpassing unguided self-play by 2.5% on LiveCodeBench and solving hard problems that all baselines fail to solve.
tags:
  - ICLR 2026
  - LLM/NLP
  - Asymmetric self-play
  - code generation
  - curriculum learning
  - RLVR
  - goal-guided
date: 2026-05-08
content_hash: faf97c8e1179cba7
---

# GASP: Guided Asymmetric Self-Play For Coding LLMs

**Conference**: ICLR 2026
**arXiv**: [2603.15957](https://arxiv.org/abs/2603.15957)
**Code**: None
**Area**: LLM Training / Code Reasoning
**Keywords**: Asymmetric self-play, code generation, curriculum learning, RLVR, goal-guided

## TL;DR
GASP introduces "goalposts" (hard target problems) into asymmetric self-play to guide the teacher in generating targeted training problems. Through a lemma (simplified variant) → lift (harder variant) curriculum structure, the framework progressively approaches difficult targets, surpassing unguided self-play by 2.5% on LiveCodeBench and solving hard problems that all baselines fail to solve.

## Background & Motivation

**Background**: Asymmetric self-play (e.g., Absolute Zero/AZR) enables LLMs to simultaneously act as teacher (problem proposer) and student (problem solver), achieving open-ended training without human-annotated data. RLVR trains code/math reasoning capabilities via verifiable rewards.

**Limitations of Prior Work**: Existing self-play is goal-agnostic — the teacher focuses solely on the learnability of problems (neither too easy nor too hard) without considering whether the generated problems are "interesting" or "beneficial for downstream tasks." Consequently, many hard problems at the learning frontier do not contribute meaningfully to improving the model's practical programming ability.

**Key Challenge**: Self-play requires exploring difficult problems to advance the capability frontier, but unguided exploration is inefficient — many "hard" problems are artificially constructed and do not represent genuine programming challenges.

**Goal**: (1) Can real-world hard problems guide self-play? (2) Does such guidance improve downstream programming performance?

**Key Insight**: Hard problems from the training set that remain unsolved after RLVR training are selected as "goalposts." The teacher is guided to generate simplified versions (lemmas) of these goalposts, and then harder variants (lifts) from the lemmas, forming a curriculum that progressively approaches the target.

**Core Idea**: Use real hard problems as goalposts to guide the self-play teacher, and progressively break through the capability frontier via a lemma–lift stepping-stone curriculum.

## Method

### Overall Architecture
Three-phase cyclic training: Phase 1 (Lemma Generation) → Phase 2 (Lift Generation) → Phase 3 (Solver Training). Teacher and student share parameters and are distinguished via role prompts. A set of 146 goalpost hard problems (filtered from 601 problems with pass@100 = 0) serves as the guiding signal.

### Key Designs

1. **Goalpost Filtering Pipeline**:

    - Function: Multi-stage filtering to identify genuinely hard problems — problems with pass@100 = 0 across all checkpoints of standard RLVR training, AZR training, and additional RL runs.
    - Mechanism: Triple filtering ensures goalposts lie strictly outside the model's current capability.
    - Design Motivation: Goalposts must be "genuinely hard" — verifiably difficult and relevant to real programming challenges.

2. **Lemma Generation (Simplified Variant)**:

    - Function: Given a goalpost $h$, the teacher generates a simpler variant $\ell_0$ that preserves high-level algorithmic themes.
    - Mechanism: Reward function $r_{\text{lemma}} = [4p(1-p)]^5$ (when $0.3 \leq p \leq 0.7$), peaked at $p = 0.5$, encouraging moderate difficulty.
    - Design Motivation: The lemma should be within the student's learnable range but still challenging, and thematically related to the goalpost.

3. **Lift Generation (Harder Variant)**:

    - Function: Starting from lemma $\ell_0$, the teacher generates a harder variant $\ell_1$ **without observing the original goalpost**.
    - Mechanism: Reward $r_{\text{lift}} = 10p\left(\frac{1-p}{0.9}\right)^9$ ($0.1 \leq p \leq 0.5$), peaked at $p = 0.1$, encouraging harder problems.
    - Design Motivation: Withholding the goalpost from the lift stage is a deliberate design choice — it prevents the teacher from superficially copying the target and encourages incrementally increasing difficulty from the student's current frontier.

4. **Difficulty Axes**:

    - I/O axis: Varying the complexity of inputs/outputs (e.g., a single list → nested lists).
    - f axis: Varying algorithmic complexity (e.g., adding constraints or composite operations).
    - Each lemma randomly selects one axis; the lift increases difficulty along the same axis.

### Loss & Training
- Task-Relative REINFORCE++ (adopted from AZR)
- Teacher and student share parameters and are updated jointly
- Three task types: Induction (primary), Deduction, Abduction (introduced in the solver phase)

## Key Experimental Results

### Main Results
LiveCodeBench v5 (Qwen2.5-Coder-7B):

| Method | pass@1 | pass@20 | Notes |
|--------|--------|---------|-------|
| Base model | baseline | baseline | No training |
| RLVR (real data) | strong | strong | Upper-bound reference |
| AZR (unguided self-play) | moderate | moderate | No goalposts |
| GASP | **strong** | **AZR + 2.5%** | Goalpost-guided |

### Goalpost Progress

| Training Stage | Solved Goalposts | Notes |
|----------------|-----------------|-------|
| Initial | 0/146 | All unsolved |
| RLVR | 0/146 | Standard RLVR still fails |
| AZR | 0/146 | Unguided self-play still fails |
| GASP | **>0/146** | Some goalposts solved! |

### Key Findings
- GASP surpasses AZR by 2.5% on pass@20, with larger gains at higher $k$ (indicating that the curriculum increases diversity).
- Most notably, GASP successfully solves a subset of goalpost problems that all baselines (RLVR/AZR) fail to solve.
- The quality of teacher-generated lemma–lift curricula improves over training — later-stage lemmas more closely approximate goalpost difficulty.
- Withholding the goalpost from the lift stage is critical — providing the goalpost directly to the lift teacher causes it to copy surface features rather than incrementally increasing difficulty.

## Highlights & Insights
- **Goal-Guided Self-Play**: Introducing an external "goal" signal into fully unsupervised self-play gives the teacher's creativity a sense of direction, analogous to goal-conditioned learning in RL.
- **Lemma–Lift Stepping Stones**: Rather than directly attacking hard problems, the framework approaches them through a simplify-then-progressively-harden curriculum. This curriculum design is generalizable to hard-problem solving in other domains.
- **Clever Design of Withholding the Goalpost from Lift**: Forcing the teacher to incrementally increase difficulty from the student's current capability, rather than jumping to replicate the target, is more consistent with the gradual nature of learning.

## Limitations & Future Work
- Validated only in the code domain; the definition and effectiveness of goalposts in math or general reasoning remain unexplored.
- Goalpost filtering requires extensive RL training (multiple seeds and checkpoints), incurring high computational cost.
- The lemma–lift curriculum spans only two stepping-stone levels; longer curriculum chains may be more effective.
- Sharing parameters between teacher and student limits the teacher's problem-proposing capacity; a dedicated teacher model may perform better.

## Related Work & Insights
- **vs. AZR**: GASP augments AZR with goalpost guidance, demonstrating the value of directed exploration. AZR is goal-agnostic; GASP provides a sense of direction.
- **vs. SOAR**: SOAR uses meta-learning to reward the teacher, whereas GASP is simpler — the teacher is not directly rewarded for goalpost improvement; solving goalposts is a byproduct of the curriculum.
- **vs. Standard RLVR**: RLVR relies on a static dataset; GASP automatically generates new training data with directional guidance.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The concept of goal-guided self-play is novel, and the lemma–lift design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comparisons with multiple baselines; goalpost progress analysis is convincing.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear; algorithmic descriptions are detailed.
- Value: ⭐⭐⭐⭐⭐ — Represents a significant advance in self-play training paradigms, breaking through the ceiling of unguided self-play.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play](../../NeurIPS2025/llm_nlp/acesearcher_bootstrapping_reasoning_and_search_for_llms_via_reinforced_self-play.md)
- [\[NeurIPS 2025\] Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs](../../NeurIPS2025/llm_nlp/triplets_better_than_pairs_towards_stable_and_effective_self-play_fine-tuning_fo.md)
- [\[ICLR 2026\] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework](ellmob_event-driven_human_mobility_generation_with_self-aligned_llm_framework.md)
- [\[ICLR 2026\] Enhancing Persona Following at Decoding Time via Dynamic Importance-Guided Token Estimation for Role-Playing Agents](enhancing_persona_following_at_decoding_time_via_dynamic_importance-guided_token.md)
- [\[ICLR 2026\] WebDevJudge: Evaluating (M)LLMs as Critiques for Web Development Quality](webdevjudge_mllm_web_development.md)

<!-- RELATED:END -->
