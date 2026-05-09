---
title: >-
  [Paper Note] Checklists Are Better Than Reward Models For Aligning Language Models
description: >-
  [NeurIPS 2025][Reinforcement Learning][RLCF] This paper proposes Reinforcement Learning from Checklist Feedback (RLCF), which decomposes instructions into dynamically generated yes/no checklists, scores each item using an AI judge and code verifier, and trains with DPO. RLCF consistently improves Qwen2.5-7B-Instruct across 5 benchmarks and is the only method that achieves positive gains on all benchmarks (FollowBench +4pt, InFoBench +6pt, Arena-Hard +3pt).
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - RLCF
  - checklist
  - reward model
  - DPO
  - instruction following
  - alignment
date: 2026-05-08
content_hash: f55c6db7c8a7275d
---

# Checklists Are Better Than Reward Models For Aligning Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2507.18624](https://arxiv.org/abs/2507.18624)
**Code**: Open-sourced (WildChecklists dataset + model + code)
**Area**: Reinforcement Learning
**Keywords**: RLCF, checklist, reward model, DPO, instruction following, alignment

## TL;DR
This paper proposes Reinforcement Learning from Checklist Feedback (RLCF), which decomposes instructions into dynamically generated yes/no checklists, scores each item using an AI judge and code verifier, and trains with DPO. RLCF consistently improves Qwen2.5-7B-Instruct across 5 benchmarks and is the only method that achieves positive gains on all benchmarks (FollowBench +4pt, InFoBench +6pt, Arena-Hard +3pt).

## Background & Motivation

**State of the Field**: Instruction following is a core requirement for practical LLMs. Current post-training pipelines typically adopt SFT + RLHF (using reward model scores for preference optimization), but reward model signals are scalar values that are too coarse for evaluating complex, multi-requirement instructions.

**Limitations of Prior Work**: The three dominant automatic feedback paradigms each have significant drawbacks: (1) verifiable instructions cover only format constraints, ignoring subjective dimensions such as style and content; (2) reward models are flexible but highly arbitrary, prone to reward hacking, and high rankings on RewardBench do not necessarily translate to effective RLHF; (3) using large models as judges requires the judge to infer evaluation criteria on its own, blurring the generator–verifier gap.

**Root Cause**: Both reward models and AI judges evaluate all aspects of all instructions with a single scalar or fixed set of dimensions, whereas the evaluation dimensions required by user instructions are highly dynamic and diverse. For example, "translate into Spanish" and "write an argument containing exactly 3 commas" require entirely different evaluation criteria.

**Paper Goals**: Design an automatic, flexible, intuitive, and general-purpose method for scoring responses—generating instruction-specific, multi-dimensional evaluation criteria (checklists) for each instruction to replace reward models in RL training.

**Starting Point**: Decompose the complex problem of response quality evaluation into a series of simple yes/no questions (checklists), each answered either by an AI judge or by automatically generated verification code. This leverages the principle that a combination of simple judgments outperforms a single complex judgment.

**Core Idea**: Dynamically generate instruction-specific checklists from instructions via a candidate-based method, score each item using an AI judge and code verifier, aggregate weighted scores, and train with DPO for preference optimization.

## Method

### Overall Architecture
The RLCF pipeline proceeds as follows: (1) generate candidate-based checklists from 130K WildChat instructions; (2) sample response pairs from the student model; (3) score each checklist item (AI judge with 25 samples + code verifier); (4) aggregate item scores into an overall score via weighted summation; (5) select the 40% of response pairs with the largest score differences for DPO training.

### Key Designs

1. **Candidate-Based Checklist Generation**:

    - **Function**: Generate high-quality, fine-grained evaluation criteria for each instruction.
    - **Mechanism**: Two-stage pipeline — first, models of varying sizes (0.5B–7B) generate multiple candidate responses of differing quality; then a 72B model analyzes all potential failure modes across these responses and summarizes them into a checklist. Each item is assigned an importance weight on a 0–100 scale.
    - **Design Motivation**: Directly extracting checklists from instructions tends to produce items that merely paraphrase the instruction with insufficient coverage. By observing responses of varying quality, the model can identify more subtle quality distinctions (e.g., formatting, factual accuracy, tone). Experiments confirm that the candidate-based method significantly outperforms the direct method in objectivity (+0.8%) and atomicity (+22%).

2. **Hybrid Scoring: AI Judge + Code Verifier**:

    - **Function**: Reliably evaluate each checklist item.
    - **Mechanism**: For each checklist item, (a) a 72B judge model samples 25 scores on a 0–100 scale and averages them, and (b) the model simultaneously determines whether the item can be precisely verified by code (e.g., "does the response contain exactly 3 commas?"); if so, a Python verification function is generated and its score is averaged with the judge score.
    - **Design Motivation**: LLMs are unreliable at judging hard constraints (e.g., counting character occurrences), whereas code verifiers handle these perfectly. Conversely, code cannot evaluate soft constraints (e.g., "is this coherent?"), necessitating an AI judge. The case analysis in Table 8 clearly illustrates the complementary nature of both approaches.

3. **Universal Requirements (Anti-Reward-Hacking)**:

    - **Function**: Prevent models from gaming checklist scores by padding responses with redundant content.
    - **Mechanism**: Two universal requirements are appended to all checklists: (1) whether the response directly addresses the request without excessive or off-topic information, and (2) whether the response matches the tone and style required by the instruction.
    - **Design Motivation**: Initial experiments revealed that models learned to prepend lengthy summaries to responses in order to inflate checklist scores, a behavior analogous to reward hacking.

### Loss & Training
- DPO training; batch size 1024; max sequence length 2048; cosine learning rate schedule (max 3e-6, min 2e-6); 2 epochs.
- 130K instructions from WildChat; 40% of response pairs with the largest score differences are selected.
- Training takes approximately 3 hours on 8×H100; scoring takes approximately 4 days (25 samples/item).

## Key Experimental Results

### Main Results (Qwen2.5-7B-Instruct)

| Method | IFEval (Avg) | InFoBench | FollowBench HSR | Arena-Hard | AlpacaEval LC |
|---|---|---|---|---|---|
| Baseline | 77.3 | 78.1 | 71.4 | 42.8 | 36.2 |
| + DPO (Skywork RM) | 76.0 | 82.0 | 69.5 | 50.3 | 41.5 |
| + DPO (ArmoRM) | 76.0 | 83.5 | 70.4 | 46.4 | 38.1 |
| + DPO (Ultrafeedback) | 74.6 | 80.0 | 72.6 | 47.9 | 38.7 |
| + DPO (AI Judge) | 75.2 | 76.1 | 70.3 | 44.4 | 33.4 |
| **+ DPO (RLCF)** | **78.6** | **84.1** | **75.3** | **48.4** | **37.1** |

RLCF is the only method that achieves positive improvements on all 5 benchmarks. RM-based methods improve on some benchmarks but regress on others (e.g., Skywork regresses on IFEval and FollowBench).

### Ablation Study

| Configuration | Effect | Notes |
|---|---|---|
| Direct vs. candidate-based checklist | Candidate-based improves IFEval/FollowBench by 2–3% | More objective and atomic checklists yield better training signal |
| Remove code verifier | Slight drop on FollowBench but improvement on Arena-Hard | Code verifier helps with hard constraints but is not decisive |
| Remove prompt-based scoring | Performance drops on format-type constraints | AI judge and code verifier are complementary |
| Reduce sampling (25→5) | IFEval/InFoBench largely unchanged; FollowBench content category drops | 5 samples saves 55% of time — a practical trade-off |
| Off-policy (Llama/OLMo) | Consistent improvements, no regression | Checklists capture general criteria, not model-family-specific patterns |

### Key Findings
- **RewardBench accuracy ≠ RLHF effectiveness**: Skywork-27B far outperforms the checklist-based reward on RewardBench (96.1 vs. 90.0 Chat), yet checklists yield better results in practice. This finding aligns with Malik et al. 2025 and Razin et al. 2025.
- **RLCF most benefits "content" constraints**: Per-category analysis on FollowBench shows that RLCF achieves the largest gains on content constraints that restrict the answer space (+6.4%), rather than format constraints. This indicates that checklists guide models to attend to the full semantic intent of instructions.
- **Strong generalization but domain bias**: Since WildChat is dominated by everyday conversational instructions, slight regressions are observed on mathematics (GSM8K −1%) and factual accuracy (TruthfulQA −1.5%). However, expanding the prompt distribution is substantially easier than retraining a reward model.

## Highlights & Insights
- **"Extreme mixture-of-evaluators" perspective**: RLCF can be understood as an infinitely large mixture-of-evaluators, where each instruction dynamically selects a dedicated set of evaluators. This provides exponentially richer feedback signals compared to fixed four-dimensional evaluation (UltraFeedback) or scalar reward models.
- **The reward model paradox**: Reward models that score higher on RewardBench perform worse in actual RLHF, because the "good response" patterns learned by the RM may not match the specific requirements of individual instructions. Checklists circumvent this problem — rather than learning what constitutes "good," they directly verify specific conditions.
- **Selective use of code verifiers**: Code is generated only when the model is fully confident that a checklist item can be precisely verified; otherwise, evaluation is delegated to the AI judge. This adaptive combination is a practical and elegant design choice.

## Limitations & Future Work
- **High scoring cost**: Scoring 130K instructions requires 4 days on 8×H100 (72B model, 25 samples/item), though reducing to 5 samples saves 55% of computation time.
- Only DPO (off-policy) is explored; on-policy methods such as GRPO are not attempted, which may better exploit checklist feedback.
- The approach relies on a 72B teacher model as judge, constituting strong-to-weak generalization.
- Training data is biased toward everyday conversational instructions, with insufficient coverage of safety and mathematical reasoning domains.

## Related Work & Insights
- **vs. UltraFeedback**: UltraFeedback evaluates all responses along 4 fixed dimensions (instruction following, helpfulness, truthfulness, honesty). RLCF uses dynamic, instruction-specific dimensions, which experiments demonstrate to be significantly more effective.
- **vs. AutoIF / IFBench**: These works synthesize instructions with verifiable constraints to train models. RLCF does not create new instructions; instead, it extracts evaluation dimensions from existing natural instructions, yielding stronger generalization.
- **vs. SPARKLE (2506.04723)**: SPARKLE finds that RL enhances knowledge integration. RLCF reveals a more fundamental issue with RL feedback signal quality — effective feedback need not come from a better reward model, but from a more structured evaluation approach.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Redefining the reward signal from a scalar to an instruction-specific checklist represents an important paradigm shift; the candidate-based generation method is clever and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five benchmarks, six baselines, cross-model-family validation, RewardBench comparison, and multiple ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Figure 1 and Tables 2–4 clearly demonstrate that only RLCF achieves consistent positive gains; the case analysis in Table 8 is highly persuasive.
- **Value**: ⭐⭐⭐⭐⭐ Exposes fundamental limitations of reward models in RLHF, provides a viable alternative, and the WildChecklists dataset offers direct practical value.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Training Language Models to Reason Efficiently](training_language_models_to_reason_efficiently.md)
- [\[NeurIPS 2025\] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models](incentivizing_reasoning_for_advanced_instruction-following_of_large_language_mod.md)
- [\[NeurIPS 2025\] Behavior Injection: Preparing Language Models for Reinforcement Learning](behavior_injection_preparing_language_models_for_reinforcement_learning.md)
- [\[NeurIPS 2025\] When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners](when_less_language_is_more_language-reasoning_disentanglement_makes_llms_better_.md)
- [\[ICLR 2026\] ParaS2S: Benchmarking and Aligning Spoken Language Models for Paralinguistic-Aware Speech-to-Speech Interaction](../../ICLR2026/reinforcement_learning/paras2s_benchmarking_and_aligning_spoken_language_models_for_paralinguistic-awar.md)

<!-- RELATED:END -->
