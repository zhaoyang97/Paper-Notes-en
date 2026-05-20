---
title: >-
  [Paper Note] A Course Correction in Steerability Evaluation: Revealing Miscalibration and Side Effects in LLMs
description: >-
  [AAAI 2026][Reinforcement Learning][steerability evaluation] This paper proposes a multi-dimensional objective-space framework for evaluating LLM steerability…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "steerability evaluation"
  - "side effects"
  - "miscalibration"
  - "RL fine-tuning"
  - "text rewriting"
date: 2026-05-08
content_hash: cb5515d72734388d
---

# A Course Correction in Steerability Evaluation: Revealing Miscalibration and Side Effects in LLMs

**Conference**: AAAI 2026
**arXiv**: [2505.23816](https://arxiv.org/abs/2505.23816)  
**Code**: [https://github.com/MLD3/steerability](https://github.com/MLD3/steerability)  
**Area**: Reinforcement Learning
**Keywords**: steerability evaluation, side effects, miscalibration, RL fine-tuning, text rewriting

## TL;DR
This paper proposes a multi-dimensional objective-space framework for evaluating LLM steerability, decomposing steering error into miscalibration and side effects (orthogonality). Experiments on text rewriting reveal that even the strongest LLMs produce severe side effects; prompt engineering proves ineffective, best-of-N sampling is prohibitively costly, and RL fine-tuning yields improvements but does not fully resolve the problem.

## Background & Motivation
LLMs continue to advance on reasoning and instruction-following benchmarks, yet such progress does not imply that models can reliably satisfy diverse, fine-grained user objectives — i.e., *steerability*. Current LLM evaluation suffers from two fundamental flaws: (1) many benchmarks draw data from real chat logs or internet text, which naturally skews toward common requests (e.g., "more formal and longer" is frequent, whereas "more formal and shorter" is rare), failing to cover the objective space uniformly; (2) most evaluations rely on scalar metrics (e.g., binary accuracy, ranking accuracy) that cannot capture multi-dimensional behavioral changes in open-ended generation — a model may satisfy the target dimension while silently altering other dimensions that should remain unchanged (side effects), and scalar metrics are entirely blind to such hidden behavioral shifts.

## Core Problem
When a user issues a multi-dimensional rewriting instruction to an LLM (e.g., "make this text harder to read and slightly longer"), can the model precisely modify only the requested dimensions without inadvertently altering others (e.g., formality, lexical diversity)? The central questions this paper addresses are: how to quantify and decompose this multi-dimensional *steering error*, and to what extent existing interventions (prompt engineering, sampling, fine-tuning) can mitigate these side effects.

## Method

### Overall Architecture
The input is a pair (source text $z_0$, target intent $z^*$); the output is a vector $\hat{z}$ obtained by mapping the LLM's rewritten text into the objective space. The core idea is to map both the user's goal and the model's output into a shared multi-dimensional objective space $\mathcal{Z} = [0,1]^{|G|}$, where each dimension corresponds to a measurable textual attribute. Steering error is then measured as the Euclidean distance between $\hat{z}$ and $z^*$, and is further orthogonally decomposed into two components: error along the direction of the user's intent (miscalibration, i.e., overshoot/undershoot) and error perpendicular to the intent direction (orthogonality, i.e., side effects). Target pairs $(z_0, z^*)$ are sampled from a uniform distribution during evaluation to avoid bias toward common requests.

### Key Designs
1. **Multi-dimensional objective space and metric functions**: Four rule-based, verifiable textual attributes are selected as objective dimensions — reading difficulty (Flesch–Kincaid grade level), formality (Heylighen–Dewaele F-score), lexical diversity (MTLD), and text length (word count). Each dimension has a deterministic metric function, making evaluation results auditable and interpretable while avoiding additional noise from learned evaluators. All dimensions are normalized to $[0,1]$ via linear mapping over the central 95% range of seed data, with clipping applied.

2. **Steerability probe construction**: A total of 8,303 seed texts are sampled from four stylistically diverse datasets (CNN/DailyMail news, RedditTIFU social media, BookSum English novels, SummScreenFD movie synopses), with density-ratio-based resampling to approximate a uniform distribution over the objective space. The primary probe set comprises 64 source texts × 32 targets = 2,048 samples. For each source text, 3 active dimensions are randomly selected and shift magnitudes are sampled in the range $[\pm 0.1, 0.7]$. Prompts follow a templated design, adding "slightly" for changes $< 0.2$ and "much" for changes $> 0.5$.

3. **Error decomposition — Miscalibration and Orthogonality**: The steering error vector is orthogonally decomposed into components along and perpendicular to the user's intent direction. Miscalibration measures overshoot/undershoot along the intent direction, normalized by the requested displacement; orthogonality measures deviation away from the intent direction, normalized by the actual displacement. Both metrics are non-negative with an optimal value of zero. This decomposition cleanly distinguishes between "correct direction but wrong magnitude" and "unintended changes along other dimensions."

### Loss & Training
RL fine-tuning employs MA-LOOP (Margin-Aware Leave-One-Out Policy Optimization), a variant of GRPO using the negative steering error as the reward. Key design choices include: (1) density-ratio resampling weights $w(z_0)$ to approximate a uniform training distribution; (2) IPO-style margin-aware regularization ensuring that the log-likelihood difference between preferred and non-preferred responses is proportional to the true reward gap; (3) per-token normalized loss to eliminate length bias. The base model is Llama3.1-8B with rank-stabilized LoRA (rank 256), trained in a 2D objective space (reading difficulty × formality).

## Key Experimental Results

| Setting | Steering Error | Miscalibration | Orthogonality |
|---------|---------------|----------------|---------------|
| Base model (pre-RL) | 0.300 | 0.986 | 0.147 |
| Best@128 (pre-RL) | 0.210 | 0.683 | 0.121 |
| Miscal-only reward | 0.210 | 0.542 | 0.366 |
| Ortho-only reward | 0.386 | 1.463 | 0.025 |
| Full steering error (post-RL) | **0.119** | **0.294** | 0.160 |

Additional key findings:
- Llama3.3-70B achieves a median steering error of 0.452 on the 4D evaluation (far above the ideal value of 0), with orthogonality = 0.718 (approaching 1, indicating severe side effects).
- Increasing model scale substantially reduces miscalibration (Llama3.1-8B: 0.667 → 70B: 0.455) but leaves orthogonality nearly unchanged.
- Steering error for anti-correlated requests (e.g., "harder to read but less formal") is significantly higher than for correlated requests (0.535 vs. 0.404).
- Post-RL, BLEU between source and rewritten text drops from 0.864 to 0.529, indicating the model no longer defaults to conservative copying.
- RL reduces the orthogonality gap between correlated and anti-correlated requests from 0.114 to 0.005.

### Ablation Study
- Optimizing only miscalibration causes orthogonality to deteriorate sharply (0.147 → 0.366); conversely, optimizing only orthogonality causes miscalibration to spike (0.986 → 1.463), demonstrating the necessity of joint optimization.
- Various prompt engineering strategies (negative prompting, chain-of-thought, added instructions) yield modest improvements in miscalibration but are largely ineffective against orthogonality.
- Best-of-N improves with increasing $N$ but extremely slowly (each doubling of $N$ reduces median steering error by at most 0.031).
- Results with uniformly resampled vs. naively sampled probes are similar, indicating that steerability failures are not confined to rare target configurations.

## Highlights & Insights
- **Elegant error decomposition**: Orthogonally decomposing total steering error into miscalibration and orthogonality (side effects) is geometrically intuitive and conceptually clean, enabling precise diagnosis of model failure modes.
- **Discovery of "dimension entanglement"**: LLMs internally couple reading difficulty and formality — requests to increase difficulty automatically increase formality. Mixed-effects model analysis confirms this coupling originates from the LLM itself rather than statistical correlations in the input data.
- **RL learns genuinely new rewriting strategies**: Rather than simple numerical optimization, the fine-tuned model learns to compensate for polysyllabic words with shorter sentences to achieve target reading difficulty while decoupling dimensions (as demonstrated in the case study in Table 9).
- **Modular framework design**: Both the objective dimensions and their metric functions are interchangeable, making the framework theoretically extensible to arbitrary textual attributes or even multimodal settings.

## Limitations & Future Work
- The framework currently focuses exclusively on rule-based, verifiable attributes (reading difficulty, formality, etc.) and does not address more subjective objective dimensions such as style or tone.
- Only single-turn settings are evaluated; real user interactions are multi-turn.
- RL experiments are limited to an 8B model and a 2D objective space; larger models and higher-dimensional spaces may introduce new challenges (e.g., combinatorial explosion of dimension interactions).
- Prompt format variations (e.g., few-shot prompting) are not evaluated.
- The framework assumes all objective dimensions are independent and all targets $z^*$ are reachable — in practice, fundamental trade-offs may exist among dimensions.
- As an evaluation framework, the paper's contribution is primarily diagnostic rather than prescriptive; the proposed solution (RL fine-tuning) offers only partial remediation.

## Related Work & Insights
- Compared with **Vafa et al. (2025)**: Vafa et al. focus on the distinction between what a model can *produce* (producible) and what it can be *guided toward* (reachable); this paper goes further by proposing a concrete multi-dimensional evaluation framework with explicit error decomposition.
- Compared with **AxBench (Wu et al., 2025)**: AxBench addresses quantitative evaluation of activation steering using concept detection scores, but operates with 1D metrics; this paper emphasizes multi-dimensionality and the explicit capture of side effects.
- Compared with **Miehling et al. (2025)**: Miehling et al. also study prompt steerability but employ a more constrained evaluation setup (questionnaire-based); this paper adopts a continuous, multi-dimensional open-ended generation setting.
- The multi-dimensional objective space combined with orthogonal decomposition can be transferred to controllable generation evaluation in VLMs (e.g., controlling multiple attributes in image generation).
- The side effect/dimension entanglement problem is pervasive in RLHF — optimizing helpfulness may degrade safety; this paper's quantitative framework offers a transferable methodology.
- The trade-off between "jointly optimizing multiple dimensions" and "optimizing each dimension independently" in RL is a direction worthy of further exploration.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The framework design is original (objective-space mapping + orthogonal decomposition), though the core ideas (multi-dimensional evaluation, side effect detection) are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple model families (GPT/Llama/DeepSeek/Qwen/o1–o3), diverse intervention strategies, complete ablation studies, and an exceptionally detailed appendix.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Narrative is fluent, figures and tables are polished, mathematical derivations are clear, and case analyses are persuasive.
- **Value**: ⭐⭐⭐⭐ Offers important insights for the LLM alignment community — side effects are a severely underestimated problem — though practical solutions remain a work in progress.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Revealing POMDPs: Qualitative and Quantitative Analysis for Parity Objectives](revealing_pomdps_qualitative_and_quantitative_analysis_for_parity_objectives.md)
- [\[AAAI 2026\] Does Self-Evaluation Enable Wireheading in Language Models?](does_self-evaluation_enable_wireheading_in_language_models.md)
- [\[ICLR 2026\] Nearly-Optimal Bandit Learning in Stackelberg Games with Side Information](../../ICLR2026/reinforcement_learning/nearly-optimal_bandit_learning_in_stackelberg_games_with_side_information.md)
- [\[AAAI 2026\] A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)
- [\[NeurIPS 2025\] Learning to Clean: Reinforcement Learning for Noisy Label Correction](../../NeurIPS2025/reinforcement_learning/learning_to_clean_reinforcement_learning_for_noisy_label_correction.md)

</div>

<!-- RELATED:END -->
