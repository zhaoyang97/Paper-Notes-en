---
title: >-
  [Paper Note] RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning
description: >-
  [NeurIPS 2025][Reinforcement Learning][generator-verifier co-training] Tango proposes a framework that alternately trains a generator and a verifier via RL — the verifier is a generative process-level LLM that evaluates reasoning step by step in natural language, trained solely with outcome-level correctness rewards (no step-level annotations), and mutually reinforced through co-evolution with the generator. On 7B/8B-scale models, Tango achieves SOTA, with a 100% relative improvement over vanilla GRPO on AIME 2025.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - generator-verifier co-training
  - process reward
  - co-evolution
  - generative verifier
date: 2026-05-08
content_hash: 172a3104ae68a0e2
---

# RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2505.15034](https://arxiv.org/abs/2505.15034)
**Code**: [github.com/kaiwenzha/rl-tango](https://github.com/kaiwenzha/rl-tango)
**Area**: Reinforcement Learning
**Keywords**: generator-verifier co-training, process reward, reinforcement learning, co-evolution, generative verifier

## TL;DR
Tango proposes a framework that alternately trains a generator and a verifier via RL — the verifier is a generative process-level LLM that evaluates reasoning step by step in natural language, trained solely with outcome-level correctness rewards (no step-level annotations), and mutually reinforced through co-evolution with the generator. On 7B/8B-scale models, Tango achieves SOTA, with a 100% relative improvement over vanilla GRPO on AIME 2025.

## Background & Motivation

**Background**: In RL post-training for LLM reasoning, the generator serves as the policy while the verifier (reward model) provides feedback. However, verifiers in existing methods are typically fixed (rule-based or frozen pretrained models) or trained discriminatively via SFT.

**Limitations of Prior Work**: (1) Fixed verifiers are prone to reward hacking — reward signals become unreliable as the generator's distribution shifts; (2) Discriminatively trained PRMs via SFT produce deterministic rewards that are brittle and generalize poorly; (3) Although PRIME jointly trains the PRM, it remains logit-based and SFT-trained, failing to fundamentally address these issues.

**Key Challenge**: Effective co-evolution requires both the generator and verifier to be sufficiently capable — a weak or static verifier imposes an upper bound on generator improvement.

**Goal**: How can the generator and verifier truly co-evolve and mutually reinforce each other?

**Key Insight**: (1) Train the verifier with RL (rather than SFT) to achieve stronger reasoning capability; (2) Make the verifier generative (outputting natural language judgments rather than implicit logits), introducing stochasticity to resist reward hacking; (3) Alternate training to realize co-evolution.

**Core Idea**: Train a generative process-level verifier with RL and alternate it with the generator in a co-evolutionary loop — a more accurate verifier leads to better generator reasoning, which produces more diverse outputs, which in turn enables the verifier to learn stronger generalization.

## Method

### Overall Architecture
Alternating training: every $N_g=3$ generator update steps, $N_v=1$ verifier update step is performed. The generator receives both outcome-level and step-level advantage signals; the verifier is trained with only outcome-level correctness rewards. The verifier undergoes a 40-step warm-up phase to learn the output format.

### Key Designs

1. **Generative Process-Level Verifier**:

    - Function: Evaluates the generator's reasoning process step by step in natural language.
    - Mechanism: Given a question $\mathbf{x}$ and a generator response $\mathbf{o}^g$, the verifier produces a judgment $\mathbf{o}^v$ containing $K$ step-level verdicts $y_{\text{step},k} \in \{-1, 1\}$ and a final verdict $y_{\text{final}} \in \{0, 1\}$.
    - Design Motivation: (1) Generative outputs (vs. discriminative) introduce sampling stochasticity, reducing reward hacking; (2) Natural language reasoning-based judgments (vs. logit scores) enable deeper reasoning in the verifier; (3) Text outputs are interpretable.

2. **Two-Level Advantage Fusion**:

    - Function: Allows the generator to leverage both outcome-level and step-level advantage signals simultaneously.
    - Mechanism: Advantages are computed at each level independently and then fused via weighted combination: $\hat{A}_{g,t}^i = (1-\alpha)\hat{A}_{g,\text{out},t}^i + \alpha\hat{A}_{g,\text{step},t}^i$, where step-level advantages are computed as cumulative rewards of subsequent steps as token-level advantages.
    - Design Motivation: Independent normalization before fusion (rather than merging rewards first) avoids training instability caused by scale mismatch. The weight $\alpha$ follows exponential decay: emphasizing step-level exploration early in training and shifting toward outcome-level stability for convergence.

3. **Category-Aware Reweighting (Verifier Training)**:

    - Function: Addresses class imbalance during the early stages of verifier training.
    - Mechanism: In early training, the generator produces mostly incorrect responses, causing the verifier to be predominantly exposed to negative samples and collapse into always predicting "incorrect." Class-aware scaling factors $s^+, s^-$ (inversely proportional to the square root of the number of positive/negative samples) are applied to reweight the advantages, amplifying the minority-class signal.
    - Design Motivation: EMA-smoothed updates of the balancing coefficients ensure stable training throughout.

### Loss & Training
Generator: standard GRPO/RLOO/REINFORCE++ with two-level advantages. Verifier: GRPO with category-aware reweighting, outcome-level correctness reward, and format reward. Base models: Qwen2.5-Math-7B (generator) + Qwen2.5-7B (verifier).

## Key Experimental Results

### Main Results

**Tango vs. Vanilla RL (GRPO, 200 steps)**

| Method | MATH500 | AIME24 | AIME25 | AMC23 | Math Avg. | OOD Avg. |
|--------|---------|--------|--------|-------|-----------|----------|
| SFT only | 66.6 | 3.3 | 3.3 | 27.5 | 25.8 | 52.8 |
| GRPO | 74.6 | 13.3 | 10.0 | 50.0 | 37.0 | 57.6 |
| **GRPO + Tango** | **81.4** | **20.0** | **20.0** | **65.0** | **46.1** | **61.1** |

**SOTA Comparison (7B/8B scale)**

| Model | AIME24 | AIME25 | AMC23 | Math Avg. |
|-------|--------|--------|-------|-----------|
| PRIME-7B | 26.7 | 13.3 | 60.0 | 44.8 |
| rStar-Math-7B | 26.7 | - | 47.5 | - |
| **Tango-Qwen-7B** | **26.7** | **23.3** | **70.0** | **49.5** |

### Ablation Study

| Configuration | Generator Accuracy | Verifier Step F1 | Notes |
|---------------|--------------------|------------------|-------|
| Tango (full) | **Highest** | **Continuously improves** | Co-evolution |
| Fixed generator | — | Improves then plateaus | Stagnates on static distribution |
| Fixed verifier | Stagnates for first 20 steps | No change | Incorrect feedback impedes early learning |

### Key Findings
- **Doubled performance on AIME 2025**: GRPO + Tango (20.0%) vs. vanilla GRPO (10.0%) — largest gains on the most challenging competition mathematics.
- **3.3× training efficiency**: Tango reaches at 60 steps what vanilla GRPO achieves at 200 steps.
- **Verifier SOTA on ProcessBench**: Outperforms 72B models on the hardest subsets of OlympiadBench and Omni-MATH (trained from a 7B base).
- **No step-level annotations required**: The verifier is trained with only outcome-level rewards yet progressively learns accurate step-level judgments.
- **Cross-model generalization**: Consistently effective on Llama-3.1-8B.

## Highlights & Insights
- **Training the verifier with RL instead of SFT is a paradigm innovation**: This corresponds to the insight that "SFT memorizes, RL generalizes" — an RL-trained verifier generalizes better and reasons more robustly.
- **The co-evolutionary positive feedback loop is elegantly designed**: More diverse generator outputs → more robust verifier judgments → more accurate reward signals → stronger generator reasoning.
- **Category-aware reweighting addresses a practical training collapse issue**: This is an important engineering finding with significant practical implications.

## Limitations & Future Work
- The verifier requires additional training computation (approximately 1/3 of training steps are dedicated to verifier updates).
- The verifier warm-up phase (40 steps) requires careful hyperparameter tuning.
- The decay schedule for $\alpha$ has a significant impact on results.
- The verifier has not been evaluated for standalone use in Best-of-N inference-time scaling.

## Related Work & Insights
- **vs. PRIME**: PRIME also performs joint training but uses SFT + logit-based rewards for the verifier; Tango uses RL + generative outputs. Tango consistently outperforms PRIME across all benchmarks.
- **vs. GenPRM / Think-PRM**: These are SFT-trained generative PRMs; Tango is the first to train a generative PRM with RL and co-evolve it with the generator.
- **vs. vanilla GRPO/RLOO**: Tango yields 25%+ improvements in mathematical reasoning and 7%+ OOD improvements across all three RL algorithms evaluated.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to train a generative process verifier with RL and co-evolve it with the generator; the design is complete and well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 math benchmarks + 4 OOD benchmarks + ProcessBench + 3 RL algorithms + ablations + Llama validation.
- Writing Quality: ⭐⭐⭐⭐ Method is clearly presented; Figure 1 conveys the core idea at a glance.
- Value: ⭐⭐⭐⭐⭐ 7B-scale SOTA, open-source code, and a significant contribution to the direction of joint RL + verifier training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation](noisyrollout_reinforcing_visual_reasoning_with_data_augmenta.md)
- [\[NeurIPS 2025\] When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners](when_less_language_is_more_language-reasoning_disentanglement_makes_llms_better_.md)
- [\[NeurIPS 2025\] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models](incentivizing_reasoning_for_advanced_instruction-following_of_large_language_mod.md)
- [\[NeurIPS 2025\] SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](swe-rl_advancing_llm_reasoning_via_reinforcement_learning_on_open_software_evolu.md)
- [\[ICLR 2026\] AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking](../../ICLR2026/reinforcement_learning/abstral_augmenting_llms_reasoning_by_reinforcing_abstract_thinking.md)

</div>

<!-- RELATED:END -->
