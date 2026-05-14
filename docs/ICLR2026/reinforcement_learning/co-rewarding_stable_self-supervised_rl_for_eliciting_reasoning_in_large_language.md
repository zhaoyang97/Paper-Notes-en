---
title: >-
  [Paper Note] Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models
description: >-
  [ICLR 2026][Reinforcement Learning][Self-supervised RL] Co-rewarding proposes a self-supervised RL framework that addresses training collapse in self-rewarding RL through two complementary supervision perspectives: a dat…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Self-supervised RL"
  - "Label-free Reasoning"
  - "Training Collapse"
  - "GRPO"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: c430d59987f21df2
---

# Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2508.00410](https://arxiv.org/abs/2508.00410)
**Code**: [https://github.com/tmlr-group/Co-rewarding](https://github.com/tmlr-group/Co-rewarding)
**Area**: Reinforcement Learning
**Keywords**: Self-supervised RL, Label-free Reasoning, Training Collapse, GRPO, Contrastive Learning

## TL;DR
Co-rewarding proposes a self-supervised RL framework that addresses training collapse in self-rewarding RL through two complementary supervision perspectives: a data-side mechanism (cross-view consistency via contrastive paraphrased questions) and a model-side mechanism (EMA teacher model providing pseudo-labels). Without any human annotations, the framework matches or surpasses RLVR (with ground-truth labels) across multiple mathematical reasoning benchmarks.

## Background & Motivation

**Background**: RLVR (Reinforcement Learning with Verifiable Rewards, e.g., GRPO in DeepSeek-R1) is the dominant approach for enhancing LLM reasoning, but relies on human-annotated ground-truth answers as reward signals.

**Limitations of Prior Work**:
   - Ground-truth annotation is costly and difficult to scale, especially for complex tasks
   - Self-rewarding methods (self-certainty, entropy-based, majority voting) can substitute GT labels but frequently suffer from **training collapse**
   - Root cause of collapse: reward signals derived from the model's own single-perspective output form a "self-consistency illusion," enabling reward hacking

**Key Challenge**: Self-supervised signals are entangled with the current policy — the model achieves high rewards by minimizing entropy or maximizing consistency, but converges to trivial solutions (repeated strings, consistent yet incorrect answers).

**Goal**:
   - How to achieve stable RL training without GT labels?
   - How to break the self-consistency illusion arising from a single perspective?
   - Can the performance of GT-label-based RLVR be matched?

**Key Insight**: Inspired by self-supervised learning (SimCLR, BYOL, DINO) — genuine reasoning ability should manifest as invariance across views/time, rather than certainty of a single output.

**Core Idea**: Introduce complementary supervisory perspectives via data-side "paraphrase cross-validation" and model-side "EMA teacher pseudo-labels," increasing the difficulty of reward hacking and thereby preventing training collapse.

## Method

### Overall Architecture
Built upon GRPO optimization, the core innovation lies in modifying the reward source within advantage estimation: instead of GT labels or the model's own single-view output, pseudo-labels from an alternative perspective are introduced as cross-references. Three instantiations are proposed: Variant I (data-side), Variant II (model-side), and Variant III (combined).

### Key Designs

1. **Co-rewarding-I (Data-side: Contrastive Consistency)**:

    - **Function**: Generates cross pseudo-labels for the original question and its semantically paraphrased version.
    - **Mechanism**: Given original question $x$ and its paraphrase $x'$, $G$ rollouts are sampled from the policy model for each. Majority voting over rollouts of $x$ yields pseudo-label $y_v$, and similarly $y_v'$ for $x'$. These labels are then **cross-applied**: $y_v'$ evaluates rollouts of $x$, and $y_v$ evaluates rollouts of $x'$. Advantage is computed as $\hat{A}_i = \frac{r(y_v', y_i) - \text{mean}(...)}{\text{std}(...)}$.
    - **Design Motivation**: Semantically equivalent paraphrases should yield identical answers (analogy-invariance). Cross-validation makes it difficult for the model to obtain high rewards by producing consistent but incorrect answers on a single input, since the paraphrase outputs "audit" the original answers.

2. **Co-rewarding-II (Model-side: Self-distillation)**:

    - **Function**: Provides pseudo-labels via an EMA-updated teacher model.
    - **Mechanism**: Maintains an EMA teacher $\tilde{\pi}_{ref}^{(k)} \leftarrow \alpha^{(k)} \tilde{\pi}_{ref}^{(k-1)} + (1-\alpha^{(k)}) \pi_{\theta_{old}}^{(k)}$, with the EMA coefficient annealed via cosine scheduling from $\alpha_{start}$ to $\alpha_{end}$. The teacher generates rollouts and produces pseudo-labels $\tilde{y}_v$ via majority voting, which are then used to evaluate the policy model's rollouts.
    - **Design Motivation**: The EMA-updated teacher is "temporally decoupled" from the online policy — its pseudo-labels are not immediately affected by policy updates, breaking the instantaneous feedback loop of self-rewarding. This is analogous to the momentum teacher in BYOL/DINO.

3. **Co-rewarding-III (Combined)**:

    - **Function**: Simultaneously employs paraphrased questions and the EMA teacher for full decoupling.
    - **Mechanism**: The teacher model generates rollouts on paraphrased questions → produces pseudo-labels → supervises the policy model's rollouts on original questions (and vice versa). Complementary perspectives are introduced along both data and model dimensions.
    - **Design Motivation**: Data-side and model-side perspectives are complementary — Variant I addresses single-perspective data issues, while Variant II addresses entanglement between supervision and policy.

### Loss & Training
- Based on GRPO: $\mathcal{J}(\theta) = \text{clipped surrogate objective} - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$
- Reward function: Binary (correct = 1, incorrect = 0), evaluated using cross pseudo-labels
- EMA teacher update: Cosine annealing schedule — faster updates early (tracking policy improvement), slower updates later (stabilizing supervision)
- Paraphrase generation: Semantically equivalent paraphrases of original math problems are generated using an LLM

## Key Experimental Results

### Main Results (MATH training set, Qwen3-8B-Base)

| Method | MATH500 | GSM8K | AMC | IFEval | MMLU-Pro |
|--------|---------|-------|-----|--------|----------|
| Before RL | 72.4 | 27.8 | 20.9 | 50.9 | 52.9 |
| GT-Reward (RLVR) | 82.6 | 87.3 | 54.2 | 52.8 | 57.1 |
| Self-Certainty | 80.2 | 80.7 | 50.8 | 51.0 | 54.2 |
| Majority-Voting | 79.8 | 89.8 | 49.1 | 51.8 | 56.9 |
| **Co-rewarding-I** | 81.2 | **93.7** | 51.2 | 55.8 | **60.0** |
| **Co-rewarding-II** | 80.8 | 92.4 | 53.5 | **60.7** | 57.5 |
| **Co-rewarding-III** | **81.4** | 91.0 | **54.1** | 53.7 | 59.1 |

### Ablation Study

| Configuration | Training Collapse? | Avg. Math Reasoning Gain |
|--------------|-------------------|--------------------------|
| Self-Certainty | Frequent | +3% |
| Entropy | Occasional | +2% |
| Majority-Voting | Occasional | +4% |
| **Co-rewarding** | **None** | **+7.49% (Llama-3.2-3B)** |

### Key Findings
- **GSM8K 94.01%**: Co-rewarding achieves 94.01% Pass@1 on GSM8K with Qwen3-8B, surpassing GT-label RLVR (87.26%) — the label-free approach outperforms the labeled one.
- **Training Stability**: All self-rewarding baselines exhibit collapse during training (sudden spikes in validation loss), while Co-rewarding training curves remain consistently stable.
- **Average Gain of +3.31%**: Across multiple mathematical reasoning benchmarks, Co-rewarding outperforms the best self-rewarding baseline by an average of 3.31%, reaching +7.49% on Llama-3.2-3B.
- **Cross-task Transfer**: Trained solely on MATH, the method yields significant improvements on Code (LiveCodeBench) and Instruction Following (IFEval).
- **Complementary Variant Strengths**: Variant I is strongest on GSM8K, Variant II on IFEval, and Variant III is the most balanced overall.

## Highlights & Insights
- **Elegant Transfer of Self-supervised Learning Philosophy**: The conceptual transfer of SimCLR's "dual-view consistency" and BYOL/DINO's "momentum teacher" to LLM RL training is clear and precise. This suggests a broader methodological principle: successful paradigms from self-supervised learning can be systematically transferred to RL.
- **Label-free Outperforming Labeled Training**: The phenomenon of 94.01% vs. 87.26% (GT-Reward) on GSM8K warrants attention. A plausible explanation is that self-supervised signals encourage more diverse exploration, while binary GT rewards may over-constrain the policy.
- **EMA Teacher + Paraphrase Cross-validation is Practically Scalable**: Computational overhead is manageable (EMA requires no additional optimizer), paraphrases can be generated offline, and the overall approach scales more readily to unlabeled data than RLVR.

## Limitations & Future Work
- Paraphrase quality affects the effectiveness of Co-rewarding-I, requiring high-quality paraphrase models.
- EMA teacher hyperparameters ($\alpha_{start}, \alpha_{end}$) require tuning.
- Validation is limited to mathematical reasoning; effectiveness on NLP reasoning, code generation, and other domains remains to be explored.
- Co-rewarding-III requires maintaining both the EMA teacher and paraphrase data, incurring higher memory overhead.
- Theoretical analysis is insufficient — formal guarantees explaining why cross-validation prevents collapse are lacking.

## Related Work & Insights
- **vs. Self-Certainty (Zhao et al.)**: Single-perspective certainty signals are prone to collapse; Co-rewarding introduces multi-perspective signals for stability.
- **vs. RLVR (GT-Reward)**: RLVR's reliance on human annotation limits scalability; Co-rewarding is label-free yet matches or exceeds RLVR in multiple settings.
- **vs. Majority-Voting (Shafayat et al.)**: Both use majority voting, but the latter applies it only within a single question — still a single-perspective approach; Co-rewarding introduces genuinely complementary perspectives via paraphrases or the teacher model.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The conceptual transfer from self-supervised learning to RL is elegant, though the individual techniques (paraphrasing + majority voting + EMA) are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple model families (Qwen3/Llama), multiple benchmarks, training stability visualizations, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Concepts are clearly articulated and the progressive relationship among the three variants is well-structured, though the notation is dense.
- **Value**: ⭐⭐⭐⭐⭐ Training stability in label-free RL is a practical pain point; this paper presents a viable solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[ICLR 2026\] AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](awm_accurate_weight-matrix_fingerprint_for_large_language_models.md)
- [\[ICLR 2026\] Robust Multi-Objective Controlled Decoding of Large Language Models](robust_multi-objective_controlled_decoding_of_large_language_models.md)
- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[ICLR 2026\] TROLL: Trust Regions improve Reinforcement Learning for Large Language Models](troll_trust_regions_improve_reinforcement_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
