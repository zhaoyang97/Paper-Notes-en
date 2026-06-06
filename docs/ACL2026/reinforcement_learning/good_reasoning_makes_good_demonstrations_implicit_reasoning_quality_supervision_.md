---
title: >-
  [Paper Note] Good Reasoning Makes Good Demonstrations: Implicit Reasoning Quality Supervision via In-Context Reinforcement Learning
description: >-
  [ACL2026][Reinforcement Learning][RLVR] This paper identifies that RLVR cannot distinguish between "high-quality reasoning for a correct answer" and "low-quality reasoning that happens to get the answer right." It propos…
tags:
  - "ACL2026"
  - "Reinforcement Learning"
  - "RLVR"
  - "Evidence Gain"
  - "in-context learning"
  - "DAPO"
  - "Mathematical Reasoning"
date: 2026-05-08
content_hash: b2bdf04df8acc432
---

# Good Reasoning Makes Good Demonstrations: Implicit Reasoning Quality Supervision via In-Context Reinforcement Learning

**Conference**: ACL2026  
**arXiv**: [2603.09803](https://arxiv.org/abs/2603.09803)  
**Code**: https://github.com/Mithas-114/IC-DAPO  
**Area**: Reinforcement Learning / LLM Reasoning / RLVR  
**Keywords**: RLVR, Evidence Gain, in-context learning, DAPO, Mathematical Reasoning  

## TL;DR
This paper identifies that RLVR cannot distinguish between "high-quality reasoning for a correct answer" and "low-quality reasoning that happens to get the answer right." It proposes using the in-context teaching utility of a demonstration, termed Evidence Gain, as an implicit quality signal. Through In-Context RLVR, mathematical reasoning accuracy and quality are improved without the need to train a PRM.

## Background & Motivation
**Background**: Reinforcement Learning with Verifiable Rewards (RLVR) has become a major paradigm for enhancing mathematical reasoning in large language models. It relies on verifiable answers, assigning positive rewards to correct results while avoiding expensive step-by-step human process labeling.

**Limitations of Prior Work**: RLVR result rewards are too coarse-grained. As long as the final answer is correct, the same reward is given regardless of whether the reasoning process is rigorous, redundant, skip-step, or a lucky guess. Consequently, low-quality reasoning trajectories are reinforced, which may damage the model's internal problem-solving strategies in the long run.

**Key Challenge**: Process Reward Models (PRM) can distinguish reasoning quality but require additional labeling or training an evaluator; using answer rewards alone fails to differentiate between "good" and "bad" correct trajectories. This work seeks to resolve whether RLVR can automatically favor high-quality reasoning trajectories without introducing a PRM.

**Goal**: Define a global signal that reflects reasoning quality and integrate it into RLVR at low cost, ensuring the training process assigns higher weights to high-quality correct trajectories and lower weights to low-quality ones.

**Key Insight**: High-quality reasoning is viewed as a "good demonstration." If a reasoning trajectory is truly clear, relevant, and transferable, putting it as an in-context demonstration before another problem should help the current policy generate high-quality reference solutions more easily.

**Core Idea**: Utilize the model's own ICL capability to measure the log-likelihood improvement (Evidence Gain) brought by a reasoning trace used as a demonstration. Instead of explicitly calculating this during training, high-quality demonstrations are added before rollouts, allowing the objective function to implicitly re-weight rewards based on Evidence Gain.

## Method
The methodology consists of two parts. First, it proves that Evidence Gain serves as a proxy for reasoning quality. Second, it applies this idea to training via In-Context RLVR.

### Overall Architecture

Given a training problem $q$ and a model-generated reasoning trajectory $r$, a held-out validation set is prepared where each sample contains a problem and a high-quality reference reasoning. Evidence Gain measures how much the log-likelihood of generating the validation reference reasoning increases when $(q,r)$ is used as a demonstration, compared to a zero-shot setting.

Directly using Evidence Gain as a reward is computationally expensive. Estimates suggest that explicitly calculating Evidence Gain for 12K samples and 100 demonstrations would take approximately 80 H800 hours. Therefore, instead of calculating rewards after rollouts, a demonstration is sampled from a demonstration set before each rollout and prepended to the current problem, followed by standard RLVR updates. This simple input-side modification constitutes In-Context RLVR.

### Key Designs

1.  **Evidence Gain as a Demonstration Utility Signal**:
    - **Function**: Measures whether a reasoning trajectory helps the model generate high-quality reference solutions, thereby indirectly reflecting the quality of the reasoning trace itself.
    - **Mechanism**: For high-quality reference reasoning in a validation set, the difference in generation log-likelihood before and after adding a candidate demonstration is compared and averaged across validation samples. High Evidence Gain indicates the reasoning provides a transferable problem-solving pattern.
    - **Design Motivation**: Unlike length, logprob, or majority vote, Evidence Gain tests the educational value of the trajectory—"can this reasoning teach the model to perform similar reasoning?"

2.  **Input-side Training Modification for In-Context RLVR**:
    - **Function**: Naturally amplifies the gradient weights of high-quality reasoning trajectories without explicit Evidence Gain calculations.
    - **Mechanism**: Before each rollout, a high-quality Q&A/reasoning pair is randomly sampled from the demonstration set and prepended to the current problem. The model generates responses based on this demonstration-conditioned input and is trained using standard verifiable answer rewards.
    - **Design Motivation**: Adding a demonstration in the input changes the sampling distribution. It is theoretically proven that this objective is equivalent to reward re-weighting in zero-shot RLVR, where the weight is approximately proportional to the exponent of Evidence Gain.

3.  **Decoupled Combination with DAPO/GRPO**:
    - **Function**: Demonstrates that the idea is an input-side enhancement module applicable to existing RLVR frameworks rather than a specific RL optimizer feature.
    - **Mechanism**: The main experiments integrate In-Context RLVR with DAPO (yielding IC-DAPO) and with GRPO on a 1.5B model (yielding IC-GRPO). Training still uses answer correctness rewards; demonstrations only change input conditions and implicit weighting.
    - **Design Motivation**: Stable gains across both DAPO and GRPO suggest that Evidence Gain re-weighting is a general training signal.

### Loss & Training

Standard RLVR optimizes the answer reward $R(q,r)$ on problem $q$. In-Context RLVR samples a demonstration $e$ first, then samples an answer from $π_{\theta}(r|e, q)$. The authors derive via Bayesian identity that this is equivalent to optimizing $R(q,r) \cdot w(q,r)$ on the base distribution $π_{\theta}(r|q)$, where $w(q,r)$ is the expectation of the demonstration likelihood ratio, and $\log w(q,r)$ is approximately Evidence Gain plus a model-dependent constant.

Training data is sourced from KlearReasoner-MathSub-30K, divided into a policy optimization training set, a demonstration set with 1,082 pairs, and a held-out set of 100 samples. Evaluation covers AIME24, AIME25, HMMT25, MATH500, AMC23, and OlympiadBench.

## Key Experimental Results

### Main Results

| Model/Method | AIME24 | AIME25 | HMMT25 | MATH500 | AMC23 | Olympiad | Average | Time/Step |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DS-R1-Distill-Qwen-1.5B | 29.2 | 24.1 | 13.1 | 86.0 | 73.7 | 51.8 | 46.3 | N/A |
| + GRPO | 33.4 | 28.1 | 16.6 | 88.3 | 79.3 | 56.2 | 50.3 | 457.4s |
| + IC-GRPO | 38.3 | 30.6 | 17.7 | 89.5 | 82.5 | 56.9 | 52.6 | 461.8s |
| + DAPO | 40.0 | 28.4 | 19.2 | 90.0 | 84.4 | 61.6 | 53.9 | 459.6s |
| + CE-GPPO | 42.8 | 32.5 | 20.5 | 91.0 | 85.8 | 61.8 | 55.7 | 464.0s |
| + IC-DAPO | 45.6 | 34.2 | 19.7 | 90.6 | 86.2 | 62.1 | 56.4 | 477.2s |

| Model/Method | AIME24 | AIME25 | HMMT25 | MATH500 | AMC23 | Olympiad | Average | Time/Step |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DS-R1-Distill-Qwen-7B | 54.5 | 39.1 | 26.2 | 93.6 | 90.6 | 67.0 | 61.8 | N/A |
| + GRPO | 55.3 | 40.3 | 24.5 | 93.7 | 88.8 | 65.6 | 61.4 | 305.6s |
| + DAPO | 62.0 | 45.9 | 27.4 | 94.1 | 92.3 | 69.9 | 65.3 | 303.1s |
| + CE-GPPO | 64.2 | 50.3 | 28.9 | 95.3 | 93.3 | 71.6 | 67.3 | 292.5s |
| + IC-DAPO | 66.5 | 49.8 | 29.4 | 95.6 | 93.7 | 71.7 | 67.8 | 315.6s |

IC-DAPO improves upon DAPO by an average of 2.5 points on both 1.5B and 7B; IC-GRPO improves upon GRPO by 2.3 points on 1.5B. While training overhead increases, the authors highlight that the additional cost of IC-DAPO is less than 5%.

### Ablation Study

| Proxy Signal | 1.5B Spearman rho | 7B Spearman rho | Description |
| :--- | :--- | :--- | :--- |
| Length | -0.147 | -0.161 | Longer reasoning does not imply better quality |
| LogProb | 0.129 | 0.178 | Confidence is only weakly correlated |
| MajorVote | 0.079 | 0.109 | Answer consistency has weak discriminative power |
| Evidence Gain | 0.405 | 0.444 | Strongest correlation with reasoning quality |

| Difficulty | DAPO 1.5B | IC-DAPO 1.5B | DAPO 7B | IC-DAPO 7B | Main Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Easy | 98.3 | 98.8 (+0.5%) | 98.6 | 99.3 (+0.7%) | Little room for improvement |
| Medium | 90.1 | 93.5 (+3.8%) | 97.8 | 98.2 (+0.4%) | Stable gains on medium tasks |
| Hard | 23.1 | 26.0 (+12.6%) | 39.2 | 43.2 (+10.2%) | Gains concentrated on hard problems |

| Demo Source | 1.5B Average | 7B Average | Description |
| :--- | :--- | :--- | :--- |
| DAPO | 53.9 | 65.3 | Without in-context demonstration |
| IC-DAPO (V3.1) | 55.7 | 66.4 | Demos from non-reasoning model DeepSeek-V3.1 |
| IC-DAPO (R1) | 56.4 | 67.8 | Best results with DeepSeek-R1 refined traces |

### Key Findings

- Evidence Gain predicts reasoning quality better than length, logprob, or majority vote, indicating it captures transferable problem-solving patterns rather than surface features.
- During training, the average Evidence Gain and quality scores of IC-DAPO grow faster; the Spearman correlation between Evidence Gain and quality stabilizes at ~0.4.
- Gains primarily come from hard problems: 1.5B hard split improved by 12.6% vs. DAPO, supporting the explanation that quality re-weighting is most suitable for tasks requiring deep reasoning.

## Highlights & Insights
- **Transformation of Reasoning Quality into Teaching Utility**: Instead of asking "Does this reasoning look good?", the authors ask, "Can it help the model solve other problems as a demonstration?" This definition inherently emphasizes transferable structure.
- **Implicit Reward Re-weighting via Input Modification**: The most valuable insight is the theoretical proof that simply adding a demonstration before rollout implicitly amplifies the gradient of high Evidence Gain trajectories. Implementation is simple but the logic is robust.
- **Process Preference without PRM**: The method bypasses the cost of process labeling and training evaluators, which is particularly useful for math/code tasks with verifiable answers.
- **Relative Ranking Stability**: While the absolute Evidence Gain is higher for the 7B model, the relative ranking of high-quality trajectories remains stable, making this signal ideal for internal model ranking.

## Limitations & Future Work

- **Task Scope**: Evaluation is limited to mathematical reasoning. Generalization to other reasoning-intensive fields like STEM, code, or open Q&A remains unverified.
- **Dependency on Strong Models**: The demonstration set currently relies on traces from models like DeepSeek-R1. Without a strong teacher model, or if the teacher has a biased style, the effectiveness may decrease.
- **Focus on Correct Answers**: RLVR still filters by answer correctness; incorrect but heuristic reasoning trajectories are not directly utilized. Future work could incorporate partially correct steps.
- **Input Length and Cost**: Demonstration concatenation increases context length. For longer problems or more demonstrations, the cost might become a bottleneck.

## Related Work & Insights
- **vs. Standard RLVR/GRPO/DAPO**: Standard methods focus on the final answer. Ours changes the sampling distribution via in-context demonstrations, effectively weighting high-quality trajectories.
- **vs. PRM**: PRMs evaluate intermediate steps explicitly but require labels/external models; Evidence Gain uses the policy’s own ICL capacity as an implicit evaluator without new reward models.
- **vs. Proxy Signals**: Statistical proxies like length or logprob show significantly lower correlation with quality compared to Evidence Gain.
- **Inspiration for Future RLHF/RLVR**: Demonstration-conditioned rollout can serve as a universal wrapper for verifiable tasks (code, theorem proving) to mitigate the "correct-but-bad-reasoning" problem using high-quality demonstrations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The definition of "demonstration utility" and the implicit re-weighting derivation are highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of multiple benchmarks, model scales, and optimizers; however, the domain remains focused on mathematics.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear closure between motivation, theory, and experiments.
- Value: ⭐⭐⭐⭐⭐ Highly practical for RLVR training, especially when aiming for reasoning quality improvements without training a PRM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AttnPO: Attention-Guided Process Supervision for Efficient Reasoning](attnpo_attention-guided_process_supervision_for_efficient_reasoning.md)
- [\[AAAI 2026\] Good-for-MDP State Reduction for Stochastic LTL Planning](../../AAAI2026/reinforcement_learning/good-for-mdp_state_reduction_for_stochastic_ltl_planning.md)
- [\[ICLR 2026\] Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment](../../ICLR2026/reinforcement_learning/reasoning_as_representation_rethinking_visual_reinforcement_learning_in_image_qu.md)
- [\[ACL 2026\] ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following](imprif_stronger_implicit_reasoning_leads_to_better_complex_instruction_following.md)
- [\[CVPR 2026\] Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision](../../CVPR2026/reinforcement_learning/reasoning-driven_anomaly_detection_and_localization_with_image-level_supervision.md)

</div>

<!-- RELATED:END -->
