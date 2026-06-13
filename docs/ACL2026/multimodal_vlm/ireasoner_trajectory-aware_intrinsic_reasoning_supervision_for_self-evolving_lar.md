---
title: >-
  [Paper Note] iReasoner: Trajectory-Aware Intrinsic Reasoning Supervision for Self-Evolving Large Multimodal Models
description: >-
  [ACL2026][Multimodal VLM][Multimodal reasoning] iReasoner enables LMMs to perform self-questioning and self-answering on unlabeled images…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Multimodal reasoning"
  - "self-evolutionary training"
  - "Chain-of-Thought"
  - "intrinsic reward"
  - "trajectory consistency"
date: 2026-05-08
content_hash: e6436488f4ebe611
---

# iReasoner: Trajectory-Aware Intrinsic Reasoning Supervision for Self-Evolving Large Multimodal Models

**Conference**: ACL2026  
**arXiv**: [2601.05877](https://arxiv.org/abs/2601.05877)  
**Code**: Our code is available here  
**Area**: Multimodal VLM / Self-supervised reasoning training  
**Keywords**: Multimodal reasoning, self-evolutionary training, Chain-of-Thought, intrinsic reward, trajectory consistency  

## TL;DR
iReasoner enables LMMs to perform self-questioning and self-answering on unlabeled images, extending final answer consistency to intermediate CoT step consistency rewards. This leads to a peak multimodal reasoning improvement of approximately +2.13 points on Qwen2.5-VL-7B.

## Background & Motivation
**Background**: Self-evolving training for multimodal large models is shifting from "reliance on human labels" to "utilizing unlabeled images for self-generated questions and answers." Proposer-Solver frameworks allow models to pose questions based on images, sample multiple solutions, and use internal consistency as a reward signal.

**Limitations of Prior Work**: Most existing self-evolving LMM methods only reward final answers or entire responses. Two reasoning trajectories giving the same answer might receive nearly identical rewards, even if one contains hallucinated intermediate steps, incorrect visual evidence, or lucky cancellation of calculation errors.

**Key Challenge**: Under unsupervised settings, there are no human-labeled answers or external judges to stably evaluate each reasoning step. However, the reliability of multimodal reasoning depends heavily on intermediate visual grounding and step-by-step inference. Training signals are too coarse if only final answers are considered.

**Goal**: To provide optimizable intrinsic supervision for intermediate reasoning steps of LMMs without introducing labels, external verifiers, or reward models, optimizing not just "getting it right" but also "how to reason to the answer."

**Key Insight**: The authors utilize the consensus among multiple Solver rollouts for the same image-question pair. If a batch of rollouts converges to the same dominant answer, the reasoning steps at the same positions in these rollouts should possess similar semantics. This cross-trajectory step consistency acts as an unsupervised reasoning quality signal.

**Core Idea**: Transforming step-level CoT agreement within the dominant-answer group into an intrinsic reward, mixed with answer self-consistency rewards, and training the Solver using KL-regularized policy gradients.

## Method

### Overall Architecture
iReasoner follows the Proposer-Solver self-evolution framework. Given an unlabeled image $x$, the Proposer generates a visually relevant question $q$. The Solver samples $N$ reasoning rollouts for $(x, q)$, each containing explicit `<think>` reasoning steps and an `<answer>` final answer. Multiple answers form an empirical distribution $p(a|x,q)$. The Proposer uses answer entropy to maintain moderate difficulty, while the Solver receives both answer-level self-consistency rewards and step-level consistency rewards.

The core of the method is not generating longer CoT, but making intermediate steps across different rollouts semantically comparable, aggregatable, and rewardable. It converts "whether the reasoning trajectory is stable under the same final answer" into a training signal.

### Key Designs
1. **Dominant-answer group**:

	- Function: To filter the set of rollouts with consistent final answers, establishing step consistency on a relatively reliable answer pattern.
	- Mechanism: Select the dominant answer $\hat{a}$ from the answer distribution $p(a|x,q)$ and form a set $\mathcal{G}$ of rollouts generating that answer. Use $\rho=(|\mathcal{G}|/N)^\gamma$ to represent the density of the majority group; the step reward is more heavily down-weighted as the dominant group becomes smaller.
	- Design Motivation: If all rollouts are scattered, directly rewarding step similarity would encourage incorrect consensus. Clustering by answer before evaluating steps reduces unsupervised reward noise.

2. **Intrinsic CoT Agreement Reward**:

	- Function: Measures whether the intermediate steps of different rollouts within the same answer group are semantically consistent.
	- Mechanism: Decompose each reasoning trajectory into steps $s_{i,j}$, representing each step with the normalized mean of the model's internal token embeddings. For each step position $j$, calculate a dominant group prototype $\mu_j$ and score using cosine similarity $r_{i,j}=\text{sim}(e_{i,j},\mu_j)$. Use decreasing weights $w_1>w_2>...$ during aggregation to emphasize early visual grounding steps.
	- Design Motivation: Early steps are typically responsible for identifying image information and establishing the problem state; errors here propagate. Position decay focuses rewards on these foundational steps rather than late template-based summaries.

3. **Reward integration with self-evolution**:

	- Function: Synthesizes answer-level and step-level rewards into a single trainable Solver reward.
	- Mechanism: The answer reward is $r_i^{ans}=p(a_i|x,q)^\alpha(1-\eta\bar{\ell}_i)$, including answer self-consistency and length penalties. The final Solver reward is $r_i^{sol}=(1-\lambda(t))r_i^{ans}+\lambda(t)r_i^{step}$, where $\lambda(t)$ gradually increases with training warmup.
	- Design Motivation: Answer consensus is unstable in early training; reinforcing step consistency too early may amplify errors. Warmup allows the model to first form basic answer stability before learning trajectory structure.

### Loss & Training
Both the Solver and Proposer are trained using KL-regularized policy gradients, with reference models frozen and an EMA baseline used to reduce variance. The Solver objective includes a REINFORCE term and a KL penalty against the reference policy. The Proposer uses answer entropy to shape rewards, keeping question difficulty within a non-degenerate range.

Training details are reserved: Initialized from Qwen2.5-VL-7B-Instruct, using LoRA to train both Proposer and Solver. The training pool consists of 2,500 unlabeled images from datasets like ChartQA, AI2D, InfoGraphic-VQA, PlotQA, ChartX, and Geometry3K. One question is sampled per image, with $N=5$ rollouts sampled for the Solver. The Proposer updates every 5 rounds. Step reward weight warms up to 0.7 over 2.5k steps, with AdamW learning rate $10^{-6}$, completing in ~35 hours on 8 AMD MI250X GPUs.

## Key Experimental Results

### Main Results
| Benchmark | Qwen2.5-VL-7B Baseline | EvoLMM | iReasoner | Gain over Baseline |
|-----------|------------------------|--------|-----------|------------------|
| InfoGraphic-VQA | 80.44 | 81.06 | 81.56 | +1.12 |
| AI2D | 82.61 | 83.41 | 83.89 | +1.28 |
| ScienceQA | 88.30 | 89.50 | 89.92 | +1.62 |
| MMMU | 51.11 | 52.01 | 52.37 | +1.26 |
| ChartQA | 84.00 | 86.70 | 85.78 | +1.78 |
| MathVista | 68.47 | 70.52 | 69.74 | +1.27 |
| MathVision | 23.91 | 24.81 | 25.29 | +1.38 |
| MathVerse | 43.78 | 44.88 | 45.91 | +2.13 |

### Ablation Study
| Configuration | ScienceQA | MMMU | ChartQA | MathVerse | Description |
|------|-----------|------|---------|-----------|------|
| Full iReasoner | 89.92 | 52.37 | 85.78 | 45.91 | Full combination of answer + step rewards |
| Soft majority reward only / EvoLMM | 89.41 | 51.92 | 86.64 | 44.71 | Stronger on short verifiable tasks, weaker transfer |
| Step-level reward only | 88.44 | 50.98 | 84.38 | 43.87 | High noise when used in isolation |
| w/o Warmup schedule | 89.26 | 51.74 | 85.02 | 45.11 | Absence causes the most significant/stable degradation |
| w/o Position decay | 89.55 | 52.02 | 85.41 | 45.49 | Weighting early steps contributes positively |
| w/o Density weighting | 89.47 | 51.88 | 85.29 | 45.32 | Majority reliability weighting prevents false consensus |

### Key Findings
- iReasoner outperforms the initial Qwen2.5-VL-7B on all 8 benchmarks, with average gains of ~+1.32 in general visual understanding and ~+1.64 in visual math.
- Compared to EvoLMM, iReasoner is stronger on InfoGraphic-VQA, AI2D, ScienceQA, MMMU, MathVision, and MathVerse, but slightly lower on ChartQA and MathVista, suggesting answer stability rewards are better suited for highly verifiable short-answer tasks.
- Step-level rewards cannot be used in isolation; they require answer-level stability to filter out obviously incorrect rollouts.
- The accuracy of the dominant answer group improves from ~76% early in training to ~93% later, showing that the majority group is not a blind source of supervision.

## Highlights & Insights
- The most valuable aspect of the paper is transforming the issue of "different reasoning trajectories for the same answer" into an optimizable signal. While many self-consistency methods only look at answer voting, iReasoner interrogates whether the reasoning paths behind the votes are stable.
- Using internal model embeddings to represent steps is lightweight and does not require external judges or human step labels, fitting the unsupervised self-evolution setting.
- The three minor designs—warmup, density weighting, and position decay—are crucial, reflecting empirical experience in handling unsupervised RL noise.
- This logic can be transferred to text reasoning, code reasoning, or agent trajectory training: as long as multiple trajectories can be sampled and a "same-result group" can be defined, intermediate step consistency can be compared.

## Limitations & Future Work
- iReasoner only uses internal signals formed by the model's own sampling and cannot directly optimize external correctness. When a majority answer group is confident but wrong, step consistency may reinforce consistent but incorrect reasoning.
- Experimental scale is still limited: training only covers 2,500 images and 2.5k steps, primarily focused on the Qwen2.5-VL series; larger scale, longer training, and more model families still need verification.
- The method requires access to model log probabilities, internal embeddings, and KL-regularized training, making it more suitable for open-weight models than black-box closed-source APIs.
- CoT itself may have faithfulness issues. The paper includes no-CoT reasoning controls, but more rigorous evaluations of causal intervention or process supervision remain worth pursuing.

## Related Work & Insights
- **vs EvoLMM / VisPlay**: These self-evolving LMMs mainly rely on answer or response-level intrinsic rewards; iReasoner pushes supervision granularity to intermediate steps.
- **vs Multimodal-CoT / R3V**: These methods focus on explicit reasoning chain quality but usually depend on labels, filtering, or external feedback; iReasoner emphasizes pure unlabeled training.
- **vs RLVR-style training**: Traditional RLVR requires verifiable answers; iReasoner provides a way to construct weak verification signals in open-ended visual tasks.
- **Inspiration for future research**: Step agreement can be extended to consistency rewards for graph structures, tool-call sequences, or visual region references, beyond just textual similarity.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Intermediate trajectory consistency as an unsupervised intrinsic reward is insightful, though built upon existing Proposer-Solver frameworks.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Main experiments, ablations, step budgets, and training dynamics are complete; model and data scale could be expanded further.
- Writing Quality: ⭐⭐⭐⭐☆ Method explanations are intuitive and diagrams are clear.
- Value: ⭐⭐⭐⭐☆ Provides practical references for multimodal self-training and process supervision, especially for open-source LMM post-training research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards](../../CVPR2026/multimodal_vlm/evolmm_self_evolving_lmm_continuous_rewards.md)
- [\[ACL 2026\] PRISM: Self-Pruning Intrinsic Selection Method for Training-Free Multimodal Data Selection](prism_self-pruning_intrinsic_selection_method_for_training-free_multimodal_data_.md)
- [\[ICML 2026\] Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training](../../ICML2026/multimodal_vlm/learn_to_think_improving_multimodal_reasoning_through_vision-aware_self-improvem.md)
- [\[ACL 2026\] VAUQ: Vision-Aware Uncertainty Quantification for LVLM Self-Evaluation](vauq_vision-aware_uncertainty_quantification_for_lvlm_self-evaluation.md)
- [\[CVPR 2026\] Self-Consistency for LLM-Based Motion Trajectory Generation and Verification](../../CVPR2026/multimodal_vlm/self-consistency_for_llm-based_motion_trajectory_generation_and_verification.md)

</div>

<!-- RELATED:END -->
