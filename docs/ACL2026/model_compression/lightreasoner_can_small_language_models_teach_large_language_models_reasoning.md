---
title: >-
  [Paper Note] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?
description: >-
  [ACL 2026][Model Compression][Small Model Teacher] LightReasoner uses the token distribution discrepancy between a weak Amateur model and a strong Expert model to automatically identify high-value reasoning steps. It the…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Small Model Teacher"
  - "Reasoning Distillation"
  - "KL Divergence"
  - "Selective Fine-Tuning"
  - "LoRA"
date: 2026-05-08
content_hash: 6f992f11c55f2bb7
---

# LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?

**Conference**: ACL 2026  
**arXiv**: [2510.07962](https://arxiv.org/abs/2510.07962)  
**Code**: https://github.com/HKUDS/LightReasoner  
**Area**: Model Compression / LLM Reasoning / Efficient Fine-Tuning  
**Keywords**: Small Model Teacher, Reasoning Distillation, KL Divergence, Selective Fine-Tuning, LoRA

## TL;DR
LightReasoner uses the token distribution discrepancy between a weak Amateur model and a strong Expert model to automatically identify high-value reasoning steps. It then applies contrastive self-distillation specifically to these steps, enabling mathematical reasoning models to match or exceed SFT performance while significantly reducing sampling, training time, and tuning tokens.

## Background & Motivation
**Background**: A common trajectory for enhancing the mathematical reasoning of LLMs is rejection-sampling SFT: first, the model generates multiple reasoning trajectories, correct ones are filtered using answers or verifiers, and the entire trajectory is used as supervised data for fine-tuning. This approach is direct and effective, and it is compatible with reasoning-enhanced paradigms like Chain-of-Thought and RFT.

**Limitations of Prior Work**: Rejection SFT is highly expensive. It requires the generation of full candidate solutions and filtering with ground truth or external verifiers, and it optimizes all tokens in the reasoning chain indiscriminately. The paper points out that many tokens are merely routine conjunctions or low-information steps; the key turning points that decide the success of reasoning are often few, meaning full-trajectory training wastes computational resources on low-return tokens.

**Key Challenge**: Strong models already possess partial latent reasoning capabilities, but existing training signals often rely on external answers or human-curated data. On the other hand, while weak models lack sufficient ability, they can expose "where things go wrong" given the same prefix. The core challenge in this paper is how to identify critical reasoning moments where the strong model has a genuine advantage over the weak model without relying on labels or complete trajectories.

**Goal**: The authors aim to construct a verifier-free reasoning enhancement framework that automatically localizes high-value tokens, trains the Expert only on these tokens, and ensures the training signal reflects the Expert's advantageous distribution relative to the Amateur rather than just the Expert's own one-hot output.

**Key Insight**: The authors observe the next-token distributions of the Expert and Amateur under the same prefix. If the two are highly consistent, the token is likely a routine step; if the KL divergence suddenly increases, it may correspond to critical reasoning points like arithmetic operations, logical transitions, or intermediate conclusions. The paper provides statistics: approximately 60% of tokens have a KL in $[0.0, 0.1)$, while only about 20% exceed 0.4. When the Expert and Amateur top-1 outputs disagree, the average KL is 1.99, compared to 0.166 when they agree.

**Core Idea**: Replace manual labels and full-trajectory SFT with the Expert-Amateur distribution discrepancy, turning the weak model into a "negative reference" to distill only the most significant reasoning advantages of the Expert over the Amateur.

## Method
LightReasoner can be understood as selective self-distillation for reasoning models. Instead of having the small model generate answers for the large model or performing dual-model contrastive decoding during inference, it compares the token distributions of both during the data construction phase and converts high-discrepancy steps into soft supervision.

### Overall Architecture
The input consists of a batch of reasoning problems; the main experiments use the GSM8K training set to generate supervised samples. For each problem, the Expert model first generates a short-prefix reasoning trajectory in a CoT manner, with the sampling rollout length limited to 128 tokens. For each prefix $s_t$ in the trajectory, both the Expert distribution $\pi_E(\cdot\mid s_t)$ and the Amateur distribution $\pi_A(\cdot\mid s_t)$ are computed.

The first stage is sampling and filtering: if $D_{KL}(\pi_E\|\pi_A)>\beta$, the step is considered an informative step. The second stage is constructing contrastive supervision: the ratio $\log \pi_E(a\mid s_t) / \pi_A(a\mid s_t)$ is computed over a masked support set of high-confidence Expert tokens and normalized into a soft target $v_C$. The third stage is fine-tuning: the same Expert is trained using LoRA to bring its output distribution closer to $v_C$, thereby strengthening the reasoning decisions where the Expert already outperforms the Amateur.

### Key Designs
1. **KL-driven Informative Step Filtering**:
	- **Function**: Filters a few high-value tokens from complete reasoning trajectories to avoid spreading the training budget evenly across all tokens.
	- **Mechanism**: Compares $D_{KL}(\pi_E(\cdot\mid s_t)\|\pi_A(\cdot\mid s_t))$ under the same prefix, using $\beta=0.4$ as the filtering threshold in main experiments. A larger KL indicates a more pronounced difference in the next-step choice between the Expert and Amateur.
	- **Design Motivation**: Reasoning errors often occur at bottleneck steps like arithmetic, symbolic transformation, or logical jumps. Using Expert-Amateur divergence as a proxy signal is closer to the actual difficulties of each trajectory than using fixed prefix lengths or manual rules.

2. **Contrastive Distribution Supervision**:
	- **Function**: Converts "where the Expert is stronger than the Amateur" into training labels, rather than simply duplicating the Expert's self-generated tokens.
	- **Mechanism**: First, Expert low-probability tail tokens are filtered using $\alpha=0.2$, keeping tokens where $\pi_E(a\mid s_t) \geq \alpha \max_b \pi_E(b\mid s_t)$. Then, a contrastive score $v'_C(a\mid s_t)=\log \pi_E(a\mid s_t)/\pi_A(a\mid s_t)$ is calculated and softmax-normalized to obtain $v_C$.
	- **Design Motivation**: One-hot supervision discards distributional information and risks treating the Expert's incidental output as the sole truth. Contrastive soft labels emphasize the advantage margin of the Expert relative to the Amateur and can weaken low-confidence noise.

3. **Short Rollout and LoRA Self-Distillation**:
	- **Function**: Keeps both supervision construction and fine-tuning at low cost while reducing false positives caused by cascading reasoning errors in later stages.
	- **Mechanism**: Sampling is limited to the first 128 tokens. The Expert is trained using LoRA for 1000 steps with 16 contrastive supervision samples per step. The loss is $D_{KL}(v_C\|\pi_E)$, equivalent to cross-entropy against the soft target.
	- **Design Motivation**: The paper argues that early reasoning steps are more stable, whereas full answers are more susceptible to error cascading. Short rollouts combined with selective token training allow LightReasoner to reduce both sampling and tuning tokens compared to rejection SFT.

### Loss & Training
The training goal is to match the Expert's output distribution to the contrastive supervision $v_C$: $\mathcal{L}(s_t)=D_{KL}(v_C(\cdot\mid s_t)\|\pi_E(\cdot\mid s_t))$. Since $v_C$ is a constant with respect to current training parameters, this objective is equivalent to $-\sum_a v_C(a\mid s_t)\log\pi_E(a\mid s_t)$. In the experiments, Experts include Qwen2.5-Math-1.5B/7B, their Instruct versions, and DeepSeek-R1-Distill-Qwen-1.5B, while the Amateur is fixed as Qwen2.5-0.5B.

## Key Experimental Results

### Main Results
Main results use zero-shot pass@1 or the corresponding evaluation settings specified in the text, covering 7 mathematical reasoning benchmarks. The table below excerpts the AVG and several representative models, demonstrating that LightReasoner can exceed or approach rejection SFT on most models.

| Expert Model | Method | GSM8K | MATH | SVAMP | ASDiv | MMLU STEM | AVG |
|--------|------|------|------|-------|-------|-----------|-----|
| Qwen2.5-Math-1.5B | Baseline | 42.5 | 34.2 | 68.8 | 68.1 | 49.8 | 42.4 |
| Qwen2.5-Math-1.5B | SFT | 69.2 | 57.1 | 64.1 | 70.2 | 47.7 | 50.1 |
| Qwen2.5-Math-1.5B | LightR | 70.6 | 59.3 | 76.0 | 79.8 | 54.9 | 54.2 |
| DeepSeek-R1-Distill-Qwen-1.5B | Baseline | 75.2 | 54.2 | 79.9 | 84.9 | 22.3 | 50.3 |
| DeepSeek-R1-Distill-Qwen-1.5B | SFT | 78.2 | 60.3 | 81.5 | 87.4 | 26.2 | 53.3 |
| DeepSeek-R1-Distill-Qwen-1.5B | LightR | 79.5 | 60.2 | 83.5 | 87.5 | 26.2 | 55.9 |
| Qwen2.5-Math-7B | Baseline | 57.5 | 51.8 | 67.9 | 72.7 | 69.8 | 50.0 |
| Qwen2.5-Math-7B | SFT | 64.4 | 63.3 | 76.2 | 76.6 | 68.5 | 54.5 |
| Qwen2.5-Math-7B | LightR | 67.9 | 57.8 | 77.2 | 80.6 | 70.5 | 54.7 |

### Ablation Study
Ablation studies on Qwen2.5-Math-1.5B gradually remove step selection and contrastive supervision. Full LightReasoner averages 54.0, higher than rejection SFT's 50.6; removing contrast drops the average to 44.8, indicating that contrastive supervision is more critical than mere token filtering.

| Configuration | GSM8K | MATH | SVAMP | ASDiv | Minerva Math | Olympiad Bench | AVG |
|------|------|------|-------|-------|--------------|----------------|-----|
| Baseline | 42.5 | 34.2 | 68.8 | 68.1 | 9.9 | 23.7 | 41.2 |
| Rejection SFT | 69.2 | 57.1 | 64.1 | 70.2 | 15.1 | 27.6 | 50.6 |
| GT Supervision | 43.4 | 34.8 | 70.4 | 69.7 | 10.2 | 19.8 | 41.4 |
| Full LightReasoner | 70.6 | 59.3 | 76.0 | 79.8 | 11.4 | 27.1 | 54.0 |
| W/o step selection, w/ contrast | 67.6 | 58.8 | 78.7 | 80.5 | 11.0 | 26.4 | 53.8 |
| W/ step selection, w/o contrast | 62.0 | 53.1 | 56.6 | 61.0 | 10.7 | 25.5 | 44.8 |
| Both removed | 55.5 | 50.2 | 50.0 | 65.4 | 10.4 | 24.0 | 42.6 |

### Key Findings
- Efficiency metrics show that SFT on Qwen2.5-Math-1.5B requires 4.0h, 3952 problems, and 1.77M tuned tokens, while LightReasoner needs only 0.5h, 1000 problems, and 0.02M tuned tokens, with the average gain increasing from +7.7% to +11.8%.
- On Qwen2.5-Math-7B, SFT takes 9.5h with 6029 problems and 2.20M tokens; LightReasoner takes 0.75h with 1000 problems and 0.02M tokens, yielding similar or slightly higher average gains.
- Overall, the paper reports accuracy improvements of up to 28.1% while saving approximately 90% of time, 80% of sampled problems, and 99% of tuned tokens.
- Mechanism analysis indicates that the effectiveness of the contrastive signal increases with an appropriate Expert-Amateur capability gap; if the Amateur is close to or stronger than the Expert, gains diminish or degrade.

## Highlights & Insights
- The ingenuity of LightReasoner lies in flipping the "weak model" from a student in traditional distillation to a reference for identifying the strong model's advantages. Instead of having the small model teach the large model answers, it allows the small model to expose its own failures, thereby alerting the large model to which tokens are most worth reinforcing.
- The method moves the idea of contrastive decoding from inference time to training time. This preserves the advantages of Expert-Amateur contrast while avoiding the latency and VRAM overhead of running two models for every inference.
- Evidence for selective token training is substantial: KL distribution, top-1 disagreement, ablation tables, and efficiency metrics all point to the conclusion that reasoning ability is not uniformly distributed across a trajectory but concentrated at a few high-leverage decision points.
- The insight for model compression and efficient fine-tuning is that compression does not necessarily mean just transferring large model knowledge to small models; it can also leverage the failure modes of small models to inversely improve large model training efficiency.

## Limitations & Future Work
- The paper primarily evaluates mathematical reasoning (GSM8K, MATH, SVAMP, ASDiv, Minerva Math, Olympiad Bench, MMLU STEM); whether it is equally effective in code reasoning, tool use, open-ended planning, etc., remains to be verified.
- Expert-Amateur pairing depends on an appropriate capability gap. A gap too small leads to insufficient contrastive signals, while a negative gap might mislead the Expert. Automatic Amateur selection or dynamic pairing are key future directions.
- both $\alpha$ masking and $\beta$ filtering are additional hyperparameters; although the paper provides defaults ($\alpha=0.2$, $\beta=0.4$), different tasks and model families may require re-tuning.
- The experiments cover small to medium-scale open-source models; scalability to larger closed-source or stronger reasoning models has not yet been proven.

## Related Work & Insights
- **vs rejection SFT / RFT**: SFT relies on complete trajectories and answer verification. LightReasoner uses distribution discrepancies of short prefixes for supervision, offering lower costs and no reliance on ground-truth, though it requires access to logits of both models.
- **vs Contrastive Decoding**: CD runs both Expert and Amateur at inference time. This paper distills the contrastive signal into the Expert, avoiding dual-model overhead during inference but requiring extra sampling and distribution calculation before training.
- **vs RHO-1 / selective token training**: Methods like RHO-1 focus on token learning value. LightReasoner differs by defining token value based on the domain capability gap between models of the same family, without needing an external reference scorer.
- **Insights for follow-up work**: Expert-Amateur KL could be used as a general "learning value probe" for localized supervision construction in code repair, tool planning, or multimodal reasoning.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Using weak model failure modes to inversely enhance strong models is a highly distinctive perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 5 Experts and 7 math benchmarks with clear ablations, though cross-domain verification is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Method motivation and efficiency arguments are clear, though some tables in the cached text are densely formatted.
- **Value**: ⭐⭐⭐⭐⭐ Direct reference value for efficient reasoning fine-tuning, label-scarce scenarios, and selective training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew](judgemenot_personalizing_large_language_models_to_emulate_judicial_reasoning_in_.md)
- [\[ACL 2026\] GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models](grasprune_global_gating_for_budgeted_structured_pruning_of_large_language_models.md)
- [\[ACL 2026\] TLoRA: Task-aware Low Rank Adaptation of Large Language Models](tlora_task-aware_low_rank_adaptation_of_large_language_models.md)
- [\[ACL 2026\] Training-Free Test-Time Contrastive Learning for Large Language Models](training-free_test-time_contrastive_learning_for_large_language_models.md)
- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](../../ICLR2026/model_compression/landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)

</div>

<!-- RELATED:END -->
