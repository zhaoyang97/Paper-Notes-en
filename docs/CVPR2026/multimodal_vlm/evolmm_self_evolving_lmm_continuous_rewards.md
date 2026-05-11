---
title: >-
  [Paper Note] EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards
description: >-
  [CVPR 2026][Multimodal VLM][Self-evolving LMM] This paper proposes EvoLMM, a fully unsupervised self-evolving framework that derives a Proposer (generating image-grounded questions) and a Solver (answering those questions) from a single LMM. A continuous self-consistency reward — replacing discrete majority voting — forms a closed-loop training signal. Using only raw images (no annotations, no external reward models), EvoLMM achieves consistent gains of approximately 2–3% across eight multimodal mathematical reasoning benchmarks.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Self-evolving LMM
  - unsupervised self-improvement
  - continuous self-consistency reward
  - Proposer-Solver
  - visual mathematical reasoning
date: 2026-05-08
content_hash: b7ab2a6add640fa6
---

# EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards

**Conference**: CVPR 2026
**arXiv**: [2511.16672](https://arxiv.org/abs/2511.16672)
**Code**: [https://github.com/mbzuai-oryx/EvoLMM](https://github.com/mbzuai-oryx/EvoLMM) (open-source)
**Area**: Multimodal VLM / Self-Evolving Learning
**Keywords**: Self-evolving LMM, unsupervised self-improvement, continuous self-consistency reward, Proposer-Solver, visual mathematical reasoning

## TL;DR
This paper proposes EvoLMM, a fully unsupervised self-evolving framework that derives a Proposer (generating image-grounded questions) and a Solver (answering those questions) from a single LMM. A continuous self-consistency reward — replacing discrete majority voting — forms a closed-loop training signal. Using only raw images (no annotations, no external reward models), EvoLMM achieves consistent gains of approximately 2–3% across eight multimodal mathematical reasoning benchmarks.

## Background & Motivation

**Background**: Large multimodal models (LMMs) have made substantial progress in visual reasoning, yet their training pipelines still depend on (a) human-annotated data and (b) external reward models or evaluators, limiting autonomy and scalability.

**Limitations of Prior Work**: Self-evolving methods exist in the LLM domain (e.g., SQLM, Proposer-Solver-Judge), but applying them directly to multimodal settings introduces problems: discrete majority-voting rewards produce a large proportion of zero-reward updates during early-stage visual reasoning training, leading to instability. Existing multimodal self-improvement methods (ViPER, Vision-Zero) still rely on structured intermediate signals.

**Key Challenge**: Self-evolution requires effective internal training signals; however, discrete rewards cannot provide meaningful gradient feedback during the early phase when model outputs are highly variable, causing optimization stagnation.

**Goal**: Enable LMMs to self-improve multimodal reasoning capabilities through internal consistency, under fully unsupervised conditions.

**Key Insight**: Replace discrete majority voting with a continuous self-consistency reward to supply smooth gradient signals; use entropy-guided Proposer rewards to realize adaptive curriculum learning.

**Core Idea**: Continuous self-consistency rewards enable the Proposer and Solver to co-evolve smoothly, continuously improving visual reasoning using only raw images.

## Method

### Overall Architecture
A pretrained LMM (e.g., Qwen2.5-VL-7B) is decomposed into two roles sharing a backbone but with separate LoRA adapters: the Proposer generates visually grounded mathematical questions from images, and the Solver attempts to answer them. For each question, the Solver samples $N=5$ responses; a continuous reward is computed based on inter-response consistency and used to update both policies via REINFORCE with KL regularization. The entire loop requires only raw images — zero human annotations, zero external models.

### Key Designs

1. **Continuous Self-Consistency Solver Reward**

   - **Function**: Quantifies the degree of consistency among multiple Solver responses to the same question and converts it into a continuous training signal.
   - **Mechanism**: The reward equals the empirical probability of an answer across $N$ samples raised to the power $\gamma$, multiplied by a length penalty term. $\gamma$ controls reward softness (lower values amplify differences at intermediate consistency levels); the length penalty encourages concise response formats.
   - **Design Motivation**: Discrete majority voting only checks whether an answer constitutes a majority — partial agreement of 2/5 and 3/5 receives identical zero signal. The continuous reward provides meaningful positive gradients even under model uncertainty, avoiding early-training stagnation. Experiments confirm that discrete rewards yield near-zero and unstable early Solver rewards, whereas continuous rewards rise smoothly.
   - **Novelty**: SQLM uses discrete majority voting, which produces a large proportion of zero-reward updates in multimodal settings and leads to learning stagnation.

2. **Entropy-Guided Continuous Proposer Reward**

   - **Function**: Encourages the Proposer to generate questions of moderate difficulty — neither too easy nor too hard.
   - **Mechanism**: A Gaussian band-pass function is applied, maximizing the reward when the entropy $H$ of Solver responses is at an intermediate value. Low $H$ indicates the question is too easy (all answers agree); high $H$ indicates the question is too hard or ambiguous. The center parameter is $\mu_H = 0.90$ with bandwidth $\sigma_H = 0.35$.
   - **Design Motivation**: This implements adaptive curriculum learning. As the Solver improves, previously moderate questions become too easy (low entropy), prompting the Proposer to generate harder yet still solvable questions to obtain high rewards, naturally forming a progressive curriculum. Figure 6 clearly illustrates this emergent behavior.
   - **Novelty**: No external Judge module or manually designed difficulty criteria are required.

3. **KL-Regularized REINFORCE Optimization**

   - **Function**: Stabilizes policy gradient training and prevents excessive deviation from the pretrained model.
   - **Mechanism**: An exponential moving average baseline is used to reduce variance; a dynamic KL coefficient adaptively controls the degree of deviation. A tighter KL constraint is applied to the Solver for stability, while a looser constraint is applied to the Proposer to permit exploration.

### Training Details
- **Base model**: Qwen2.5-VL-7B; backbone frozen; two LoRA adapters
- **Training data**: ~6K raw images (no QA annotations) from ChartQA, AI2D, InfographicVQA, PlotQA, ChartX, and Geometry3K
- **Hardware**: 8× AMD MI250X GPUs, bfloat16 precision
- **Training steps**: 6,000 steps, batch size 1; Proposer updated every 5 steps
- **Hyperparameters**: $N=5$ samples, $\gamma=0.7$, learning rate $1\text{e-}6$

## Key Experimental Results

### Main Results (8 Multimodal Reasoning Benchmarks)

| Model | ChartQA | MathVista | MathVision | MathVerse | AI2D | ScienceQA | MMMU |
|-------|---------|-----------|------------|-----------|------|-----------|------|
| Qwen2.5-VL-7B Base | 84.00 | 68.46 | 23.91 | 43.78 | 82.61 | 88.30 | 51.11 |
| + Discrete Reward | 84.62 | 68.88 | 22.52 | 42.10 | 82.18 | 87.98 | 50.84 |
| + Continuous Reward (Ours) | 86.70 | 70.52 | 24.81 | 44.88 | 83.41 | 89.50 | 52.01 |
| Gain | +2.7 | +2.06 | +0.9 | +1.1 | +0.8 | +1.2 | +0.9 |

### Ablation Study (Parameter Update Strategies)

| Strategy | ChartQA | MathVista | ScienceQA | Notes |
|----------|---------|-----------|-----------|-------|
| LoRA | 86.70 | 70.52 | 89.50 | Best; preserves pretrained capabilities |
| QLoRA | 85.32 | 68.92 | 88.73 | Slight degradation from quantization noise |
| Full Fine-tune | 84.20 | 68.41 | 88.12 | Overfits without external supervision |

### Cross-Model Generalization

| Base Model | ChartQA Gain | MathVista Gain |
|-----------|-------------|----------------|
| Qwen2.5-VL-7B | 84.00 → 86.70 | 68.46 → 70.52 |
| InternVL3-8B | 82.40 → 84.97 | 65.20 → 67.20 |

### Key Findings
- **Continuous vs. discrete rewards**: Discrete rewards yield negative gains on MathVision (−1.39) and MathVerse (−1.68), while continuous rewards produce positive gains across all eight benchmarks.
- **LoRA >> Full Fine-tuning**: Under unsupervised self-evolution without external supervision, parameter-efficient fine-tuning outperforms full fine-tuning — the latter tends to overfit internal signals.
- **Emergent adaptive curriculum**: During training, the Proposer naturally transitions from generating overly simple or overly difficult questions toward moderate-difficulty questions; the entropy distribution shifts from a U-shape to a bell curve centered at intermediate values.
- **Cross-model effectiveness**: Consistent gains on both Qwen2.5-VL-7B and InternVL3-8B confirm the generality of the approach.

## Highlights & Insights
- **Continuous self-consistency reward** is the core contribution: using the empirical answer probability raised to the power $\gamma$ as a continuous signal avoids the all-or-nothing problem of discrete voting. This is the key innovation for upgrading self-consistency from an evaluation metric to a differentiable training signal — an insight generalizable to any scenario requiring internal consistency as a training objective.
- **Entropy band-pass Proposer reward** realizes curriculum learning without human intervention: no external difficulty annotations are needed, as the Solver's answer entropy naturally reflects question difficulty. This mechanism is transferable to any self-play training requiring adaptive difficulty regulation.
- **The comparison in Figures 3 and 4** is particularly instructive: it clearly demonstrates the fundamental difference in training dynamics between discrete and continuous rewards, providing key visualization for understanding the advantage of continuous rewards.
- **The cleanliness of the experimental setup** is commendable: the framework genuinely operates under a minimalist setting of "raw images + pretrained model only," with no hidden external dependencies.

## Limitations & Future Work
- Performance gains are modest (~2–3%), with a notable gap compared to supervised methods.
- Validation is limited to mathematical and chart reasoning; generalization to open-domain visual understanding remains unexplored.
- Training uses only 6K images and 6,000 steps; scaling laws have not been investigated.
- The quality of Proposer-generated questions has not been evaluated by human annotators; semantically vacuous questions may exist.
- Continuous rewards may not be robust to answers that are semantically equivalent but lexically distinct (e.g., "3.14" vs. "$\pi$").

## Related Work & Insights
- **vs. SQLM [Huang et al.]**: SQLM is the LLM-domain predecessor of EvoLMM, relying on discrete majority voting. EvoLMM demonstrates that multimodal settings necessitate replacing discrete rewards with continuous ones.
- **vs. Vision-Zero [Xu et al.]**: Vision-Zero achieves self-evolution through a "Who's the Spy" game but depends on GPT-4o/Gemini to generate image pairs. EvoLMM requires no external models whatsoever.
- **vs. ViPER [Zhang et al.]**: ViPER uses reconstruction objectives as self-supervision; EvoLMM uses consistency — the latter is more general and does not require image generation capabilities.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Continuous self-consistency rewards and entropy band-pass Proposer rewards are technical highlights, though the overall framework inherits from SQLM.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive ablations across 8 benchmarks, 4 backbones, and 3 fine-tuning strategies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The visual comparison of discrete vs. continuous rewards (Figures 3 and 4) is highly intuitive.
- **Value**: ⭐⭐⭐⭐ — Offers important reference value for unsupervised multimodal self-evolution, though absolute performance gains are limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory](evolving_contextual_safety_in_multi-modal_large_language_models_via_inference-ti.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] Evolving Prompt Adaptation for Vision-Language Models](evolving_prompt_adaptation_for_vision-language_models.md)
- [\[CVPR 2026\] EvoPrompt: Evolving Prompt Adaptation for Vision-Language Models](evolving_prompt_adaptation_for_vision-language_models.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr_turbo_merged_checkpoint_free_teacher.md)

</div>

<!-- RELATED:END -->
