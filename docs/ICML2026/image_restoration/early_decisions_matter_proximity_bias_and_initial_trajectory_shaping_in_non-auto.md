---
title: >-
  [Paper Note] Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models
description: >-
  [ICML 2026][Image Restoration][dLLM] This paper systematically characterizes the failure mechanisms of masked diffusion language models (dLLMs) under **fully non-autoregressive (NAR) decoding**—where proximity bias cause…
tags:
  - "ICML 2026"
  - "Image Restoration"
  - "dLLM"
  - "Non-autoregressive decoding"
  - "proximity bias"
  - "EOS overflow"
  - "lightweight planner"
date: 2026-05-08
content_hash: f95a9445913737b6
---

# Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models

**Conference**: ICML 2026  
**arXiv**: [2604.10567](https://arxiv.org/abs/2604.10567)  
**Code**: None  
**Area**: LLM Efficiency / Diffusion Language Models / Non-autoregressive Decoding  
**Keywords**: dLLM, Non-autoregressive decoding, proximity bias, EOS overflow, lightweight planner

## TL;DR
This paper systematically characterizes the failure mechanisms of masked diffusion language models (dLLMs) under **fully non-autoregressive (NAR) decoding**—where proximity bias causes confidence-based sampling to degrade into reverse autoregression and premature EOS saturation. By introducing a 5M-parameter lightweight planner and EOS temperature annealing to intervene in unmasking positions **only during the first step**, NAR decoding for LLaDA 8B is improved by 2.8–4.3 points on reasoning tasks like GSM8K with almost no extra overhead.

## Background & Motivation

**Background**: dLLMs (LLaDA, Dream, MDLM, etc.) model text via mask-and-predict mechanisms. Theoretically, they offer two major advantages: **parallelism** (decoding multiple tokens at once) and **bidirectionality** (utilizing both prefix and suffix context), making them potential alternatives to autoregressive LLMs.

**Limitations of Prior Work**: In practical deployment, **fully NAR decoding** almost never produces coherent text. State-of-the-art methods (LLaDA, Block Diffusion, etc.) often revert to **semi-autoregressive (semi-AR)** decoding—generating tokens block-by-block—which sacrifices the advantages of bi-directional parallelism and is particularly detrimental to tasks requiring global structural planning.

**Key Challenge**: Is the failure due to the insufficient capability of the dLLM backbone itself, or a structural flaw in the NAR decoding strategy? Previous research mostly relied on the empirical conclusion that "semi-AR is more stable," lacking a breakdown of NAR temporal dynamics. Consequently, improvements have relied on bypassing the issue by adding more AR priors.

**Goal**: (1) Identify the root causes behind the failure of confidence-based NAR decoding; (2) Design a *minimal intervention* that makes fully NAR decoding viable for reasoning tasks without fine-tuning the backbone or introducing block structures.

**Key Insight**: By tracking unmasking positions along the *time axis*, the authors identified two mutually reinforcing biases—**proximity bias** (newly decoded tokens tend to be adjacent to previous ones) and **EOS dominance** (the EOS logit is typically the largest under high uncertainty). This leads to a critical asymmetry: **the position decisions in the first step have a disproportionately large impact on the final quality of the entire trajectory**.

**Core Idea**: Instead of injecting token-level temperature across all steps, interventions should be concentrated on the *position selection of the very first step*. Using a lightweight planner to select the initial unmasking positions and applying temperature annealing to EOS logits in early stages is sufficient to flip the entire trajectory quality.

## Method

### Overall Architecture
The method is built upon standard MDLM reverse decoding: at each step $d$, the model must (a) predict tokens for all masked positions and (b) select a subset $\mathcal{U}_d$ to unmask. This work **does not change** the backbone $\theta$ or the greedy strategy of subsequent steps. It only replaces the "Top-1 confidence" position selection $\mathcal{U}_1$ in the first step with a planner score and applies time-decaying inverse-temperature scaling to EOS logits across all steps. The pipeline uses a progressive schedule where $|\mathcal{U}_d|<L/T$ in early stages, releasing more tokens later to allow the planner better discriminative power.

### Key Designs

1.  **Proximity Bias Diagnosis and Temporal Asymmetry**:
    *   **Function**: Quantify "why NAR fails" into two measurable observations to provide precise targets for intervention.
    *   **Mechanism**: Fix $L=256$ and scan $T\in\{32,...,256\}$. NAR performance is found to *decrease* monotonically as $T$ increases (contrary to the intuition that more steps are better for dLLMs). Visualizations confirm that: (i) Step 1 consistently unmasks EOS at the end of the sequence; (ii) subsequent steps unmask positions adjacent to the previous ones, creating a "reverse AR from tail to head" pattern; (iii) on GSM8K, an average of 144.6 out of 256 slots are occupied by EOS. Pass@k benchmarks show that injecting randomness *only* into Step 1 position selection (greedy otherwise) scores 7+ points higher than full-trajectory temperature sampling. Resampling experiments on 256 trajectories show that the final accuracy gap between correct and incorrect early trajectories is $\sim 16{-}33$ points with non-overlapping confidence intervals, proving "the first step determines the whole."
    *   **Design Motivation**: To frame "NAR instability" as a mechanism involving *proximity bias × EOS dominance × temporal asymmetry*, providing a theoretical basis for intervention only at the first step.

2.  **Lightweight Planner $\pi_\phi(\mathcal{U}_1\mid h_\mathcal{S})$**:
    *   **Function**: Select the optimal $\mathcal{U}_1^\star$ from $P=32$ random candidate position sets $\{\mathcal{S}^i\}$ in Step 1 to maximize the final task reward $R(\mathbf{z}_0)$.
    *   **Mechanism**: The planner is a 2-layer Transformer encoder with a position-wise scoring head. It **only processes the backbone's last-layer hidden states** $h_\mathcal{S}$ at the candidate positions $\mathcal{S}$ (ignoring the full context). It outputs a scalar score for each token, which is averaged for the candidate set. Training data is generated offline: for each sample, $S=32$ candidates $\mathcal{U}_1$ are sampled, and trajectories are completed greedily to obtain 0/1 labels. The planner is trained via BCE to $\max_\phi \mathbb{E}_{\mathcal{U}_1\sim\pi_\phi}[R(\mathbf{z}_0)]$. At inference, the highest-scoring candidate is chosen for $\mathcal{U}_1$, and *all subsequent steps follow confidence-based greedy decoding*.
    *   **Design Motivation**: To view NAR decoding as an "opening move problem." Since the first step is decisive, a 5M parameter "opening coach" (negligible compared to the 8B backbone) is highly efficient. Looking only at hidden states at candidate positions prevents the planner from simply mimicking backbone confidence.

3.  **EOS Temperature Annealing**:
    *   **Function**: Apply time-decaying inverse-temperature $\lambda_d$ to the EOS token logit across all steps (linear annealing from $\lambda_T=3$ to $1$). This **only affects the ranking of position selection** and does not change actual token prediction.
    *   **Mechanism**: Dividing the EOS logit by $\lambda_d$ before the softmax significantly lowers its priority during unmasking selection. Once positions are chosen, tokens are selected via standard greedy argmax, preserving the model's natural stopping behavior in later steps. Combined with a progressive schedule, this increases the number of effective (non-EOS) tokens on GSM8K from 157.2 to 188.6.
    *   **Design Motivation**: Proximity bias propagates forward once it starts from early EOS tokens. The most economical disruption is not retraining but temporarily weakening EOS influence during the "who to unmask" decision phase.

### Loss & Training
- Planner Loss: Trajectory-level BCE using 0/1 correctness labels from offline rollouts.
- Only the planner $\phi$ is trained; the backbone $\theta$ remains frozen. Planner $\approx$ 5M parameters.
- Progressive Schedule: $B<L/T$ in early steps to make differences between candidate positions more separable.
- Hyperparameters: $T=32, L=256, P=32, \lambda_T=3$ linear annealing.

## Key Experimental Results

### Main Results (LLaDA 8B Instruct, $T=32, L=256$, Fully NAR)

| Dataset | Top-1 (greedy) | + Planner | + EOS Anneal | + Both | Prob Margin + Both |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GSM8K | 46.6 | 55.0 | 50.9 | **56.8** | **58.6** |
| MATH | 19.2 | 22.4 | 22.4 | 22.8 | **23.0** |
| Countdown | 42.2 | 44.1 | **46.1** | 43.8 | 45.3 |
| Sudoku | 71.2 | 65.2 | 63.6 | 67.0 | 69.5 |
| **Avg** | 44.8 | 46.7 | 45.7 | **47.6** | **49.1** |

> Under the same settings, semi-AR averages only 27.0; random initial position averages 37.1; full token temperature sampling is 44.4. This proves NAR can be "rescued" and pushed beyond semi-AR performance.

### Generalization Across Budgets ($T=64,128$, planner trained only at $T=32$)

| Setting ($T$) | Top-1 | Ancestral | Temperature | Init.Pos | + Planner | + EOS | Both |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Avg @ 64 | 39.7 | 26.6 | 41.9 | 41.6 | 48.0 | 47.4 | **50.2** |
| Avg @ 128 | 37.7 | 28.2 | 39.2 | 43.0 | 47.8 | 48.1 | **52.8** |

Performance of the baseline decreases as $T$ increases (EOS dominance intensifies), while metadata methods improve, showing the planner learns a universal "opening priority" rather than over-fitting a specific step count.

### Key Findings
- **Proximity bias + premature EOS is the dual engine of NAR failure**: Neither phenomenon alone explains the results; together, they explain why larger $T$ performs worse and why semi-AR is usually superior.
- **Strong Temporal Asymmetry**: Delaying randomness from Step 1 to Step 5 causes a collapse in pass@k scores. Correct vs. incorrect early trajectories lead to a stable $\sim 16{-}33$ point gap.
- **Sensitivity of Structured Tasks**: In 1-shot Sudoku with strict templates, "blind randomness" is worse than greedy (71.2 $\to$ 48.3). However, the learned planner captures structural priors, reducing the loss from 23 points to 6.
- **Orthogonality with Heuristics**: Combining the planner with Probability Margin on GSM8K improves scores from 47.2 $\to$ 58.6.
- **Almost Zero Overhead**: 5M planner only runs at Step 1; EOS annealing is a simple scalar scaling. Gains saturate at $P=32$ candidates.

## Highlights & Insights
- **Counter-intuitive "Opening Move" Theory**: In a theoretically "globally parallel" dLLM, critical decisions are concentrated in the first step. This flips the default assumption of "uniform stochasticity" in diffusion sampling.
- **Diagnosis-Driven Minimal Intervention**: The intervention (first-step only + EOS logit suppression) follows directly from the *proximity bias × EOS × temporal asymmetry* diagnosis, providing a model for "explain-then-fix" research.
- **Transferability of Planner Design**: The paradigm of using *local hidden states $\to$ global trajectory reward* with a 5M model can be migrated to any mask-and-predict model (code generation, image token diffusion, etc.).

## Limitations & Future Work
- **Dependency on Task-Level 0/1 Rewards**: It is difficult to get clear labels for sparse reward open generation (writing, dialogue); process rewards or LLM judges are needed.
- **Scale and Domain**: Validated primarily on LLaDA/Dream 7-8B; performance on larger dLLMs or multilingual tasks is unverified.
- **Comparison with First-Step Fine-tuning**: It remains to be seen if RL fine-tuning the backbone's first step would outperform a 5M planner.
- **Sub-greedy Performance on Structured Tasks**: On Sudoku, "Both" (67.0) is still lower than "Top-1 greedy" (71.2), suggesting the planner might sacrifice task priors for diversity.

## Related Work & Insights
- **vs. Rainbow Padding (Kim et al., 2026)**: Both address EOS overflow, but Rainbow requires token replacement and fine-tuning. This work is zero-backbone-change and attributes EOS dominance to proximity bias.
- **vs. Block Diffusion / LLaDA semi-AR**: semi-AR uses strong structural priors to bypass NAR instability but loses parallelism. This work enables fully NAR to exceed semi-AR accuracy (47.6 vs 27.0).
- **vs. Convolutional NAR (Seo et al., 2025)**: While Seo focuses on *spatial* consistency via filters, this work focuses on the *temporal* axis and identifies first-step asymmetry.
- **vs. Per-step planners (Peng et al., 2025)**: Per-step planners are active throughout and have high overhead. This work shows most gains are captured by intervening only at step 1.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Consistent Diffusion Language Models](consistent_diffusion_language_models.md)
- [\[ICML 2026\] Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models](plan_for_speed_dilated_scheduling_for_masked_diffusion_language_models.md)
- [\[ICML 2026\] Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges](structured_diffusion_bridges_inductive_bias_for_denoising_diffusion_bridges.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](../../ICLR2026/image_restoration/activation_steering_for_masked_diffusion_language_models.md)
- [\[ICML 2026\] Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner](coevolutionary_continuous_discrete_diffusion_make_your_diffusion_language_model_.md)

</div>

<!-- RELATED:END -->
