---
title: >-
  [Paper Note] HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench
description: >-
  [ICML 2026][Code Intelligence][SWE-bench] Traditional PPL on SWE-bench is disrupted by the "long-context tax" and fails to predict post-SFT agent capabilities. This paper proposes the "Entropy Compression Hypothesis" and…
tags:
  - "ICML 2026"
  - "Code Intelligence"
  - "SWE-bench"
  - "Mid-training Evaluation"
  - "Top-k Entropy"
  - "High-entropy Decision Points"
  - "Entropy Compression"
date: 2026-05-08
content_hash: b260efd350726202
---

# HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench

**Conference**: ICML 2026  
**arXiv**: [2601.20255](https://arxiv.org/abs/2601.20255)  
**Code**: None (Meituan LongCat team)  
**Area**: Code Intelligence / LLM Evaluation / Mid-training Metrics  
**Keywords**: SWE-bench, Mid-training Evaluation, Top-k Entropy, High-entropy Decision Points, Entropy Compression

## TL;DR
Traditional PPL on SWE-bench is disrupted by the "long-context tax" and fails to predict post-SFT agent capabilities. This paper proposes the "Entropy Compression Hypothesis" and the HE-SNR metric. By calculating the Signal-to-Noise Ratio only at "high-entropy decision points" where Top-10 entropy exceeds $(\ln 3 + \ln 4)/2$, the method achieves a Pearson correlation of 0.96 and a Kendall consistency of 0.98 with downstream SWE-bench scores.

## Background & Motivation

**Background**: SWE-bench has become the de facto standard for evaluating LLM software engineering capabilities. SOTA systems (SWE-RL, Kimi-Dev, SWE-Dev) rely on applying SFT to instruction models. The mid-training phase (between PT and SFT) determines a model's SWE "potential," but the only way to evaluate a mid-training checkpoint is to consume massive compute for full SFT and subsequent SWE-bench testing.

**Limitations of Prior Work**: (1) PPL/BPC correlates poorly with downstream SWE-bench scores; when Top-1 accuracy exceeds 90%, PPL measures "parroting" rather than reasoning. (2) When using RoPE to extend context, models immediately suffer a "long-context tax"—Top-1 and PPL temporarily degrade while actual SWE capabilities improve, causing PPL to move in the opposite direction. (3) Improvements like LongPPL only address positional bias in retrieval-based long-context tasks (RULER) and are not tailored for key tokens in agentic reasoning.

**Key Challenge**: Does intelligence equal "compression"? The traditional Compression-Intelligence Hypothesis uses scalar PPL as a measure of compression, but PPL measures "repeater precision." True reasoning requires "rational hesitation" among multiple candidates—this is compression at the distribution level, which scalar information theory fails to capture.

**Goal**: (1) Identify an SFT-invariant mid-training signal, (2) Resist the "long-context tax," and (3) Provide reliable downstream performance predictions using minimal data (500 trajectories).

**Key Insight**: By plotting Top-10 entropy distributions across multiple models and checkpoints, the authors found a universal law: predicted tokens (excluding Top-1) concentrate at "natural boundaries" such as $\ln 2$ and $\ln 3$. Stronger models compress "scattered $\ln 4$ uncertainty" into "rational $\ln 3$ hesitation." This suggests that reasoning capability can be measured by how well a model "folds" uncertainty into specific set sizes.

**Core Idea**: Upgrade "compression" from the scalar PPL level to the distribution level. Calculate the Signal-to-Noise Ratio (HE-SNR) of target token probability vs. entropy only at "high-entropy decision points" where Top-10 entropy exceeds the midpoint of $\ln 3$-$\ln 4$. Strictly filter out style tokens in the chain-of-thought to evaluate only the executable logic of action tokens.

## Method

### Overall Architecture
The framework consists of two parts: (1) **Data Filtering**—Retaining only Action segments from 500 SWE-bench-Verified trajectories (discarding Thoughts as they are dominated by SFT style), then using regex to remove XML tags, AST to remove comments, and stripping whitespace; (2) **Metric Calculation**—Calculating Top-10 entropy $H_{top10}(x_t)$ for each retained token, filtering the high-entropy decision set $\mathcal{H} = \{t : H_{top10} > \epsilon \text{ and } x_t \in C_{10}\}$, and computing HE-SNR as the mean ratio of target probability to entropy over the high-entropy set.

### Key Designs

1.  **Entropy Compression States and "$\ln 3$" Transition**:
    - **Function**: Infers the size of the candidate set the model is "hesitating" over based on Top-$k$ entropy peak positions.
    - **Mechanism**: According to Jensen's inequality, the upper bound of Top-$k$ re-normalized entropy is $\ln k$, achieved if and only if candidate probabilities are uniform. Observing entropy peaks at $\ln 2, \ln 3, \ln 4$ means the model is hesitating equally among 2, 3, or 4 candidates. Strong models exhibit a "Shift to $\ln 3$," folding uncertainty from $\ln 4$ to $\ln 3$.
    - **Design Motivation**: Traditional PPL mixes all candidates into a scalar, obscuring set size. Top-$k$ entropy peaks directly reveal the number of candidates under consideration. "Fewer but rational hesitations" are a hallmark of reasoning depth. In MoE-A26B, observation tokens exhibit a $\ln 10$ peak (corresponding to random digits 0-9), confirming $\ln 10$ represents aleatoric uncertainty rather than intelligence.

2.  **High-Entropy SNR (HE-SNR) Metric**:
    - **Function**: Quantifies the ratio of target token signal to entropy noise at high-entropy decision points that are resistant to SFT "stylization."
    - **Mechanism**: $\text{HE-SNR} = \frac{1}{|\mathcal{H}|}\sum_{t \in \mathcal{H}} \frac{p(x_t)}{H_{top10}(x_t)}$, where $\mathcal{H} = \{t : H_{top10}(x_t) > \epsilon, x_t \in C_{10}(x_t)\}$, with a threshold $\epsilon = (\ln 3 + \ln 4)/2 \approx 0.897$. The condition $x_t \in C_{10}$ filters out style tokens that deviate completely, preventing skewing by extreme samples.
    - **Design Motivation**: SFT compresses many originally high-entropy tokens to the $\ln 1$-$\ln 3$ range. Therefore, the "residual uncertainty between $\ln 3$ and $\ln 4$" represents the parts SFT cannot easily change—the true "hard cases" of reasoning. HE-SNR measures relative confidence in target tokens for these cases, correlating highly with downstream SWE-bench scores.

3.  **Token-level Filtering Pipeline**:
    - **Function**: Extracts tokens that reflect executable logic from raw trajectories.
    - **Mechanism**: Discards Observations (input context) and Thoughts (style-dominated). For Action segments, regex removes XML tags and markdown formatting; Python AST parsing removes code comments; finally, redundant whitespace is cleaned. Tokens are mapped from character-level labels via offset alignment.
    - **Design Motivation**: Ablations show that switching from Thinking to Action improves Pearson correlation from 0.558 to 0.967. Layering XML/whitespace/AST comment filtering pushes Kendall $\tau$ to a peak of 0.979. This proves SFT style artifacts are the primary source of interference for predictive signals.

### Loss & Training
HE-SNR is an evaluation metric, not a training loss. During validation, HE-SNR from multiple mid-training checkpoints is correlated with the average of 3 SWE-bench-Verified Pass@1 evaluations after SFT.

## Key Experimental Results

### Main Results

| Metric | Calculation Scope | Pearson $r$ vs SWE-bench | Resists "Long-context Tax" |
| :--- | :--- | :--- | :--- |
| PPL | All tokens | Weak (Negative) | No, inverted at Step 200 |
| HE-PPL | High-entropy set, all tokens | Moderate | No |
| HE-PPL | High-entropy set, filtered Action | Strong | No |
| **HE-SNR** | High-entropy set, filtered Action | **Strongest** (Linear + Monotonic) | **Yes** |

Model scales covered: MoE-A3B (68B total / 3B active) and MoE-A26B (560B total / 26B active), with context extended from 32K up to 128K.

### Ablation Study

| Token Type | Filtering Strategy | Pearson $r$ | Kendall $\tau$ |
| :--- | :--- | :--- | :--- |
| Thinking | No filtering | 0.558 | 0.519 |
| Action | No filtering | 0.967 | 0.944 |
| Action | + Remove XML | 0.953 | 0.956 |
| Action | + Remove Whitespace/Symbols | 0.952 | 0.968 |
| Action | + AST Comment Removal (Full) | **0.965** | **0.979** |

Threshold Sensitivity: In the range of $\ln 2$ to $\ln 5$, $(\ln 3 + \ln 4)/2$ is near-optimal and forms a robust plateau, requiring no fine-tuning.

### Key Findings
- **The $\ln 3$ Transition Phenomenon Generalizes**: A three-peak structure ($\ln 1, \ln 2, \ln 3$) is observed in Qwen2.5-72B (Dense), DeepSeek-V3 (MoE), and across 5000 math QA pairs. $\ln 2$ corresponds to "general reasoning" (common in NL), while $\ln 3$ corresponds to "strict logical reasoning" (common in code/math).
- **The SFT "Alignment Tax" is Exposed in High-Entropy Sets**: Post-SFT global PPL improves, but HE-PPL and HE-SNR actually degrade in high-entropy sets. This suggests SFT trades "complex reasoning" for "stylization," providing a mechanistic explanation for the "alignment tax."
- **$|\mathcal{H}|$ Spikes at 128K training step 200**, where traditional HE-PPL also degrades. However, HE-SNR (signal-to-noise ratio) remains stable, indicating that "quantity of hard problems" and "mastery of hard problems" should be viewed separately.

## Highlights & Insights
- **From Scalar to Distributional Compression**: Elevates the "compression = intelligence" hypothesis to the distributional level and identifies $\ln k$ as quantifiable "natural boundaries," offering higher theoretical value than purely engineering metrics.
- **"Rational Hesitation" Perspective**: While traditional views treat high entropy as noise, this paper treats $\ln 2, \ln 3$ as "healthy states of self-aware uncertainty," which are signs of high-quality reasoning. This provides a new framework for "exploration vs. exploitation" in RL chains-of-thought.
- **AST + Offset Alignment Filtering Pipeline**: By aligning character-level masking with token-level evaluation, this pipeline can be reused for any LLM evaluation scenario requiring the removal of noise tokens based on code structure.
- **Translatability to PRM**: High-entropy decision points coincide with the "forking points" of interest for Process Reward Models (PRM). The HE-SNR concept can be seamlessly migrated to PRM training data construction.

## Limitations & Future Work
- The threshold $\epsilon$ is static. Since "rational hesitation boundaries" may vary across different tasks, the authors acknowledge the need for adaptive thresholds.
- High-entropy tokens are influenced by coding style. Different ways of writing the same logic result in different high-entropy sets; the authors propose using code canonicalization or style transfer for standardization.
- Primary validation focused on MoE-A3B and MoE-A26B. Although the "$\ln 3$ phenomenon" was validated on Qwen2.5-72B and DeepSeek-V3, full HE-SNR vs. SWE-bench correlation curves for other model families are pending.
- Currently limited to SWE-bench tasks. Whether higher-order ($k \geq 4$) entropy compression states appear in more complex tasks remains an open question.

## Related Work & Insights
- **vs. PPL / BPC** (Kaplan, Huang): Scalar compression metrics only reflect repetition. This paper uses distributional entropy peaks to capture the "reasoning depth" dimension.
- **vs. LongPPL** (Fang et al., ICLR 2025): LongPPL addresses positional bias in retrieval tasks for instructed models; HE-SNR targets base models in agentic SWE tasks with a different focus.
- **vs. Beyond 80/20 High-Entropy Token RL** (Wang et al., 2025): They identified high-entropy tokens as keys for RL optimization; this paper reverses that observation to prove they are also keys for "evaluating reasoning potential."

## Rating
- Novelty: ⭐⭐⭐⭐ The "Entropy Compression Hypothesis + HE-SNR" is a relatively novel theoretical framing, though the underlying components are simple combinations of Top-$k$ entropy.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough validation covering 3B-560B checkpoints, 32K/128K stages, and Dense + MoE architectures.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear narrative progressing from PPL failure to HE-SNR, supplemented by detailed case studies of $\ln k$ candidate distributions in the appendix.
- Value: ⭐⭐⭐⭐ Directly addresses the high-cost pain point of "selecting the right mid-training checkpoint." Using 500 trajectories (12M tokens) for reliable downstream prediction offers significant practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale](swe-rebench_v2_language-agnostic_swe_task_collection_at_scale.md)
- [\[ICML 2026\] Entropy-informed Decoding: Adaptive Information-Driven Branching](entropy-informed_decoding_adaptive_information-driven_branching.md)
- [\[ICML 2026\] Probability-Entropy Calibration: An Elastic Indicator for Adaptive Fine-tuning](probability-entropy_calibration_an_elastic_indicator_for_adaptive_fine-tuning.md)
- [\[ICML 2026\] Pull Requests as a Training Signal for Repo-Level Code Editing](pull_requests_as_a_training_signal_for_repo-level_code_editing.md)
- [\[NeurIPS 2025\] Searching Latent Program Spaces](../../NeurIPS2025/code_intelligence/searching_latent_program_spaces.md)

</div>

<!-- RELATED:END -->
