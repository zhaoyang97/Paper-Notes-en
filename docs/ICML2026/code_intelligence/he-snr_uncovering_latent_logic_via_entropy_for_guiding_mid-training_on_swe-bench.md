---
title: >-
  [Paper Note] HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench
description: >-
  [ICML 2026][Code Intelligence][SWE-bench] On SWE-bench, traditional PPL is affected by the "long context tax" and cannot predict post-SFT agent capabilities. This paper proposes the "entropy compression hypothesis" and t…
tags:
  - "ICML 2026"
  - "Code Intelligence"
  - "SWE-bench"
  - "Mid-Training Evaluation"
  - "Top-k Entropy"
  - "High-Entropy Decision Points"
  - "Entropy Compression"
date: 2026-05-08
content_hash: 42fe548a9f4ad9bc
---

# HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench

**Conference**: ICML 2026  
**arXiv**: [2601.20255](https://arxiv.org/abs/2601.20255)  
**Code**: None (Meituan LongCat Team)  
**Area**: Code Intelligence / LLM Evaluation / Mid-Training Metrics  
**Keywords**: SWE-bench, Mid-Training Evaluation, Top-k Entropy, High-Entropy Decision Points, Entropy Compression

## TL;DR
On SWE-bench, traditional PPL is affected by the "long context tax" and cannot predict post-SFT agent capabilities. This paper proposes the "entropy compression hypothesis" and the HE-SNR metric, which computes the signal-to-noise ratio only at "high-entropy decision points" where Top-10 entropy exceeds $(\ln 3 + \ln 4)/2$. This achieves a Pearson correlation of 0.96 and Kendall consistency of 0.98 with downstream SWE-bench scores.

## Background & Motivation

**Background**: SWE-bench has become the de facto standard for evaluating LLM software engineering capabilities. SOTA systems (SWE-RL, Kimi-Dev, SWE-Dev) all rely on SFT over instruction models. The mid-training phase (between PT and SFT) determines the model's "potential" for SWE, but to assess the quality of a mid-training checkpoint, the only way is to expend significant compute to run full SFT and then test on SWE-bench.

**Limitations of Prior Work**: (1) PPL/BPC correlates poorly with downstream SWE-bench scores, especially when Top-1 accuracy exceeds 90%; PPL mainly measures "parroting" rather than reasoning. (2) When using RoPE to extend context, models immediately suffer from the "long context tax"—Top-1 and PPL both temporarily worsen, but actual SWE capability improves, with PPL even reversing. (3) Improvements like LongPPL only address positional bias in retrieval-style long context (RULER), not the key tokens in agentic reasoning tasks.

**Key Challenge**: Is intelligence equivalent to "compression"? The traditional Compression-Intelligence Hypothesis uses scalar PPL as a compression metric, but PPL measures "repetition accuracy." True reasoning requires "reasonable hesitation" among multiple candidates—a distributional compression that scalar information theory cannot capture.

**Goal**: (1) Find a mid-training signal that is SFT-invariant, (2) robust to the "long context tax," (3) reliably predict downstream performance with minimal data (500 trajectories).

**Key Insight**: By plotting Top-10 entropy distributions across multiple models and checkpoints, a universal pattern emerges—non-Top-2 predicted tokens cluster at "natural boundaries" like $\ln 2, \ln 3$, etc. Stronger models compress "scattered $\ln 4$ uncertainty" into "$\ln 3$ reasonable hesitation." This suggests reasoning ability can be measured by how much uncertainty the model folds into a smaller candidate set.

**Core Idea**: Upgrade "compression" from scalar PPL to the distributional level—compute the signal-to-noise ratio (HE-SNR) of target token probability to entropy only at "high-entropy decision points" where Top-10 entropy exceeds the midpoint between $\ln 3$ and $\ln 4$, strictly filtering out style tokens in the chain-of-thought and evaluating only the executable logic of action tokens.

## Method

### Overall Architecture
Two components: (1) **Data Filtering**—from 500 SWE-bench-Verified trajectories, retain only the Action segments (excluding Thought, as it is SFT-style dominated), then use regex to remove XML tags, AST to remove comments, and strip whitespace; (2) **Metric Calculation**—for each retained token, compute Top-10 entropy $H_{top10}(x_t)$, filter out the high-entropy decision set $\mathcal{H} = \{t : H_{top10} > \epsilon \text{ and } x_t \in C_{10}\}$, then compute HE-SNR as the mean of target probability over entropy on the high-entropy set.

### Key Designs

1. **Entropy Compression State and "$\ln 3$" Shift Phenomenon**:

    - Function: Use the specific peak positions of Top-$k$ entropy to infer "how many candidate tokens the model hesitates among."
    - Mechanism: By Jensen's inequality, the upper bound of Top-$k$ normalized entropy is $\ln k$, achieved only when candidate probabilities are uniform. Thus, entropy peaks at $\ln 2, \ln 3, \ln 4$ indicate the model hesitates equally among 2, 3, or 4 candidates; strong models shift from $\ln 4$ to $\ln 3$ ("Shift to $\ln 3$"), folding uncertainty into a smaller set.
    - Design Motivation: Traditional PPL mixes all candidates into a scalar, obscuring "set size." The peak of Top-$k$ entropy directly reveals how many candidates the model hesitates among; "few but reasonable hesitation" is a hallmark of deep reasoning. MoE-A26B shows a $\ln 10$ peak on Observation tokens, corresponding to "random digits 0-9," confirming that $\ln 10$ reflects aleatoric uncertainty, not intelligence.

2. **High-Entropy SNR (HE-SNR) Metric**:

    - Function: Quantifies the ratio of target token signal to entropy noise at high-entropy decision points, which are hard for SFT to "stylize."
    - Mechanism: $\text{HE-SNR} = \frac{1}{|\mathcal{H}|}\sum_{t \in \mathcal{H}} \frac{p(x_t)}{H_{top10}(x_t)}$, where $\mathcal{H} = \{t : H_{top10}(x_t) > \epsilon, x_t \in C_{10}(x_t)\}$, with threshold $\epsilon = (\ln 3 + \ln 4)/2 \approx 0.897$. The condition $x_t \in C_{10}$ filters out "completely off-style tokens," avoiding bias from extreme samples.
    - Design Motivation: SFT compresses many originally high-entropy tokens to $\ln 1$-$\ln 3$, so the "residual uncertainty between $\ln 3$ and $\ln 4$" is precisely the part SFT cannot change—the model's true "reasoning hard cases." HE-SNR measures "how much relative confidence the model assigns to the target token on these hard cases," which should correlate highly with downstream SWE-bench scores.

3. **Token-Level Filtering Pipeline**:

    - Function: Retain only tokens reflecting executable logic from raw trajectories.
    - Mechanism: Discard Observation (input context) and Thought (style-dominated); for Action segments, use regex to remove XML tags and markdown, Python AST to remove code comments, and finally clean up extra whitespace; label at the character level, then align to token level via offset mapping.
    - Design Motivation: Ablation shows switching from Thinking to Action raises Pearson from 0.558 to 0.967; adding XML/whitespace/AST comment filtering further boosts Kendall $\tau$ from 0.944 to 0.979. This demonstrates that SFT style artifacts are the largest source of prediction noise.

### Loss & Training
HE-SNR is an evaluation metric, not a training loss. During validation, HE-SNR of multiple mid-training checkpoints is correlated with post-SFT SWE-bench-Verified Pass@1 (average of 3 runs).

## Key Experimental Results

### Main Results

| Metric | Scope | Pearson $r$ vs SWE-bench | Robust to "Long Context Tax" |
|--------|-------|--------------------------|------------------------------|
| PPL | All tokens | Weak (inverse) | No, inversion at Step 200 |
| HE-PPL | High-entropy set, all tokens | Medium | No |
| HE-PPL | High-entropy set, filtered Action | Strong | No |
| **HE-SNR** | High-entropy set, filtered Action | **Strongest** (linear + monotonic) | **Yes** |

Model scale: MoE-A3B (68B total / 3B active) and MoE-A26B (560B total / 26B active), context length from 32K to 128K.

### Ablation Study

| Token Type | Filtering Strategy | Pearson $r$ | Kendall $\tau$ |
|------------|-------------------|-------------|----------------|
| Thinking | None | 0.558 | 0.519 |
| Action | None | 0.967 | 0.944 |
| Action | +Remove XML | 0.953 | 0.956 |
| Action | +Remove whitespace/symbols | 0.952 | 0.968 |
| Action | +AST comment removal (full) | **0.965** | **0.979** |

Threshold sensitivity: In the range $\ln 2$ to $\ln 5$, $(\ln 3 + \ln 4)/2$ is empirically near-optimal and forms a robust plateau, requiring no fine-tuning.

### Key Findings
- **$\ln 3$ shift generalizes across architectures**: Qwen2.5-72B (Dense), DeepSeek-V3 (MoE), and 5000 math QA samples all show the $\ln 1, \ln 2, \ln 3$ tri-modal structure. $\ln 2$ corresponds to "general reasoning" (common in natural language), $\ln 3$ to "strict logical reasoning" (common in code/math).
- **SFT "alignment tax" is exposed in the high-entropy set**: SFT improves global PPL, but HE-PPL and HE-SNR degrade on the high-entropy set—indicating SFT trades "stylization" for "complex reasoning," providing a mechanistic explanation for the "alignment tax."
- **$|\mathcal{H}|$ spikes at 128K training step 200**; traditional HE-PPL also degrades, but HE-SNR (signal-to-noise ratio) remains robust, suggesting "number of hard cases" and "mastery of hard cases" should be considered separately.

## Highlights & Insights
- **From scalar to distributional compression**: Elevates the classic "compression = intelligence" hypothesis to the distributional level, identifying $\ln k$ as quantifiable "natural boundaries," with higher theoretical value than pure engineering metrics.
- **Perspective of "reasonable hesitation"**: Traditionally, high entropy is seen as noise; this work treats $\ln 2, \ln 3$ as "healthy states of model self-awareness of uncertainty," a hallmark of high-quality reasoning, and offers a new framework for "exploration vs exploitation" in RL chain-of-thought.
- **AST + offset-aligned filtering pipeline**: Aligns character-level labeling with token-level evaluation, reusable for any LLM evaluation scenario requiring "removal of noisy tokens by code structure."
- **Transferable to PRM**: High-entropy decision points are precisely the "forking points" of Process Reward Models (PRM); the HE-SNR approach can be seamlessly transferred to PRM training data construction.

## Limitations & Future Work
- The threshold $\epsilon$ is static; the "reasonable hesitation boundary" may differ across tasks. The authors acknowledge the need for adaptive thresholds.
- High-entropy tokens are affected by code style—different implementations of the same logic yield different high-entropy sets. The authors suggest code canonicalization/style transfer for standardization.
- Main validation is on MoE-A3B and MoE-A26B; although Qwen2.5-72B and DeepSeek-V3 also show the "$\ln 3$ phenomenon," full HE-SNR vs SWE-bench correlation curves have not been plotted for other models.
- Currently only SWE-bench tasks are considered; whether higher-order ($k \geq 4$) entropy compression states appear in more complex tasks remains an open question.

## Related Work & Insights
- **vs PPL / BPC** (Kaplan, Huang): Scalar compression metrics only reflect parroting; this work supplements the "reasoning depth" dimension via distributional entropy peaks.
- **vs LongPPL** (Fang et al., ICLR 2025): LongPPL addresses positional bias in retrieval tasks (RULER) for instruct models; HE-SNR targets base models + agentic SWE tasks, with a completely different focus.
- **vs Beyond 80/20 High-Entropy Token RL** (Wang et al., 2025): They found high-entropy tokens are key for RL optimization; this work reverses the observation, showing high-entropy tokens are also key for "evaluating reasoning potential."

## Rating
- Novelty: ⭐⭐⭐⭐ "Entropy compression hypothesis + HE-SNR" is a relatively novel theoretical package, though the underlying method is a simple combination of Top-$k$ entropy.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple checkpoints from 3B to 560B, two context lengths (32K/128K), and both Dense and MoE architectures; validation is very thorough.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative is clear, tracing the failure of PPL to the proposal of HE-SNR, with an appendix providing concrete case studies of "$\ln k$ candidate probability distributions," making it highly readable.
- Value: ⭐⭐⭐⭐ Directly addresses the high-cost pain point of "which mid-training checkpoint to select" in industry; with just 500 trajectories and 12M tokens, downstream performance can be reliably predicted, offering significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Training Software Engineering Agents and Verifiers with SWE-Gym](../../ICML2025/code_intelligence/training_software_engineering_agents_and_verifiers_with_swe-gym.md)
- [\[ACL 2025\] UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench](../../ACL2025/code_intelligence/utboost_rigorous_evaluation_of_coding_agents_on_swe-bench.md)
- [\[ICLR 2026\] Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering](../../ICLR2026/code_intelligence/ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)
- [\[NeurIPS 2025\] Searching Latent Program Spaces](../../NeurIPS2025/code_intelligence/searching_latent_program_spaces.md)
- [\[ICLR 2026\] Training Large Language Models To Reason In Parallel With Global Forking Tokens](../../ICLR2026/code_intelligence/training_large_language_models_to_reason_in_parallel_with_global_forking_tokens.md)

</div>

<!-- RELATED:END -->
