---
title: >-
  [Paper Note] Taming Extreme Tokens: Covariance-Aware GRPO with Gaussian-Kernel Advantage Reweighting
description: >-
  [ACL2026][LLM Alignment][GRPO] This paper attributes entropy instability in GRPO training to the "log probability-advantage" covariance contribution of a few extreme tokens. It introduces a Gaussian kernel with no additi…
tags:
  - "ACL2026"
  - "LLM Alignment"
  - "GRPO"
  - "Covariance Reweighting"
  - "Entropy Stabilization"
  - "Mathematical Reasoning"
  - "RLVR"
date: 2026-05-08
content_hash: 5b42ee2366e0e58a
---

# Taming Extreme Tokens: Covariance-Aware GRPO with Gaussian-Kernel Advantage Reweighting

**Conference**: ACL2026  
**arXiv**: [2605.11538](https://arxiv.org/abs/2605.11538)  
**Code**: Not disclosed  
**Area**: LLM Alignment / RL Post-training  
**Keywords**: GRPO, Covariance Reweighting, Entropy Stabilization, Mathematical Reasoning, RLVR

## TL;DR
This paper attributes entropy instability in GRPO training to the "log probability-advantage" covariance contribution of a few extreme tokens. It introduces a Gaussian kernel with no additional hyperparameters to softly suppress the advantage of these tokens, consistently improving performance on 1.5B and 7B mathematical reasoning models.

## Background & Motivation
**Background**: In reasoning training with verifiable rewards, GRPO has become a standard post-training approach for DeepSeek-R1 style models because it eliminates the need for an additional value model. It estimates advantage through the relative reward of multiple responses under the same prompt and updates the policy using PPO-style probability ratios.

**Limitations of Prior Work**: The issue with GRPO is not a lack of reward signal, but rather its susceptibility to oscillations between exploration and exploitation. Excessive exploitation causes the model to prematurely commit to suboptimal reasoning templates, while excessive exploration leads to drastic fluctuations in training entropy, often resulting in performance degradation at later checkpoints.

**Key Challenge**: The authors identify a granular mechanism: policy entropy changes are mathematically related to the covariance between token log-probabilities and advantages. Consequently, not all tokens impact the exploration-exploitation balance equally; a few tokens with extreme absolute covariance values amplify gradient directions, dragging the overall policy entropy toward instability.

**Goal**: First, to quantify whether extreme covariance tokens truly exist in GRPO training; second, to suppress them without introducing manual thresholds; and third, to maintain effective learning signals from moderate-covariance tokens so that the model neither collapses in entropy nor over-diverges.

**Key Insight**: Instead of directly clipping advantages or adjusting global KL coefficients, this work focuses on token-level covariance. This perspective directly aligns with entropy dynamics and allows "dangerous updates" to be localized at the token level rather than crudely weakening the reward of an entire response.

**Core Idea**: Use the empirical standard deviation of covariance as the Gaussian kernel bandwidth to softly down-weight updates for tokens with abnormally large absolute covariance, while keeping the original GRPO objective and training process largely unchanged.

## Method

### Overall Architecture
This method can be viewed as inserting a covariance-aware advantage reweighting layer into the token-level loss of GRPO. Given a prompt, the policy model samples a group of responses, and an external verifier scores each response. GRPO first calculates response-level advantages based on group mean and standard deviation. Subsequently, CW-GRPO does not simply copy the same advantage to every token; instead, it calculates the product of each token's centered log-probability and its centered advantage as that token's covariance contribution to entropy change. Tokens with extreme contributions are down-weighted using a Gaussian kernel. Finally, the reweighted advantages are used in the original probability ratio term, maintaining compatibility with KL penalties and the overall training framework.

### Key Designs
1. **Explaining GRPO Entropy Instability via Covariance**:

	- Function: Converts "volatile policy entropy in late-stage training" into a measurable token-level diagnostic.
	- Mechanism: Borrowing from the relationship between entropy changes and natural policy gradients, the authors approximate entropy change as $\Delta H \approx -\eta \cdot Cov_t(\log \pi_\theta(o_t), A_i)$. Thus, the further a token's log-probability and its response's advantage are from their respective means, the stronger its pull on entropy.
	- Design Motivation: Traditional GRPO only evaluates the relative quality of entire responses and cannot distinguish between "useful reasoning tokens" and "extreme tokens causing entropy instability." The covariance perspective provides a direct stability metric.

2. **Soft Suppression of Extreme Token Updates via Gaussian Kernel**:

	- Function: Automatically reduces the advantage contribution of tokens with large covariance, preventing a few outliers from dominating policy updates.
	- Mechanism: For each token, compute $c_{i,t}=(\log \pi_\theta(o_{i,t})-\overline{\log \pi})(A_i-\overline{A})$ and derive a weight $w_{i,t}=\exp(-c_{i,t}^2/(2\sigma^2))$, where $\sigma$ is the empirical standard deviation of the current set of token covariances. Tokens with moderate covariance receive weights near 1, while tokens with extreme positive or negative covariance are smoothly suppressed.
	- Design Motivation: Hard thresholds or clipping can introduce new hyperparameters or abruptly cut off useful gradients. The Gaussian kernel is continuous, symmetric for extreme values, and adapts to different batch scales via the empirical standard deviation.

3. **Weight Normalization to Maintain GRPO Update Scale**:

	- Function: Adjusts the relative contribution between tokens without systematically scaling the overall loss due to decreased average weights.
	- Mechanism: The authors normalize the raw Gaussian weights as $\tilde{w}_{i,t}=w_{i,t} \cdot N / \sum_{j,k} w_{j,k}$, where $N$ is the total number of tokens in the group, and replace the advantage in the token loss with $\tilde{w}_{i,t}A_i$.
	- Design Motivation: This makes the method act like "reallocating the gradient budget" rather than "reducing the learning rate," preserving the GRPO training pace while shifting the budget from dangerous outliers to more stable tokens.

### Loss & Training
The training follows the basic GRPO/RLVR workflow: the model generates 12 responses per math problem, with a verifier assigning 0/1 for correctness and checking `<think>` tag formatting. Rewards are group-standardized to form advantages. High-quality math problems (7,000) from Open-RS serve as the training set, with evaluations covering AIME24, MATH-500, AMC23, Minerva, and OlympiadBench. Implementation uses HuggingFace TRL for training and Lighteval for evaluation. Key hyperparameters include a learning rate of $1e-6$, batch size of 12, gradient accumulation of 4, 100 training steps, temperature of 0.7, and a maximum completion length of 4096. No additional thresholds or temperature parameters are introduced for the reweighting itself.

## Key Experimental Results

### Main Results

| Model & Method | AIME24 | MATH-500 | AMC23 | Minerva | OlympiadBench | Average |
|----------------|--------|----------|-------|---------|---------------|------|
| 1.5B Base | 28.8 | 82.8 | 62.9 | 26.5 | 43.3 | 48.9 |
| 1.5B GRPO | 33.3 | 85.0 | 67.5 | 27.2 | 49.9 | 52.6 |
| 1.5B Clip-Cov | 33.3 | 85.5 | 70.0 | 29.0 | 50.0 | 53.6 |
| 1.5B CW-GRPO | 30.0 | 87.0 | 77.5 | 29.8 | 52.0 | 55.3 |
| 7B Base | 3.3 | 82.6 | 47.5 | 33.1 | 40.4 | 41.4 |
| 7B GRPO | 10.0 | 82.2 | 55.0 | 33.1 | 40.3 | 44.1 |
| 7B Clip-Cov | 10.0 | 82.4 | 57.5 | 32.4 | 41.3 | 44.7 |
| 7B CW-GRPO | 13.3 | 82.8 | 62.5 | 32.0 | 42.7 | 46.7 |

CW-GRPO achieves an average score of 55.3 on the 1.5B model, which is 2.7 points higher than vanilla GRPO and 1.7 points higher than Clip-Cov. On the 7B model, the average of 46.7 is 2.6 points higher than GRPO. Significant gains are concentrated in AMC23 and OlympiadBench, which demand higher reasoning stability.

### Ablation Study

| Method | Training Step | MATH-500 | OlympiadBench | Phenomenon |
|--------|---------------|----------|---------------|------------|
| GRPO | 100 | 85.0 | 49.9 | Strong early performance |
| GRPO | 150 (Low Entropy) | 82.0 | 49.9 | MATH drop after entropy fall |
| GRPO | 200 (High Entropy) | 79.8 | 47.8 | Continued degradation after entropy bounce |
| CW-GRPO | 100 | 87.0 | 52.0 | Higher starting point |
| CW-GRPO | 150 (Low Entropy) | 86.2 | 53.9 | Stable despite entropy fluctuations |
| CW-GRPO | 200 (High Entropy) | 86.4 | 53.5 | No late-stage collapse |

### Covariance Distribution Analysis

| Quantile | Positive Threshold | Negative Threshold | Interpretation |
|----------|--------------------|--------------------|----------------|
| 0.01% | 11.52 | -13.62 | A tiny fraction of tokens contribute far more than the main distribution |
| 1.00% | 3.32 | -3.34 | Top 1% already deviate significantly |
| 20.00% | 0.58 | -0.36 | Most tokens reside in a moderate range |
| 40.00% | 0.33 | -0.22 | Majority covariance is very small |
| 100.00% | 0.06 | -0.04 | Tail dominates the overall covariance |

### Key Findings
- Extreme covariance indeed exists and is not just minor noise: the covariance magnitude of the top 0.01% tokens is an order of magnitude higher than the main distribution, explaining why vanilla GRPO entropy is easily skewed by a few tokens.
- Stable entropy is highly correlated with downstream performance: while GRPO dropped from 85.0 to 79.8 on MATH-500 after step 100, CW-GRPO remained stable between 86.2 and 87.0.
- The method is effective for both 1.5B and 7B models, suggesting it is a general token-level update stabilization technique rather than a specific hyperparameter optimization for a single model.

## Highlights & Insights
- The primary highlight is explaining GRPO training instability as a token-level covariance outlier problem rather than simply blaming KL coefficients, learning rates, or reward noise. This diagnosis is closer to the actual dynamics of policy entropy and facilitates local corrections.
- Gaussian kernel reweighting is restrained: it does not change the reward model, requires no value network training, and avoids manual thresholds. This "plug-and-play loss stabilizer" is easier to implement in engineering than rewriting RL pipelines.
- The paper serves as a reminder that response-level advantage is quite coarse for long reasoning chains. Even when rewards are only verifiable at the answer level, token-level statistics can help identify updates that jeopardize the exploration-exploitation balance.

## Limitations & Future Work
- Experimental scale is limited to 7B and focused on math reasoning. This mechanism requires validation on larger models, longer contexts, multi-turn dialogues, and open-ended tasks.
- The Gaussian kernel assumes "the more extreme the covariance, the more dangerous," but in some tasks, extreme tokens might correspond to truly critical reasoning breakthroughs. Future designs could integrate correctness, position, or reasoning step types into the weighting.
- Training only evaluated short-range behavior (100–200 steps). Monitoring is needed to see if new degradation patterns emerge in longer training, such as the model actively avoiding high-covariance tokens.

## Related Work & Insights
- **vs GRPO**: GRPO replaces value models with group relative rewards, which is efficient but lacks token-level stability control; this work retains the GRPO framework while adding covariance-aware reweighting before advantages enter the loss.
- **vs Clip-Cov**: Clip-Cov also focuses on covariance but uses hard constraints for extremes; CW-GRPO uses a soft Gaussian decay with adaptive scaling via empirical standard deviation, avoiding manual clipping boundaries.
- **vs KL/entropy regularization**: Conventional approaches constrain entropy or KL globally, which can suppress useful exploration; this work targets specific token updates, providing a more fine-grained stabilization path.
- **Insights**: This logic can be transferred to preference optimization (DPO), code generation RL, or tool-calling RL, especially in scenarios with sparse rewards and long sequences. Identifying which token statistics dominate training dynamics allows for targeted soft adjustments of outliers.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Clear idea of deriving token-level reweighting from entropy-covariance relationships with better mechanistic explanation than standard tuning.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Solid across two model scales and five math benchmarks with entropy/covariance analysis, though the task domain remains narrow.
- Writing Quality: ⭐⭐⭐⭐☆ Smooth logical flow between motivation, formulas, and results; tables strongly support core arguments.
- Value: ⭐⭐⭐⭐☆ Practical for RLVR post-training, particularly as a low-cost stability enhancement for GRPO.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO](mitigating_selection_bias_in_large_language_models_via_permutation-aware_grpo.md)
- [\[NeurIPS 2025\] DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO](../../NeurIPS2025/llm_alignment/deepvideor1_video_reinforcement_finetuning_via_difficultyawa.md)
- [\[ICLR 2026\] No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping](../../ICLR2026/llm_alignment/no_prompt_left_behind_exploiting_zero-variance_prompts_in_llm_reinforcement_lear.md)
- [\[ICML 2026\] UDM-GRPO: Stable and Efficient GRPO for Unified Discrete Diffusion Models](../../ICML2026/llm_alignment/udm-grpo_stable_and_efficient_group_relative_policy_optimization_for_uniform_dis.md)
- [\[ICML 2026\] F-TIS: Harnessing Diverse Models in Collaborative GRPO](../../ICML2026/llm_alignment/f-tis_harnessing_diverse_models_in_collaborative_grpo.md)

</div>

<!-- RELATED:END -->
