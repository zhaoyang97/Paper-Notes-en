---
title: >-
  [Paper Note] Verifier-Free RL for LLMs via Intrinsic Gradient-Norm Reward
description: >-
  [ACL2026][Reinforcement Learning][verifier-free RL] VIGOR utilizes the gradient norm of the teacher-forced NLL for each completion under the current model parameters as an intrinsic reward…
tags:
  - "ACL2026"
  - "Reinforcement Learning"
  - "verifier-free RL"
  - "GRPO"
  - "intrinsic reward"
  - "gradient norm"
  - "length correction"
date: 2026-05-08
content_hash: 63bdc664a7eb729e
---

# Verifier-Free RL for LLMs via Intrinsic Gradient-Norm Reward

**Conference**: ACL2026  
**arXiv**: [2605.09920](https://arxiv.org/abs/2605.09920)  
**Code**: https://github.com/ZJUSCL/VIGOR  
**Area**: reinforcement_learning  
**Keywords**: verifier-free RL, GRPO, intrinsic reward, gradient norm, length correction  

## TL;DR
VIGOR utilizes the gradient norm of the teacher-forced NLL for each completion under the current model parameters as an intrinsic reward, favoring outputs with lower gradient norms. It stabilizes GRPO through $\sqrt{T}$ length correction and intra-group rank shaping, thereby improving mathematical and code reasoning without the need for gold answers or external verifiers.

## Background & Motivation
**Background**: RLVR has become a critical post-training paradigm for enhancing the reasoning capabilities of LLMs. Mathematical tasks often employ exact-match verifiers, while code tasks use unit tests or execution results as rewards. Algorithms like GRPO sample a set of completions for each prompt and standardize intra-group rewards to calculate advantages.

**Limitations of Prior Work**: Verifiable rewards are not available for all tasks. Mathematical answers require extractability and exact matching, and code requires test cases; constructing reliable verifiers for open-ended QA, long-form generation, or weakly supervised tasks is difficult. Existing verifier-free methods utilize majority voting, likelihood, entropy, or self-certainty to construct intrinsic signals, but these signals are often exploited by the model during training, leading to reward proxy degeneration, length expansion, or late-stage performance regression.

**Key Challenge**: RL post-training requires rewards to guide the model toward better outputs. However, if the reward originates from the model itself, it must be independent of labels and resistant to manipulation. Local distribution signals like token probability or entropy are computationally cheap but easily influenced by surface-level behaviors; external verifiers are more reliable but lack universality.

**Goal**: The authors aim to identify an intrinsic reward that relies solely on the policy model itself, has low requirements for output format, and remains stable during training, allowing GRPO to continue improving reasoning capabilities in the absence of answer labels and task verifiers.

**Key Insight**: The paper approaches this from an optimization perspective: if a completion induces a smaller negative log-likelihood (NLL) gradient norm under teacher-forced conditions for the current parameters, it suggests that the completion is closer to the smooth/stable region of the current policy, leading to a gentler update direction. Conversely, a large gradient norm may imply that the output is inconsistent with the current model or requires drastic parameter adjustments.

**Core Idea**: The rank of length-corrected gradient norms for completions is converted into intra-group rewards. Completions with low gradient norms receive high rewards, while those with high gradient norms receive low rewards, which are then used to update the policy via GRPO.

## Method
The intuition behind VIGOR is as follows: for the same prompt, the model samples 8 candidate solutions. Each candidate can be viewed as a teacher-forced training sample to calculate the gradient of its token-mean NLL with respect to current parameters. If a candidate results in a flatter loss surface and smaller gradient in the parameter space, the authors treat it as a more "self-consistent" and stable output. VIGOR does not need to know if the final answer is correct; instead, it ranks candidates within the same group based on this stability signal to serve as a relative reward for GRPO.

### Overall Architecture
Given a prompt $x$, the current policy $\pi_\theta$ samples a set of completions $\{y_i\}_{i=1}^{G}$; in the paper's experiments, $G=8$. For each completion $y=(y_1,\ldots,y_T)$, the token-mean NLL $\ell_{mean}(x,y)=\frac{1}{T}\sum_{t=1}^{T}\ell_t(x,y)$ is first calculated, followed by the $\ell_2$ norm of the gradient $g(x,y)=\nabla_\theta \ell_{mean}(x,y)$. This norm is not used for further backpropagation but serves as a scalar reward signal.

The raw average gradient norm has a significant issue: as the length of the completion increases, token-level gradients are more likely to cancel each other out during averaging, causing $\|g\|_2$ to shrink roughly by a factor of $1/\sqrt{T}$. If low gradient norms were rewarded directly, the model would tend to generate longer text to "cheat" the reward. Therefore, VIGOR employs $S_{GN}(x,y)=-\sqrt{T}\|g(x,y)\|_2$, using $\sqrt{T}$ to counteract the length bias, while the negative sign converts the "smaller is better" gradient norm into a "larger is better" reward.

Finally, the $G$ values of $S_{GN}$ for the same prompt are ranked. The worst completion is mapped to -1, the best to +1, with intermediate values distributed uniformly. This is then used to obtain the advantage through GRPO's intra-group standardization to update the policy.

### Key Designs
1. **Gradient Norm as Verifier-Free Intrinsic Reward**:
	- **Function**: Generates comparable rewards for each completion in the absence of gold answers, executors, or external judges.
	- **Mechanism**: Treats completions as teacher-forced sequences and calculates the gradient norm of the token-mean NLL relative to current parameters. A lower gradient norm indicates the output is smoother for the current model and less likely to cause drastic updates, thus it is deemed superior within the same group of candidates.
	- **Design Motivation**: Unlike entropy or self-certainty which depend directly on vocabulary probability distributions, the gradient norm aggregates changes in high-dimensional parameter space, making it theoretically more difficult to manipulate through simple token-level surface patterns.

2. **$\sqrt{T}$ Length Correction**:
	- **Function**: Prevents the model from artificially reducing the average gradient norm by generating longer completions.
	- **Mechanism**: Empirical findings show that the raw gradient norm drops from approximately 180 to 90 across length bins of 250/500/750/1000 tokens, but stays stable between $2.85\times10^3$ and $2.91\times10^3$ when multiplied by $\sqrt{T}$. Thus, the reward uses $S_{GN}=-\sqrt{T}\|g\|_2$.
	- **Design Motivation**: Without length correction, the reward misinterprets "long" as "good," particularly leading to length expansion and accuracy collapse in 3B models. Length correction decouples the gradient signal from length factors.

3. **Intra-group Rank-based Reward Shaping**:
	- **Function**: Eliminates scale differences in gradient norms across different prompts, making the reward more suitable for GRPO's relative group optimization.
	- **Mechanism**: Ranks the $G$ values of $S_{GN}$ for the same prompt, setting ranks from 0 to $G-1$. The reward is $R_{GN}(x,y_i)=2\frac{rank_i}{G-1}-1$. This maps the worst to -1 and the best to +1, preserving only the relative order and not the raw magnitude.
	- **Design Motivation**: Gradient norms for different problems might vary significantly in scale; using raw values could allow a few extreme prompts to dominate the updates. Rank shaping ensures balanced contributions within each prompt and reduces the impact of reward outliers.

### Loss & Training
VIGOR integrates directly into GRPO. For each prompt, $G=8$ completions are sampled, length-corrected gradient norm rewards are calculated, rank normalization is applied, and intra-group mean-std standardization is performed to derive the advantage $\hat{A}_i$. The policy objective is similar to GRPO: maximizing the product of the clipped probability ratio and the advantage, while adding a KL divergence penalty to constrain the current policy relative to a reference policy. A key implementation detail is the stop-gradient: although the reward is calculated from the gradient norm of the current parameters, it is detached as a constant during training to avoid second-order gradient overhead and instability.

## Key Experimental Results

### Main Results
Experiments were conducted on Qwen2.5-3B-Base and Qwen2.5-7B-Base, using the MATH training set and a subset of CodeContests for post-training. During training, VIGOR does not use reference answers and only uses the problems as prompts; gold answers are used only for evaluation and the GT-Reward baseline. Evaluation covers MATH-500, GSM8K, AMC, LiveCodeBench v6, CRUX, MMLU-Pro, and IFEval.

| Training Set / Model | Method | Math Avg. | Code Avg. | MMLU-Pro | IFEval |
|-------------|------|-----------|-----------|----------|--------|
| MATH / Qwen2.5-3B | Base | 48.34 | 16.98 | 36.92 | 28.30 |
| MATH / Qwen2.5-3B | INTUITOR | 57.10 | 26.79 | 24.48 | 29.11 |
| MATH / Qwen2.5-3B | VIGOR | 59.14 | 27.95 | 32.65 | 31.72 |
| MATH / Qwen2.5-7B | Base | 42.58 | 9.69 | 47.21 | 35.90 |
| MATH / Qwen2.5-7B | INTUITOR | 66.46 | 38.51 | 43.04 | 34.91 |
| MATH / Qwen2.5-7B | VIGOR | 69.77 | 40.42 | 43.09 | 37.03 |

| Training Set / Model | Method | GSM8K | MATH500 | AMC | Math Avg. | LiveCodeBench | CRUX | Code Avg. |
|-------------|------|-------|---------|-----|-----------|---------------|------|-----------|
| CodeContests / Qwen2.5-3B | Base | 67.93 | 54.80 | 22.28 | 48.34 | 9.57 | 24.38 | 16.98 |
| CodeContests / Qwen2.5-3B | INTUITOR | 75.13 | 58.60 | 22.59 | 52.11 | 11.47 | 39.38 | 25.43 |
| CodeContests / Qwen2.5-3B | VIGOR | 77.10 | 62.80 | 29.82 | 56.57 | 11.65 | 35.62 | 23.64 |

### Ablation Study

| Model | Configuration | Math Avg. | Code Avg. | MMLU-Pro | IFEval | Note |
|------|------|-----------|-----------|----------|--------|------|
| Qwen2.5-3B | Full VIGOR | 59.14 | 27.95 | 32.65 | 31.72 | Full method |
| Qwen2.5-3B | w/o $\sqrt{T}$ | 20.71 | 0.00 | 36.39 | 29.30 | GSM8K/AMC near collapse without length correction; code migration drops to zero |
| Qwen2.5-3B | w/o rank | 58.00 | 27.07 | 33.44 | 30.19 | Slight math drop; general capability trade-off |
| Qwen2.5-7B | Full VIGOR | 69.77 | 40.42 | 43.09 | 37.03 | Full method |
| Qwen2.5-7B | w/o $\sqrt{T}$ | 68.29 | 41.34 | 41.34 | 37.23 | 7B more stable against length bias but still lower than full |
| Qwen2.5-7B | w/o rank | 69.21 | 40.06 | 34.19 | 38.32 | Significant MMLU-Pro degradation; rank shaping protects general capabilities |

| Rank | Step 10 Acc. | Step 20 Acc. | Meaning |
|------|--------------|--------------|------|
| 1 (best) | 70.50 | 72.30 | Completions with the best gradient norm rank are most often correct |
| 2 | 68.20 | 71.10 | High rank completions maintain higher accuracy |
| 4 | 67.00 | 67.40 | Median completions significantly lower than top rank |
| 6 | 60.70 | 66.00 | Accuracy drops for lower-ranked samples |
| 8 (worst) | 52.70 | 63.40 | Gap between best and worst rank is 17.8/8.9 points |

### Key Findings
- In MATH post-training, VIGOR improves math average by +3.31 and code average by +1.91 on 7B compared to INTUITOR; it also improves math, code, and IFEval on 3B.
- VIGOR exhibits clear cross-domain transfer: when trained only on MATH, the code averages for 3B/7B improve from 16.98/9.69 to 27.95/40.42 respectively.
- The code training experiment serves as a lightweight sanity check. VIGOR increases the math average from 48.34 to 56.57 on CodeContests, though the code average of 23.64 is lower than INTUITOR's 25.43, suggesting gradient norms may not be most sensitive to discrete algorithmic choices.
- Reward reliability analysis shows that completions in the top 25% by gradient norm are more stable during training than those in INTUITOR, avoiding the late-stage degradation seen with self-certainty rewards.
- $\sqrt{T}$ correction is the most critical component; notably on the 3B model, removing it leads to length hacking, where GSM8K drops to 0.08 and AMC to 1.66.

## Highlights & Insights
- This paper shifts the reward from "surface output probability" to "local geometry in parameter space," offering a novel perspective. The gradient norm does not judge answer correctness directly but rather whether the completion is consistent with the stable region of the current policy.
- The length correction is executed critically and transparently. The authors do not just propose the gradient norm but explicitly point out the $1/\sqrt{T}$ bias in average NLL gradients and demonstrate through experiments that failing to correct it leads to failure.
- Rank shaping is well-suited for the relative group context of GRPO. It acknowledges that absolute values of intrinsic rewards are not comparable across prompts and only utilizes intra-prompt ranking, which stabilizes the method.
- This approach can be migrated to weakly verifiable tasks: when external rewards are unavailable, internal signals like gradients, curvature, or consistency can be used for candidate ranking, combined with minimal manual or automated judge verification.

## Limitations & Future Work
- The paper primarily validates tasks like math and code where correctness can still be evaluated. Whether low gradient norms correspond to better outputs for open-ended writing, dialogue, or safety alignment remains uncertain.
- Calculating the gradient norm for each completion is more expensive than forward-only rewards like entropy or likelihood, as it requires automatic differentiation; although a simplified LM-head-only approximation is provided in the appendix, it might still be a bottleneck at larger scales.
- The gradient norm is essentially a proxy. Models might later learn to generate "low-gradient but useless" patterns, leaving a risk of reward exploitation.
- On code tasks, VIGOR's performance on CRUX is inferior to INTUITOR, indicating that parameter space smoothness cannot fully replace execution feedback in scenarios requiring discrete algorithmic correctness.

## Related Work & Insights
- **vs RLVR / GT-Reward**: RLVR uses exact match or executors, providing reliable rewards but depending on task verifiers; VIGOR does not use labels or verifiers, making it more general, though the reward is an indirect proxy.
- **vs INTUITOR / RLIF**: INTUITOR uses internal policy confidence/likelihood signals, which are prone to late-stage degradation; VIGOR uses gradient norms and rank shaping for more stable training dynamics.
- **vs Majority Voting Pseudo-labeling**: Methods like TTRL/Co-rewarding rely on extractable and aggregatable final answers; VIGOR does not require answer extraction, making it theoretically better suited for free-form completions.
- **vs entropy-based intrinsic reward**: Entropy is derived from token distributions and easily influenced by local probability patterns; the gradient norm aggregates parameter space signals, acting more like a consistency check between the completion and the model state.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Utilizes teacher-forced gradient norms as a verifier-free reward, a distinct perspective from common confidence/entropy methods.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Main experiments, cross-domain transfer, training dynamics, rank-accuracy, and ablations are all sufficient; validation on open-ended tasks is missing.
- Writing Quality: ⭐⭐⭐⭐☆ Method derivation is clear, and length bias explanation is thorough; some training cost details are mainly in the appendix, requiring further investigation by the reader.
- Value: ⭐⭐⭐⭐☆ Highly valuable for RL post-training in scenarios without verifiers, though gradient calculation costs may impact the actual scope of deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Free Energy-Driven Reinforcement Learning with Adaptive Advantage Shaping for Unsupervised Reasoning in LLMs](free_energy-driven_reinforcement_learning_with_adaptive_advantage_shaping_for_un.md)
- [\[NeurIPS 2025\] RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning](../../NeurIPS2025/reinforcement_learning/rl_tango_reinforcing_generator_and_verifier_together_for_lan.md)
- [\[ICML 2026\] From Reward-Free Representations to Preferences: Rethinking Offline Preference-Based Reinforcement Learning](../../ICML2026/reinforcement_learning/from_reward-free_representations_to_preferences_rethinking_offline_preference-ba.md)
- [\[ACL 2026\] RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization](rl-plus_countering_capability_boundary_collapse_of_llms_in_reinforcement_learnin.md)
- [\[ACL 2026\] LearnAlign: Data Selection for LLM Reinforcement Learning with Improved Gradient Alignment](learnalign_data_selection_for_llm_reinforcement_learning_with_improved_gradient_.md)

</div>

<!-- RELATED:END -->
