---
title: >-
  [Paper Note] The Unlearnability Phenomenon in RLVR for Language Models
description: >-
  [ICML 2026][LLM Safety][RLVR] The authors discover a class of "unlearnable examples" in RLVR (GRPO) training: even when correct rollouts are sampled and reward signals are non-zero…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "RLVR"
  - "GRPO"
  - "Unlearnable Examples"
  - "Gradient Similarity"
  - "Representation Deficiencies"
date: 2026-05-08
content_hash: 457970f316ea4685
---

# The Unlearnability Phenomenon in RLVR for Language Models

**Conference**: ICML 2026  
**arXiv**: [2605.16787](https://arxiv.org/abs/2605.16787)  
**Code**: https://github.com/yulinchen99/unlearnability-rlvr  
**Area**: LLM Reasoning / RLVR / GRPO  
**Keywords**: RLVR, GRPO, Unlearnable Examples, Gradient Similarity, Representation Deficiencies

## TL;DR
The authors discover a class of "unlearnable examples" in RLVR (GRPO) training: even when correct rollouts are sampled and reward signals are non-zero, the model fails to learn them throughout the entire training process. The root cause is not a scarcity of positive samples, clipping, or KL regularization on the optimization side; rather, these samples are "gradient outliers" under the initial policy, reflecting representation deficiencies that require mid-training instead of RL post-training for remediation.

## Background & Motivation

**Background**: RLVR (Reinforcement Learning with Verifiable Reward), represented by GRPO, has become a primary method for enhancing the mathematical, coding, and agent reasoning capabilities of LLMs. Intuitively, the prerequisite for GRPO to function is that "within $k$ rollouts of the same prompt, there exist both positive and negative samples." Consequently, extensive recent work (DAPO, curriculum learning, entropy weighting, etc.) has focused on "generating positive reward signals for extremely difficult samples."

**Limitations of Prior Work**: The authors observe a counter-intuitive phenomenon: when training samples are partitioned into "easy," "learnable-hard," and "unlearnable-hard" based on initial success rates, the **unlearnable-hard samples** show stagnant training rewards. Even when correct rollouts are consistently observed (i.e., non-zero outcome rewards), the rewards do not increase. These samples account for 30.2% of hard samples in Qwen2.5-0.5B/MATH-Easy and 21.9% in Llama-3.2-3B/MATH-Hard, indicating they are not marginal cases.

**Key Challenge**: The existing RLVR paradigm assumes that "as long as positive samples exist, the model can learn." This study empirically falsifies this implicit assumption. Common optimization-side interventions—such as more positive rollouts, experience replay, higher clipping thresholds, and removing KL terms—all prove ineffective, suggesting the root cause lies beyond optimization and requires a different explanatory framework.

**Goal**: (1) Rigorously define and quantify the existence of "unlearnable samples"; (2) systematically investigate common optimization-side hypotheses (positive sample scarcity, clipping, KL regularization); (3) provide a "representation-side" root cause to explain the phenomenon; and (4) examine whether data augmentation and mid-training can fix the issue.

**Key Insight**: The authors approach this from the perspective of *cross-example gradient similarity*. By treating the correct rollout of each sample as a gradient vector and examining the cosine similarity between different samples, they determine whether the "knowledge learned from one sample" can generalize to others.

**Core Idea**: Use "gradient similarity" to elevate the distinction between learnable and unlearnable samples from "reward curve observations" to the "geometric properties of the optimization space." Unlearnable samples are isolated outliers in the optimization space, reflecting inherent representation deficiencies that outcome-based RL alone cannot repair.

## Method

This is a *diagnostic* paper rather than a proposal for a new algorithm. It uses a series of carefully designed controlled experiments to quantify, attribute, and localize the "unlearnability" phenomenon to the representation level. The overall research framework follows four stages: "Phenomenon → Hypotheses Exclusion → New Hypothesis Formation → Solution Validation."

### Overall Architecture
The research process consists of four steps:
1.  **Phenomenon Definition**: Under GRPO with dynamic sampling, three independent training runs are executed. Hard samples (initial pass@1 < 0.1) are categorized into $\mathcal{D}_l$ (learnable) and $\mathcal{D}_u$ (unlearnable) based on whether their final pass@1 remains < $\tau=0.1$ (estimated via $N=32$ rollouts). Samples that never encountered a positive rollout are excluded, and the intersection of the three runs is taken to reduce noise.
2.  **Optimization Hypothesis Exclusion**: Oversampling-with-replay is used to address "positive rollout scarcity," and higher clipping thresholds or removing KL terms are used for "gradient regularization." All fail.
3.  **Representation Attribution**: Using two independent signals—cross-example gradient similarity and reasoning-quality scores from GPT-5-mini—it is demonstrated that $\mathcal{D}_u$ consists of gradient outliers with low-quality reasoning chains.
4.  **Fix Validation**: Comparing data augmentation (similar problems and sub-problems) with mid-training (OctoThinker), the former fails while the latter succeeds.

### Key Designs

1.  **Operational Definition and Three-Way Split of Unlearnable Samples**:
    - **Function**: Converts the intuitive sense of "being unable to learn" into reproducible sample subsets for subsequent gradient and reasoning quality analysis.
    - **Mechanism**: After full training with GRPO and dynamic sampling, samples with an *initial success rate* $\geq 0.1$ are categorized as *easy*. For the remaining *hard* samples, the *final* pass@1 is estimated ($N=32$ rollouts); those with final pass@1 $<\tau=0.1$ are assigned to $\mathcal{D}_u$, otherwise to $\mathcal{D}_l$. Samples without any positive rollouts during training are explicitly excluded to ensure the research question—"positive reward exists but the model still fails to learn"—is well-defined.
    - **Design Motivation**: Previous discussions often conflated "no positive samples" with "positive samples exist but cannot be learned." This work separates the latter to avoid interventions that simply "add positive samples" appearing effective when they are not solving the core issue.

2.  **Oversampling-with-Replay to Disprove "Positive Sample Scarcity"**:
    - **Function**: Retrains the model while ensuring each prompt has exactly $k_{\text{pos}}=1$ positive sample and $k-k_{\text{pos}}=7$ negative samples per batch to see if $\mathcal{D}_u$ becomes learnable.
    - **Mechanism**: $4k$ rollouts are sampled per prompt and downsampled to $k=8$. If positive samples are insufficient, they are reused from an experience replay buffer (up to twice). Advantage is calculated after replay/downsampling: $\hat{A}_i = \frac{\mathbb{1}[y_i=y^*] - \text{mean}}{\text{std}}$. Results show this significantly slows learning for $\mathcal{D}_l$ (proving the intervention is active), but the reward curve for $\mathcal{D}_u$ remains identical to the baseline.
    - **Design Motivation**: To rule out the natural explanation that positive samples are too few and gradients are "submerged." If the gap persists even when forcing one positive sample per batch or increasing $k$ to 64, the cause is not sample count.

3.  **Cross-Example Gradient Similarity Targeting Representation Deficiencies**:
    - **Function**: Transitions the "unlearnability" claim from reward curves to the geometric characteristics of the optimization space—gradient directions for $\mathcal{D}_u$ are inconsistent with others, preventing generalization.
    - **Mechanism**: 100 samples are taken from each group, and 1,000 rollouts are sampled under the *initial policy*. Correct rollouts are filtered to compute GRPO loss gradients. Gradients are averaged within and then across responses to produce one vector per sample. To manage compute, a fixed randomly initialized LoRA adapter is used (confirmed to correlate highly with full-parameter similarity). Cosine similarity $\cos(g_i, g_j)$ is computed. Results show *easy* samples are highly aligned, *learnable* are intermediate, and *unlearnable* show low similarity to all groups. GPT-5-mini scoring reveals that correct rollouts in $\mathcal{D}_u$ often rely on shortcuts or heuristics, confirming that outcome reward can reinforce "fake reasoning."
    - **Design Motivation**: To move from "phenomenon" to "mechanism," an observable metric directly linked to training dynamics is required. Gradient similarity explains why learning on other samples does not transfer and why oversampling fails.

### Loss & Training
Standard GRPO with dynamic sampling is used. The GRPO objective is (with clipping $\varepsilon$ and KL coefficient $\beta$):

$$\mathcal{L}_{\text{GRPO}}(\theta,(x,y^*)) = -\frac{1}{k}\sum_i\frac{1}{|y_i|}\sum_t \min(r_{i,t}\hat{A}_i, \text{clip}(r_{i,t},1-\varepsilon,1+\varepsilon)\hat{A}_i) - \beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})$$

where $r_{i,t}=\pi_\theta(y_{i,t}|x,y_{i,<t})/\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})$. Dynamic sampling filters prompts where $\text{std}(\{\mathbb{1}[y_i=y^*]\})=0$ to improve efficiency. Ablations include "clip-higher" and "no-KL" variants. Mid-training experiments utilize OctoThinker-3B-Hybrid/Long-Base as initial policies.

## Key Experimental Results

### Main Results

**Table 1 — Proportion of Unlearnable Samples** (percentage relative to total hard samples with initial pass@1 $<0.1$):

| Model / Dataset | $\mathcal{D}_u$ (%) | $\mathcal{D}_l$ (%) | No Pos Reward (%) |
|---|---|---|---|
| Qwen2.5-0.5B / MATH-Easy | 30.2 | 25.6 | 23.5 |
| Llama-3.2-3B-Instruct / MATH-Hard | 21.9 | 31.6 | 37.7 |
| Qwen2.5-3B / DeepScaleR | 16.7 | 14.2 | 47.2 |

Unlearnable samples are not marginal and constitute a significant portion of hard samples alongside those with no positive rewards.

### Ablation Study

**Comparison of Optimization-side vs. Representation/Data-side Interventions**:

| Intervention | Hypothesis Targeted | Effective for $\mathcal{D}_u$ | Key Observation |
|---|---|---|---|
| Oversampling + replay (1 pos / 7 neg per batch) | Positive sample scarcity | ✗ | $\mathcal{D}_l$ slowed down, $\mathcal{D}_u$ unchanged |
| SFT distillation of correct answers on $\mathcal{D}_u$ | Lack of supervision | ✗ | Gap remains |
| RL on $\mathcal{D}_u$ only + $k=64$ rollouts | Insufficient exploration | ✗ | Gap remains |
| Clip-higher | Clipping suppresses gradients | ✗ | Clipping ratios nearly identical across groups |
| No KL term | KL constraint limits updates | ✗ | Reward dynamics unchanged |
| Similar problems $\mathcal{D}_u^{sim}$ augmentation | Lack of related signals | ✗ | Augmented problems learned; original $\mathcal{D}_u$ stays unlearned |
| Sub-problems $\mathcal{D}_u^{sub}$ augmentation | Skill decomposition | ✗ | Sub-problems learned faster than $\mathcal{D}_l$; original still unlearned |
| Mid-training (OctoThinker-3B-Hybrid/Long) | Representation defect | ✓ | Gradient similarity to training distribution significantly increased |

### Key Findings
- **Unlearnability stems from representation, not optimization**: Five types of optimization/data-side interventions failed; only changing the base model (mid-training) worked, strongly suggesting the issue exists prior to RL.
- **Gradient similarity is a strong proxy for learnability**: $\mathcal{D}_u$ contains isolated gradient outliers. This aligns perfectly with reward curve grouping and persists through step 50, indicating it is not an initialization fluke.
- **Correct Answer ≠ Correct Reasoning**: GPT-5-mini scores show $\mathcal{D}_u$ correct rollouts rely on shortcuts. Case studies show models arriving at correct answers through clearly flawed logic, highlighting reward-hacking risks in outcome-only RL.
- **Semantic Similarity ≠ Optimization Similarity**: Synthetic similar problems generated by GPT-5 remain structurally identical but do not improve gradient similarity for $\mathcal{D}_u$, meaning these samples are "isolated islands" in the optimization space.
- **Divergence increases with training**: Reasoning quality for $\mathcal{D}_l$ improves continuously, while $\mathcal{D}_u$ stagnates. Curriculum learning fails to transfer improvements from easy/learnable categories to unlearnable ones.

## Highlights & Insights
- **Geometric interpretation of "unlearnability"**: Gradient similarity explains why updates from other samples do not generalize and why oversampling is futile. The LoRA-only gradient approximation makes this analysis feasible at scale.
- **Empirical evidence of "answers aren't enough"**: Quantifying the gap between outcome and process rewards via GPT-5-mini provides motivation for process supervision or mid-step verifiers.
- **Model of "Negative Results" reporting**: The study systematically eliminates hypotheses (oversampling, distillation, etc.) before arriving at mid-training, providing a diagnostic paradigm transferable to other areas like forgetting in SFT.
- **Transferable Trick**: Using the "Gradient Similarity / Reasoning Quality / pass@k" triad to characterize the "optimization properties" of training data, allowing for smarter data pruning or curriculum labeling before training begins.

## Limitations & Future Work
- Experiments are limited to 0.5B–3B mathematical reasoning models. Whether unlearnable samples persist at 30B+ scales or in code/agent domains is unverified.
- "Unlearnability" depends on a hard threshold ($\tau=0.1$). While mitigated by the intersection of three runs, a continuous sensitivity analysis of $\tau$ is missing.
- No active algorithm for "fixing" these samples is proposed; mid-training is left as an open question regarding what data and algorithms are most effective.
- The concluding lack of success for similar-problem augmentation depends on synthetic data quality from GPT-5/Gemini-2.5-pro.
- The geometric explanation for gradient similarity remains high-level; future work could investigate whether a low-rank subspace explains $\mathcal{D}_u$ outliers.

## Related Work & Insights
- **vs. Sun et al. 2025b (Fine-grained reward assignment)**: While they assume better reward design enables learning, Ours proves that even with positive outcome rewards, some samples remain unlearnable due to representation.
- **vs. Yue et al. 2025 (RL cannot teach new skills not in base model)**: Ours aligns with this "RL capability ceiling" but provides a micro, quantifiable view based on gradient geometry.
- **vs. DAPO / Clip-higher / No KL**: Key interventions from DAPO were directly ablated and shown to benefit $\mathcal{D}_l$ rather than $\mathcal{D}_u$.
- **vs. OctoThinker / mid-training (Wang et al. 2025)**: Ours provides a new motivation—mid-training is not just "making the base stronger" but "aligning gradients of hard samples with the distribution."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First systematic characterization of "unlearnability despite positive reward" with a geometric mechanism.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid across multiple models/datasets, though scale is limited to $\leq$ 3B.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Exceptionally clear organization of negative results via the "method of exclusion."
- **Value**: ⭐⭐⭐⭐⭐ Directly challenges the "Positive Reward $\Rightarrow$ Learnability" assumption, providing evidence for mid-training and process rewards.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] dgMARK: Decoding-Guided Watermarking for Diffusion Language Models](dgmark_decoding-guided_watermarking_for_diffusion_language_models.md)
- [\[ICML 2026\] Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models](forget_to_know_remember_to_use_context-aware_unlearning_for_large_language_model.md)
- [\[ICML 2026\] COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models](coft_counterfactual-conformal_decoding_for_fair_chain-of-thought_reasoning_in_la.md)
- [\[ACL 2026\] Topic-Based Watermarks for Large Language Models](../../ACL2026/llm_safety/topic-based_watermarks_for_large_language_models.md)
- [\[ICML 2026\] Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models](towards_fine-grained_robustness_attention-guided_test-time_prompt_tuning_for_vis.md)

</div>

<!-- RELATED:END -->
