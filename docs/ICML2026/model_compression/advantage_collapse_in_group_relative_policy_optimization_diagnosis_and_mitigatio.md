---
title: >-
  [Paper Note] Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation
description: >-
  [ICML 2026][Model Compression][GRPO] This paper identifies that GRPO loses gradient signals under binary verifiable rewards due to identical intra-group rewards. It proposes the ACR metric for real-time diagnosis of this…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "GRPO"
  - "RLVR"
  - "Advantage Collapse"
  - "Training Diagnosis"
  - "Virtual Samples"
date: 2026-05-08
content_hash: d989c927c35c7214
---

# Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation

**Conference**: ICML 2026  
**arXiv**: [2605.21125](https://arxiv.org/abs/2605.21125)  
**Code**: https://github.com/hexixiang/Advantage-Collapse-Rate  
**Area**: Reinforcement Learning / LLM Reasoning  
**Keywords**: GRPO, RLVR, Advantage Collapse, Training Diagnosis, Virtual Samples

## TL;DR
This paper identifies that GRPO loses gradient signals under binary verifiable rewards due to identical intra-group rewards. It proposes the ACR metric for real-time diagnosis of this "advantage collapse" and introduces AVSPO, which injects virtual reward samples to restore intra-group variance, achieving consistent 4-6 percentage point improvements across multiple Qwen2.5 mathematical reasoning models.

## Background & Motivation
**Background**: Post-training for LLM mathematical reasoning increasingly relies on RLVR, which optimizes models using binary rewards from automated verifiers. GRPO is a representative algorithm in this paradigm; it avoids training a critic by estimating advantage through relative comparisons among multiple sampled responses for the same problem, making it more memory-efficient and scalable to long reasoning tasks than actor-critic methods like PPO.

**Limitations of Prior Work**: GRPO's advantage estimation depends on the mean and standard deviation ofRewards within a group. When all $G$ sampled responses for a problem are either all correct or all wrong, the intra-group reward variance becomes 0, and the advantage for all samples becomes 0. Consequently, such batches consume expensive LLM rollouts without providing effective gradients for policy updates; metrics such as loss, average reward, or even accuracy in training logs may not promptly expose this waste.

**Key Challenge**: Binary verifiable rewards are simple and reliable but highly prone to producing "all 0 / all 1" homogeneous rewards. Since GRPO omits the critic, it places the source of the learning signal entirely on intra-group relative differences. Therefore, sparser rewards or excessively easy/difficult problems increase the likelihood of situations where computation is completed but gradients are zero.

**Goal**: The authors aim to solve two sub-problems. First, how to quantify the number of groups entering an invalid gradient state during training. Second, how to allow these wasted samples to generate learning signals again without re-sampling or additional model calls once such groups are identified.

**Key Insight**: Instead of modifying the reward model or sampling strategy, the paper returns to the GRPO advantage formula itself. Simply monitoring the intra-group reward standard deviation reveals whether a group will generate effective gradients. Injecting low-cost virtual rewards to alter the normalization statistics can potentially restore non-zero advantages.

**Core Idea**: Use ACR to directly measure the proportion of "near-zero reward variance" in GRPO batches and inject virtual rewards into the normalization statistics of collapsed groups to transform invalid rollouts into updatable samples.

## Method
The proposed method consists of a diagnoser and a lightweight intervenor. The diagnoser, ACR, quantifies how much computation is wasted in the current batch. The intervenor, AVSPO, inserts a controllable normalization reference into these wasted groups, allowing real samples to gain directional advantage. The key design principle is that virtual samples are not new text outputs and do not participate in policy gradients; they only modify the computation of reward mean/std, thus introducing no additional LLM forward overhead.

### Overall Architecture
The training follows the basic GRPO workflow. For each problem $q$, the old policy samples $G$ responses, and the automated verifier provides binary rewards $r_i \in \{0, 1\}$. Standard GRPO computes intra-group advantage as $\hat{A}_i = (r_i - \mu_R) / (\sigma_R + \epsilon)$ for the clipped policy objective.

AVSPO inserts three steps into this process. First, it calculates the reward standard deviation for each problem group; if $\sigma_R < \tau$, the group is diagnosed with advantage collapse. Second, it calculates the ACR at the batch level and uses a dynamic threshold to decide whether to trigger intervention. Third, for collapsed groups triggering intervention, it constructs $K$ virtual rewards, merges real and virtual rewards to recalculate $\mu_{R'}$ and $\sigma_{R'}$, and finally computes the new $\hat{A}'_i$ only for the real samples to update the model.

### Key Designs
1.  **Advantage Collapse Rate (ACR) Diagnostic Metric**:
    - **Function**: ACR measures the proportion of problem groups in a batch that fail to generate effective gradients due to insufficient reward variance.
    - **Mechanism**: For $N$ problem groups in a batch, it checks $\sigma_{R_j} < \tau$ for each and computes $ACR = \frac{1}{N} \sum_j \mathbb{I}(\sigma_{R_j} < \tau)$. When ACR is near 0, most groups have reward differences; when ACR is near 1, almost all rollouts are in a zero-gradient state.
    - **Design Motivation**: ACR reuses existing reward statistics from GRPO without needing a critic, manual labels, or extra inference. It transforms "training stagnation" from an ex-post accuracy observation into a real-time observable signal.

2.  **Adaptive Virtual Sample Policy (AVSPO)**:
    - **Function**: Restores reward variance in all-correct or all-wrong groups, enabling real samples to obtain non-zero advantages.
    - **Mechanism**: When the batch ACR exceeds a dynamic threshold and a specific group collapses, $K = \max(1, \min(G, \lceil G \cdot ACR^\alpha \rceil))$ virtual rewards are constructed. If the real group is all-correct, virtual rewards decrease hierarchically from near 1. If it is all-wrong, a non-zero set of virtual rewards is constructed using a small positive anchor reward. $R' = R \cup V$ is then used to compute the mean and standard deviation, but the policy gradient is still only applied to the original $G$ real responses.
    - **Design Motivation**: Neither all-wrong nor all-correct groups are uninformative. All-wrong groups indicate the policy should move away from these failure trajectories, while all-correct groups indicate successful trajectories worth reinforcing. Virtual rewards serve as a statistical reference so that directional information is not erased by the normalization formula.

3.  **Dynamic Triggering and Bounded Bias Control**:
    - **Function**: Determines when and to what intensity to intervene, avoiding crude corrections of all collapses.
    - **Mechanism**: The trigger threshold $\tau_{adapt}$ is initialized at 0.5 and adjusted dynamically based on training progress. The number of virtual samples scales with $ACR^\alpha$ (default $\alpha=0.5$). Theoretical analysis shows that when $K \leq G$, the uniform advantage magnitude generated by AVSPO in homogeneous groups satisfies $|A^c(K)| \leq \sqrt{K/G} \leq 1$, and the bias upper bound shrinks with ACR.
    - **Design Motivation**: A fixed low threshold causes over-intervention and increases variance, while a fixed high threshold misses early collapses. The dynamic threshold incorporates training progress into trigger rules, making AVSPO an on-demand fix rather than a permanent reward-shaping term.

### Loss & Training
The training objective for AVSPO remains the GRPO clipped surrogate, but the advantage $\hat{A}_i$ is replaced by $\hat{A}'_i$ calculated using the augmented reward set. Virtual rewards only enter $\mu_{R'}$ and $\sigma_{R'}$ and do not generate $\nabla_\theta \log \pi_\theta$ terms. Experiments used a group size of 8, a training temperature of 1.0, and greedy decoding for evaluation. AVSPO-specific hyperparameters include an initial threshold of 0.5, $\alpha=0.5$, threshold learning rate of 0.01, collapse threshold of $10^{-6}$, and an anchor reward of 0.1.

## Key Experimental Results

### Main Results
The paper evaluates the method by training six Qwen2.5 series models for 500 steps. The training set is Level 3-500 from the MATH training split. Evaluations cover MATH-500, GSM8K, Minerva, OlympiadBench, AMC, AIME24, and MMLU-Pro. The core finding is that AVSPO simultaneously reduces ACR and improves average accuracy, with more significant gains for models with higher baseline ACR.

| Model | GRPO ACR | AVSPO ACR | GRPO Avg Acc | AVSPO Avg Acc | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-0.5B | 0.45 | 0.18 | 16.5 | 21.0 | +4.5 |
| Qwen2.5-3B | 0.37 | 0.14 | 27.9 | 32.2 | +4.3 |
| Qwen2.5-3B-Instruct| 0.35 | 0.13 | 39.7 | 43.4 | +3.7 |
| Qwen2.5-14B | 0.28 | 0.11 | 49.9 | 54.5 | +4.6 |
| Qwen2.5-Math-1.5B | 0.40 | 0.15 | 33.5 | 39.6 | +6.1 |
| Qwen2.5-Math-7B | 0.33 | 0.14 | 42.2 | 45.9 | +3.7 |

Compared to other baselines, AVSPO demonstrates stronger average performance. It outperforms DCPO by approximately +2.9 and shows even larger margins against INTUITOR and RENT. This suggests that directly fixing batch-level reward diversity is more stable than solely modifying clipping, encouraging low entropy, or using confidence-based rewards.

### Ablation Study
The method of constructing virtual samples is the most critical ablation factor. While random sampling, fixed partial credit, and exponential decay all reduce ACR, the stratified reward assignment performs best, indicating that simply having variance is insufficient; the structure of virtual rewards affects advantage directionality and stability.

| Configuration | ACR | MATH-500 | Description |
| :--- | :--- | :--- | :--- |
| GRPO (No Augmentation)| 0.40 | 58.6 | Advantage is 0 for all collapsed groups |
| Random Uniform Virtual | 0.22 | 62.1 | Reduces collapse but introduces random variance |
| Fixed Partial Credit | 0.19 | 63.5 | Simple and stable, but lacks reward hierarchy |
| Exponential Decay | 0.18 | 64.2 | Finer than fixed values, but trails stratified |
| AVSPO Stratified | 0.15 | 67.2 | Lowest ACR, highest accuracy |

Mechanism isolation experiments are also revealing. Fixing only all-wrong groups reduces all-wrong collapse from 24.8% to 9.1% (Accuracy: 63.2). Fixing only all-correct groups reduces all-correct collapse from 15.2% to 4.2% (Accuracy: 60.8). Full AVSPO compresses both to 8.7% and 6.3%, reaching 67.2 on MATH-500. In fixed threshold comparisons, the best fixed threshold reached 60% accuracy in 380 steps, whereas the adaptive threshold required only 295 steps and achieved a final accuracy of 67.2.

### Key Findings
- **ACR is a strong diagnostic signal**: The correlation coefficient between ACR in the first 100 steps and final MATH-500 accuracy is $r = -0.785$ (linear fit $R^2 = 0.617$), meaning early ACR explains about 62% of final performance variance.
- **Collapse is not a rare anomaly**: Standard GRPO experiences full advantage collapse in 28%-45% of batch groups in these math reasoning settings, creating a bottleneck for training efficiency.
- **Moderate difficulty samples are best for RLVR**: Tasks that are too easy (all-correct) or too hard (all-wrong) increase ACR; Level 3-4 difficulty allows for more natural reward diversity.
- **Augmentation outperforms filtering**: On Qwen2.5-Math-7B, "Filter-Drop" uses only 62.4% of samples, and DAPO costs about 1.8x. AVSPO maintains 100% sample utilization and 1.0x cost while reaching 69.7/74.1 on GSM8K/MATH.

## Highlights & Insights
- The most valuable contribution is framing GRPO failure modes as a measurable training diagnosis rather than just reporting "RL instability." ACR is simple but directly corresponds to the zero-variance condition in the advantage formula.
- The virtual sample design of AVSPO is clever: it avoids generating pseudo-text or introducing new reward models, modifying only normalization statistics. This keeps engineering overhead near zero and avoids additional rollout costs.
- The inclusion of all-correct collapse is significant. While many focus on the lack of learning signals in all-wrong groups, all-correct groups similarly prevent GRPO from further reinforcing successful trajectories; AVSPO provides a unified treatment.
- The sensitivity analysis of ACR to data difficulty, temperature, and group size can be transferred to other RLVR pipelines. Even without AVSPO, practitioners can use ACR to early-stop inefficient configurations or tune sampling parameters.
- The theoretical component is not merely decorative. It demonstrates that virtual samples decrease failure probabilities in all-wrong groups and increase success probabilities in all-correct groups, with PPO clipping mitigating concerns about virtual rewards misleading the policy.

## Limitations & Future Work
- Experiments primarily focus on mathematical reasoning and binary deterministic verifiers. The suitability of ACR thresholds and virtual reward designs for open-ended preference rewards, multi-level rewards, or noisy verifiers needs further validation.
- AVSPO addresses zero intra-group reward variance but cannot fix verifier errors, problem distribution bias, or insufficient model capacity. The authors noted smaller gains on competition-level tasks (AMC/AIME), suggests bottlenecks may shift to base model capability.
- Virtual rewards inherently change advantage normalization, introducing bounded bias. While the paper provides convergence discussions, whether this bias accumulates into specific policy preferences in long-horizon, multi-turn tool-use agent tasks warrants long-term study.
- The current method assumes all collapsed groups can be fixed statistically. Future work could integrate process rewards or error-type diagnostics to intervene only in groups with high "learning value," avoiding signal creation in truly uninformative or unreliable samples.

## Related Work & Insights
- **vs. GRPO**: GRPO saves the critic using intra-group relative rewards but has zero gradients when rewards are identical. AVSPO preserves the GRPO structure while patching its failure mode by recomputing statistics in collapsed groups.
- **vs. PPO/GAE**: PPO with GAE mitigates variance issues via a value baseline but requires a critic, increasing memory and complexity. This work maintains the critic-less advantage of GRPO by intervening at the reward-statistics level.
- **vs. PRM / Dense Reward**: Process Reward Models provide fine-grained supervision but require expensive labeling or reward model training. AVSPO uses only final-answer verifiers, making it ideal for tasks with deterministic checkers like math or code.
- **vs. DAPO / DCPO**: These methods improve GRPO via clipping, dynamic sampling, or optimization details. AVSPO operates at a lower level by targeting batch reward diversity, making it complementary to system-level GRPO recipes.
- **Insight**: For any RL algorithm based on group comparisons, one should monitor the "effective gradient proportion" rather than just the mean reward. Metrics similar to ACR can be extended to binary verification tasks such as code generation, tool use, and automated theorem proving.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The diagnostic metric is simple, but systematically quantifying advantage collapse and fixing it with virtual rewards is a direct and effective supplement to the GRPO mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multi-scale models, multiple math benchmarks, ACR correlations, ablations, and cost comparisons, though non-binary or open-ended RLHF scenarios are not yet verified.
- Writing Quality: ⭐⭐⭐⭐☆ Clear problem definition; the method and experiments are well-aligned with the collapse thesis; theoretical analysis supports the main claims.
- Value: ⭐⭐⭐⭐⭐ Highly practical for those training RLVR/GRPO; ACR is worth integrating into training logs even solely as a diagnostic metric.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MetaGDPO: Alleviating Catastrophic Forgetting with Metacognitive Knowledge through Group Direct Preference Optimization](../../AAAI2026/model_compression/metagdpo_alleviating_catastrophic_forgetting_with_metacognitive_knowledge_throug.md)
- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](entropy-aware_on-policy_distillation_of_language_models.md)
- [\[ICML 2026\] Active Tabular Augmentation via Policy-Guided Diffusion Inpainting](active_tabular_augmentation_via_policy-guided_diffusion_inpainting.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[NeurIPS 2025\] ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation](../../NeurIPS2025/model_compression/orpo-distill_mixed-policy_preference_optimization_for_cross-architecture_llm_dis.md)

</div>

<!-- RELATED:END -->
