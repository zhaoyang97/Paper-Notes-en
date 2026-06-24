---
title: >-
  [Paper Note] Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs
description: >-
  [ICLR 2026][Reinforcement Learning][RLVR] Addressing the debate over whether RLVR truly improves reasoning capabilities or merely enhances sampling efficiency, this paper proposes a new metric, CoT-Pass@K (requiring both correct answers and correct reasoning). Using a theoretical framework for GRPO, it proves that as long as the base model possesses a "logic prior" where correct CoT more likely leads to correct answers, binary rewards based solely on answer correctness will *…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "RLVR"
  - "GRPO"
  - "Chain-of-Thought"
  - "Pass@K"
  - "Reasoning Capability Boundary"
date: 2026-05-08
content_hash: 06c6f01786d71629
---

# Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=jGbRWwIidy](https://openreview.net/forum?id=jGbRWwIidy)  
**Code**: Verification data is public at [HuggingFace: AIME24-25_CoT_Verification](https://huggingface.co/datasets/XumengWen/AIME24-25_CoT_Verification)  
**Area**: LLM Reasoning / Reinforcement Learning / RLVR  
**Keywords**: RLVR, GRPO, Chain-of-Thought, Pass@K, Reasoning Capability Boundary

## TL;DR
Addressing the debate over whether RLVR truly improves reasoning capabilities or merely enhances sampling efficiency, this paper proposes a new metric, CoT-Pass@K (requiring both correct answers and correct reasoning). Using a theoretical framework for GRPO, it proves that as long as the base model possesses a "logic prior" where correct CoT more likely leads to correct answers, binary rewards based solely on answer correctness will **implicitly** drive up the probability of generating correct reasoning, thereby authentically extending the reasoning boundary of base models.

## Background & Motivation
**Background**: After DeepSeek-R1 successfully implemented long Chain-of-Thought (CoT) reasoning using GRPO, RLVR (Reinforcement Learning with Verifiable Rewards) has become the mainstream paradigm for teaching LLMs to reason. The model acts as the policy, the CoT sequence as the actions, and a deterministic verifier provides binary rewards based only on the correctness of the final answer. The expectation is that models can "learn from experience" through free exploration.

**Limitations of Prior Work**: Several studies found that while RLVR improves Pass@1, it often fails to match or even falls behind the base model on Pass@K (obtaining at least one correct answer in K samples). Yue et al. (2025) proposed a provocative hypothesis: **all correct reasoning paths already exist in the base model, and RLVR merely increases their sampling probability at the cost of narrowing the overall reasoning capacity**. This hypothesis received significant support, though counter-examples exist in competitive programming and specific puzzle difficulties.

**Key Challenge**: The root of the debate lies in the unreliability of the **Pass@K** metric itself. Math answers are often simple integers; a base model might "produce incorrect reasoning but stumble upon the correct answer," especially after multiple samples on difficult problems. Since Pass@K counts "guessing correctly" as a success, it makes base models appear deceptively strong, masking the true contribution of RLVR.

**Goal**: (1) Identify an evaluation metric that distinguishes "actual reasoning" from "guessing"; (2) theoretically explain why rewarding only the answer correctness improves the reasoning process; (3) verify through training dynamics and CoT quality experiments that this reasoning improvement is genuine.

**Key Insight**: The authors argue that reasoning evaluation must incorporate the correctness of the CoT. They identify a critical difference between LLMs and traditional RL: LLMs possess strong knowledge/logic priors after pre-training. The probability of a "correct reasoning" leading to a correct answer is naturally different from that of an "incorrect reasoning." this gap serves as the lever for GRPO.

**Core Idea**: Use **CoT-Pass@K**, a dual-judgment metric (answer + reasoning), to reveal hidden capability gains. Then, use the "logic prior" hypothesis to prove that GRPO gradients monotonically increase the generation probability of correct CoTs, elevating RLVR's effectiveness from an empirical observation to an interpretable mechanism.

## Method

### Overall Architecture
The paper does not propose a new algorithm but rather a framework of "re-evaluation + theoretical explanation + training dynamic verification + quality review" to answer whether RLVR truly strengthens reasoning. It consists of four collaborative components: First, using **CoT-Pass@K** to exclude "guessing" from success and re-measure reasoning boundaries in math/code; second, providing a theoretical framework for **GRPO implicit incentive**, deriving that "the expected advantage of correct CoT is positive, while that of incorrect CoT is negative" based on the logic prior; third, replicating DAPO training to verify that training dynamics align with the theorem; and finally, using **SFT for reverse CoT quality measurement** by fine-tuning a base model on CoTs generated by models at different stages.

The targets of the study are the base model Qwen2.5-32B and its RLVR version DAPO-Qwen-32B (math), and the distilled model DeepSeek-R1-Distill-Qwen-7B and its RLVR version AceReason-Nemotron-7B (code).

### Key Designs

**1. CoT-Pass@K: A Reliable Metric to Exclude "Guessing"**

Pass@K's fatal flaw is its focus only on the answer, allowing base models to appear competent by guessing simple integer answers through repeated sampling. The authors define CoT-Pass@K: success is achieved only if the final answer $I_{\text{Ans}}(a_i)=1$ **and** the intermediate reasoning $I_{\text{CoT}}(c_i)=1$. $I_{\text{CoT}}$ is defined as "intermediate tokens expressing accurate and necessary logic to reach the correct answer." Since manual verification is difficult, the authors use an **LLM-as-a-CoT-Judge** paradigm with DeepSeek-R1-0528-Qwen3-8B. Using this metric, RLVR models maintain a consistent and significant lead over base models across all K values (up to 1024) on AIME 2024/2025. This debunks the illusion that base models catch up on Pass@K; they catch up on guessing, not reasoning.

**2. Logic Prior Hypothesis + Theorem 1: Why Answer Rewards Support Correct Reasoning**

This is the theoretical core. The authors introduce the **Logic Prior** hypothesis—correct CoTs are more likely to yield correct answers than incorrect CoTs:

$$P(I_{\text{Ans}}=1 \mid I_{\text{CoT}}=1)=\alpha > P(I_{\text{Ans}}=1 \mid I_{\text{CoT}}=0)=\beta$$

In GRPO, given the advantage $\hat{A}(y_i)=\dfrac{R(y_i)-\mu_Y}{\sigma_Y}$ and reward $R(y_i)=I_{\text{Ans}}(a_i)$, **Theorem 1** proves that under this hypothesis and given a large enough group size $G$:

$$\mathbb{E}\big[\hat{A}(y_i)\mid I_{\text{CoT}}(c_i)=1\big]>0,\qquad \mathbb{E}\big[\hat{A}(y_i)\mid I_{\text{CoT}}(c_i)=0\big]<0$$

Thus, the policy gradient monotonically increases the probability $p_c^\theta$ of generating correct CoT. The driving force is $\alpha - \beta > 0$. As $\alpha$ increases (reasoning becomes robust) and $\beta$ decreases (false correlations diminish) during training, the gap widens, accelerating positive reinforcement.

**3. Training Dynamic Metrics: Verifying Early Incentive and Generalization**

The authors replicate DAPO training and define two metrics for $G$ responses: answer pass count $C=\sum_i I_{\text{Ans}}(a_i)$ and "CoT and answer both pass" count $D=\sum_i I_{\text{CoT}}(c_i)\cdot I_{\text{Ans}}(a_i)$. They monitor $P(CA)(q)=\tfrac{C}{G}$ and $P(CC\mid CA)(q)=\tfrac{D}{C}$. Observations show that for optimized training problems, $P(CC\mid CA)$ rises as $P(CA)$ approaches 1, confirming implicit reasoning optimization. Generalization gains in CoT-Pass@K appear from the **earliest stages** of training.

**4. SFT for CoT Quality: Scoring Reasoning via Generalization**

To measure CoT quality beyond a binary judge, the authors use a learning perspective: high-quality CoTs should result in better generalization when used for SFT. They fine-tune the base Qwen2.5-32B on CoTs generated by models at various RLVR steps. Results show that models fine-tuned on later RLVR stages progressively achieve better Pass@1. CoTs from DAPO allow an SFT model to nearly replicate the performance of DAPO-Qwen-32B, suggesting that **expensive RLVR capabilities can be reproduced at low cost via SFT** if post-RLVR CoT data is available.

## Key Experimental Results

### Main Results (Reasoning Boundary Extension)

| Task/Benchmark | Comparison | Metric | Conclusion |
|------|------|------|------|
| AIME 2024 / 2025 (Math) | DAPO-Qwen-32B vs Qwen2.5-32B | Pass@K | Base model ties or exceeds as K increases (aligns with Yue et al.) |
| AIME 2024 / 2025 (Math) | Same as above | **CoT-Pass@K** | RLVR model consistently leads across all K (≤1024) |
| MATH-500 / AMC23 (Math) | Same as above | CoT-Pass@K | RLVR effect less pronounced (easier problems or potential contamination) |
| LiveCodeBench v1–v6 (Code) | AceReason-Nemotron-7B vs DeepSeek-R1-Distill-Qwen-7B | Pass@K | RLVR model leads consistently; code execution renders "guessing" nearly impossible |

### Training Dynamics & CoT Quality Analysis

| Analysis | Key Metric | Finding |
|------|---------|------|
| Optimization Effect | $P(CA)(q)$, $P(CC\mid CA)(q)$ | $P(CC\mid CA)$ rises as $P(CA)\to 1$, verifying implicit incentive |
| Generalization Behavior | Pass@K / CoT-Pass@K | Improvements appear from early training stages |
| DAPO Limitation | Median $P(CC\mid CA)$ | Approaches ~0.7 after 400 steps; reward signal vanishes once $P(CA) \approx 1$ |
| CoT Quality (SFT Proxy) | Test Pass@1 | Quality increases monotonically with RLVR steps; SFT reproduces RLVR gains |

### Key Findings
- **Metrics determine conclusions**: Pass@K suggests RLVR is useless; CoT-Pass@K proves RLVR extends boundaries. The debate is about measurement, not facts.
- **Base model Pass@K is "fake strong"**: Base models guess simple answers on hard problems with wrong CoT, inflating their Pass@K metrics.
- **DAPO Ceiling**: Once training problems are fully optimized (variance in group reward becomes zero), signal to correct the remaining ~30% flawed CoTs is lost. This explains R1-Zero's readability issues.

## Highlights & Insights
- **One metric ends the debate**: By upgrading "correct answer" to "correct answer + reasoning," CoT-Pass@K exposes the systematic bias in Pass@K.
- **Logic Prior as a dividing line**: The theorem transforms the simple inequality $\alpha - \beta > 0$ into a rigorous conclusion that answer-rewards implicitly supervise reasoning.
- **Replicating RLVR via SFT**: Using post-RLVR CoT for SFT can approximate RLVR gains, suggesting an engineering path: "run expensive RL once, distill to many."
- **Scalability of LLM-as-a-Judge**: The paradigm provides a way to verify unstructure reasoning at scale, though it highlights the need for reliable judge benchmarks.

## Limitations & Future Work
- **Reliance on LLM judges**: CoT-Pass@K depends on the accuracy of the judge model (DeepSeek-R1).
- **Failure of Logic Prior**: If a base model has a stubborn bias where an incorrect CoT leads to a correct answer, GRPO will reinforce the error.
- **Domain coverage**: Experiments are limited to $\le$ 32B models and math/code domains.
- **Vanishing signals**: Pure outcome rewards cannot correct remaining CoT flaws once the problem is solved, requiring process-level supervision or curriculum learning.

## Related Work & Insights
- **vs. Yue et al. (2025)**: While they concluded RLVR only shifts sampling probability, this paper shows that their metric (Pass@K) was the issue.
- **vs. Chen et al. (2025b)**: Consistent with their findings in code, this paper extends the context to math and explains why code Pass@K is more reliable than math Pass@K.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ CoT-Pass@K + Logic Prior theorem elevates the empirical debate to a mechanistic understanding.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple domains, training dynamics, and SFT validation, though limited to 32B models.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, seamless theory-to-experiment transition, and honest discussion of limitations.
- Value: ⭐⭐⭐⭐⭐ directly addresses a core debate in the community and provides practical engineering insights for SFT.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains](rubrics_as_rewards_reinforcement_learning_beyond_verifiable_domains.md)
- [\[ICLR 2026\] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [\[ICLR 2026\] From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for RL of Open-ended Generation](from_verifiable_dot_to_reward_chain_harnessing_verifiable_reference-based_reward.md)
- [\[ICLR 2026\] RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents](rlvmr_reinforcement_learning_with_verifiable_meta-reasoning_rewards_for_robust_l.md)
- [\[ICLR 2026\] Lookahead Tree-Based Rollouts for Enhanced Trajectory-Level Exploration in Reinforcement Learning with Verifiable Rewards](lookahead_tree-based_rollouts_for_enhanced_trajectory-level_exploration_in_reinf.md)

</div>

<!-- RELATED:END -->
