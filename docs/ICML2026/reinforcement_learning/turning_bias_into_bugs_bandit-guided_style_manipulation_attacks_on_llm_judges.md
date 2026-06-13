---
title: >-
  [Paper Note] Turning Bias into Bugs: Bandit-Guided Style Manipulation Attacks on LLM Judges
description: >-
  [ICML 2026][Reinforcement Learning][LLM-as-a-Judge] Treating known style preferences of LLM judges (such as verbosity, lists, and emojis) as systematically exploitable attack surfaces…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "LLM-as-a-Judge"
  - "Style Bias"
  - "Contextual Bandit"
  - "LinUCB"
  - "Black-box Attack"
date: 2026-05-08
content_hash: fac2329b7eb13821
---

# Turning Bias into Bugs: Bandit-Guided Style Manipulation Attacks on LLM Judges

**Conference**: ICML 2026  
**arXiv**: [2605.26156](https://arxiv.org/abs/2605.26156)  
**Code**: https://github.com/xianglinyang/llm-as-a-judge-attack  
**Area**: AI Safety / LLM Evaluation / Adversarial Attacks  
**Keywords**: LLM-as-a-Judge, Style Bias, Contextual Bandit, LinUCB, Black-box Attack

## TL;DR
Treating known style preferences of LLM judges (such as verbosity, lists, and emojis) as systematically exploitable attack surfaces, the authors model the attack as a contextual bandit. Using LinUCB within a 25-query budget, they adaptively select from 8 semantic-preserving style rewriting actions, achieving a >65% attack success rate and +1~2 point score inflation (out of 9) across 5 mainstream judges, while bypassing style control defenses.

## Background & Motivation

**Background**: LLM-as-a-Judge has become the de facto standard for chatbot evaluation, preference dataset construction, RLHF reward modeling, and even automated peer review (e.g., MT-Bench, AlpacaEval, Arena-Hard). This paradigm is built on the assumption that LLM judges are objective and reliable.

**Limitations of Prior Work**: Another line of research continuously reveals that LLM judges exhibit systematic biases—favoring their own model family (self-preference), lengthy answers, answers using lists/markdown/emojis, and even "well-written but factually incorrect" responses. However, existing work primarily treats these as "limitations" or "side effects requiring calibration," and none have explicitly weaponized them.

**Key Challenge**: Judges are deployed at scale in high-risk pipelines (leaderboards, RLHF, AI peer review) while possessing predictable style preferences. This combination naturally constitutes an exploitable security vulnerability, yet the security community has not systematically characterized this threat. Existing attack methods (e.g., BadJudge backdoors, universal adversarial perturbations, null models) either require training-time intervention or use conspicuous triggers easily detected by defenses.

**Goal**: In a strict black-box setting with a limited budget (25 queries), this paper aims to answer: (1) Can style biases be systematically exploited to manipulate scores on demand? (2) Do different judges have distinct bias profiles? (3) Can attacks maintain semantic equivalence while bypassing style control defenses?

**Key Insight**: The selection of which style edit most effectively inflates a score from a set of known perturbations is modeled as a **contextual bandit**. The context is the semantic embedding of the (question, current answer) pair, the arms are 8 style rewriting actions, and the reward is the marginal score increase from the judge. This abstraction naturally fits the core trade-off of "exploring new biases vs. exploiting known ones" and supports learning a "personalized fingerprint" for each judge.

**Core Idea**: The authors use LinUCB to perform adaptive selection over the style action space, approximating the judge's style preference curve within 25 steps. By performing only semantic-preserving style rewrites, the attack remains both stealthy (appearing as a mere change in writing style) and efficient (learning a specific strategy for each target model).

## Method

### Overall Architecture
BITE maintains a candidate answer pool $\mathcal{P}$ of capacity $K$, initialized with $\{(a_0, S_0)\}$, where $S_0$ is the judge's score for the seed answer $a_0$. Each arm (action) $b \in \mathcal{B}$ corresponds to a style rewrite (8 types in total, e.g., changing verbosity, altering tone, adding lists); each arm maintains its own LinUCB parameters $(\mathbf{A}_b, \bm{v}_b)$. In each round: a parent answer is randomly sampled from the pool $\rightarrow$ encoded as context $\rightarrow$ LinUCB selects an arm $\rightarrow$ a helper LLM executes the rewrite $\rightarrow$ the result is submitted to the target judge $\rightarrow$ the marginal reward is calculated $\rightarrow$ the arm's LinUCB model is updated $\rightarrow$ the new answer is added back to the pool (replacing the lowest score if full).

### Key Designs

1.  **Contextual Bandit Modeling + LinUCB Arm Selection**:
    - **Function**: Adaptively balances "exploring new biases" and "exploiting known biases" to approximate near-optimal strategies within a tight 25-query budget.
    - **Mechanism**: For each round $t$, a pretrained embedding $\phi(q, a_{t-1}) = \bm{x}_t \in \mathbb{R}^d$ encodes the current context. The arm selection rule follows a linear reward estimate with a UCB upper bound: $b_t = \arg\max_{b \in \mathcal{B}} (\bm{x}_t^\top \hat{\bm{\theta}}_b + \alpha \sqrt{\bm{x}_t^\top \mathbf{A}_b^{-1} \bm{x}_t})$, where the first term represents exploitation (expected reward) and the second term represents exploration (uncertainty), with $\alpha$ controlling the trade-off. Each arm independently maintains $\mathbf{A}_b \in \mathbb{R}^{d \times d}$ and $\bm{v}_b \in \mathbb{R}^d$, performing a rank-one update after observing $(\bm{x}_t, r_t)$: $\mathbf{A}_{b_t} \leftarrow \mathbf{A}_{b_t} + \bm{x}_t \bm{x}_t^\top$, $\bm{v}_{b_t} \leftarrow \bm{v}_{b_t} + r_t \bm{x}_t$, and re-estimating $\hat{\bm{\theta}}_{b_t} \leftarrow \mathbf{A}_{b_t}^{-1} \bm{v}_{b_t}$.
    - **Design Motivation**: Each judge has a unique "vulnerability fingerprint"—some prefer verbosity, others prefer markdown lists. Using a contextual bandit allows the attack to learn this personalized preference curve online without any access to model parameters. The linear model choice involves an explicit trade-off: despite potential model misspecification, linear models provide the sample efficiency required to converge within 25 steps.

2.  **8 Semantic-Preserving Actions Distilled from Bias Literature**:
    - **Function**: Compresses the "infinite space of linguistic transformations" into a small, effective discrete action space, ensuring dense reward signals for LinUCB learning.
    - **Mechanism**: The authors systematically reviewed LLM judge bias literature (verbosity bias, list bias, markdown bias, tone bias, self-preference, etc.) and extracted 8 established style transformations as arms. Each action is implemented via a helper LLM $\psi$: "Given original answer $a_{t-1}$ and action $b_t$, produce a semantically equivalent answer with modified style $a_t = \psi(a_{t-1}, b_t)$." The rewriting process strictly requires maintaining the semantic content.
    - **Design Motivation**: An excessively large action space would prevent LinUCB from exploring sufficiently within 25 steps; the action space serves as a "known bias prior," providing a much higher hit rate than random linguistic perturbations.

3.  **Elite Candidate Pool + Marginal Reward Signal**:
    - **Function**: Allows the attack to stack style rewrites on top of "already high-scoring answers" in an evolutionary manner, rather than restarting from $a_0$ in every round.
    - **Mechanism**: An elite pool of size $K$ is maintained, and a parent answer is sampled uniformly each round. The reward is defined as the marginal improvement relative to the parent answer $r_t = S_t - S_{t-1}$, **not the absolute score**. This forces LinUCB to learn "which style further increases the score in this context" rather than "which style has a high score in general." When the pool is full, the lowest-scoring element is eliminated to maintain a high-scoring, diverse population.
    - **Design Motivation**: Marginal rewards avoid the "score ceiling effect"—if absolute scores were used as rewards, all actions would yield near-zero rewards when $S_0 \approx 9$. Eliminating the lowest score instead of the oldest entry ensures the pool preserves "high-score diversity," providing more useful directions for the next round.

### Loss & Training
The attack is black-box and online, requiring no training phase. Theoretically, the authors provide a key result: given a model misspecification level $\zeta_T$, the pseudo-regret of BITE satisfies $R_T = \tilde{O}(dK\sqrt{T} + \zeta_T dKT)$, implying that even with a gap between the linear model and the true non-linear reward, the regret remains controlled by a statistical $\sqrt{T}$ term and a linear $\zeta_T$ term. The technical contribution involves extending Abbasi-Yadkori’s LinUCB analysis to a multi-arm, misspecified setting.

## Key Experimental Results

### Main Results

| Judge | Naive Injection | Fake Completion | PAIR (Jailbreak) | AutoDAN | **BITE (Ours)** |
|--------|-----------|-----------------|------------------|---------|-----------------|
| Qwen3-235b | 0.833 | 1.287 | 1.284 | 0.941 | **2.010** |
| DeepSeek-R1 | 1.016 | 1.214 | 1.856 | 1.365 | **1.909** |
| Llama-3.3-70B | 0.763 | 1.080 | 1.233 | 1.166 | **1.347** |
| Gemini-2.5-flash | 0.862 | 1.089 | 1.337 | 1.489 | **1.731** |
| o3-mini | 0.650 | 1.103 | 0.869 | 1.091 | **1.356** |

BITE significantly outperforms both prompt injection (Naive/Fake Completion/Escape) and jailbreak (PAIR/TAP/AutoDAN) baselines across all 5 judges.

### Ablation Study (MLRBench Automated Peer Review Scenario)

| Judge | Initial Score | Iterative Rewrite | Random Action | **BITE (Ours)** |
|--------|--------|-------------------|----------------|-----------------|
| DeepSeek-R1-0528 | 5.67 ± 0.97 | 6.84 ± 0.22 | 7.31 ± 0.23 | **7.63 ± 0.29** |
| Gemini-2.5-flash | 5.28 ± 0.66 | 6.59 ± 0.39 | 7.18 ± 0.28 | **7.44 ± 0.40** |
| Llama-3.3-70B | 7.90 ± 0.07 | 8.17 ± 0.24 | 8.34 ± 0.19 | **8.38 ± 0.18** |

Iterative Rewrite uses only a single rewriting action; Random Action uses the full action space but selects randomly; BITE uses LinUCB for adaptive selection.

### Key Findings
- **Contribution of action space diversity exceeds adaptive strategy**: Random Action significantly outperforms Iterative Rewrite, suggesting that simply "selecting among 8 biased actions" is much more effective than "repeated full-text rewriting by an LLM"—confirming that "known bias priors" are the primary driver of attack success.
- **LinUCB adds an extra layer of gain on top of diversity**: The improvement of BITE over Random Action validates the value of the adaptive strategy itself, with the gap being particularly pronounced in pairwise scenarios.
- **Equally effective for objective/factual questions**: BITE consistently inflates scores even on factual problems, confirming that LLM judges' assessments of "objective correctness" are also susceptible to style interference—a dangerous finding implying that no category on a leaderboard is safe.
- **Bias profiles do not transfer across judges**: The authors found that each judge's "vulnerability fingerprint" is unique; the attack does not transfer directly, which validates the necessity of BITE's adaptive, per-target personalized learning.
- **Stealth**: >90% of attack responses maintain semantic equivalence with the original answer under LLM similarity evaluations, bypassing style control defenses and automated detectors designed for style manipulation.

## Highlights & Insights
- **"Known bias as a prior" is the core design philosophy**: Rather than letting the agent explore all linguistic space from scratch, the authors explicitly encode 8 literature-verified biases as actions. This allows the 25-query budget to suffice for convergence, providing a paradigm for "few-shot black-box attacks": using prior literature to compress the action space.
- **Marginal Reward + Elite Pool**: Grafting population-based evolutionary concepts onto a bandit framework avoids the collapse of reward signals due to absolute score saturation. This trick is applicable to other optimization scenarios where the target is already near its upper bound (e.g., aligning high-scoring responses or jailbreaking models with high refusal rates).
- **Regret bound explicitly tracks misspecification**: This is a valuable theoretical contribution, notifying the user of the degree of model mismatch under which the attack is still provably convergent, providing "theoretical security" for the attack.
- **Exposing paradigm-level flaws**: The true value of the paper lies not just in "another attack," but in falsifying the foundation of the LLM-as-a-Judge paradigm—that "judges are style-neutral." This has direct policy implications for leaderboard design, RLHF data cleaning, and AI peer review.

## Limitations & Future Work
- The action space consists of 8 manually encoded biases, which could be "reverse-sanitized" by defenders: if a defender trains a judge to normalize these 8 styles, the attack may fail immediately. The paper does not discuss automatic expansion of the action space or closed-loop adversarial scenarios.
- The 25-query budget is a realistic assumption for chatbot scenarios, but for large-scale RLHF data generation, the budget is nearly infinite, potentially making attacks stronger but also easier to detect via traffic anomalies. This trade-off is not characterized.
- Semantic maintenance relies on another LLM for similarity evaluation, creating a circular vulnerability where the "referee LLM and judge LLM share correlated biases." Ideally, human evaluation should be used for final semantic consistency verification.
- Non-transferability is both a highlight and a limitation: attacking each new model requires the full 25-query budget to relearn the strategy, which poses a non-negligible cost for large-scale attacks (e.g., across multiple leaderboards).
- Discussion on the defense side is relatively weak—while the paper proves that style control and existing detectors fail, it does not provide constructive solutions for "how to defend."

## Related Work & Insights
- **vs. BadJudge (Tong et al., 2025)**: BadJudge implants backdoor triggers during training, whereas BITE is purely black-box during inference. BadJudge requires supply chain poisoning, while BITE requires only API access, making the threat surface much broader.
- **vs. PAIR / AutoDAN (Jailbreak variants)**: These methods optimize adversarial prefixes to force restricted behaviors; BITE changes style without altering content. Jailbreak prefixes are easily recognized by safety training, whereas BITE's style shifts reside in the blind spots of "safety + instruction following" training.
- **vs. Universal Adversarial Perturbations (Shi et al., 2024)**: Universal perturbations seek universality at the cost of high detectability. BITE takes the opposite route—personalizing for each target while maintaining high stealth and effectiveness.
- **vs. Null Model (Zheng et al., 2025)**: Null Model uses heuristics to exploit evaluation protocol flaws (e.g., irrelevant answers getting high scores). BITE uses learning algorithms to systematically explore the bias curve, making it more general and effective on harder benchmarks like Arena-Hard.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Combining "known bias as a prior" with LinUCB in the LLM judge safety context is a truly fresh perspective, not just a variant of common jailbreaks.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 5 judges, two benchmarks, two grading modes, three types of defenses, plus an MLRBench peer review case study.
- Writing Quality: ⭐⭐⭐⭐ The three RQs are clear, and the algorithm diagrams are intuitive. The theoretical section might be challenging for readers without a bandit background, but main results are clearly stated.
- Value: ⭐⭐⭐⭐⭐ Directly challenges the foundation of the LLM-as-a-Judge paradigm. It has policy-level impact on leaderboards, RLHF, and AI peer review, making it a rare work capable of changing community practices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning](llm-guided_communication_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[ICML 2026\] Adaptive Bandit Algorithms for Contextual Matching Markets](adaptive_bandit_algorithms_for_contextual_matching_markets.md)
- [\[ICML 2026\] One Bias After Another: Mechanistic Reward Shaping and Persistent Biases in Language Reward Models](one_bias_after_another_mechanistic_reward_shaping_and_persistent_biases_in_langu.md)
- [\[AAAI 2026\] A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](../../AAAI2026/reinforcement_learning/a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)
- [\[ICLR 2026\] ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation](../../ICLR2026/reinforcement_learning/reform_reflected_flows_for_on-support_offline_rl_via_noise_manipulation.md)

</div>

<!-- RELATED:END -->
