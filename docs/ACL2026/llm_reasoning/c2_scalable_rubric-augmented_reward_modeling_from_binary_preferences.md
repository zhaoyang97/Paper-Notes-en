---
title: >-
  [Paper Note] C2: Scalable Rubric-Augmented Reward Modeling from Binary Preferences
description: >-
  [ACL 2026][LLM Reasoning][Reward Model] Addressing the "double-edged sword" problem where self-generated rubrics often mislead the reward model…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Reward Model"
  - "Rubric"
  - "DPO"
  - "GRPO"
  - "Cooperative Communication"
date: 2026-05-08
content_hash: c7549bf38f495152
---

# C2: Scalable Rubric-Augmented Reward Modeling from Binary Preferences

**Conference**: ACL 2026  
**arXiv**: [2604.13618](https://arxiv.org/abs/2604.13618)  
**Code**: https://github.com/asahi-research/C2 (Available)  
**Area**: LLM Reasoning / RLHF / Reward Modeling  
**Keywords**: Reward Model, Rubric, DPO, GRPO, Cooperative Communication

## TL;DR
Addressing the "double-edged sword" problem where self-generated rubrics often mislead the reward model, the authors utilize LM likelihood margins to automatically label 16 self-sampled rubrics into "helpful / misleading" pairs. Subsequently, a cooperative rubric generator is trained using DPO, and a "critical" verifier—which first assesses the credibility of a rubric before making a judgment—is trained using GRPO. Relying solely on binary preference data, C2 achieves an improvement of up to 6.5 points (RM-Bench) over reasoning RMs trained with GRPO across four preference benchmarks, increases the LC win rate of downstream DPO by 6 points, and allows an 8B model to match the performance of schemes using a 4× larger model (Qwen3-32B) to provide rubrics.

## Background & Motivation

**Background**: Reward models (RMs) are the core of RLHF, yet scalar RMs are easily deceived by surface-level features (e.g., length, format). A recent trend treats preference prediction as a reasoning task (e.g., J1, Think-RM) using GRPO, requiring the RM to output `<analyze>` before delivering a judgment. Another line of work is rubric-augmented verification—generating a set of scoring criteria first, then letting the verifier evaluate accordingly.

**Limitations of Prior Work**: Rubric-based methods depend on human-written or larger proprietary model-generated rubrics, which are costly and incompatible with existing binary preference corpora. An intuitive alternative is allowing the base model to self-generate rubrics. However, experiments on RM-Bench hard (Fig 2) reveal: (1) most self-generated rubrics result in nearly zero change in verifier confidence, making them useless; (2) high-quality rubrics can improve Tulu3-8B accuracy by +8.2 and Qwen3-8B by +13.6, but low-quality rubrics cause both to drop to 39.6% / 49.3%, performing worse than having no rubric at all.

**Key Challenge**: Rubric quality is a double-edged sword—good rubrics provide significant gains, while bad rubrics cause even greater damage. Once a verifier is misled by a rubric, it loses the ability to "reject a bad rubric" independently. This is a "cooperation failure" problem.

**Goal**: (1) Utilize binary preference data—the most common and inexpensive supervisory signal—to train both the rubric generator and the verifier; (2) Enable the verifier to adopt rubrics "critically"—listening to good ones and ignoring bad ones, reverting to rubric-free reasoning when necessary.

**Key Insight**: The authors draw from Grice’s Cooperative Principle—successful interpersonal communication does not rely on the "speaker always being reliable," but on the "listener learning whom to trust, and the speaker learning how to be useful." This dynamic is transferred to the rubric ↔ verifier relationship.

**Core Idea**: A base model self-samples $K=16$ rubrics, which are labeled as (helpful $r^+$, misleading $r^-$) contrastive pairs based on their impact on the verifier’s log-likelihood margin. DPO is used to train the generator to produce more $r^+$, while GRPO is used to train the verifier to provide both correct preferences and accurate assessments of whether a rubric is helpful or misleading. During inference, the verifier adopts the rubric only if it is judged as "helpful."

## Method

### Overall Architecture
Two components, $G_\phi$ (generator) and $V_\theta$ (verifier), are initialized from the same base model $M$. The pipeline consists of three steps:
1. **Synthesizing Contrastive Pairs**: Operating in dual roles ( $M_g$ as generator, $M_v$ as verifier), $M$ calculates the rubric-free margin $m_\emptyset = \log p_{M_v}(l|c) - \log p_{M_v}(\bar{l}|c)$ for each sample $(c=(x,y_A,y_B),l)$. Then, 16 sampled rubrics are used to calculate $m(r_k)$. $r^+$ is selected as the max from $\mathcal{R}^+ = \{r_k : m(r_k) > \max(0, m_\emptyset)\}$, and $r^-$ is selected as the min from $\mathcal{R}^- = \{r_k : m(r_k) < \min(0, m_\emptyset)\}$. Samples where both sets are empty are discarded.
2. **Training**: DPO trains $G_\phi$ to prefer $r^+$ over $r^-$. GRPO trains $V_\theta$ on two tasks: rubric-free prediction of $\hat l$, and rubric-augmented prediction of both $\hat l$ and $q\in\{\text{helpful}, \text{misleading}\}$. The reward consists of format $R_f$ + preference $R_p$ + rubric $R_r$ (augmented task only).
3. **Selective Inference**: Given $c$, a rubric $r \sim G_\phi$ is sampled, and $V_\theta$ outputs $(\hat l, q)$. If $q=\text{helpful}$, $\hat l$ is used; otherwise, it reverts to rubric-free mode for a re-evaluation.

The rubric structure includes a reasoning paragraph + a series of (criterion, yes/no question) pairs.

### Key Designs

1. **Margin-based Synthesizing of Helpful/Misleading Rubrics**:
    - **Function**: Automatically labels each rubric with its "actual effect" using binary preferences, eliminating the need for human-authored rubrics.
    - **Mechanism**: $m(r) = \log p_{M_v}(l|c,r) - \log p_{M_v}(\bar l|c,r)$ represents the verifier's preference for the correct answer after adding the rubric. A helpful rubric must "push the verifier toward the correct answer"—specifically, if $m_\emptyset>0$, the rubric must increase the margin; if $m_\emptyset<0$, it must flip the margin to positive. Misleading rubrics do the opposite. The most extreme $r^+$ and $r^-$ are selected from the qualified sets, retaining approximately 95–98% of the data.
    - **Design Motivation**: Previous rubric supervision relied on expensive GPT-5 scoring (1–5) or non-scalable human writing. Margin-based labels internalize the question of "whether a rubric truly helps" into the verifier's own statistics, making it label-free and noise-resistant.

2. **DPO-trained Cooperative Generator**:
    - **Function**: Encourages the generator to produce high-quality rubrics that the verifier can actually use, rather than rubrics that are superficially plausible but practically useless or harmful.
    - **Mechanism**: DPO is performed using the synthesized $\{(c, r^+, r^-)\}$ where chosen = $r^+$ and rejected = $r^-$. The original rubric prompt template is maintained. Post-training, GPT-5 evaluation of rubric quality improved from 2.11 → 2.66 (Tulu3-8B) and 3.15 → 3.52 (Qwen3-8B), approaching the levels of larger models like Tulu3-70B (2.85) and Qwen3-32B (3.62).
    - **Design Motivation**: Direct SFT only teaches "imitation of a qualified rubric" but fails to teach "avoiding bad ones." DPO's contrastive signal simultaneously encourages the good and punishes the bad, making it more suitable for these "double-edged" problems. Ablation shows that removing negative rubrics (SFT only on good ones) is the most detrimental variant (dropping up to 3.6 points).

3. **Critical Verifier + Selective Inference (GRPO)**:
    - **Function**: Enables the verifier to provide $\hat l$ while also predicting whether a rubric is trustworthy, avoiding being dragged down by low-quality rubrics.
    - **Mechanism**: During GRPO training, rubric-free tasks (format + preference reward only) and rubric-augmented tasks (plus rubric reward $R_r$: +1 if $q=q^*$) are mixed. This ensures the model retains its base rubric-free capability while learning to distinguish between helpful and misleading rubrics. The final verifier prompt requires output in the format `<analyze> → <rubric>helpful|misleading</rubric> → <answer>A|B</answer>`. During inference, if $q=$ misleading, the rubric is discarded and a rubric-free judgment is requested.
    - **Design Motivation**: Existing rubric methods assume the verifier must follow the rubric. This work grants the verifier "veto power." Once the verifier learns to reject, robustness to rubric distribution improves significantly: Fig 5 shows that when the ratio of good to bad rubrics shifts from 9:1 to 1:9, Reasoning RM accuracy plummets from 73% to 52%, whereas C2 only drops from 76% to 70%.

### Loss & Training
- **Data**: Synthesized contrastive pairs from 5k UltraFeedback samples (4,903 retained for Tulu3-8B, 4,648 for Qwen3-8B), creating a total of 14k+ training samples for rubric-free + rubric-augmented tasks.
- **GRPO**: lr 5e-7, batch 64, rollout=8, temperature=1.0 (Tulu3) / 0.6 (Qwen3), max prompt 8192 / response 2048, 1 epoch for C2 (Reasoning RM used 3 epochs to align compute due to having 3× less data).
- **DPO Generator**: lr 5e-7, β=0.1, 3 epochs, max seq 4096.
- **Reward Weights**: Grid search leads to $(w_p, w_r, w_f) = (0.6, 0.3, 0.1)$ for C2.
- **Hardware**: All training conducted on 8× A100 80GB.

## Key Experimental Results

### Main Results
Preference prediction accuracy (%), averaged over 3 seeds:

| Base | Method | RewardBench | RM-Bench | RewardBench2 | JudgeBench | Avg |
|------|------|------|------|------|------|------|
| Tulu3-8B | Base Model | 67.2 | 56.1 | 35.2 | 22.7 | 45.3 |
| Tulu3-8B | Reasoning RM (GRPO) | 73.7 | 64.9 | 45.6 | 35.8 | 55.0 |
| Tulu3-8B | + Self-Rubric | 70.8 | 64.2 | 40.8 | 35.2 | 52.8 |
| Tulu3-8B | + External-Rubric (32B) | 84.9 | 77.7 | 59.6 | 59.2 | 70.4 |
| Tulu3-8B | **C2 (Ours)** | **77.2** | **65.6** | **50.7** | **39.8** | **58.3** |
| Qwen3-8B | Reasoning RM | 89.8 | 81.3 | 67.6 | 60.1 | 74.7 |
| Qwen3-8B | + Self-Rubric | 90.8 | 81.3 | 69.4 | 60.8 | 75.6 |
| Qwen3-8B | + External-Rubric (32B) | 91.3 | 84.6 | 73.9 | 63.9 | 78.4 |
| Qwen3-8B | **C2 (Ours)** | **91.8** | **87.8** | 71.0 | 63.5 | **78.5** |

Downstream DPO + AlpacaEval 2.0 / Arena-Hard:

| Base | Method | AE2 WR | AE2 LC | AH WR |
|------|------|------|------|------|
| Tulu3-8B | DPO w/ Reasoning RM | 13.1 | 19.0 | 21.3 |
| Tulu3-8B | DPO w/ **C2** | **18.3** | **25.0** | **26.8** |
| Qwen3-8B | DPO w/ Reasoning RM | 41.2 | 38.2 | 71.8 |
| Qwen3-8B | DPO w/ **C2** | **44.0** | **40.9** | **74.6** |

### Ablation Study
Averages of RB / RM-Bench / RewardBench2:

| Variant | Tulu3-8B Avg | Qwen3-8B Avg |
|------|------|------|
| C2 (Full) | **64.5** | **83.5** |
| w/o Cooperative Generator | 63.3 | 82.0 |
| w/o Critical Verifier | 62.7 | 81.2 |
| w/o Negative Rubrics | 60.9 | 80.7 |

Robustness under different rubric quality ratios (Tulu3-8B / Qwen3-8B):

| High:Low | Reasoning RM | C2 |
|------|------|------|
| 9:1 | 53% / 73% | 51% / 76% |
| 1:9 | 39% / 52% | 46% / 70% |

### Key Findings
- **Self-Rubric actually decreases performance for Reasoning RMs** (Tulu3-8B 55.0 → 52.8), confirming the motivational experiment's conclusion that self-generated rubrics are often ineffective or harmful.
- **8B models match 32B models using self-generated rubrics**: Qwen3-8B + C2 (78.5) ≈ Qwen3-8B + Qwen3-32B external rubric (78.4), suggesting a larger rubric oracle is unnecessary.
- **Removing negative rubrics results in the largest drop** (Tulu3-8B -3.6, Qwen3-8B -2.8), proving that "learning to reject" is more critical than "learning to generate good ones."
- **C2 exhibits minimal ranking drift**: Fig 5 shows that while Reasoning RM collapses when rubrics are mostly bad, C2 remains stable. This robustness is a highly valuable attribute for real-world deployment.
- **C2 outperforms even when compute-matched**: Allowing Reasoning RM to use 2.5× tokens (via N×2.5 voting) still does not beat C2 (which uses 2.3-2.4× Reasoning RM tokens), indicating that C2’s gains are structural rather than just a result of extra compute.

## Highlights & Insights
- Using the "verifier's own likelihood margin" for rubric labeling is an ingenious self-supervision method. It avoids manual effort and distillation from larger models, internalizing rubric evaluation into the RM training loop and allowing for seamless reuse of any binary preference dataset.
- The "selective inference + retry" is a cost-effective design with significant impact—it essentially transforms the verifier into an ensemble of two policy modes (rubric-aware vs rubric-free), dynamically selected based on rubric quality.
- Analogizing the RM to a "listener" and the rubric generator to a "speaker" and designing the loop around the cooperative principle is highly insightful. This approach can be extended to tasks like tool-use ("to call or not call a tool") or RAG ("to trust retrieval or not").
- The ablation priority "negative > critical verifier > cooperative generator" provides a clear hierarchy—subsequent work with limited resources should prioritize the "verifier learning to reject" component.

## Limitations & Future Work
- Weak base models may struggle with the critical verifier: Fig 5 shows C2 slightly lower than Reasoning RM for Tulu3-8B at the 9:1 ratio, suggesting small models might unnecessarily reject rubrics even when they are mostly helpful.
- High inference cost: Rubric generation + possible retries make C2 2.3–2.4× slower than Reasoning RM (latency of 4–5 seconds/sample). High-traffic scenarios might require more efficient rubric caching or selective generation.
- Training data is limited to 5k UltraFeedback; cross-domain generalization (e.g., code, math) has not been fully verified, with smaller gains observed in RewardBench / JudgeBench math subsets.
- Future directions: (1) Use hierarchical rubrics (coarse-grained decision on whether to use a rubric first) to reduce generator costs; (2) feed retry signals back into RL to let the verifier learn "early rejection"; (3) explore rubric transfer across different base models.

## Related Work & Insights
- **vs Reasoning RM (J1, Think-RM)**: All use GRPO to train verifiers for reasoning. This work adds "explicit rubrics + selective adoption," introducing a critical layer; it gains +3.3–3.5 points on average using the same base model and data.
- **vs Rubric as Reward (Gunjal 2025) / Checklists (Viswanathan 2025)**: These treat rubrics as reward signals but assume they are correct. This work explicitly handles rubric quality uncertainty.
- **vs CARMO / Prometheus (rubric from larger LLM)**: Previous works relied on larger proprietary models for rubrics. This work demonstrates that an 8B model, through self-generation and contrastive training, can match the performance of external rubrics from 4× larger models.
- **Insight**: "Margin-based data synthesis + selective inference" is a generalizable template applicable to any setting where external signals might mislead basic reasoning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of cooperative communication for rubric-verifier collaboration, margin-based self-labeling, and selective inference is novel and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 4 RM benchmarks + downstream DPO + Best-of-N + Rejection Sampling + compute-matched + rubric noise stress test + 3 ablations + GPT-5 scoring + error taxonomy; very comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation experiment in Section 3 clearly defines the problem before introducing the solution. The narrative flow is excellent, and equations + Fig 2 are very persuasive.
- **Value**: ⭐⭐⭐⭐ Provides an RM trained entirely on binary preferences that can replace those using "external large model rubrics." Open-source code and standard HF models make it directly applicable to RLHF practices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](efficient_process_reward_modeling_via_contrastive_mutual_information.md)
- [\[ICML 2026\] Reward Modeling from Natural Language Human Feedback](../../ICML2026/llm_reasoning/reward_modeling_from_natural_language_human_feedback.md)
- [\[ACL 2026\] Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards](process_reward_models_meet_planning_generating_precise_and_scalable_datasets_for.md)
- [\[ICLR 2026\] mR3: Multilingual Rubric-Agnostic Reward Reasoning Models](../../ICLR2026/llm_reasoning/mr3_multilingual_rubric-agnostic_reward_reasoning_models.md)
- [\[ICLR 2026\] Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling](../../ICLR2026/llm_reasoning/fixing_the_broken_compass_diagnosing_and_improving_inference-time_reward_modelin.md)

</div>

<!-- RELATED:END -->
