---
title: >-
  [Paper Note] Conditional Advantage Estimation for Reinforcement Learning in Large Reasoning Models
description: >-
  [ICLR 2026][LLM Reasoning][RLVR] CANON eschews predefined directional priors such as "higher entropy is better" or "shorter length is better." Instead, it sorts sampled responses for the same query by a target metric (entropy or length) and splits them into two groups. By utilizing **inter-group comparisons** to automatically discover which metric tre
tags:
  - ICLR 2026
  - LLM Reasoning
  - RLVR
  - GRPO
date: 2026-05-08
content_hash: a7e895fae9818bc2
---
# Conditional Advantage Estimation for Reinforcement Learning in Large Reasoning Models

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=CTEXdHB1BB](https://openreview.net/forum?id=CTEXdHB1BB)  
**Code**: [https://github.com/the-secret-key/CANON](https://github.com/the-secret-key/CANON) (§CANON in the paper, subject to repository name)  
**Area**: LLM Reasoning / RLVR  
**Keywords**: RLVR, GRPO, Advantage Estimation, Conditional Grouping, Entropy, Token Efficiency, Reasoning Models  

## TL;DR
CANON eschews predefined directional priors such as "higher entropy is better" or "shorter length is better." Instead, it sorts sampled responses for the same query by a target metric (entropy or length) and splits them into two groups. By utilizing **inter-group comparisons** to automatically discover which metric trend favors accuracy and **intra-group comparisons** to select superior responses within the same trend, CANON amplifies the effective influence of target metrics without the need for manual penalty term tuning.

## Background & Motivation
**Background**: Reinforcement Learning with Verifiable Rewards (RLVR) has become a core method for enhancing the mathematical and logical reasoning capabilities of Large Reasoning Models (LRMs). GRPO and its variant DR.GRPO are the mainstream algorithms. Researchers have observed that certain training metrics (e.g., entropy, response length) correlate strongly with reasoning behavior, leading to efforts to inject these "human priors" into training.

**Limitations of Prior Work**: Mainstream approaches rely on **reward shaping** (e.g., length penalties for efficiency) or **advantage shaping** (e.g., using entropy signals to maintain exploration). These methods inherently assume a "higher-is-better" or "lower-is-better" relationship for a given metric, implemented via manual penalties or preference coefficients. If hyperparameters are poorly tuned, these directional priors can over-bias the model, forcing metrics up or down at the expense of performance.

**Key Challenge**: The "optimal direction" for a single metric is often **context-dependent**. High-entropy responses favor exploration and help solve complex problems, while low-entropy responses reflect higher certainty and better accuracy on problems within the model's current capability. Presetting a single direction inevitably fails across diverse scenarios.

**Goal**: To **amplify the impact of target metric variations on advantage estimation** without presetting directional preferences, allowing the model to discover "which direction to move" from its own rollouts, thereby naturally acquiring beneficial behaviors (e.g., enhanced exploration or improved token efficiency).

**Core Idea**: **Conditional Grouping + Dual-Path Advantage**. Responses are sorted and split into two groups based on metric values. An "inter-group advantage" identifies the optimal direction, while an "intra-group advantage" addresses selection. These are fused via a weight $\mu$—where DR.GRPO emerges as a special case with $\mu=0.5$.

## Method

### Overall Architecture
For each query, $G$ responses are sampled as in GRPO. CANON introduces a **conditional regrouping** step before calculating advantages: responses are sorted by a target metric (e.g., per-token entropy or response length) and split into a high-value group $G^+_q$ and a low-value group $G^-_q$. Subsequently, two types of advantages are calculated in parallel: **Inter-group advantage**, which compares each response against the mean of the "opposite group" to reveal which metric trend yields higher rewards; and **Intra-group advantage**, which compares responses against the mean of their "own group" to select superior samples within the same trend. Finally, these are weighted by $\mu$ to form the final advantage for a PPO/GRPO-style policy update.

```mermaid
flowchart TD
    A[query q: Sample G responses] --> B[Sort by target metric<br/>Entropy or Length]
    B --> C[Split into equal groups<br/>G+ High / G- Low]
    C --> D[Inter-group Advantage A_inter<br/>Compare to opposite group mean]
    C --> E[Intra-group Advantage A_intra<br/>Compare to own group mean]
    D --> F[A_CANON = μ·A_inter + (1-μ)·A_intra]
    E --> F
    F --> G[Policy Update]
```

### Key Designs

**1. Conditional Regrouping: Replacing vague global baselines with distinct contrastive groups.** GRPO uses the mean of all responses in a group as a baseline, which can provide an unclear contrast and noisy optimization signals. CANON introduces an arbitrary condition $c$, denoting responses satisfying the condition as $C^+_q=\{o\mid o\text{ satisfies }c\}$ and the remainder as $C^-_q=G_q\setminus C^+_q$. This paper focuses on "relative conditions"—sorting by entropy or length—ensuring that the grouping carries contrastive semantics (high vs. low), providing clear anchors for comparison.

**2. Inter-group Advantage: Allowing the model to discover "favorable directions" automatically.** The inter-group advantage compares each response to the mean reward of the **opposite group**:

$$\hat A^{\text{inter}}_{q,o,t}=\begin{cases}R_o-\text{mean}(\{R_{o'}\mid o'\in G^+_q\}), & o\in G^-_q\\ R_o-\text{mean}(\{R_{o'}\mid o'\in G^-_q\}), & o\in G^+_q\end{cases}$$

Taking entropy as an example: if the low-entropy group (more certain) yields a higher average reward, the inter-group advantage will favor "low-entropy and correct" responses. This avoids presetting the entropy direction; the direction is determined automatically by the reward difference between the two groups. The paper provides **Theorem 1** to prove that when groups are of equal size, $|\hat A^{\text{inter}}|/|\hat A^{\text{DR.GRPO}}|>1$, meaning the inter-group advantage **amplifies the influence of the grouping metric** relative to DR.GRPO. **Theorem 2** ensures this amplification is "selective"—it only amplifies the metric used for grouping and does not amplify independent factors.

**3. Intra-group Advantage: Selection within trends and rescuing correct "underdog" responses.** The intra-group advantage compares responses to their **own group** mean:

$$\hat A^{\text{intra}}_{q,o,t}=\begin{cases}R_o-\text{mean}(\{R_{o'}\mid o'\in G^+_q\}), & o\in G^+_q\\ R_o-\text{mean}(\{R_{o'}\mid o'\in G^-_q\}), & o\in G^-_q\end{cases}$$

While similar to DR.GRPO within a smaller subset, the difference in group means is crucial: when a group has a lower average reward, correct responses within that group receive a **larger** advantage ($1-\text{mean}(G^+_q)>1-\text{mean}(G^-_q)$ when $\text{mean}(G^+_q)<\text{mean}(G^-_q)$). In the context of entropy, this provides more incentive to exploratory "high-entropy but correct" rollouts, encouraging effective exploration.

**4. Unified Weighting and Training Alignment: Connecting to DR.GRPO and tuning efficiency.** The two advantages are fused as $\hat A^{\text{CANON}}_{q,o,t}=\mu\hat A^{\text{inter}}_{q,o,t}+(1-\mu)\hat A^{\text{intra}}_{q,o,t}$. Setting $\mu=0.5$ exactly recovers DR.GRPO (Eq. 7), while $\mu=1$ or $\mu=0$ yields CANON-Inter or CANON-Intra, respectively. $\mu$ can also be scheduled during training. Additionally, a weight $\alpha$ can be applied to one group in the inter-group calculation:

$$\hat A^{\text{inter}}_{q,o,t,\alpha}=\begin{cases}R_o-\alpha\cdot\text{mean}(\{R_{o'}\mid o'\in G^+_q\}), & o\in G^-_q\\ \alpha\cdot R_o-\text{mean}(\{R_{o'}\mid o'\in G^-_q\}), & o\in G^+_q\end{cases}$$

By designating the long-response group as $G^+_q$ and setting $\alpha=0.9$, length can be significantly compressed with minimal performance drop, achieving high token efficiency.

## Key Experimental Results

### Main Results
Based on Qwen2.5-Math-7B, comparison with various advantage estimation baselines (Average Acc across 6 math benchmarks / Average Tokens; ZebraLogic High-complexity Logic Acc):

| Method | Math Acc | Math Tokens | Logic Acc | Logic Tokens |
|------|----------|-------------|-----------|--------------|
| GRPO | 53.8 | 3730 | 17.2 | 9406 |
| DR.GRPO ($\mu=0.5$) | 55.7 | 1522 | 26.2 | 4896 |
| Entropy Adv | 56.3 | 2389 | 18.5 | 8207 |
| Clip-Cov | 56.1 | 1344 | 26.5 | 4045 |
| CANON-Inter (by Entropy) | **57.6** | 1466 | 25.7 | 4415 |
| CANON-Intra (by Entropy) | 54.7 | 2959 | 29.1 | 3101 |
| CANON-Dynamic (by Entropy) | 56.7 | 1452 | 29.2 | 3535 |
| CANON-Inter (by Length) | 55.3 | **1008** | **29.5** | 3652 |

Key takeaways: Inter-group advantage (by entropy) achieves the highest math accuracy (57.6), an improvement of ~1.9 points over DR.GRPO. Intra-group advantage excels on complex logic tasks (up to 5.2 point gain on XLarge subset).

### Ablation Study
Robustness of CANON-Dynamic ($\mu$ scheduling) across three models and two tasks:

| Model | Method | Math Acc | Logic Acc |
|------|------|----------|-----------|
| Qwen2.5-Math-7B | DR.GRPO | 55.7 | 26.2 |
| Qwen2.5-Math-7B | First-Inter-Later-Intra | **57.0** | 28.3 |
| Qwen2.5-Math-1.5B | DR.GRPO | 46.4 | 12.8 |
| Qwen2.5-Math-1.5B | First-Inter-Later-Intra | 46.8 | **17.0** |
| Llama3.1-8B | DR.GRPO | 22.0 | 14.9 |
| Llama3.1-8B | Cosin-First-Inter-Later-Intra | 22.6 | **18.9** |

### Key Findings
- **Directional Adaptivity**: The relative importance of inter-group vs. intra-group varies by task—math benefits more from inter-group (finding the right direction), while complex logic benefits more from intra-group (incentivizing exploration in weak groups). Dynamic scheduling captures both.
- **Efficiency Frontier**: Grouping by length with $\alpha<1$ establishes a new Pareto frontier for performance-cost. It achieves a 2.63× math performance gain under low token budgets and a 45.5% reduction in token consumption for the same performance level.
- **Theoretical Support**: As per Theorem 1, equal-size grouping amplifies the target metric's influence relative to DR.GRPO, and per Theorem 2, it does not pollute other independent factors.

## Highlights & Insights
- **Conceptual Innovation of "No Pre-set Direction"**: Changing "where to go" from a hyperparameter to an inference based on reward differences between data groups bypasses the most fragile assumptions in reward/advantage shaping.
- **Elegant Unified Perspective**: Viewing DR.GRPO as CANON($\mu=0.5$) clarifies that mainstream group-relative advantages are a uniform mix of inter- and intra-group signals. CANON decouples these and allows for customized proportions.
- **Dual-Purpose Framework**: Switching the grouping metric (from entropy to length) shifts the goal from "enhancing exploration" to "enhancing efficiency," demonstrating high flexibility.

## Limitations & Future Work
- Grouping is limited to "binary equal splits," and theoretical guarantees (Theorem 1) depend on this equality. Optimal grouping strategies for multiple groups, unequal splits, or continuous conditions remain unexplored.
- $\mu$ scheduling strategies (e.g., First-Inter-Later-Intra, cosine) remain somewhat heuristic. Optimal schedules vary across models (e.g., Llama prefers the cosine variant), requiring further research into automated scheduling.
- Metrics are restricted to entropy and length. It remains to be verified whether CANON can generalize to abstract reasoning behavior metrics (e.g., frequency of reflection, sub-goal decomposition).

## Related Work & Insights
- **Advantage Estimation Taxonomy**: Progressing from GAE in PPO to critic-free methods like ReMax, RLOO, GRPO, and REINFORCE++, CANON extends the group-relative lineage by upgrading from a "single baseline" to "dual-group contrast."
- **RLVR + Behavior Shaping**: Compared to length penalties (Arora & Zanette, Luo et al.) or entropy advantage shaping (Chen et al., Cheng et al.), CANON differs by avoiding penalty terms and preset directions, theoretically avoiding the unintended amplification of irrelevant factors.
- **Inspiration**: The approach of replacing directional priors with data-driven direction discovery could be transferred to other RLHF/alignment scenarios where preference intensity typically requires human specification.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — "Conditional regrouping + dual-path advantage without preset directions" is a clean and novel perspective. The unification of DR.GRPO as a special case is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers three models, two tasks, and nine benchmarks total, including dynamic scheduling and efficiency frontiers, supported by two theorems. However, grouping variations are somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logical flow from motivation to theory and experiments; formulas and diagrams are well-presented.
- **Value**: ⭐⭐⭐⭐ — Provides a practical, low-tuning, and transferable paradigm for advantage shaping in RLVR. The efficiency frontier results have direct industrial significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Quantile Advantage Estimation: Stabilizing RLVR for LLM Reasoning](quantile_advantage_estimation_stabilizing_rlvr_for_llm_reasoning.md)
- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](../../ACL2026/llm_reasoning/revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)
- [\[ICLR 2026\] A Simple "Motivation" Can Enhance Reinforcement Finetuning of Large Reasoning Models](a_simple_motivation_can_enhance_reinforcement_finetuning_of_large_reasoning_mode.md)
- [\[ICLR 2026\] Learning What Reinforcement Learning Can't: Interleaved Online Fine-Tuning for Hardest Questions](learning_what_reinforcement_learning_cant_interleaved_online_fine-tuning_for_har.md)
- [\[ICLR 2026\] NFT: Bridging Supervised Learning and Reinforcement Learning in Math Reasoning](nft_bridging_supervised_learning_and_reinforcement_learning_in_math_reasoning.md)

</div>

<!-- RELATED:END -->
