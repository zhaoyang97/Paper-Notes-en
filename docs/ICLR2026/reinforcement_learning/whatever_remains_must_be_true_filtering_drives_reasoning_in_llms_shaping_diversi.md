---
title: >-
  [Paper Note] Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity
description: >-
  [ICLR 2026][Reinforcement Learning][α-divergence] This paper proposes the DMVR framework and the α-DPG algorithm. By explicitly defining a target distribution that "filters out incorrect answers" and approximating it via the α-divergence family, the work unifies RLVR (Reverse KL) and rejection sampling fine-tuning (Forward KL), achieving Pareto-optimal performance on the accuracy–coverage frontier for Lean theorem proving.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - α-divergence
  - distributional matching
  - RLVR
  - diversity preservation
  - theorem proving
date: 2026-05-08
content_hash: 94b7d0c30df58b35
---

# Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity

**Conference**: ICLR 2026
**arXiv**: [2512.05962](https://arxiv.org/abs/2512.05962)
**Code**: [https://github.com/naver/alpha-dpg](https://github.com/naver/alpha-dpg)
**Area**: LLM NLP / Reinforcement Learning / LLM Reasoning
**Keywords**: α-divergence, distributional matching, RLVR, diversity preservation, theorem proving

## TL;DR
This paper proposes the DMVR framework and the α-DPG algorithm. By explicitly defining a target distribution that "filters out incorrect answers" and approximating it via the α-divergence family, the work unifies RLVR (Reverse KL) and rejection sampling fine-tuning (Forward KL), achieving Pareto-optimal performance on the accuracy–coverage frontier for Lean theorem proving.

## Background & Motivation

**Background**: Reinforcement learning with verifiable rewards (RLVR, e.g., GRPO/PPO) has become the standard approach for tuning LLM reasoning capabilities. However, growing evidence shows that RLVR-trained models suffer from **severe diversity loss** (mode collapse)—while pass@1 improves, the diversity of the generation policy drops substantially, causing pass@k (for large k) to fall below that of the base model.

**Limitations of Prior Work**: RLVR methods (GRPO/PPO, etc.) implicitly optimize the Reverse KL divergence to the target distribution, a mode-seeking divergence that concentrates the model on a few high-reward regions while ignoring other valid solutions. When $\beta=0$, the objective degenerates to pure REINFORCE with no diversity protection whatsoever. Existing mitigation strategies (KL penalties, Rw-Ulkly, etc.) treat symptoms rather than causes.

**Key Challenge**: There is a fundamental trade-off between accuracy (pass@1) and coverage (pass@k). Existing RL methods can only favor the accuracy end, and lack a systematic mechanism to control this trade-off.

**Goal**: How can one preserve the solution diversity already present in the base model while maintaining correctness? How can a continuously controllable mechanism for the accuracy–coverage trade-off be provided?

**Key Insight**: The paper re-examines RLVR from a **distributional matching** perspective—explicitly defining the target distribution as one that filters out incorrect answers while preserving the relative probability of correct answers: $p_x(y) \propto \pi_{\text{base}}(y|x) \cdot v(y,x)$. The α-divergence family is then used to approximate this target distribution, with different values of α corresponding to different accuracy–diversity trade-offs.

**Core Idea**: The root cause of diversity loss lies not in the target distribution (filtering itself), but in the choice of divergence used to approximate it—replacing Reverse KL with α-divergences enables systematic control over the accuracy–diversity balance.

## Method

### Overall Architecture
The DMVR (Distributional Matching with Verifiable Rewards) framework proceeds as follows: (1) Define the target distribution $p_x(y) \propto \pi_{\text{base}}(y|x) \cdot v(y,x)$ (filtering out errors, preserving the original relative probability of correct answers) → (2) Choose the α-divergence $D_{f_\alpha}(\pi_\theta \| p_x)$ as the optimization objective → (3) Train via the policy gradient of the f-DPG algorithm → (4) The α parameter controls a continuous transition from mode-seeking (α→1, resembling RLVR) to mass-covering (α→0, resembling rejection sampling fine-tuning).

### Key Designs

1. **Explicit Definition of the Target Distribution**:

    - Function: Defines the "ideal" training target—filtering all incorrect responses while leaving the relative probability of correct responses unchanged.
    - Mechanism: $p_x(y) \propto \pi_{\text{base}}(y|x) \cdot v(y,x)$. This is the unique distribution satisfying two conditions: (i) all outputs pass the verifier $v$, and (ii) among all distributions satisfying (i), it minimizes the Forward KL to the base model (the I-projection in information geometry).
    - Design Motivation: Unlike the implicit RLVR target $p_{x,\beta}(y) \propto \pi_{\text{base}}(y|x) \cdot \exp(v(y,x)/\beta)$ (a smoothed approximation), the explicit definition separates "what the target is" from "how to approximate it," allowing the approximation strategy to be chosen independently.

2. **Equivalence Between RLVR and Reverse KL (Theoretical Contribution)**:

    - Function: Proves that the implicit optimization of RLVR is equivalent to minimizing the Reverse KL to $p_{x,\beta}$.
    - Mechanism: **Lemma 1** proves $\nabla_\theta \mathbb{E}_x[KL(\pi_\theta \| p_{x,\beta})] = -\frac{1}{\beta} \nabla_\theta \mathbb{E}_{x,y\sim\pi_\theta}[v(y,x) - \beta \log \frac{\pi_\theta}{\pi_{\text{base}}}]$, i.e., maximizing the RLVR pseudo-reward is equivalent to minimizing the Reverse KL. **Lemma 2** proves $\lim_{\beta\to 0} p_{x,\beta} = p_x$.
    - Design Motivation: Explains why RLVR necessarily leads to diversity loss—Reverse KL is zero-forcing, permitting the model to ignore entire modes of the target distribution.

3. **α-DPG Algorithm**:

    - Function: Parameterizes f-DPG with the α-divergence family to achieve a continuously controllable accuracy–diversity trade-off.
    - Mechanism: The α-DPG pseudo-reward is $\hat{R}_\theta(y,x) = \min\left(\left(\frac{p_x(y)}{\pi_\theta(y|x)}\right)^{1-\alpha} - 1, M\right)$. As α→1, the algorithm recovers REINFORCE (mode-seeking); as α→0, it recovers KL-DPG / rejection sampling fine-tuning (mass-covering); α=0.5 corresponds to the Hellinger distance. A leave-one-out mean is used as a baseline to reduce variance, and a clipping value of M=10 prevents variance explosion at low α.
    - Design Motivation: The unified framework encompasses RLVR (Reverse KL, α≈1), KL-DPG (Forward KL, α=0), and rejection sampling fine-tuning (RS-FT) as special cases. A single hyperparameter α suffices to traverse the entire accuracy–coverage Pareto frontier.

4. **Online Estimation of the Partition Function**:

    - Function: Computes the normalization constant $Z_x$ of the target distribution.
    - Mechanism: $Z_x = \mathbb{P}_{y\sim a(\cdot|x)}[v(y,x)=1]$, i.e., the base model's pass rate. It is estimated online from the current batch, introducing no additional computational overhead. A lower-bound clamp of $\epsilon = 1e^{-4}$ prevents division by zero.
    - Design Motivation: Avoids the cost of additional sampling or model copies.

### Loss & Training
- Pseudo-reward: $\hat{R}_\theta(y,x) = \min\!\left(\!\left(\frac{p_x(y)}{\pi_\theta(y|x)}\right)^{\!1-\alpha} - 1,\, M\right)$
- Gradient: $\nabla_\theta \mathcal{L} = \mathbb{E}_{x,y\sim\pi_\theta}[-\hat{A}^f(y,x) \nabla_\theta \log \pi_\theta(y|x)]$
- Baseline: leave-one-out mean pseudo-reward per context
- Training details: 4×A100, 512 sequences/step, 200 iterations (~3 epochs), maximum response length 1024 tokens, float16

## Key Experimental Results

### Main Results
pass@k results on the Lean theorem proving task (10K training problems, 200 test problems):

| Method | pass@1 | pass@16 | pass@256 | Characteristics |
|--------|--------|---------|----------|-----------------|
| Base-SFT | Low | Medium | Medium-High | Diverse but imprecise |
| GRPO (β=0) | **High** | Medium | Low | Precise but diversity collapse |
| GRPO (High-KL) | Medium-High | Medium-High | Medium-High | KL penalty alleviates collapse |
| Rw-Ulkly | Medium-High | Medium-High | Medium-High | Ranking preference preserves diversity |
| Pass@k Training | Medium | Medium-High | High | Optimizes coverage directly |
| α-DPG (α=0.999) | **High** | **High** | Medium-High | Near-RLVR accuracy + better coverage |
| α-DPG (α=0.25) | Medium | **High** | **Highest** | Best coverage |

### Ablation Study

| α value | Behavior | Accuracy (pass@1) | Coverage (pass@256) |
|---------|----------|-------------------|---------------------|
| α=0.25 | Strong mass-covering | Moderate improvement | Highest, surpasses all methods |
| α=0.5 (Hellinger) | Balanced | Moderate | High |
| α=0.75 | Mild mode-seeking | Relatively high | Medium-High |
| α=0.999 | Near Reverse KL | Highest | Comparable to GRPO |
| Pareto frontier (all α) | All on or near frontier | Continuously controllable | Continuously controllable |

### Key Findings
- **α-DPG models lie almost entirely on the Pareto frontier**: A single hyperparameter α continuously spans the accuracy–coverage trade-off space.
- **α=0.999 generally dominates GRPO and pure RL methods**: Similar accuracy with better coverage.
- **α=0.25 achieves the highest pass@256 among all methods**: Coverage surpasses Base-SFT, pass@k training, and KL regularization.
- **Problem-difficulty shift analysis**: GRPO and α=0.999 render many medium-difficulty problems easy but also make some hard problems entirely unsolvable; α=0.25 and High-KL are more conservative, losing solvability on only 3 problems.
- **Diversity analysis**: Strategy/premise diversity (Shannon index) correlates positively with pass@256 and negatively with pass@1.
- **Perplexity analysis**: Sequences generated by all models have low perplexity under the base model, confirming that RL does not create new capabilities but reweights existing behaviors.

## Highlights & Insights
- **Core insight—"diversity loss lies in the divergence, not the target"**: The problem is reframed from "is RL harmful?" to "which divergence is used to approximate the same target distribution?" This perspective shift is profound: the target distribution (filtering) is entirely reasonable; the problem stems from the mode-seeking nature of Reverse KL.
- **Grand unifying framework**: α-DPG subsumes REINFORCE/GRPO (α≈1), KL-DPG (α=0), and rejection sampling fine-tuning (α=0, offline) under a single framework distinguished only by a scalar α. This theoretical elegance is highly valuable.
- **Pareto controllability**: A single hyperparameter continuously traverses the accuracy–coverage frontier, which is more intuitive and more effective than tuning the KL penalty coefficient β.
- **Reflection on RLVR**: The paper rigorously proves RLVR ≡ Reverse KL to the filtered distribution. Combined with recent findings that "RL does not create but only reranks," this clearly explains why RLVR models underperform the base model under large sampling budgets.

## Limitations & Future Work
- **Validation only on Lean theorem proving**: Generalization to other verifiable tasks such as code generation and mathematical reasoning remains unknown.
- **Only 7B models tested**: Effects on larger models (70B+) are not evaluated.
- **Training instability at low α**: α≤0.5 requires clipping, which introduces bias.
- **Noisy partition function estimation**: Online estimation of $Z_x$ is inaccurate for hard problems with very low pass rates.
- **Promising directions**: Curriculum learning over α (starting low for coverage, ending high for accuracy); extension to non-binary reward settings; integration with search strategies such as MCTS.

## Related Work & Insights
- **vs. GRPO/PPO (RLVR)**: These methods implicitly optimize Reverse KL, necessarily sacrificing diversity. α-DPG at α≈1 recovers their accuracy while retaining more coverage.
- **vs. KL-DPG (Khalifa et al.)**: KL-DPG uses Forward KL, preserving diversity but at the cost of accuracy. α-DPG unifies both extremes.
- **vs. Rw-Ulkly (He et al.)**: Preserves diversity through ranking preference penalties but lacks theoretical grounding. α-DPG is supported by the information-geometric theory of divergence families.
- **vs. Pass@k training**: Directly optimizes pass@k but lacks the theoretical perspective of distributional matching. α-DPG achieves superior coverage.
- This paper offers deep insight into understanding "what post-training RL is actually doing"—when mode collapse is observed, the divergence choice should be examined before the objective function.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unifying and explaining the diversity problem of RLVR through the lens of divergence selection, and proposing α-DPG, constitutes an elegant conceptual contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Experiments on the Lean task are comprehensive (Pareto analysis, difficulty shift, diversity analysis, perplexity), but are limited to a single task and model size.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous; both citations and exposition reflect substantial depth and precision.
- Value: ⭐⭐⭐⭐⭐ Provides a theoretical framework and practical solution for a core problem in the RLVR field; the Pareto controllability of α has substantial practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reasoning Boosts Opinion Alignment in LLMs](reasoning_boosts_opinion_alignment_in_llms.md)
- [\[ICLR 2026\] AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking](abstral_augmenting_llms_reasoning_by_reinforcing_abstract_thinking.md)
- [\[ICLR 2026\] MVR: Multi-view Video Reward Shaping for Reinforcement Learning](mvr_multi-view_video_reward_shaping_for_reinforcement_learning.md)
- [\[ICLR 2026\] How LLMs Learn to Reason: A Complex Network Perspective](how_llms_learn_to_reason_a_complex_network_perspective.md)
- [\[ICLR 2026\] AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization](autoqd_diverse_behaviors.md)

</div>

<!-- RELATED:END -->
