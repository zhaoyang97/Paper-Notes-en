---
title: >-
  [Paper Note] ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning
description: >-
  [ICML 2026][LLM Reasoning][GRPO] ResRL theoretically decomposes the "negative sample gradient polluting positive sample" phenomenon (Lazy Likelihood Displacement) in RLVR into two components: "logit × representation." It…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "GRPO"
  - "Negative Sample Projection"
  - "SVD Subspace"
  - "Lazy Likelihood Displacement"
  - "Pass@k"
date: 2026-05-08
content_hash: 99d44cd1bd030b91
---

# ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.00380](https://arxiv.org/abs/2605.00380)  
**Code**: https://github.com/1229095296/ResRL.git (available)  
**Area**: Reinforcement Learning / LLM Reasoning / RLVR  
**Keywords**: GRPO, Negative Sample Projection, SVD Subspace, Lazy Likelihood Displacement, Pass@k

## TL;DR
ResRL theoretically decomposes the "negative sample gradient polluting positive sample" phenomenon (Lazy Likelihood Displacement) in RLVR into two components: "logit × representation." It then applies a projection residual at the representation layer using the SVD low-rank subspace of positive samples, assigning each negative token a gradient weight in $[\xi,1]$ based on its "orthogonal component energy"—the more similar the representation to positive samples (smaller residual), the lighter the penalty; only purely erroneous components are heavily penalized. This preserves Pass@1 while maintaining Pass@k diversity. On Qwen3-4B math tasks, Avg@16 improves by 9.4% and Pass@128 by 7.0% over NSR.

## Background & Motivation

**Background**: RLVR (Reinforcement Learning with Verifiable Rewards) has become mainstream for LLM post-training—DeepSeek-R1 uses GRPO to significantly enhance complex reasoning. Its variant NSR (Negative Sample Reinforcement) improves Pass@1 while maintaining diversity (Pass@k) by increasing negative sample gradient weights.

**Limitations of Prior Work**: Both GRPO and NSR penalize all negative sample tokens equally, but "positive and negative sample responses overlap heavily in grammar, partial reasoning steps, and common expressions." When NSR increases suppression of negative samples, these **shared legitimate token distributions** are also suppressed, making it harder to generate key tokens for positive samples—this is the LLD (Lazy Likelihood Displacement) phenomenon: after training, $\ln \pi(y^+|c)$ actually decreases. NSR, with larger negative weights, suffers this side effect more than vanilla GRPO, so while NSR excels at Pass@k, its improvement on Pass@1 (i.e., Avg@1) is limited.

**Key Challenge**: The semantic distributions of positive and negative samples significantly overlap in token representation space, but the gradient direction targets "all tokens in the response" without distinguishing whether "this token is a unique negative error (should be heavily penalized)" or "this token is a legitimate expression shared by both (should be lightly penalized)." Ideally, only the gradient component in negative samples "orthogonal to positive samples" should be penalized.

**Goal**: To truly improve Pass@1 while retaining NSR's Pass@k advantage—specifically, to design a token-level, representation-aware gradient modulation mechanism that restricts negative sample penalties to directions orthogonal to positive sample representations.

**Key Insight**: Starting from a first-order expansion of LLD, the authors rigorously prove that LLD is proportional to the "inner product of positive and negative sample output head gradients" (Eq.2). Leveraging the linear output head $z=Wx$ structure, they show the gradient inner product can be decomposed as $\langle \delta_1, \delta_2 \rangle \cdot \langle x_1, x_2 \rangle$ (Lemma 1)—a logit component and a representation component. The logit component is the "backprop signal" shape known during forward pass, which is costly; **the representation component can be estimated with a single forward pass**, and Transformer representations empirically exhibit anisotropy and approximate low-rank properties, making SVD subspace approximation feasible.

**Core Idea**: Use the orthogonal component energy $e(x)$ of each negative token's hidden representation relative to the "positive sample SVD low-rank subspace" as a proxy for "alignment with positive sample representations." Tokens with low alignment (large orthogonal residual) are heavily penalized, while those with high alignment (within the positive subspace) are lightly penalized—thus protecting shared semantics and only suppressing independent errors.

## Method

### Overall Architecture
ResRL extends GRPO with token-wise advantage reweighting. For a prompt $c$, $G$ trajectories are sampled. The positive sample group $\mathcal{P}$ (advantage $>0$) is weakly anchored with a small constant $\lambda_{\text{pos}} = 0.1$ (to prevent mode collapse); each token in the negative sample group (advantage $\leq 0$) receives a dynamic weight $\omega_{i,t} \in [\xi, 1]$, determined by a three-step process: (1) Take the penultimate hidden state $h_{i,t}$, apply LayerNorm and subtract the positive sample centroid $\mu^+$ to obtain centered representation $x_{i,t}$; (2) Perform truncated SVD on the positive sample subset $\hat{X}^+$ to obtain rank-$k$ principal directions $V_k$, constructing the projector $P_S = V_k V_k^\top$; (3) For each negative token, compute the orthogonal residual energy $\mathcal{R}_{i,t} = \frac{1}{d}\|(I-P_S) x_{i,t}^-\|_2^2$, normalize via group-relative quantiles to $[\xi, 1]$, and use this as the final token-wise weight. The entire process introduces no additional trainable parameters, only modifies the advantage shape.

### Key Designs

1. **Theoretical Framework: LLD and Gradient Decomposition**:

    - Function: Starting from first-order Taylor expansion and the algebraic structure of the linear head, proves "why projection residual is a reasonable proxy."
    - Mechanism: **(a)** Defines the change in positive sample log-likelihood before and after training $\Delta(c) = \ln \pi_{\theta_{\text{fin}}}(y^+|c) - \ln \pi_{\theta_{\text{init}}}(y^+|c)$, first-order approximation yields $\Delta(c) \approx -\eta \sum_{(i,t) \in \mathcal{N}(c)} \langle \nabla_W \ell^+, g^-_{i,t} \rangle$, showing LLD is determined by the "inner product of positive and negative output head gradients" (Eq.2). **(b)** Lemma 1: From $\nabla_W \ell = \delta x^\top$ ($\delta$ is the backprop signal at the logit, $x$ is the representation), $\langle \nabla_W \ell_1, \nabla_W \ell_2 \rangle = \langle \delta_1, \delta_2 \rangle \cdot \langle x_1, x_2 \rangle$—cleanly decomposing the gradient inner product into logit and representation terms. **(c)** Lemma 2 (Alignment bound): For $x^+ \in S$ subspace, $\langle x, x^+ \rangle^2 \leq \|x^+\|^2 (\|x\|^2 - d \cdot e(x))$—increasing the orthogonal component energy $e(x)$ monotonically decreases the upper bound of similarity with any positive sample representation. **(d)** Theorem 1: Combining Lemma 1+2, $|\langle x^-, x^+ \rangle| \leq \|P_S x^+\|_2 \sqrt{\|x^-\|^2 - d\cdot e(x^-)} + \|x^-\|_2 \sqrt{d \cdot e(x^+)}$, under the assumption that the subspace sufficiently covers positive samples ($e(x^+) \leq \varepsilon_+$), $e(x^-)$ serves as a conservative upper bound proxy for the gradient inner product.
    - Design Motivation: This theoretical framework elevates the use of "projection residual" from heuristic to provable "upper bound proxy," requiring only a single forward pass for estimation (unlike direct token-wise full-parameter gradient computation, which needs extra backward and full-parameter communication), greatly improving computational feasibility.

2. **SVD Low-Rank Subspace Construction + Projection Residual Token Weights**:

    - Function: Turns the theoretical $e(x)$ proxy into a lightweight operator that can be computed online in the GRPO training loop.
    - Mechanism: Within each prompt group, **(1)** uniformly sample $M$ tokens from the positive sample pool, apply LayerNorm and subtract the centroid to form matrix $\hat{X}^+ \in \mathbb{R}^{M \times d}$, perform truncated SVD: $\hat{X}^+ = U \Sigma V^\top$, take the top $k$ right singular vectors to construct $V_k \in \mathbb{R}^{d \times k}$ and projector $P_S = V_k V_k^\top$. **(2)** For each negative token, compute $\mathcal{R}_{i,t} = \frac{1}{d}\|(I-P_S) x^-_{i,t}\|^2$. **(3)** Use group-relative quantiles $q_{\text{low}} = \mathcal{Q}(\mathbf{D}, \alpha)$, $q_{\text{high}} = \mathcal{Q}(\mathbf{D}, \beta)$ instead of min/max for robust normalization: $z_{i,t} = \text{clamp}((\mathcal{R}_{i,t} - q_{\text{low}}) / (q_{\text{high}} - q_{\text{low}} + \epsilon), 0, 1)$. **(4)** Map to $[\xi,1]$: $\omega_{i,t} = \xi + (1-\xi) z_{i,t}$. **(5)** Token-wise advantage: $\tilde{A}_{i,t} = \lambda_{\text{pos}} \hat{A}_i$ if $\hat{A}_i > 0$, $\omega_{i,t} \hat{A}_i$ if $\hat{A}_i \leq 0$; substitute into the standard GRPO clipped objective.
    - Design Motivation: Sampling + low-rank SVD (instead of all tokens, full rank) reduces complexity to acceptable levels; quantile normalization replaces min-max to prevent outliers from distorting the overall weight distribution; using $[\xi,1]$ instead of $[0,1]$ ensures a minimum penalty even for perfectly aligned representations (ξ is the lower bound), preventing the model from completely ignoring those errors. The penultimate layer is used for representations instead of the final layer because the final layer directly feeds the output head and is biased by the token prediction objective; the penultimate layer better captures "semantic abstraction."

3. **Positive Sample Weak Anchoring + Length-Scaled Reward**:

    - Function: Prevents mode collapse due to lack of positive sample reinforcement when only negative samples are heavily penalized, and curbs verbosity (a common RL training side effect).
    - Mechanism: For positive advantage tokens, scale by $\lambda_{\text{pos}} = 0.1$—do not completely remove positive sample gradients, but retain a small "weak reward anchor" to prevent the policy from focusing solely on "avoiding errors" and forgetting "achieving correctness." A length-scaled reward mechanism (formula in the appendix) is introduced as a "safety valve"—ensuring ResRL does not produce excessively long chain-of-thoughts in pursuit of diversity.
    - Design Motivation: The authors follow Zhu 2025a (NSR)'s "small positive anchor" approach, as removing positive gradients entirely destabilizes training; length reward as a safeguard is necessary because diversity RL often induces verbose outputs, and length explosion hampers inference speed and quality.

### Loss & Training

$$
\mathcal{L}_{\text{ResRL}}(\theta) = \mathbb{E}_{x, \mathcal{G}}\left[\frac{1}{G}\sum_i \frac{1}{T_i} \sum_t \min(\rho_{i,t} \tilde{A}_{i,t}, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \tilde{A}_{i,t}) \right]
$$
- Experimental setup: Qwen3-1.7B/4B/8B backbone, 4096 max response length, group size $G$ as in GRPO default, SVD rank $k$, sample size $M$, quantile $(\alpha, \beta)$, and $\xi$ are all grid searched in ablation.

## Key Experimental Results

### Main Results

| Method (Qwen3-4B) | AIME24 | AIME25 | AMC23 | MATH500 | Minerva | Olympiad | Avg |
|---|---|---|---|---|---|---|---|
| Backbone | 20.0 | 17.3 | 56.9 | 77.8 | 36.9 | 48.2 | 35.5 |
| GRPO | 37.1 | 27.7 | 87.2 | 79.9 | 31.5 | 55.1 | 53.1 |
| DAPO | 23.5 | 18.9 | 63.4 | 80.8 | 39.1 | 51.2 | 46.2 |
| FlowRL | 35.4 | 30.2 | 74.5 | 84.7 | 38.9 | 58.1 | 53.6 |
| NSR | 38.5 | 33.1 | 79.8 | 77.4 | 33.5 | 50.1 | 52.1 |
| **ResRL** | **45.2** | **38.6** | **89.4** | 77.8 | **38.6** | 52.3 | **57.0** |

| Method (Qwen3-8B) | AIME24 | AIME25 | AMC23 | MATH500 | Minerva | Olympiad | Avg |
|---|---|---|---|---|---|---|---|
| Backbone | 25.4 | 18.1 | 61.4 | 77.6 | 39.2 | 48.6 | 45.1 |
| GRPO | 36.3 | 29.2 | 78.0 | 89.4 | 42.1 | 62.0 | 56.2 |
| FlowRL | 47.7 | 33.3 | 85.8 | 92.1 | 44.6 | 68.5 | 62.1 |
| NSR | 55.4 | 38.5 | 89.8 | 87.3 | 40.0 | 60.6 | 61.9 |
| **ResRL** | 50.8 | **41.1** | 89.7 | **92.7** | **46.0** | 68.1 | **64.7** |

| Code (Qwen3-4B) | LiveCodeBench Avg/Pass@16 | CodeForces Rating (Pct.) | HumanEval+ Pass@16 |
|---|---|---|---|
| Backbone | 30.5 / 40.9 | 578.8 (1.2) | 89.0 |
| GRPO | 39.5 / 55.1 | 1267.9 (63.1) | 95.7 |
| NSR | 32.8 / 52.3 | 1340.9 (69.3) | 96.9 |
| **ResRL** | **43.2 / 59.9** | **1469.5 (78.9)** | **97.0** |

| Agent / Tool Use | ALFWorld All | WebShop Succ. | BFCL Overall |
|---|---|---|---|
| Prompting ReAct | 31.2 | 19.5 | - |
| PPO | 80.4 | 68.7 | - |
| EMPG | 78.5 | 69.3 | - |
| ResT-8B | - | - | strong in several |
| **ResRL** | **86.7** | **71.5** | best in several |

### Ablation Study
The authors report ablations on rank $k$, sample size $M$, hidden layer selection, quantile $(\alpha, \beta)$, and $\xi$ (see Section 5+ for full tables; only core conclusions are summarized here).

| Key Hyperparameter | Conclusion |
|---|---|
| Rank $k$ | Too low ($k=1$) lacks expressiveness, too high ($k \approx d$) degenerates to no projection; $k \approx 8-16$ is optimal |
| Penultimate vs final layer hidden | Penultimate is significantly better, supporting the hypothesis that the final layer is biased by the prediction objective |
| Quantile $(\alpha, \beta)$ | (0.1, 0.9) is more robust than (0, 1) min-max; outliers do not distort the overall weight distribution |
| $\xi$ (minimum weight) | $\xi \approx 0.3-0.5$ is most stable; $\xi=0$ causes some tokens to be completely unsupervised, leading to drift |

### Key Findings
- **Simultaneous improvement in Pass@1 (Avg@16) and Pass@128**: NSR mainly excels at Pass@k with little Pass@1 gain; ResRL improves both—on Qwen3-4B math, Avg@16 +9.4%, Pass@128 +7.0%, directly addressing NSR's weakness.
- **Theory-to-algorithm consistency**: Theorem 1 predicts "the larger $e(x^-)$, the tighter the gradient alignment upper bound, and the heavier the penalty should be," and ablation shows quantile normalization + $\xi$ lower bound are consistent with theory.
- **Task generality**: Achieves SOTA across math (AIME/MATH500), code (LiveCodeBench/CodeForces), long-horizon agent (ALFWorld/WebShop), and function call (BFCL), indicating "projection residual weighting" is a general RL improvement, not task-specific.
- **CodeForces +9.6% rating**: From NSR 1340 → ResRL 1469, percentile from 69% → 78%, a substantial real-world leap—showing "protecting shared semantics" is especially important for structured code generation.
- **Small constant $\lambda_{\text{pos}}=0.1$ is critical**: Removing positive gradients causes immediate collapse; retaining a weak anchor stabilizes training—an engineering insight shared by NSR-family methods.

## Highlights & Insights
- **Lemma 1's output head gradient decomposition**: $\langle \nabla_W \ell_1, \nabla_W \ell_2 \rangle = \langle \delta_1, \delta_2 \rangle \cdot \langle x_1, x_2 \rangle$—a remarkably clean equation, providing the theoretical foundation for "why single forward can proxy token-wise gradient interactions"; future work on "representation-level RL control" can leverage this.
- **Positive sample SVD subspace as 'legitimate semantic manifold'**: Conceptually expresses "what is an acceptable token" as a low-rank subspace; large energy outside the projection indicates "independent errors" to be heavily penalized—this "semantic manifold + orthogonal residual" view is more interpretable than the purely geometric "gradient direction angle" perspective.
- **Group-relative quantile gating**: Each prompt group independently normalizes via quantiles, avoiding scale differences across prompts from contaminating a unified threshold; this "relative, not absolute" design extends the GRPO spirit to the token level.
- **Penultimate vs final layer**: Though seemingly an engineering detail, it reflects "which layer's representation best captures semantics without prediction bias," offering lessons for future representation engineering.
- **Theory + empirical dual loop**: Theorem 1 provides a conservative bound, experiments tune its tightness—this "theory guides, experiment tunes" paradigm is more convincing than pure empirical or pure theoretical work.

## Limitations & Future Work
- SVD must be performed for each prompt group; even with sampling and low-rank, there is computational overhead. For very large group sizes or long sequences, costs may rise significantly; the authors mitigate this with sampling but do not provide detailed wall-clock comparisons.
- Subspace rank $k$, sample size $M$, quantile $(\alpha, \beta)$, and $\xi$ all require grid search, increasing hyperparameter tuning burden in deployment.
- The "semantic" assumption of the penultimate hidden state is empirical, not systematically validated across various LLM architectures (Mistral, Llama, GLM).
- Length-scaled reward is a safeguard, but the authors do not provide comparisons showing "what happens without length penalty"—ResRL itself may have a verbosity tendency.
- Only validated under RLVR (binary reward); applicability to dense reward or preference learning (DPO/RLHF) is not discussed.
- No comparison with ConsisCo, RPO, and other RL algorithms focusing on representation geometry; comparison with PRM-based methods is limited to simple baselines.

## Related Work & Insights
- **vs GRPO (DeepSeek-R1) / DAPO**: Their token-wise advantage is determined by group normalization, without distinguishing shared vs independent tokens; ResRL adds representation-aware weights to negative tokens for higher precision.
- **vs NSR (Zhu 2025a)**: NSR increases negative sample gradients but does not address LLD, so Pass@1 improvement is limited; ResRL uses projection residuals for selective suppression, retaining NSR's diversity advantage while improving Pass@1.
- **vs FlowRL (Zhu 2025b)**: FlowRL controls policy distribution via flow matching, while ResRL uses representation geometry—both target the "diversity vs precision" tradeoff; on 8B, ResRL's overall performance (64.7) surpasses FlowRL (62.1).
- **vs LLD studies (Deng 2025c, 2025b)**: They identify the LLD phenomenon; ResRL is the first to proxy, differentiate, and integrate the "gradient inner product" of LLD into the GRPO training loop—moving from diagnosis to remedy.
- **vs Token-level loss balancing / Curriculum**: These approaches use heuristic-based weights (loss magnitude, length, etc.); ResRL's weights are based on theoretical upper bounds, offering more principled guidance.
- **vs Representation engineering (Repeng, Steering Vectors)**: Those works modify activations at inference; ResRL modifies gradients during training—both leverage representation geometry but in opposite directions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Decomposes "gradient inner product = logit × representation" + SVD low-rank projection + group-relative gating into a principled framework, balancing theory and engineering
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 benchmarks + 4 task types + 3 model scales + multiple ablations; Qwen3-1.7B/4B/8B all win
- Writing Quality: ⭐⭐⭐⭐ Clear Lemma → Theorem → Algorithm chain, rigorous formulas; some derivations in appendix, main text is dense
- Value: ⭐⭐⭐⭐⭐ Provides an actionable improvement for the RLVR community (open-source code), and a theoretical template for future representation-aware RL work

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](../../ICLR2026/llm_reasoning/temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)
- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)
- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](../../ACL2026/llm_reasoning/revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)
- [\[NeurIPS 2025\] The Impact of Quantization on Large Reasoning Model Reinforcement Learning](../../NeurIPS2025/llm_reasoning/the_impact_of_quantization_on_large_reasoning_model_reinforcement_learning.md)
- [\[AAAI 2026\] SERL: Self-Examining Reinforcement Learning on Open-Domain](../../AAAI2026/llm_reasoning/serl_self-examining_reinforcement_learning_on_open-domain.md)

</div>

<!-- RELATED:END -->
