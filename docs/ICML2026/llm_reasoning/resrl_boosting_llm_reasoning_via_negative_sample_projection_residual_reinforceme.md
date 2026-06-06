---
title: >-
  [Paper Note] ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning
description: >-
  [ICML 2026][LLM Reasoning][GRPO] ResRL theoretically decomposes the "negative sample gradient contamination of positive samples" phenomenon (Lazy Likelihood Displacement…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "GRPO"
  - "Negative Sample Projection"
  - "SVD Subspace"
  - "Lazy Likelihood Displacement"
  - "Pass@k"
date: 2026-05-08
content_hash: 5b5982d393c15a08
---

# ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.00380](https://arxiv.org/abs/2605.00380)  
**Code**: https://github.com/1229095296/ResRL.git (Available)  
**Area**: Reinforcement Learning / LLM Reasoning / RLVR  
**Keywords**: GRPO, Negative Sample Projection, SVD Subspace, Lazy Likelihood Displacement, Pass@k

## TL;DR
ResRL theoretically decomposes the "negative sample gradient contamination of positive samples" phenomenon (Lazy Likelihood Displacement, LLD) in RLVR into two components: "logit $\times$ representation." It then uses the low-rank SVD subspace of positive samples to calculate projection residuals in the representation layer. Each negative token is assigned a gradient weight in the $[\xi, 1]$ interval based on its "orthogonal component energy"—representations more similar to positive samples (smaller residuals) receive lighter penalties, while purely incorrect components are heavily penalized. This preserves Pass@1 without sacrificing Pass@k diversity. On Qwen3-4B mathematics tasks, it achieves a 9.4% improvement in Avg@16 and a 7.0% improvement in Pass@128 compared to NSR.

## Background & Motivation

**Background**: Reinforcement Learning with Verifiable Rewards (RLVR) has become a mainstream LLM post-training approach—DeepSeek-R1 utilized GRPO to significantly enhance complex reasoning. Its variant, NSR (Negative Sample Reinforcement), improves Pass@1 while maintaining diversity (Pass@k) by increasing the gradient weight of negative samples.

**Limitations of Prior Work**: Both GRPO and NSR penalize negative sample tokens indiscriminately. However, positive and negative responses highly overlap in grammar, certain reasoning steps, and common expressions. When NSR intensifies the suppression of negative samples, these **shared legitimate token distributions** are also suppressed. This leads to the LLD (Lazy Likelihood Displacement) phenomenon: $\ln \pi(y^+|c)$ actually decreases after training. Due to the larger negative weights in NSR, this side effect is even more severe than in vanilla GRPO, resulting in limited Pass@1 (Avg@1) gains despite strong Pass@k performance.

**Key Challenge**: The semantic distributions of positive and negative samples significantly overlap in the token representation space, but the gradient direction suppresses all tokens in a negative response. There is no mechanism to distinguish whether a token is a "negative-sample-specific error pattern (should be heavily penalized)" or a "shared legitimate expression (should be lightly penalized)." Ideally, only the gradient components of negative samples that are "orthogonal to positive samples" should be penalized.

**Goal**: To bridge the gap in Pass@1 while maintaining the Pass@k advantages of NSR. Specifically, the goal is to design a token-level, representation-aware gradient modulation mechanism that limits negative sample penalties to directions orthogonal to positive sample representations.

**Key Insight**: Starting from the first-order expansion of LLD, the authors rigorously prove that LLD is proportional to the "inner product of the output head gradients of positive and negative samples" (Eq.2). Utilizing the structure of the linear output head $z=Wx$, they prove that the gradient inner product can be decomposed into $\langle \delta_1, \delta_2 \rangle \cdot \langle x_1, x_2 \rangle$ (Lemma 1)—a logit component and a representation component. While the logit component (the backprop signal) is computationally expensive to determine during the forward pass, the **representation component can be estimated via a single forward pass**. Furthermore, since Transformer representations empirically exhibit anisotropy and approximate low-rank properties, they can be approximated using an SVD subspace.

**Core Idea**: Use the orthogonal component energy $e(x)$ of each negative token's hidden representation relative to the "positive sample low-rank SVD subspace" as a proxy for its alignment with positive samples. Lower alignment (larger orthogonal residual) leads to heavy penalties, while higher alignment (falling within the positive sample subspace) leads to lighter penalties. This protects shared semantics and only suppresses independent errors.

## Method

### Overall Architecture
ResRL is a token-wise advantage reweighting extension of GRPO. For a prompt $c$, $G$ trajectories are sampled. Positive samples $\mathcal{P}$ (advantage $>0$) are consistently anchored with a small constant $\lambda_{\text{pos}} = 0.1$ to prevent mode collapse. Each token in the negative sample group (advantage $\leq 0$) receives a dynamic weight $\omega_{i,t} \in [\xi, 1]$ generated through a three-step process: (1) Extract the penultimate hidden state $h_{i,t}$, perform LayerNorm, and subtract the positive sample centroid $\mu^+$ to obtain the centered representation $x_{i,t}$; (2) Perform truncated SVD on the positive sample subset $\hat{X}^+$ to obtain rank-$k$ principal directions $V_k$ and construct the projector $P_S = V_k V_k^\top$; (3) Calculate the orthogonal residual energy $\mathcal{R}_{i,t} = \frac{1}{d}\|(I-P_S) x_{i,t}^-\|_2^2$ for each negative token, map it to $[\xi, 1]$ via group-relative quantile normalization, and use it as the final token-wise weight. This process requires no additional trainable parameters and only modifies the advantage shape.

### Key Designs

1.  **Theoretical Framework: LLD and Gradient Decomposition**:
    *   **Function**: Proves why the projection residual is a reasonable proxy based on first-order Taylor expansion and the algebraic structure of the linear head.
    *   **Mechanism**: **(a)** Define the change in positive sample log-likelihood before and after training as $\Delta(c) = \ln \pi_{\theta_{\text{fin}}}(y^+|c) - \ln \pi_{\theta_{\text{init}}}(y^+|c)$. First-order approximation yields $\Delta(c) \approx -\eta \sum_{(i,t) \in \mathcal{N}(c)} \langle \nabla_W \ell^+, g^-_{i,t} \rangle$, indicating that LLD is determined by the "inner product of positive and negative output head gradients" (Eq.2). **(b)** Lemma 1: From $\nabla_W \ell = \delta x^\top$ (where $\delta$ is the logit backprop signal and $x$ is the representation), it follows that $\langle \nabla_W \ell_1, \nabla_W \ell_2 \rangle = \langle \delta_1, \delta_2 \rangle \cdot \langle x_1, x_2 \rangle$—the gradient inner product decomposes cleanly into logit and representation terms. **(c)** Lemma 2 (Alignment bound): For $x^+ \in S$ subspace, $\langle x, x^+ \rangle^2 \leq \|x^+\|^2 (\|x\|^2 - d \cdot e(x))$—increasing the orthogonal component energy $e(x)$ monotonically decreases the upper bound of similarity with any positive sample representation. **(d)** Theorem 1: Combining Lemma 1 and 2, the inner product is bounded by $e(x^-)$. Under the assumption that the subspace sufficiently covers positive samples ($e(x^+) \leq \varepsilon_+$), $e(x^-)$ becomes a conservative upper bound proxy for the gradient inner product.
    *   **Design Motivation**: This theory transforms the "why use projection residuals" question from a heuristic into a provable "upper bound proxy" that only requires single forward pass estimation (unlike direct token-wise full-parameter gradient calculation which requires extra backward passes and communication), significantly improving computational feasibility.

2.  **SVD Low-rank Subspace Construction + Projection Residual Token Weighting**:
    *   **Function**: Converts the theoretical $e(x)$ proxy into a lightweight operator that can be calculated online during the GRPO training loop.
    *   **Mechanism**: Within each prompt group, **(1)** $M$ tokens are uniformly sampled from the positive sample pool. After LayerNorm and centroid subtraction, a matrix $\hat{X}^+ \in \mathbb{R}^{M \times d}$ is formed. Truncated SVD is performed: $\hat{X}^+ = U \Sigma V^\top$, and the top $k$ right singular vectors are used to construct $V_k \in \mathbb{R}^{d \times k}$ and the projector $P_S = V_k V_k^\top$. **(2)** $\mathcal{R}_{i,t} = \frac{1}{d}\|(I-P_S) x^-_{i,t}\|^2$ is calculated for each negative token. **(3)** Robust normalization is performed using group-relative quantiles $q_{\text{low}} = \mathcal{Q}(\mathbf{D}, \alpha)$ and $q_{\text{high}} = \mathcal{Q}(\mathbf{D}, \beta)$ instead of min/max: $z_{i,t} = \text{clamp}((\mathcal{R}_{i,t} - q_{\text{low}}) / (q_{\text{high}} - q_{\text{low}} + \epsilon), 0, 1)$. **(4)** Mapping to [$\xi$, 1]: $\omega_{i,t} = \xi + (1-\xi) z_{i,t}$. **(5)** Token-wise advantage: $\tilde{A}_{i,t} = \lambda_{\text{pos}} \hat{A}_i$ if $\hat{A}_i > 0$, and $\omega_{i,t} \hat{A}_i$ if $\hat{A}_i \leq 0$. This is then substituted into the standard GRPO clipped objective.
    *   **Design Motivation**: Sampling and low-rank SVD are used to reduce complexity to an acceptable level. Quantile normalization is used to prevent outliers from distorting the overall weight distribution. The [$\xi$, 1] interval ensures a minimum penalty even for fully aligned representations, preventing the model from ignoring those errors entirely. The penultimate layer is chosen because the final layer directly feeds into the output head and is biased by the token prediction objective; the penultimate layer better reflects "semantic abstraction."

3.  **Positive Sample Weak Anchoring + Length-scaled Reward**:
    *   **Function**: Prevents mode collapse from a lack of positive reinforcement while curbing verbosity (a common side effect of RL training).
    *   **Mechanism**: Positive advantage tokens are scaled by $\lambda_{\text{pos}} = 0.1$. This retains a small "weak reward anchor" to prevent the policy from fully shifting towards "avoiding errors" while forgetting how to "reach correctness." Additionally, a length-scaled reward mechanism (formula in the appendix) is introduced as an "anti-inflation valve" to ensure ResRL does not produce excessively long chains-of-thought.
    *   **Design Motivation**: The authors follow the "small positive anchor" approach from Zhu 2025a (NSR) because removing the positive gradient entirely destabilizes training. Length rewards serve as a safeguard because diversity-focused RL often induces verbose generation, which hurdles inference speed and effectiveness.

### Loss & Training
$$
\mathcal{L}_{\text{ResRL}}(\theta) = \mathbb{E}_{x, \mathcal{G}}\left[\frac{1}{G}\sum_i \frac{1}{T_i} \sum_t \min(\rho_{i,t} \tilde{A}_{i,t}, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \tilde{A}_{i,t}) \right]
$$
- Experimental Setup: Qwen3-1.7B/4B/8B backbones, 4096 max response length, group size $G$ as per standard GRPO. SVD rank $k$, sample count $M$, quantiles $(\alpha, \beta)$, and $\xi$ were determined via grid search in ablation studies.

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
| ResT-8B | - | - | Multiple Strong |
| **ResRL** | **86.7** | **71.5** | Multiple Best |

### Ablation Study
The authors reported several sets of ablations for rank $k$, sample count $M$, hidden layer choice, quantiles $(\alpha, \beta)$, and $\xi$.

| Key Hyperparameter | Conclusion |
|---|---|
| Rank $k$ | Too low ($k=1$) lacks representation; too high ($k \approx d$) is equivalent to no projection. $k \approx 8-16$ is optimal. |
| Penultimate vs final layer | Penultimate is significantly better, validating the hypothesis regarding prediction objective bias. |
| Quantile $(\alpha, \beta)$ | (0.1, 0.9) is more robust than (0, 1) min-max normalization. |
| $\xi$ (Min Weight) | $\xi \approx 0.3-0.5$ is most stable. $\xi=0$ can cause drift by leaving some tokens unsupervised. |

### Key Findings
- **Simultaneous improvement in Pass@1 (Avg@16) and Pass@128**: While NSR mainly excels in Pass@k, ResRL refreshes both dimensions—achieving +9.4% Avg@16 and +7.0% Pass@128 on Qwen3-4B mathematics.
- **Consistency between theory and algorithm**: Theorem 1 predicts that larger $e(x^-)$ leads to a tighter gradient alignment bound and should be penalized more heavily; ablation results for quantile normalization and the $\xi$ lower bound align with this theory.
- **Cross-task universality**: SOTA results achieved across math (AIME/MATH500), code (LiveCodeBench/CodeForces), long-horizon agents (ALFWorld/WebShop), and function calling (BFCL).
- **CodeForces +9.6% rating**: Jumping from NSR's 1340 to ResRL's 1469 (Pct. 69% to 78%) represents a massive leap in practical skill, suggesting that "protecting shared semantics" is crucial for structured code generation.
- **Small constant $\lambda_{\text{pos}}=0.1$ is critical**: Completely removing positive gradients causes immediate collapse; retaining a weak anchor ensures stability.

## Highlights & Insights
- **Lemma 1 Gradient Decomposition**: $\langle \nabla_W \ell_1, \nabla_W \ell_2 \rangle = \langle \delta_1, \delta_2 \rangle \cdot \langle x_1, x_2 \rangle$. This clean equation provides the theoretical foundation for using a single forward pass to proxy token-wise gradient interactions.
- **Positive Sample SVD Subspace as a "Legitimate Semantic Manifold"**: This conceptually represents "acceptable tokens" as a low-rank subspace. Large energy outside this projection represents "independent errors."
- **Group-relative quantile gating**: Quantile normalization performed independently per prompt group avoids scale differences across prompts, pushing GRPO's "relative vs absolute" spirit to the token level.
- **Penultimate vs Final Layer**: This detail reflects which representation layer best represents semantics without prediction bias interference.
- **Theory-Empirical Closed Loop**: Theorem 1 provides a conservative bound, and experiments calibrate its tightness.

## Limitations & Future Work
- SVD must be performed for each prompt group; even with sampling, this adds computational overhead that may scale poorly with large group sizes or long sequences.
- Multiple hyperparameters ($k, M, \alpha, \beta, \xi$) require grid searching, increasing the tuning burden for practical deployment.
- The "semantic" assumption of penultimate hidden states is empirical and not systematically validated across diverse LLM architectures (Mistral, Llama, GLM).
- Comparison with representation-geometric RL algorithms (ConsisCo, RPO) is missing, and PRM-based comparisons are limited to simple baselines.
- Only validated in RLVR (binary reward) settings; applicability to dense rewards or preference learning (DPO/RLHF) remains undiscussed.

## Related Work & Insights
- **vs GRPO (DeepSeek-R1) / DAPO**: These methods calculate token-wise advantages via group normalization without distinguishing shared vs. independent tokens.
- **vs NSR (Zhu 2025a)**: NSR increases negative gradients but fails to solve LLD, limiting Pass@1 gains. ResRL uses projection residuals to selectively suppress, resolving NSR’s pain point.
- **vs FlowRL (Zhu 2025b)**: FlowRL controls policy distribution via flow matching. ResRL uses representation geometry. Both target the diversity vs. precision tradeoff, but ResRL shows higher overall performance (64.7 vs 62.1 on 8B).
- **vs LLD Research (Deng 2025c, 2025b)**: While previous work identified LLD, ResRL is the first to proxy, differentiate, and integrate the "gradient inner product" into the GRPO training loop.
- **vs Representation Engineering**: While those works modify activations during inference, ResRL modifies gradients during training.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ principled framework for logit $\times$ representation decomposition.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 benchmarks across 4 task types and 3 model scales.
- Writing Quality: ⭐⭐⭐⭐ Clear chain of logic; however, some derivations are dense or moved to the appendix.
- Value: ⭐⭐⭐⭐⭐ Provides an actionable RLVR improvement and a theoretical template for representation-aware RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning](../../ACL2026/llm_reasoning/templaterl_structured_template-guided_reinforcement_learning_for_llm_reasoning.md)
- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](../../ICLR2026/llm_reasoning/temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)
- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)
- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](../../ACL2026/llm_reasoning/revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)
- [\[ICML 2026\] Deliberate Evolution: Agentic Reasoning for Sample-Efficient Symbolic Regression with LLMs](deliberate_evolution_agentic_reasoning_for_sample-efficient_symbolic_regression_.md)

</div>

<!-- RELATED:END -->
