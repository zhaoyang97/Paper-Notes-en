---
title: >-
  [Paper Note] ∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space
description: >-
  [ICLR 2026][LLM Reasoning][test-time scaling] This paper proposes ∇-Reasoner, which upgrades inference-time search from zeroth-order (sampling + evaluation) to first-order (gradient descent). By applying Differentiable T…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "test-time scaling"
  - "gradient-based optimization"
  - "differentiable optimization"
  - "reward model"
  - "inference-time reasoning"
date: 2026-05-08
content_hash: c34ce170bb394aef
---

# ∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space

**Conference**: ICLR 2026  
**arXiv**: [2603.04948](https://arxiv.org/abs/2603.04948)  
**Code**: [https://github.com/VITA-Group/Nabla-Reasoner](https://github.com/VITA-Group/Nabla-Reasoner)  
**Area**: Optimization  
**Keywords**: test-time scaling, gradient-based optimization, differentiable optimization, reward model, inference-time reasoning

## TL;DR
This paper proposes ∇-Reasoner, which upgrades inference-time search from zeroth-order (sampling + evaluation) to first-order (gradient descent). By applying Differentiable Textual Optimization (DTO) in the token logits space—jointly leveraging reward gradients and LLM likelihood—the framework iteratively refines the decoding strategy, achieving 10–40% accuracy gains on mathematical reasoning tasks while reducing model calls by 10–40%.

## Background & Motivation
**Background**: Inference-time compute scaling has emerged as a critical approach to enhancing LLM reasoning. Existing methods such as Best-of-N, Self-Consistency, Tree-of-Thought, and RAP improve answer quality through repeated sampling and evaluation.

**Limitations of Prior Work**: These methods are fundamentally **zeroth-order searches**—they exploit only the scalar reward value to filter candidates, entirely ignoring gradient direction information. As the search space grows exponentially with sequence length, undirected search becomes inefficient, and performance saturates as the computational budget increases.

**Key Challenge**: Reward models are inherently differentiable (transformer-based classifiers), making gradient information readily available yet completely unused. Zeroth-order methods fail to exploit the structural information embedded in the reward landscape.

**Goal**: How can reward gradients be leveraged at inference time to efficiently guide LLM outputs toward high-reward regions while preserving generation fluency?

**Key Insight**: LLM reasoning is reformulated as a continuous optimization problem—performing gradient descent in the token logits space, with a straight-through estimator bridging the discrete and continuous domains.

**Core Idea**: Replace zeroth-order search with first-order gradient descent for test-time policy optimization, jointly maximizing reward and LLM likelihood in the logits space.

## Method

### Overall Architecture
∇-Reasoner is an iterative decoding framework. Given a prompt, the LLM first generates a complete rollout along with its logits. DTO then optimizes these logits, from which the first token is resampled. Rejection sampling determines whether the new token is accepted, after which generation proceeds to the next token. This process advances token by token, with each token potentially undergoing gradient-based optimization.

### Key Designs

1. **Differentiable Textual Optimization (DTO)**:

    - **Function**: Performs gradient descent in the token logits space, jointly optimizing reward and LLM likelihood.
    - **Mechanism**: The optimization objective is $\mathcal{L}(\mathbf{y}) = -\lambda r(\mathbf{y}|\mathbf{x}) - \log \pi_{LLM}(\mathbf{y}|\mathbf{x})$, where the reward term provides directional guidance and the NLL term prevents deviation from the LLM distribution (mitigating reward hacking). Discrete tokens are parameterized as continuous logits via a Gumbel-softmax straight-through estimator, enabling gradient flow.
    - **Design Motivation**: Gradients propagate **bidirectionally**—prefix tokens are regularized via NLL to maintain consistency with subsequent tokens, while later tokens back-propagate reward signals to earlier ones through attention, achieving a global look-ahead optimization effect.

2. **Iterative Decoding + Rejection Sampling**:

    - **Function**: Embeds DTO into the token-by-token decoding loop, accepting only token modifications that improve the reward.
    - **Mechanism**: After DTO optimization, the first token $\tilde{y}_1$ is resampled from $\text{softmax}(\tilde{\mathbf{z}}_1/\tau)$. If $\tilde{y}_1 \neq y_1$, a new continuation is generated and its reward is compared: the new token is accepted only if the new continuation achieves a higher reward.
    - **Design Motivation**: Rejection sampling ensures that every modification is beneficial, preventing noisy gradient updates from degrading generation quality. Experiments show that DTO reduces the rejection rate from ~66% to ~29–40%.

3. **Acceleration Strategies (three components)**:

    - **Gradient Caching**: Since one-hot tokens change infrequently during optimization, $\partial\mathcal{L}/\partial\mathbf{y}$ is cached and reused, recomputed only when a token flip occurs.
    - **Rollout Reuse**: When a token modification is rejected, its rollout trajectory is directly reused as the rollout for the next step.
    - **Confidence- and Gradient-Guided Token Selection**: DTO is applied only to high-entropy, high-gradient tokens, skipping tokens with high confidence or low gradient magnitude.

### Loss & Training
No training is required (purely an inference-time method). The DTO optimization objective is: $\mathcal{L} = -\log \pi_{LLM}(\mathbf{y}|\mathbf{x}) - \lambda \cdot r(\mathbf{y}|\mathbf{x})$, where $\lambda$ balances the reward term and the NLL regularization. The paper theoretically demonstrates that DTO's sample-space gradient descent is equivalent to PPO's Wasserstein gradient flow (Theorem 4.1), unifying the theoretical frameworks of pretraining scaling and inference-time scaling.

## Key Experimental Results

### Main Results

| Model + Benchmark | Greedy | SC (N=8) | BoN (N=8) | RAP | GRPO | ∇-Reasoner |
|---|---|---|---|---|---|---|
| Qwen-2.5-7B MATH-500 | 43.8 | 69.8 | 70.2 | 68.6 | 70.8 | **71.0** |
| Qwen-2.5-7B AMC | 33.0 | 49.4 | 50.1 | 50.1 | 52.8 | 51.5 |
| Qwen-2.5-7B-Inst MATH-500 | 71.2 | 76.6 | 77.8 | 80.2 | - | **80.4** |
| Qwen-2.5-7B-Inst AMC | 43.0 | 55.5 | 55.9 | 54.6 | - | **56.8** |
| Qwen-2.5-7B-Inst AIME24 | 5.3 | 25.0 | 22.5 | 1.6 | - | **26.6** |
| Llama-3.1-8B-Inst MATH-500 | 40.6 | 54.8 | 52.2 | 55.4 | - | **55.8** |

### Ablation Study

| Configuration | Result | Note |
|------|------|------|
| DTO rejection rate (Qwen-Inst) | 28.9% | Compared to 66.5% for the no-DTO baseline—substantial reduction |
| DTO rejection rate (Llama-Inst) | 40.1% | Compared to baseline 66.9% |
| Reward model 4B vs. 8B (MATH-500) | 80.4 vs. 80.8 | Larger reward model yields only +0.4% improvement |
| Model call count | Reduced by 10–40% | Compared to BoN/SC |

### Key Findings
- DTO reduces the rejection rate from the theoretical baseline of 66% to approximately 30%, confirming that gradient optimization effectively improves the policy at each step.
- Computational efficiency advantage: the parallel execution of transformers makes gradient computation comparable in cost to a single forward pass; confidence/gradient-guided selection skips a large fraction of tokens that do not require optimization.
- Performance is insensitive to reward model scale (gap between 4B and 8B is <1%).
- On test-time scaling curves, ∇-Reasoner's Pareto frontier consistently dominates both BoN and SC.

## Highlights & Insights
- **Paradigm shift from zeroth-order to first-order**: A fundamental advancement in test-time scaling, providing the first demonstration that first-order gradients are applicable and more efficient at inference time.
- **Theoretical elegance**: The equivalence between DTO's sample-space gradient descent and PPO's Wasserstein gradient flow is formally proven, unifying pretraining scaling (parameter-space optimization) and inference-time scaling (sample-space optimization).
- **Transferable gradient caching trick**: The observation that one-hot tokens change infrequently—a consequence of softmax sharpening—can be generalized to other settings that require gradient-based optimization over discrete structures.

## Limitations & Future Work
- Performance is bounded by the joint capability ceiling of the base model and the reward model; the method cannot surpass this combined bottleneck.
- The base model and reward model must share the same vocabulary to enable end-to-end logit optimization, restricting flexibility in model composition.
- Validation is currently limited to mathematical reasoning tasks; performance on code generation, open-domain QA, and other settings remains unexplored.
- Integration with serving engines (e.g., vLLM) requires additional engineering effort to insert backpropagation into the decoding loop.

## Related Work & Insights
- **vs. Best-of-N / SC**: These methods rely on pure zeroth-order sampling and filtering; ∇-Reasoner directly optimizes via first-order gradients, achieving superior results with fewer samples.
- **vs. ToT / RAP**: While these methods also guide search, they depend on heuristic tree search and Q-value estimation; ∇-Reasoner employs differentiable optimization for a more direct and efficient solution.
- **vs. GRPO (training-time method)**: ∇-Reasoner achieves performance comparable to GRPO without modifying model weights, and the mathematical equivalence between the two is formally established.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Paradigm shift from zeroth-order to first-order; both theory and method are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage of mathematical reasoning, but limited task diversity.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, figures are intuitive, and the narrative is well-structured.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for test-time scaling and is poised to become an important baseline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](on_the_design_of_kl-regularized_policy_gradient_algorithms_for_llm_reasoning.md)
- [\[ICML 2026\] Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning](../../ICML2026/llm_reasoning/beyond_test-time_memory_state-space_optimal_control_for_llm_reasoning.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](../../ACL2026/llm_reasoning/parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[ICLR 2026\] ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](atts_asynchronous_test-time_scaling_via_conformal_prediction.md)
- [\[ICLR 2026\] Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs](plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)

</div>

<!-- RELATED:END -->
