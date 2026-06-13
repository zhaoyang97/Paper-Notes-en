---
title: >-
  [Paper Note] Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning
description: >-
  [NeurIPS 2025][LLM Reasoning][chain-of-thought] This paper proposes Re-FORC, a lightweight adapter that predicts the future expected reward $\psi(t|x,z…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "chain-of-thought"
  - "reasoning efficiency"
  - "reward prediction"
  - "adaptive computation"
  - "test-time compute allocation"
  - "Gittins index"
date: 2026-05-08
content_hash: 32b9c4068cc7a7e9
---

# Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2511.02130](https://arxiv.org/abs/2511.02130)  
**Code**: Available  
**Area**: LLM Reasoning
**Keywords**: chain-of-thought, reasoning efficiency, reward prediction, adaptive computation, test-time compute allocation, Gittins index

## TL;DR
This paper proposes Re-FORC, a lightweight adapter that predicts the future expected reward $\psi(t|x,z,\pi)$ in real time during CoT reasoning. The framework models reasoning compute allocation as a Pandora's box problem, enabling adaptive early stopping (26% compute savings), joint model-and-compute selection (+4% accuracy at equal compute, or −55% compute at equal accuracy), and test-time compute scaling (+11% accuracy). Users can freely adjust the accuracy–efficiency trade-off at inference time via a cost coefficient $\lambda$, without any retraining.

## Background & Motivation

**Background**: Chain-of-thought reasoning in large language models (e.g., DeepSeek-R1, QwQ) significantly improves accuracy on complex tasks such as mathematics by extending the reasoning chain. However, longer chains incur higher computational costs, and not every problem warrants equally lengthy reasoning.

**Limitations of Prior Work**: Existing CoT reasoning lacks an adaptive mechanism for determining when to stop thinking. Simple problems waste large numbers of tokens on unnecessary reasoning, while hard problems may receive insufficient computation. Current methods either require retraining (e.g., RL-based chain compression) or rely on simple heuristics (e.g., fixed-length truncation), and cannot flexibly adjust at inference time.

**Key Challenge**: Optimal allocation of test-time compute is fundamentally a sequential decision problem under uncertainty—given a partial reasoning trace, the question of whether to continue (spending more compute) or stop (accepting the current answer) depends on the expected marginal improvement from further reasoning, which is itself unknown during generation.

**Goal**: Design a lightweight "reasoning forecaster" that evaluates the marginal value of continued reasoning in real time during CoT generation, supporting: (1) when to stop a single reasoning trace; (2) which model and how much compute to use; and (3) how to optimally allocate a test-time compute budget.

**Key Insight**: Drawing on metareasoning and sequential decision theory, the paper analogizes CoT reasoning to the Pandora's box problem and uses the Gittins index to provide a theoretically grounded optimal stopping strategy, while designing a lightweight adapter that can be paired with any frozen LLM.

**Core Idea**: Train an adapter to predict the expected reward of continued reasoning, transforming CoT compute allocation into an optimal stopping/selection problem with theoretical guarantees.

## Method

### Overall Architecture
Re-FORC consists of two components: (1) a **frozen reasoning LLM** $\pi$ (e.g., the Qwen3 family) that generates chain-of-thought tokens $z$; and (2) a **lightweight Forecaster adapter** $\psi$ that reads the partial reasoning trace $(x, z_{1:t})$ generated so far and predicts the expected reward upon completion. The system objective is to maximize net utility $J = \mathbb{E}[R^*] - \lambda \cdot T_{\text{total}}$, where $\lambda$ is a user-specified compute cost coefficient.

### Key Designs

1. **Forecaster Adapter**:

    - **Function**: Given a question $x$ and a generated reasoning prefix $z_{1:t}$, predict the expected reward distribution upon completion of reasoning.
    - **Mechanism**: A self-attention pooling layer and a linear projection layer are appended to the LLM's hidden states to output Beta distribution parameters $(\alpha, \beta)$, modeling the bounded $[0,1]$ reward as $\psi(t|x,z,\pi) \sim \text{Beta}(\alpha, \beta)$.
    - **Design Motivation**: The Beta distribution is naturally suited to probabilistic prediction over a bounded interval and can express a wide range of uncertainty states from uniform to high-confidence. Self-attention pooling extracts a fixed-dimensional representation from variable-length reasoning sequences.

2. **Pandora's Box Optimal Decision Framework**:

    - **Function**: Based on the Forecaster's predictions, provide a theoretically optimal policy for "continue vs. stop" and "which model to select."
    - **Mechanism**: Each (model, compute) pair is treated as a "box" with opening cost $\lambda \cdot t$ and an internal reward given by the forecaster. The Gittins index $\sigma_i = \sup\{s : \mathbb{E}[\max(R_i - s, 0)] \geq \lambda \cdot c_i\}$ yields the optimal policy—select the box with the highest index, and stop when all indices fall below the current best value.
    - **Design Motivation**: The Pandora's box problem admits a known optimal solution (Weitzman 1979), avoiding the need to train a separate policy for each value of $\lambda$. Users can switch between efficiency and accuracy preferences at inference time simply by adjusting $\lambda$.

3. **Monte Carlo Training Procedure**:

    - **Function**: Construct (reasoning prefix, future reward) training pairs via sampling to train the Forecaster.
    - **Mechanism**: For each question $x$, $K$ complete reasoning traces $z^{(k)}$ are sampled. Prefixes are extracted at predefined grid points $T = \{0, 512, \ldots, 8192\}$, and corresponding ground-truth rewards $r^{(k)}$ are computed. The Forecaster is trained with a Beta NLL loss: $\mathcal{L} = -\sum \log p_{\text{Beta}}(r|\alpha, \beta)$.
    - **Design Motivation**: Monte Carlo sampling directly leverages the frozen LLM to generate training data without human annotation. Grid-based prefix extraction enables efficient training with coverage across varying reasoning depths.

### Loss & Training
- Training is based on the DeepScaleR-Preview dataset (40K mathematical reasoning problems), with multiple reasoning traces sampled per problem.
- Qwen3-1.7B/4B/8B models are frozen; only adapter parameters are trained.
- The Forecaster's predictions improve progressively during reasoning, with Pearson correlation increasing monotonically with both reasoning depth and model scale.

## Key Experimental Results

### Early Stopping & Joint Model-Compute Selection

| Strategy | Accuracy | Compute Savings | Notes |
|------|--------|----------|------|
| No early stopping (baseline) | 100% (relative) | 0% | Qwen3-8B full inference |
| Re-FORC early stopping | ≈100% (maintained) | **26%** | Adaptive truncation of low-value reasoning |
| Re-FORC model selection (equal compute) | **+4%** | — | Dynamic routing across multiple models |
| Re-FORC model selection (equal accuracy) | — | **55%** | Small models handle easy problems |

### Test-Time Compute Scaling

| Compute Regime | Re-FORC vs. Baselines | Notes |
|----------|---------------------|------|
| High compute | **+11%** accuracy | Precise allocation to hard problems |
| Low compute | **+7%** accuracy | Efficient truncation of low-yield reasoning |

### Key Findings
- Forecaster prediction quality improves monotonically with reasoning depth: more tokens yield higher Pearson correlation.
- Reasoning traces from larger models are more predictable: reward prediction accuracy follows 8B > 4B > 1.7B.
- Users can seamlessly navigate the efficiency–accuracy curve at inference time via $\lambda$ without retraining any parameters.

## Highlights & Insights
- **Elegant integration of theory and practice**: The Pandora's box/Gittins index formulation casts CoT compute allocation as a sequential decision problem with a known optimal solution, avoiding ad hoc heuristic design.
- **Inference-time adjustable cost coefficient $\lambda$** is the core selling point—train once, then freely control the accuracy–efficiency trade-off at deployment via $\lambda$, greatly enhancing practical usability.
- **Lightweight adapter design** enables plug-and-play compatibility with any frozen LLM without modifying base model weights, making engineering deployment straightforward.
- The 55% compute savings at equal accuracy is highly significant for cost control in large-scale inference serving.

## Limitations & Future Work
- Validation is limited to mathematical reasoning (DeepScaleR); generalization to other CoT settings such as code generation and logical reasoning remains unexplored.
- The Forecaster requires a separately trained adapter for each base model, incurring a non-zero switching cost.
- The grid of truncation points $T = \{0, 512, \ldots, 8192\}$ is relatively coarse; finer-grained or continuous prediction may yield additional gains.
- No direct comparison with chain-of-thought compression methods such as STILL and S1.
- The Beta distribution assumption may be ill-suited for multimodal reward distributions.

## Related Work & Insights
- **vs. CoT compression (STILL/S1, etc.)**: These methods compress CoT via RL or distillation by retraining the model. Re-FORC leaves the model unchanged and makes adaptive decisions at inference time; the two approaches are complementary.
- **vs. Best-of-N sampling**: BoN selects the best output after completing all reasoning traces. Re-FORC dynamically forecasts value during generation, enabling compute savings at the generation stage itself.

## Rating
- Novelty: ⭐⭐⭐⭐ — Introduces metareasoning theory into CoT efficiency optimization; the Pandora's box perspective is novel, though the idea of predicting rewards via an adapter is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three application scenarios validated across multiple model scales, though evaluation is confined to mathematical reasoning.
- Writing Quality: ⭐⭐⭐⭐ — Clear logical flow from theoretical framework to method to experiments; mathematical notation is well-defined.
- Value: ⭐⭐⭐⭐⭐ — Inference-time adjustable compute allocation has significant practical value for LLM serving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Note 7: Value-Guided Search - Efficient Chain-of-Thought Reasoning](polymath_evaluating_mathematical_reasoning_in_multilingual_contexts.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](latent_chain-of-thought_for_visual_reasoning.md)
- [\[NeurIPS 2025\] On Learning Verifiers and Implications to Chain-of-Thought Reasoning](on_learning_verifiers_and_implications_to_chain-of-thought_reasoning.md)
- [\[ACL 2026\] ETR: Entropy Trend Reward for Efficient Chain-of-Thought Reasoning](../../ACL2026/llm_reasoning/etr_entropy_trend_reward_for_efficient_chain-of-thought_reasoning.md)
- [\[NeurIPS 2025\] Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning](mind_the_gap_bridging_thought_leap_for_improved_chain-of-thought_tuning.md)

</div>

<!-- RELATED:END -->
