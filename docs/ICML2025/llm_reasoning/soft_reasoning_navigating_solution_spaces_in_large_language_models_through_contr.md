---
title: >-
  [Paper Note] Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration
description: >-
  [ICML 2025 Spotlight][Reasoning][Embedding perturbation] This paper introduces Soft Reasoning, which injects Gaussian perturbations into the embedding space of the first generated token and utilizes Bayesian optimization to search for the optimal perturbation vector. This guides LLMs to explore better solution spaces during inference in a black-box manner without requiring access to model parameters or external verifiers. It outperforms baselines like temperature scaling and…
tags:
  - "ICML 2025 Spotlight"
  - "Reasoning"
  - "Embedding perturbation"
  - "Bayesian optimization"
  - "decoding strategy"
  - "reasoning diversity"
  - "test-time computation"
date: 2026-05-08
content_hash: a6620d0f6674f397
---

# Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2505.24688](https://arxiv.org/abs/2505.24688)  
**Code**: [alickzhu/Soft-Reasoning](https://github.com/alickzhu/Soft-Reasoning)  
**Area**: LLM Reasoning  
**Keywords**: Embedding perturbation, Bayesian optimization, decoding strategy, reasoning diversity, test-time computation

## TL;DR

This paper introduces Soft Reasoning, which injects Gaussian perturbations into the embedding space of the first generated token and utilizes Bayesian optimization to search for the optimal perturbation vector. This guides LLMs to explore better solution spaces during inference in a black-box manner without requiring access to model parameters or external verifiers. It outperforms baselines like temperature scaling and Best-of-N on mathematical reasoning tasks with extremely low computational overhead.

## Background & Motivation

### Background

While LLMs excel at simple reasoning tasks, they still face significant challenges in complex tasks such as multi-step mathematical reasoning. Existing mainstream strategies to improve reasoning quality can be categorized into two paradigms:

**Diverse Sampling Methods**: These introduce generation diversity via temperature scaling, top-k, and nucleus sampling, aiming to hit the correct solution among multiple candidate answers.

**Planning and Search Methods**: Such as Chain-of-Thought (CoT), Tree of Thoughts (ToT), and MCTS, which explore reasoning paths through verbal instructions or tree-structure search.

### Limitations of Prior Work

- **Limitations of Temperature Scaling**: Raising the temperature parameter flattens the entire token distribution, indiscriminately boosting the sampling probability of all low-probability tokens. This introduces significant noise rather than meaningful exploration, leading to degraded generation quality without guaranteed coverage of correct answers.
- **Inefficiency of Heuristic Search**: Methods like ToT and MCTS rely on prompt-level heuristic strategies that do not directly act on the model's internal representations. This results in low search efficiency and a high dependency on prompt variations, often leading to aimless random searches.
- **Prohibitive Computational Overhead**: The Best-of-N approach requires heavy sampling (e.g., $N=64$), where computational costs scale linearly with $N$. Methods like ToT and TSE involve multi-round tree search and backtracking, which also incur substantial overhead.

### Key Challenge

How to efficiently explore the solution space of LLMs while preserving generation quality and coherence? Existing methods either sacrifice quality for diversity (temperature sampling) or sacrifice efficiency for coverage (large-scale sampling/search).

### Key Insight

Rather than introducing indiscriminate perturbations at the token probability level (e.g., temperature scaling), it is superior to perform controlled perturbations directly in the embedding space and leverage Bayesian optimization to guide the search direction. The key insight is that **the embedding of the first token exerts a decisive influence on the entire subsequent reasoning chain**. Consequently, optimizing only the first token is sufficient to control the overall generation trajectory.

## Method

### Overall Architecture

The core mechanism of Soft Reasoning is to formulate the LLM reasoning problem as an optimization problem in the embedding space:

1. **Embedding Perturbation**: Inject a Gaussian noise vector $\mathbf{z}$ into the embedding of the first answer token, yielding the perturbed embedding $\mathbf{e}_1 + \mathbf{z}$
2. **Deterministic Decoding**: Given the perturbed embedding, all subsequent tokens are generated using greedy decoding, ensuring that each perturbation vector $\mathbf{z}$ uniquely maps to a single generated sequence
3. **Reward Evaluation**: Assess the correctness and coherence of the generated sequence using a verifier (which can be the LLM itself) to obtain a reward signal $r(\mathbf{z})$
4. **Bayesian Optimization**: Based on the observed $(\mathbf{z}, r)$ pairs, utilize Bayesian Optimization to select the next most promising perturbation vector

The overall process forms a closed loop: **perturbation $\to$ generation $\to$ verification $\to$ optimization $\to$ better perturbation**.

### Key Designs

#### 1. First-Token Embedding Perturbation

Traditional temperature scaling modifies the softmax distribution:

$$P(w^{(t)} \mid w^{(1:t-1)}; \theta, \tau) = \frac{\exp(\ell_{t,w^{(t)}} / \tau)}{\sum_w \exp(\ell_{t,w} / \tau)}$$

The temperature $\tau$ scales all tokens indiscriminately, where a low $\tau$ degenerates to greedy decoding, and a high $\tau$ approaches a uniform distribution.

Soft Reasoning adopts a different perturbation mechanism: it injects a Gaussian vector $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ into the embedding of the first token, directly altering the representation at this position to **selectively** steer the generation path of subsequent tokens. This yields three advantages:

- **Controllability**: Each $\mathbf{z}$ uniquely determines a generation sequence (via greedy decoding), making the search deterministic and reproducible.
- **Flexibility**: Perturbations in the embedding space can yield richer distributional changes compared to simple temperature scaling.
- **Locality**: Only the first token is perturbed; subsequent tokens propagate the differences naturally, minimizing computational overhead.

#### 2. Bayesian Optimization Search

Formulate the reasoning task as a black-box optimization problem:

$$\mathbf{z}^* = \arg\max_{\mathbf{z}} r(\mathbf{z})$$

where $r(\mathbf{z})$ represents the reward value of the sequence generated under the perturbation vector $\mathbf{z}$. The core components of Bayesian optimization include:

- **Gaussian Process (GP) Surrogate Model**: Constructs a probabilistic model of $r(\mathbf{z})$ based on the observed $\{(\mathbf{z}_i, r_i)\}$ to predict the expected reward and uncertainty of unexplored regions.
- **Acquisition Function**: Balances exploration and exploitation to select the most informative next perturbation vector $\mathbf{z}$.
- **Iterative Update**: Updates the GP model upon receiving each new observation, progressively narrowing down the search space.

The key distinction from MCTS / ToT is that Soft Reasoning performs optimization in a continuous embedding space rather than combinatoric search over discrete token/path spaces.

#### 3. Self-Verification Mechanism

A major highlight of Soft Reasoning is that it **obviates the need for external verifiers**. The LLM itself simultaneously acts as both generator and verifier:

- Given a question and a candidate answer, the same LLM is prompted to evaluate the correctness and coherence of the answer.
- The reward function jointly considers correctness (whether the final answer is correct) and coherence (whether the reasoning steps are logically self-consistent).
- This renders the method completely model-agnostic, allowing seamless plug-and-play integration with any LLM.

### Loss & Training

Soft Reasoning is a **pure test-time method** that requires no training or fine-tuning:

- **No Training Required**: It does not modify model parameters or require additional training data.
- **No Model Parameter Access**: It only requires access to the model's embedding interface and generation API (black-box).
- **Controllable Iterative Budget**: The number of Bayesian optimization iterations can be flexibly adjusted based on the computational budget.
- **Combination with Greedy Decoding**: It leverages greedy decoding post-perturbation, eliminating sampling stochasticity.

## Key Experimental Results

### Main Results

| Method | Search Space Type | GSM8K | MATH | Inference Cost |
|------|-------------|-------|------|----------|
| Greedy Decoding | — | Baseline | Baseline | 1× |
| Temperature Sampling (τ=0.7) | token distribution | +Limited Improvement | +Limited Improvement | N× |
| Best-of-N (N=64) | independent sampling | Significant Improvement | Significant Improvement | 64× |
| ToT | discrete paths | Moderate Improvement | Moderate Improvement | High |
| **Soft Reasoning** | **continuous embedding** | **Optimal** | **Optimal** | **Far below Best-of-64** |

> Soft Reasoning achieves accuracy superior to or on par with Best-of-64 under a budget far below 64 generations.

### Methods Characteristics Comparison

| Method | Search Space | External Verifier Required? | Model Parameter Access Required? | Computational Controllability |
|------|----------|-------------------|-----------------|-----------|
| Temperature Scaling | token distribution | No | No | Low (Temperature adjustment only) |
| Best-of-N | independent sampling | Yes (Select best) | No | Poor (Linear growth) |
| Tree of Thoughts | discrete paths | Yes | Partial | Poor (Exponential growth) |
| MCTS | discrete paths | Yes | Partial | Moderate |
| **Soft Reasoning** | **continuous embedding** | **No (Self-verification)** | **No** | **High (BO iterations controllable)** |

### Cross-Model Generalizability

The paper validates the effectiveness of Soft Reasoning across various LLMs, spanning different scales and architectures. Since the method is fully black-box, switching models only requires replacing the base LLM without altering the search pipeline. Experimental results demonstrate consistent improvements across diverse models.

## Highlights & Insights

1. **Empirical Validation of First-Token Determinism**: The approach is grounded in the insight that "the embedding of the first token has a decisive influence on the entire reasoning chain," which aligns with the findings of Wang & Zhou (2024). Perturbing only the first token rather than the entire sequence drastically reduces the search dimensionality.

2. **Continuous vs. Discrete Space Search**: In contrast to ToT/MCTS which search in discrete token spaces, Soft Reasoning searches in a continuous embedding space. This naturally fits continuous optimization paradigms like Bayesian optimization, leading to higher search efficiency.

3. **Elegant Self-Verification Design**: Utilizing the LLM as its own verifier eliminates dependency on external reward models. Although self-verification accuracy may fall short of specialized verifiers, the Bayesian optimization framework is robust enough to leverage noisy reward signals to effectively guide the search.

4. **Deterministic Guarantees of Greedy Decoding**: Using greedy decoding post-perturbation ensures a one-to-one mapping $\mathbf{z} \to \text{sequence}$. This yields a well-structured search space, making it easier for the Bayesian optimization surrogate model to fit.

5. **Model-Agnostic Plug-and-Play Architecture**: Requiring only an embedding input interface and a text output interface, the method is decoupled from specific model architectures, parameters, or training procedures, granting it exceptional applicability.

## Limitations & Future Work

1. **Limitations of First-Token Perturbation**: The assumption of perturbing only the first token may prove insufficient in scenarios requiring mid-course corrections of the reasoning trajectory. Extremely complex problems might necessitate intervention at intermediate steps.

2. **Dimensionality Constraints of Bayesian Optimization**: Embedding spaces are typically high-dimensional (e.g., 4096 dimensions). Standard Bayesian optimization exhibits limited efficiency in such high dimensions, potentially requiring dimensionality reduction techniques (e.g., random projections or PCA).

3. **Self-Verification Quality**: The accuracy of the LLM as its own verifier is bounded by the model's inherent capabilities. In domains where the model itself struggles to evaluate correctness (e.g., advanced mathematics), self-verification may supply misleading reward signals.

4. **Lack of Theoretical Guarantees**: There is a lack of rigorous theoretical analysis regarding how embedding perturbations propagate to alter subsequent token distributions. The prior assumption of Gaussian perturbations may not be optimal.

5. **Non-Zero Computational Cost**: Though significantly more efficient than Best-of-64, each iteration still demands a complete sequence generation and validation step, which may introduce latency in real-time inference scenarios.

## Related Work & Insights

- **Decoding Strategies**: Complementary to nucleus sampling (Holtzman et al., 2020), min-p sampling (Minh et al., 2025), and D3 (Bao et al., 2024). Soft Reasoning approaches diversity from the embedding space rather than the logit space.
- **Test-Time Compute Optimization**: Snell et al. (2025) suggest that optimizing test-time compute allocation is more effective than scaling model size; Soft Reasoning represents a concrete execution within this paradigm.
- **Tree Search Methods**: While ToT (Yao et al., 2023) and TSE (Zhang & Liu, 2024) search in discrete spaces, Soft Reasoning generalizes this to the continuous embedding space.
- **Mutual Reasoning and Self-Play**: The MCTS + self-play ideas of Qi et al. (2025b) and Yan et al. (2024) share similarities with the self-verification mechanism in Soft Reasoning.
- **Significance of the First Token**: Research by Wang & Zhou (2024) demonstrates that the choice of the first token critically impacts the reasoning output, providing an empirical foundation for the core design of this paper.
- **Bayesian Optimization in NLP**: Applying BO for hyperparameter search is common, but utilizing it for test-time embedding search is a novel endeavor.

## Rating

- **Novelty**: ⭐⭐⭐⭐☆ — Transforming the reasoning task into a Bayesian optimization problem in the embedding space presents a fresh perspective. The first-token perturbation design is elegant and simple.
- **Experimental Thoroughness**: ⭐⭐⭐☆☆ — It covers multiple LLMs and reasoning tasks, but cache truncation limits full evaluation of all experimental details.
- **Writing Quality**: ⭐⭐⭐⭐☆ — Clear motivation, systematic methodology description, and thorough comparisons with prior works.
- **Value**: ⭐⭐⭐⭐☆ — Offers a novel and practical reasoning enhancement strategy, with its model-agnostic nature showing wide application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Improving Rationality in the Reasoning Process of Language Models through Self-playing Game](improving_rationality_in_the_reasoning_process_of_language_models_through_self-p.md)
- [\[CVPR 2026\] ReLaX: Reasoning with Latent Exploration for Large Reasoning Models](../../CVPR2026/llm_reasoning/relax_reasoning_with_latent_exploration_for_large_reasoning_models.md)
- [\[NeurIPS 2025\] Topology of Reasoning: Understanding Large Reasoning Models through Reasoning Graph Properties](../../NeurIPS2025/llm_reasoning/topology_of_reasoning_understanding_large_reasoning_models_through_reasoning_gra.md)
- [\[ICML 2025\] Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models](emergent_symbolic_mechanisms_support_abstract_reasoning_in_large_language_models.md)
- [\[AAAI 2026\] Efficient Thought Space Exploration Through Strategic Intervention](../../AAAI2026/llm_reasoning/efficient_thought_space_exploration_through_strategic_intervention.md)

</div>

<!-- RELATED:END -->
