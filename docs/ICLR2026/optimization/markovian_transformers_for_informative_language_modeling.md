---
title: >-
  [Paper Note] Markovian Transformers for Informative Language Modeling
description: >-
  [ICLR 2026][Optimization][Markov constraint] This paper proposes the Markovian Language Model (MLM) framework, which enforces CoT to serve as a causally necessary reasoning bottleneck through **structural constraints** (removing the original question during answer prediction, so that the answer is derived solely from the CoT). Analogous to the narrow latent layer in an autoencoder, this approach is combined with GRPO-style policy gradient training, improving accuracy on GSM8K from 19.6% to 57.1%. The learned CoT also transfers across model architectures (Llama→Mistral/Phi/GPT-2), demonstrating that CoT encodes natural language reasoning rather than steganography.
tags:
  - ICLR 2026
  - Optimization
  - Markov constraint
  - CoT faithfulness
  - reasoning bottleneck
  - autoencoder analogy
  - GRPO training
  - information theory
date: 2026-05-08
content_hash: 952828d852118fed
---

# Markovian Transformers for Informative Language Modeling

**Conference**: ICLR 2026
**arXiv**: [2404.18988](https://arxiv.org/abs/2404.18988)
**Authors**: Scott W. Viteri, Max Lamparth, Peter Chatain, Clark Barrett (Stanford University)
**Code**: [GitHub](https://github.com/scottviteri/MarkovianTraining/)
**Area**: Optimization
**Keywords**: Markov constraint, CoT faithfulness, reasoning bottleneck, autoencoder analogy, GRPO training, information theory

## TL;DR

This paper proposes the Markovian Language Model (MLM) framework, which enforces CoT to serve as a causally necessary reasoning bottleneck through **structural constraints** (removing the original question during answer prediction, so that the answer is derived solely from the CoT). Analogous to the narrow latent layer in an autoencoder, this approach is combined with GRPO-style policy gradient training, improving accuracy on GSM8K from 19.6% to 57.1%. The learned CoT also transfers across model architectures (Llama→Mistral/Phi/GPT-2), demonstrating that CoT encodes natural language reasoning rather than steganography.

## Background & Motivation

**The ubiquity of CoT faithfulness issues**: Although Chain-of-Thought reasoning improves LLM performance, substantial prior work (Turpin et al., 2023; Lanham et al., 2023) shows that CoT does not necessarily faithfully reflect the model's true reasoning process—perturbing CoT text may leave the final answer unchanged, indicating that CoT is not truly "load-bearing."

**Limitations of Prior Work**: Methods such as STaR (Zelikman et al., 2022) and DeepSeek-R1 (Guo et al., 2025) improve CoT quality through fine-tuning, but the model still has access to the original question when predicting the answer, leaving an architectural bypass through which the model can answer directly without relying on the CoT.

**Absence of an information-theoretic framework**: There is a need for a framework that makes CoT the sole information channel from question to answer, such that corrupting the CoT necessarily degrades answer quality—providing causal necessity rather than mere statistical correlation.

**Gap between structural and optimization-based constraints**: Pure optimization approaches (e.g., regularization or supervised signals) impose only soft constraints on CoT quality. This work instead pursues a hard architectural constraint—fundamentally severing the direct question→answer pathway.

**Insight from the autoencoder analogy**: CoT is analogized to the narrow latent layer of an autoencoder: all information flowing from input ($Q$) to output ($A$) must pass through the limited-bandwidth bottleneck ($\text{CoT}$), compelling the model to compress its reasoning into interpretable natural language steps.

**Empirical refutation of steganography risk**: In principle, a model could encode answer information in CoT using human-unreadable schemes (steganography). This possibility is empirically ruled out through KL-divergence penalties and cross-model transfer experiments.

## Method

### 1. Formalization of the Markovian Language Model (MLM)

The MLM is defined as $M = (\mathcal{O}, \mathcal{S}, \pi, u, s_1)$, where:
- $\mathcal{O}$: observation space (questions and answers)
- $\mathcal{S}$: state space (CoT reasoning text)
- $\pi: \mathcal{S} \to \Delta(\mathcal{O})$: policy — **predicts the next observation from the state only**
- $u: \mathcal{O} \times \mathcal{S} \to \Delta(\mathcal{S})$: state update function
- $s_1 \in \mathcal{S}$: initial state

Key constraint: when predicting answer $o_2$, $\pi$ **can only observe the CoT state $s_2$** and not the original question $o_1$, enforcing the Markov chain structure $A \to B \to C$.

### 2. Informativeness Objective

The reward is defined as the log-probability improvement of the trained model relative to a frozen baseline:

$$R_\theta(\tau) = \sum_{t=1}^{T}\left[\ln\pi_\theta(x_t|s_t) - \ln\pi'(x_t|s'_t)\right]$$

The objective is: $J(\theta) = \mathbb{E}_{\tau \sim P, u_\theta, u'}[R_\theta(\tau)]$

Maximizing $J(\theta)$ ensures that the CoT generated by the state update function $u_\theta$ is sufficiently informative for predicting future observations (relative to the baseline).

### 3. GRPO-Style Policy Gradient Training

The loss function consists of three terms:

$$\mathcal{L} = \mathcal{L}_{PG} + \mathcal{L}_{AR} + \mathcal{L}_{KL}$$

- $\mathcal{L}_{PG} = -\ln u_\theta(\text{CoT}|q, \text{CoT}_{init}) \cdot A^{detach}$ (policy gradient term)
- $\mathcal{L}_{AR} = -A$ (actor-reward gradient term — a key contribution of this paper)
- $\mathcal{L}_{KL} = \beta_{KL} D_{KL}(u_\theta \| u')$, $\beta_{KL}=0.1$ (KL regularization to prevent steganography)

### Key Design Choices

- **Actor-Reward Gradient**: Since the same Transformer defines both the sampling distribution $u_\theta$ and the reward $R_\theta$, the chain rule yields two gradient terms—a standard policy gradient and a direct reward gradient $\nabla_\theta R_\theta(\tau)$. Both are used simultaneously.
- **Parallel Sampling**: Each batch contains $B$ copies of the same $(q, a)$ pair; the model generates $B$ diverse CoT candidates. GRPO-style within-batch advantage normalization eliminates the need for a critic model.
- **Frozen Baseline CoT′**: The untuned model generates a reference CoT′ as a local baseline; local subtraction followed by within-batch normalization is applied.
- **Coding-Theoretic Interpretation**: $-\log\pi_\theta(C|B)$ is the encoding cost of the answer given the CoT; $-\log u'(B|A)$ is the prior encoding cost of the CoT. Training seeks a short textual state $B$ that minimizes both costs simultaneously.
- **Time-Limited Complexity Argument**: The CoT provides $|B|$ additional forward passes for reasoning. Since the model has only $|A|$ forward passes while reading the question, it cannot reliably solve hard problems without leveraging the CoT.

## Key Experimental Results

### Table 1: Main Accuracy Comparison (Llama 3.1 8B)

| Dataset | Baseline | Expert Iteration | No Reward Gradient | **Markovian (Ours)** | Non-Markovian |
|--------|------|-----------------|-----------|-------------------|-----------|
| GSM8K | 19.6% | 61.6% | 62.2% | **57.1%** | 63.3% |
| ARC-Challenge | 36.1% | 65.6% | 79.3% | **79.9%** | 78.6% |
| MMLU | 21.4% | 53.2% | 46.6% | **55.5%** | 68.7% |
| SVAMP | 18.0% | 38.7% | 40.7% | **42.3%** | 43.3% |
| Arithmetic | 1.0% | 76.0% | 81.0% | **98.0%** | 97.0% |
| **Average** | 19.2% | 59.0% | 62.0% | **66.6%** | 70.2% |

### Table 2: Perturbation Vulnerability on Wikipedia Continuation ($\Delta\ln P$ = Markovian drop − Non-Markovian drop)

| Perturbation Level | Char Substitution | Deletion | Digit Substitution | Suffix Truncation | Prefix Truncation | **Row Mean** |
|---------|---------|------|---------|-------|-------|-----------|
| 20% | +0.457 | +0.459 | +0.016 | +0.254 | -0.009 | **+0.235** |
| 40% | +0.849 | +0.836 | +0.025 | +0.368 | +0.121 | **+0.440** |
| 60% | +1.042 | +1.002 | +0.035 | +0.596 | +0.284 | **+0.592** |
| 80% | +1.079 | +1.069 | +0.038 | +1.020 | +0.622 | **+0.766** |
| 100% | +1.084 | +1.263 | +0.039 | +1.258 | +1.262 | **+0.981** |

### Table 3: Perturbation Vulnerability on QA Tasks (Accuracy Δ; positive = Markovian more vulnerable)

| Dataset | Char Substitution | Deletion | Digit Substitution | Suffix Truncation | Prefix Truncation | **Mean** |
|--------|---------|------|---------|-------|-------|---------|
| ARC | +0.320 | +0.424 | -0.004 | +0.069 | +0.439 | **+0.250** |
| SVAMP | +0.154 | +0.204 | +0.081 | +0.076 | +0.046 | **+0.112** |
| GSM8K | +0.059 | +0.069 | -0.013 | +0.105 | +0.044 | **+0.053** |
| MMLU | +0.056 | +0.124 | +0.004 | +0.038 | -0.001 | **+0.044** |

## Key Findings

- **The cost of the Markovian constraint is small**: Across five datasets, the Markovian model trails the non-Markovian counterpart by only 3.6 pp on average (66.6% vs. 70.2%)—a modest accuracy cost in exchange for a structural guarantee of CoT causal necessity.
- **Markovian outperforms non-Markovian on ARC-Challenge and Arithmetic**: ARC 79.9% vs. 78.6%, Arithmetic 98.0% vs. 97.0%—suggesting that for certain reasoning tasks, forcing an information bottleneck has a beneficial regularization effect.
- **Perturbation vulnerability increases monotonically**: As perturbation intensity increases from 20% to 100%, $\Delta\ln P$ grows from +0.235 to +0.981, confirming that Markovian CoT carries significantly more causal information.
- **Cross-model transfer succeeds**: CoT generated by Llama is informative for Mistral, Phi-3.5, and even GPT-2, demonstrating that the CoT encodes natural language reasoning steps rather than model-specific steganography.
- **Actor-reward gradient is critical**: Ablation experiments show that removing the $\nabla_\theta R_\theta$ term degrades performance on multiple tasks (e.g., MMLU drops from 55.5% to 46.6%).

## Highlights & Insights

- **Elegance of the autoencoder analogy**: $Q$ = input data, $\text{CoT}$ = latent layer, $A$ = reconstruction. The narrow latent layer forces compression, so reasoning must be expressed linguistically. The coding-theoretic (MDL) interpretation endows this analogy with mathematical rigor.
- **Methodological contribution of structure over optimization**: CoT necessity is guaranteed via architectural constraints (removing the attention path from $q$ to $A$) rather than loss-level constraints—a stronger guarantee than optimization-based methods such as FRODO.
- **Cross-model transfer as anti-steganography evidence**: If CoT were merely an encrypted encoding, substituting a different decoder would break it; yet even GPT-2 (far smaller than the training model) can exploit the CoT, implying it must be expressed in natural language.
- **Qualitative change on arithmetic tasks**: Before training, Llama's CoT merely lists numbers and erroneous sums; after training, correct step-by-step calculations emerge—a behaviorally observable emergence of reasoning ability.

## Limitations & Future Work

- **Larger gap on MMLU** (55.5% vs. 68.7%): For tasks requiring factual retrieval, the CoT bottleneck restricts information flow from question to answer, making the accuracy cost more pronounced.
- **No guarantee of complete faithfulness**: The model may have already computed the answer during the forward passes over the question, subsequently generating a plausible but non-genuine CoT. The Markovian constraint ensures CoT sufficiency but not that it reflects internal computation.
- **Validated only on 7–8B models**: Larger models have stronger steganographic encoding capacity; it remains open whether the KL penalty remains sufficient at scale.
- **No human evaluation**: Interpretability is currently validated indirectly through perturbation vulnerability and cross-model transfer; direct assessment of human readability and usefulness is absent.
- **High training cost**: Approximately \$20K total (H100/H200), with a single training run taking roughly 10 hours—not easily accessible for academic groups.

## Related Work & Insights

### vs. FRODO (Paul et al., 2024)
FRODO employs causal mediation analysis and a two-module training framework to increase the causal effect of CoT on the answer, but still allows the model to observe the original question when generating the answer—an **optimization-level** soft constraint. The present work directly removes the question-to-answer attention pathway at the **architectural level**, providing a stronger causal necessity guarantee. Experiments show this approach performs better on multiple datasets and uniquely enables cross-model CoT transfer.

### vs. DeepSeek-R1 / STaR / QuietSTaR
These methods also use RL or self-training to improve CoT reasoning quality, but allow the model to observe the full context when generating reasoning tokens—no Markovian structure is enforced. The key distinction of this work is the **information bottleneck**: the answer is derived solely from the CoT, providing a structural guarantee of CoT faithfulness rather than merely a performance improvement. DeepSeek-R1 pursues stronger reasoning capability without addressing causal necessity; STaR/QuietSTaR improve CoT iteratively but lack an architectural mechanism to prevent bypassing the CoT.

### vs. Lyu et al. (2023) Faithful CoT
This work similarly considers restricting model access to the original input, but rewrites questions into formal language or code before execution. The present work uses natural language as the reasoning state, preserving cross-task interpretability and generality without depending on an external executor.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Introduces the autoencoder bottleneck concept into CoT faithfulness; the theoretical framework (MLM + MDL) is elegant and unified.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Five QA datasets + Wikipedia + perturbation analysis + cross-model transfer + ablations; lacks scaling experiments and human evaluation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The autoencoder analogy is precise; the logical chain from definitions to algorithms to experiments is clear and complete.
- **Value**: ⭐⭐⭐⭐ Foundational methodological contribution to interpretable AI and CoT faithfulness; efficiency and scaling challenges remain for practical deployment.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Optimality and NP-Hardness of Transformers in Learning Markovian Dynamical Functions](../../NeurIPS2025/optimization/optimality_and_np-hardness_of_transformers_in_learning_markovian_dynamical_funct.md)
- [\[ICLR 2026\] Learning to Recall with Transformers Beyond Orthogonal Embeddings](learning_to_recall_with_transformers_beyond_orthogonal_embeddings.md)
- [\[ICLR 2026\] COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics](cold-steer_steering_large_language_models_via_in-context_one-step_learning_dynam.md)
- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](../../NeurIPS2025/optimization/streaming_federated_learning_with_markovian_data.md)
- [\[NeurIPS 2025\] Training Robust Graph Neural Networks by Modeling Noise Dependencies](../../NeurIPS2025/optimization/training_robust_graph_neural_networks_by_modeling_noise_dependencies.md)

<!-- RELATED:END -->
