---
title: >-
  [Paper Note] On the Power of Context-Enhanced Learning in LLMs
description: >-
  [ICML 2025 (Main Conference)][Interpretability][Context-Enhanced Learning] This paper formally defines "context-enhanced learning" (CEL), proving that its sample efficiency is **exponentially higher** than that of standard learning under simplified settings, and reveals at a mechanistic level that its advantage stems from more precise gradient learning signals.
tags:
  - "ICML 2025 (Main Conference)"
  - "Interpretability"
  - "Context-Enhanced Learning"
  - "In-Context Learning"
  - "Sample Efficiency"
  - "Gradient Signal"
  - "Multi-Step Reasoning"
  - "Data Security"
date: 2026-05-08
content_hash: f7e54dda7146c488
---

# On the Power of Context-Enhanced Learning in LLMs

**Conference**: ICML 2025 (Main Conference)  
**arXiv**: [2503.01821](https://arxiv.org/abs/2503.01821)  
**Code**: —  
**Area**: Interpretability  
**Keywords**: Context-Enhanced Learning, In-Context Learning, Sample Efficiency, Gradient Signal, Multi-Step Reasoning, Data Security  

## TL;DR

This paper formally defines "context-enhanced learning" (CEL), proving that its sample efficiency is **exponentially higher** than that of standard learning under simplified settings, and reveals at a mechanistic level that its advantage stems from more precise gradient learning signals.

## Background & Motivation

In-context learning (ICL) in LLMs—learning new tasks via examples in the context at inference time—has been a hot research topic in recent years. However, a variant has been relatively overlooked: **context-enhanced learning**.

**Standard Learning**: Autoregressive loss is calculated on training texts to update parameters.

**Context-Enhanced Learning (CEL)**: During training, additional data is placed in the context window, but **no autoregressive gradients are computed on this context data**; gradients are only computed on the target text. Formal definition:

$$\mathcal{L}_{\text{CEL}}(\theta) = -\sum_{t=1}^{T} \log p_\theta(x_t | c_1, \ldots, c_K, x_1, \ldots, x_{t-1})$$

where $c_1, \ldots, c_K$ are the context enhancement data, and gradients are only backpropagated with respect to the $x_t$ portion.

This setting appears in some recent works (such as data augmentation and retrieval-augmented training) but lacks theoretical understanding. Core problems:

1. Why is CEL effective? How much better can it theoretically perform compared to standard learning?
2. Can learning materials in the context be detected or recovered? (Implications for data security)

## Method

### Theoretical Framework

#### Multi-Step Reasoning Task Setup

A multi-step compositional reasoning task is constructed: given a $k$-step reasoning chain $a_1 \to a_2 \to \ldots \to a_k \to y$, where each transition $a_i \to a_{i+1}$ is chosen from a set of rules.

**Limitations of Standard Learning**: It requires simultaneously learning combinations of all $k$-step rules from the training data, resulting in a sample complexity of:

$$n_{\text{standard}} = \Omega(R^k)$$

where $R$ is the number of alternative rules for each step. That is, the sample requirement of standard learning grows exponentially with the number of reasoning steps.

**Advantage of CEL**: Placing a portion of the reasoning rules into the context. The model can directly read the rules in the context through its ICL capability, needing only to learn the remaining rules. The sample complexity is reduced to:

$$n_{\text{CEL}} = O(\text{poly}(R, k))$$

#### Core Theorem

**Theorem 1**: For models with ICL capability, the sample efficiency of context-enhanced learning can be **exponentially higher** than that of standard learning. Specifically, there exists a family of multi-step reasoning tasks such that:

$$\frac{n_{\text{standard}}}{n_{\text{CEL}}} = \Omega\left(\frac{R^k}{\text{poly}(R, k)}\right) = \text{super-polynomial growth}$$

### Mechanistic Analysis

#### Gradient Signal Precision

The core advantage of CEL lies in a **more precise gradient signal**. Intuitive explanation:

- **Standard Learning**: Gradients contain significant noise because the model must simultaneously deduce rules for all reasoning steps.
- **CEL**: Rules in the context act as "anchors" for the model, making the gradient signal more focused.

Formally, the gradient variance of CEL satisfies:

$$\text{Var}[\nabla_\theta \mathcal{L}_{\text{CEL}}] \ll \text{Var}[\nabla_\theta \mathcal{L}_{\text{standard}}]$$

The reduction in gradient noise is proportional to the amount of information provided in the context.

### Data Security Analysis

An important question studied experimentally: can learning materials in the context be detected or recovered ex-post?

Through experiments with membership inference and data extraction attacks, it is found that **learning materials in the context are very difficult to detect or recover**. This has twofold implications:
- Positive: CEL does not leak context data.
- Negative: It could be used to circumvent copyright protection by training with protected data enhancing the context.

## Experiments

### Synthetic Task: Sample Efficiency Comparison

| Reasoning Steps $k$ | Samples Required for Standard Learning | Samples Required for CEL | Efficiency Ratio |
|-------------|----------------|-------------|--------|
| 2 | ~$R^2$ | ~$R$ | $R$x |
| 3 | ~$R^3$ | ~$R$ | $R^2$x |
| 4 | ~$R^4$ | ~$R$ | $R^3$x |
| 5 | ~$R^5$ | ~$R^{1.2}$ | ~$R^{3.8}$x |

When $R=10$, CEL is approximately 6,000 times more sample-efficient than standard learning on a 5-step reasoning task.

### Natural Language Experiments

| Setup | Accuracy (5000 Samples) | Accuracy (50000 Samples) |
|------|-------------------|---------------------|
| Standard Fine-tuning | 42.3% | 68.7% |
| CEL Fine-tuning | 71.5% | 82.1% |
| ICL (No Fine-tuning) | 35.8% | 35.8% |

The advantage of CEL is particularly prominent in few-sample scenarios.

### Data Security Experiments

| Attack Method | Success Rate of Detecting/Recovering Context Data |
|----------|--------------------------|
| Membership Inference | ~52% (close to random guess) |
| Data Extraction | < 5% |
| Perplexity-based Detection | ~55% |

Learning materials in the context are almost undetectable ex-post.

## Highlights & Insights

- **First Theoretical Analysis of CEL**: Proves an exponential advantage in sample efficiency, establishing a theoretical foundation for methods like retrieval-augmented training.
- **Deep Mechanistic Insight**: The advantage stems from gradient signal precision rather than model capacity.
- **Double-Edged Sword of Data Security**: CEL protects the privacy of context data, but can also be abused.
- Mutual verification of theory and experiments; a comprehensive 77-page paper (accepted at the Main Conference).

## Limitations & Future Work

- The theoretical analysis is based on a simplified Transformer model (single-layer attention); the gap with actual deep models remains to be verified.
- The setup of the multi-step reasoning task is relatively synthetic; more experiments are required to see if it can generalize to real-world reasoning tasks.
- The data security analysis only covers basic attack methods; more advanced attacks might alter the conclusions.
- The experimental scale is constrained by academic resources, and has not been validated on 100B+ models.

## Related Work & Insights

- **ICL Theory (Garg et al., 2022; Akyürek et al., 2023)**: Understands ICL as implicit gradient descent.
- **Retrieval-Augmented Training (Borgeaud et al., 2022)**: A practical form of CEL.
- **Data Contamination (Shi et al., 2024)**: Training data detection methods.
- This work provides a theoretical foundation for CEL/RAT, proving that its advantages are fundamental (exponential) rather than merely empirical.

## Rating

⭐⭐⭐⭐⭐ — An ICML 2025 main conference paper with solid theory (exponential separation proof), clear mechanistic explanations, and consideration of data security implications. It is a significant contribution to the learning theory of LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Understanding Prompt Tuning and In-Context Learning via Meta-Learning](../../NeurIPS2025/interpretability/understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)
- [\[ICLR 2026\] Hessian-Enhanced Token Attribution (HETA): Interpreting Autoregressive LLMs](../../ICLR2026/interpretability/hessian-enhanced_token_attribution_heta_interpreting_autoregressive_llms.md)
- [\[ICLR 2026\] Comparing the learning dynamics of in-context learning and fine-tuning in language models](../../ICLR2026/interpretability/comparing_the_learning_dynamics_of_in-context_learning_and_fine-tuning_in_langua.md)
- [\[ICML 2025\] Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective](evolving_prompts_in-context_an_open-ended_self-replicating_perspective.md)
- [\[ICLR 2026\] Understanding Task Vectors in In-Context Learning: Emergence, Functionality, and Limitations](../../ICLR2026/interpretability/understanding_task_vectors_in_in-context_learning_emergence_functionality_and_li.md)

</div>

<!-- RELATED:END -->
