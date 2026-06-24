---
title: >-
  [Paper Note] Binary Classifier Optimization for Large Language Model Alignment
description: >-
  [ACL 2025][LLM (Other)][LLM alignment] Proposes Binary Classifier Optimization (BCO), mathematically proving that the binary cross-entropy (BCE) loss is an upper bound of the DPO loss. This theoretical link enables LLM alignment using only "thumbs-up/down" binary feedback instead of pairwise preference data. By introducing a novel reward shift technique to tighten the upper bound, BCO performs comparably to DPO on paired preference datasets and outperforms both DPO and KTO on…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "LLM alignment"
  - "binary feedback"
  - "DPO"
  - "binary classifier"
  - "reward shift"
date: 2026-05-08
content_hash: 4cbba5a96cc3823a
---

# Binary Classifier Optimization for Large Language Model Alignment

**Conference**: ACL 2025  
**arXiv**: [2404.04656](https://arxiv.org/abs/2404.04656)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: LLM alignment, binary feedback, DPO, binary classifier, reward shift

## TL;DR
Proposes Binary Classifier Optimization (BCO), mathematically proving that the binary cross-entropy (BCE) loss is an upper bound of the DPO loss. This theoretical link enables LLM alignment using only "thumbs-up/down" binary feedback instead of pairwise preference data. By introducing a novel reward shift technique to tighten the upper bound, BCO performs comparably to DPO on paired preference datasets and outperforms both DPO and KTO on real-world Likert-5 annotated data.

## Background & Motivation
**Background**: RLHF and DPO are standard methods for LLM alignment, but they require pairwise preference data (chosen vs. rejected), which is expensive to collect.

**Limitations of Prior Work**: In real-world services (such as ChatGPT, Gemini, etc.), users typically provide only binary feedback ("👍/👎") rather than comparison-based feedback between two responses. Converting binary feedback into pairwise preference data requires additional engineering effort.

**Key Challenge**: The feedback format that is easiest to collect (binary signals) does not match the data format required by existing alignment methods (pairwise preferences).

**Goal**: How can LLMs be effectively aligned using only binary feedback (thumbs-up/down)? What is its theoretical connection to DPO?

**Key Insight**: Treatment of alignment as a binary classification problem: $\{ \text{prompt}, \text{good response} \} \to 1$ and $\{ \text{prompt}, \text{bad response} \} \to 0$, where the logit of the classifier serves as the implicit reward.

**Core Idea**: The BCE loss for training a binary classifier is a strict upper bound on the DPO loss, meaning that minimizing the former implicitly minimizes the latter.

## Method

### Overall Architecture
By treating the implicit reward of LLMs, $r_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$, as the logit of a binary classifier where positive responses are labeled 1 and negative ones labeled 0, alignment can be achieved simply by training with the BCE loss.

### Key Designs
1. **BCE ↔ DPO Upper Bound (Theorem 1)**:

    - **Function**: Proves that the BCE loss is strictly greater than the DPO loss.
    - **Mechanism**: Utilizes Lemma 2 ($\log\sigma(x+y) > \log\sigma(x) + \log\sigma(y)$) to decompose the DPO loss $-\log\sigma(r_w - r_l)$ into the separable BCE form $-\log\sigma(r_w) - \log\sigma(-r_l)$.
    - **Design Motivation**: Establishes a theoretical bridge between binary-signal alignment and preference alignment, justifying the effectiveness of the former.

2. **Reward Shift (Theorem 3 & 4)**:

    - **Function**: Tightens the gap between BCE and DPO via a shift parameter $\delta$.
    - **Mechanism**: The error term $e^{-(r_w - \delta)} + e^{r_l - \delta}$ is minimized when $\delta = (r_w + r_l)/2$. In practice, $\delta = \frac{\mathbb{E}_{D^+}[r_\theta] + \mathbb{E}_{D^-}[r_\theta]}{4}$ (computed using an Exponential Moving Average) is used.
    - **Design Motivation**: The direct BCE upper bound can be loose; reward shift significantly tightens this bound, improving alignment performance.

3. **Key Differences from KTO**:

    - BCO optimizes $\log\sigma$ (whose gradient is $\sigma(-r)\nabla\log\pi$), whereas KTO optimizes $\sigma$ (whose gradient contains an extra factor of $\sigma(r)$, which excessively diminishes gradients for samples with large rewards).
    - The shift $\delta$ in BCO is optimally derived from theory, whereas KTO's $z_{\text{ref}}$ uses the in-batch average reward with a max(0,·) truncation, which lacks standard theoretical justification.

## Key Experimental Results

### Paired Preference Dataset (Anthropic HH-RLHF)

| Method | Win Rate vs SFT |
|------|-----------------|
| DPO | ~Baseline |
| KTO | Lower than DPO |
| BCO | Comparable to DPO |

### Real-World Likert-5 Annotated Dataset

| Method | Qwen-1.5-0.5B | Qwen-1.5-7B | Llama-3-8B |
|------|---------------|-------------|------------|
| DPO | Suboptimal | Suboptimal | Suboptimal |
| KTO | Worst | Worst | Worst |
| BCO | **Best** | **Best** | **Best** |

### Ablation Study

| Configuration | Performance |
|------|------|
| BCE only (no shift) | Effective but unstable |
| BCE + reward shift | Consistently outperforms no-shift |
| KTO $z_{\text{ref}}$ | $z_{\text{ref}}$ is pinned at 0 in early training, delaying alignment |

### Key Findings
- BCO consistently outperforms DPO and KTO on real user data across 4 base LLMs.
- Reward shift is crucial for training stability; the EMA calculation yields smoother training compared to batch-level calculation.
- KTO's $\sigma(r)\sigma(-r)$ gradient structure leads to gradient vanishing on samples with large rewards.
- BCO performs comparably to DPO on paired data, validating the effectiveness of the upper bound.

## Highlights & Insights
- Elegant theoretical analysis: a simple Lemma (the superadditivity of log-sigmoid) bridges BCE and DPO.
- The concept of reward shift stems from error-term minimization, providing a theoretically optimal solution rather than a heuristic.
- High practical value: platforms like ChatGPT are already collecting thumbs-up/down data, allowing BCO to be directly applied.

## Limitations & Future Work
- The theory assumes independent distributions of chosen and rejected data; correlation may exist in practice.
- Lack of direct comparison with RLHF (PPO).
- The EMA hyperparameter for reward shift requires tuning.
- Validated only on small-to-medium scale LLMs (0.5B-8B).

## Related Work & Insights
- **vs. DPO**: BCO does not require paired preference data and performs better on real-world data.
- **vs. KTO**: Both align using binary signals, but BCO features a more rigorous theoretical foundation and better gradient properties.
- **vs. NCA**: NCA requires multiple completions per prompt, whereas BCO requires only one response per prompt.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Clear theoretical contributions (BCE as an upper bound to DPO), and the reward shift is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated across 4 base models, paired and real-world data, with thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous and clear theoretical derivations; the logical progression from motivation to analysis, methodology, and evaluation is complete.
- **Value**: ⭐⭐⭐⭐⭐ Directly applicable to production-level LLM alignment pipelines, reducing data collection costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] IPO: Your Language Model is Secretly a Preference Classifier](ipo_your_language_model_is_secretly_a_preference_classifier.md)
- [\[ACL 2025\] Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models](gapo_multi_objective_alignment.md)
- [\[ACL 2025\] Improve Language Model and Brain Alignment via Associative Memory](improve_language_model_and_brain_alignment_via_associative_memory.md)
- [\[ACL 2025\] Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence In Large Language Models](direct_confidence_alignment_aligning_verbalized_confidence_with_internal_confide.md)
- [\[ACL 2025\] From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment](from_neurons_to_semantics_evaluating_cross-linguistic_alignment_capabilities_of_.md)

</div>

<!-- RELATED:END -->
