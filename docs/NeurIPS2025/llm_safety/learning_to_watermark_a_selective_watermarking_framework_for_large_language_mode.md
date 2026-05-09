---
title: >-
  [Paper Note] Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization
description: >-
  [NeurIPS 2025][LLM Safety][LLM watermarking] This paper proposes LTW (Learning to Watermark), a framework that employs a lightweight selector network to adaptively determine when to apply watermarks based on sentence embeddings, token entropy, and the current watermarking ratio. By leveraging multi-objective optimization via MGDA, LTW achieves a Pareto-optimal balance between detectability and text quality, substantially improving watermarked text quality without compromising detection performance.
tags:
  - NeurIPS 2025
  - LLM Safety
  - LLM watermarking
  - selective watermarking
  - multi-objective optimization
  - text quality
  - watermark detectability
date: 2026-05-08
content_hash: 002a4eb7a5b7d686
---

# Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization

**Conference**: NeurIPS 2025  
**arXiv**: [2510.15976](https://arxiv.org/abs/2510.15976)  
**Code**: [https://github.com/fattyray/learning-to-watermark](https://github.com/fattyray/learning-to-watermark)  
**Area**: AI Safety  
**Keywords**: LLM watermarking, selective watermarking, multi-objective optimization, text quality, watermark detectability

## TL;DR
This paper proposes LTW (Learning to Watermark), a framework that employs a lightweight selector network to adaptively determine when to apply watermarks based on sentence embeddings, token entropy, and the current watermarking ratio. By leveraging multi-objective optimization via MGDA, LTW achieves a Pareto-optimal balance between detectability and text quality, substantially improving watermarked text quality without compromising detection performance.

## Background & Motivation

**Background**: The rapid development of LLMs has introduced risks related to copyright infringement and misuse, making watermarking techniques (e.g., KGW, Unigram) essential tools for detecting AI-generated text. KGW embeds detectable signals by partitioning the vocabulary into "green lists" and "red lists" and biasing token selection toward green-list tokens.

**Limitations of Prior Work**: Existing watermarking methods face a fundamental trade-off between detectability and text quality—systematically shifting token selection degrades semantic coherence. TS-watermark restricts user flexibility in parameter tuning; NS-watermark significantly slows generation speed and yields fragile detectability; EXP-edit incurs prohibitively slow detection.

**Key Challenge**: Selective watermarking—applying watermarks only to a subset of tokens—is a promising direction for resolving the quality–detectability trade-off. However, the existing method SWEET relies solely on an entropy threshold as the selection criterion, requiring extensive grid search for manual tuning and ignoring important information such as semantic context.

**Goal**: To automatically learn optimal watermark-application decisions—determining when to watermark a token and when not to.

**Key Insight**: The selective watermarking decision is modeled as the output of a learnable network, with multi-objective optimization simultaneously targeting the two conflicting objectives of detectability and text quality.

**Core Idea**: A lightweight MLP is trained as a "selector" that integrates semantic embeddings, token entropy, and the watermarking ratio, using MGDA to find the Pareto-optimal solution between detectability and quality.

## Method

### Overall Architecture
At each token generation step of the LLM, the selector network determines whether to apply a watermark to the current token based on the current context. During training, continuous outputs enable differentiable optimization; at inference time, an adaptive threshold discretizes the output into a binary decision. During detection, the same selection decisions are reconstructed, and only the selected tokens are evaluated.

### Key Designs

1. **Selector Network**:

    - Function: Decides at each generation step whether the current token should be watermarked.
    - Mechanism: The input is a concatenation of three components: (a) sentence embeddings of the preceding $k$ tokens $\mathcal{E}_{\text{sem}}(s_{n-k+1:n})$ (provided by SimCSE); (b) the Shannon entropy $e$ of the current token probability distribution; and (c) the current proportion of watermarked tokens $r$. The MLP outputs $\mathbf{m}_{\text{wm}} \in [0,1]$:
    $\mathbf{m}_{\text{wm}} = \mathcal{M}_\theta(\mathcal{E}_{\text{sem}}(s_{n-k+1:n}),\; e,\; r)$
    - Design Motivation: Entropy alone is insufficient—semantic context can identify which word classes (e.g., prepositions, conjunctions) are less suitable for watermarking; the watermarking ratio provides a global control signal.

2. **Differentiable Watermarking Framework**:

    - Function: Relaxes the standard KGW watermarking process into a differentiable form to support gradient-based training.
    - Mechanism: The original logits are modified as $\tilde{\mathbf{l}}_{t+1} = \mathbf{l}_{t+1} + \delta \cdot \mathbf{m}_{\text{wm}}^{[t+1]} \mathbf{m}_{\text{green}}$, where $\delta$ is the watermark strength. During training, $\mathbf{m}_{\text{wm}}$ retains continuous values; at inference time it is discretized via threshold $\tau$: $m_{\text{wm}} = \mathbb{1}[\mathbf{m}_{\text{wm}} > \tau]$.
    - The z-score detection formula is relaxed into a differentiable form:
    $z = \frac{\sum_{t=1}^T p_{gr}^{[t]} \cdot m_{\text{wm}}^{[t]} - \gamma \sum_{t=1}^T m_{\text{wm}}^{[t]}}{\sqrt{\sum_{t=1}^T m_{\text{wm}}^{[t]} \gamma(1-\gamma)}}$

3. **Adaptive Threshold**:

    - Function: Dynamically adjusts the binarization threshold based on the current watermarking ratio.
    - Mechanism: When the watermarking ratio is low, the threshold is lowered so that more tokens are watermarked to ensure detectability; when the ratio is high, the threshold is raised to reduce watermarking and preserve quality.
    - Design Motivation: A static threshold cannot balance the demands of different generation stages; dynamic adjustment maintains a global equilibrium between detectability and quality.

### Loss & Training

Two optimization objectives are jointly optimized via MGDA (Multiple Gradient Descent Algorithm):

**Quality-oriented loss**:
$$\mathcal{L}_Q = \lambda_{\text{sim}} \mathcal{L}_S + \lambda_{\text{entropy}} \mathcal{L}_{\text{entropy}} + \lambda_{\text{fix}} \mathcal{L}_{\text{output\_fix}}$$

- $\mathcal{L}_S = -\cos_{\text{sim}}(E_w, E_s)$: maximizes semantic similarity between watermarked and non-watermarked text.
- $\mathcal{L}_{\text{entropy}} = \text{BCE}(m_{\text{wm}}, \sigma(\lambda_e(e - \mu_e)))$: encourages watermarking at high entropy and non-watermarking at low entropy.
- $\mathcal{L}_{\text{output\_fix}} = -\frac{1}{T}\sum_{t=1}^T (m_{\text{wm}}^{[t]} - 0.5)^2$: regularizes outputs toward binary mask behavior near 0 or 1.

**Detectability-oriented loss**:
$$\mathcal{L}_D = -\lambda_z z + \lambda_{\text{wm}} \mathcal{L}_{\text{wm\_ratio}} + \lambda_{\text{fix}} \mathcal{L}_{\text{output\_fix}}$$

- $-\lambda_z z$: maximizes the z-score.
- $\mathcal{L}_{\text{wm\_ratio}} = \text{MSE}(m_{\text{wm}}^{[t]}, f(r_t))$: encourages more watermarking when the ratio is low and less when it is high.

Training is conducted on OPT-1.3b using the C4 RealNewsLike subset (10k training, 500 test samples).

## Key Experimental Results

### Main Results

Comparison of watermarking methods on OPT-6.7B:

| Method | TPR@2% | AUROC | Best F1 | Perplexity↓ | Semantic Similarity↑ |
|--------|--------|-------|---------|-------------|----------------------|
| KGW | 0.998 | 0.9999 | 0.999 | 20.52 | 0.547 |
| Unigram | 1.000 | 1.0000 | 1.000 | 17.68 | 0.531 |
| EXP-edit | 0.972 | 0.9874 | 0.980 | 23.93 | 0.505 |
| SWEET | 1.000 | 1.0000 | 1.000 | 17.93 | 0.555 |
| TS-watermark | 0.996 | 0.9997 | 0.996 | 14.08 | 0.565 |
| **LTW-1 (Ours)** | **1.000** | **1.0000** | **1.000** | **13.62** | **0.570** |
| **LTW-0 (Ours)** | **1.000** | **1.0000** | **1.000** | **13.62** | **0.574** |

Key finding: LTW achieves perfect detectability (AUROC = 1.0) while attaining a perplexity of only 13.62, substantially lower than KGW (20.52) and SWEET (17.93), and also lower than TS-watermark (14.08).

### Ablation Study — Adaptive Threshold Module

| Configuration | $\delta$=1.5 | $\delta$=2 | $\delta$=2.5 | $\delta$=3 | $\delta$=3.5 | $\delta$=4 |
|---------------|-------------|-----------|-------------|-----------|-------------|-----------|
| With adaptive - PPL | 11.69 | 12.35 | 13.00 | 13.41 | 13.98 | 14.03 |
| With adaptive - z-score | 6.84 | 9.29 | 11.24 | 12.67 | 13.78 | 14.50 |
| Without adaptive - PPL | 11.80 | 12.52 | 13.22 | 13.40 | 13.92 | 14.31 |
| Without adaptive - z-score | 6.83 | 9.03 | 11.12 | 12.44 | 13.57 | 14.37 |

### Key Findings
- LTW achieves a superior Pareto frontier over KGW (higher detection rate + lower perplexity) across all watermark strength settings.
- The adaptive threshold improves the z-score in all 6 out of 6 watermark strength configurations and reduces perplexity in 4 out of 6.
- Network outputs are highly correlated with part-of-speech: the selector tends to avoid watermarking prepositions, conjunctions, punctuation, and symbols (which affect semantic coherence), while preferring to watermark adverbs and adjectives (which have more synonymous substitutes).
- LTW-0 (based on Unigram) demonstrates stronger robustness under paraphrase attacks, as its fixed green/red list assignments are unaffected by changes in the preceding token.
- Results on GPT-J-6B are consistent, with perplexity reduced from 16.55 (KGW) to 9.56 (LTW-0).

## Highlights & Insights
- **Learnable selection strategy**: This work is the first to replace hand-crafted rules with a trained neural network for deciding when to apply watermarks. The network automatically learns to avoid watermarking function words and prefer content words—a strategy aligned with linguistic intuition yet requiring no manual encoding.
- **Elegant application of multi-objective optimization**: MGDA handles the conflicting objectives of detectability and quality, avoiding manual tuning of trade-off hyperparameters. This multi-objective framework is transferable to other AI safety scenarios involving conflicting objectives.
- **Principled training–inference separation**: Continuous masks are used during training to ensure differentiability, while hard binary masks are applied at inference to enhance robustness; the two are seamlessly bridged through output regularization that pushes values toward 0 or 1.

## Limitations & Future Work
- The selector network is trained on OPT-1.3b; whether it transfers optimally to larger models remains unverified.
- The choice of sentence embedding window size involves a trade-off discussed in the paper but is not systematically optimized.
- Under paraphrase attacks, LTW-1 exhibits slightly weaker robustness than the Unigram baseline, indicating room for improvement in the robustness of selective watermarking built on KGW.
- Evaluation is limited to text completion tasks; performance in other settings such as dialogue and code generation remains to be validated.

## Related Work & Insights
- **vs. KGW**: LTW-1 extends KGW with selectivity, reducing perplexity from 20.5 to 13.6 (a 33% reduction) with no loss in detectability.
- **vs. SWEET**: Both employ selective watermarking, but SWEET relies on manual entropy threshold grid search, whereas LTW learns the selection automatically via a network. LTW additionally leverages semantic embeddings and the watermarking ratio as auxiliary information.
- **vs. TS-watermark**: TS-watermark trains generators for $\gamma$ and $\delta$ to adaptively adjust watermarking hyperparameters; LTW approaches the problem by selecting *which tokens* to watermark, more directly protecting semantically critical tokens.
- **vs. NS-watermark**: NS-watermark minimizes the number of watermarked tokens while keeping the z-score just above the threshold, making it highly vulnerable to attacks; LTW maintains a high z-score while reducing quality degradation through selective application.

## Rating
- Novelty: ⭐⭐⭐⭐ Replacing rule-driven selection with a learnable network is a clear and well-motivated contribution; the application of multi-objective optimization is natural and principled.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The evaluation covers two LLMs, multiple baselines, paraphrase attacks, Pareto frontier comparisons across watermark strengths, ablation studies, and part-of-speech analysis—comprehensive overall.
- Writing Quality: ⭐⭐⭐⭐ The method is described in detail with complete derivations, though some notation is slightly redundant.
- Value: ⭐⭐⭐⭐ The work provides a practical and effective improvement to LLM watermarking, an active research area, and the code is publicly available.

## Related Papers

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] On the Empirical Power of Goodness-of-Fit Tests in Watermark Detection](on_the_empirical_power_of_goodness-of-fit_tests_in_watermark_detection.md)
- [\[NeurIPS 2025\] Exploring the Limits of Strong Membership Inference Attacks on Large Language Models](exploring_the_limits_of_strong_membership_inference_attacks_on_large_language_mo.md)
- [\[NeurIPS 2025\] Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values](distributive_fairness_in_large_language_models_evaluating_alignment_with_human_v.md)
- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](../../CVPR2026/llm_safety/multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)
- [\[NeurIPS 2025\] FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models](fedrw_efficient_privacy-preserving_data_reweighting_for_enhancing_federated_lear.md)

<!-- RELATED:END -->
