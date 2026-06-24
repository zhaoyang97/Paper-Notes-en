---
title: >-
  [Paper Note] Robust Hallucination Detection in LLMs via Adaptive Token Selection
description: >-
  [NeurIPS 2025][Hallucination Detection][multiple instance learning] HaMI frames hallucination detection as a Multiple Instance Learning (MIL) problem, treating each generated sequence as a bag of token instances. By jointly optimizing token selection and hallucination detection, it adaptively identifies the most informative tokens, achieving substantial AUROC improvements over all existing methods across four QA benchmarks (up to 11.9%).
tags:
  - "NeurIPS 2025"
  - "Hallucination Detection"
  - "multiple instance learning"
  - "adaptive token selection"
  - "internal representation"
  - "predictive uncertainty"
date: 2026-05-08
content_hash: 5dd0c526bc77dcc2
---

# Robust Hallucination Detection in LLMs via Adaptive Token Selection

**Conference**: NeurIPS 2025
**arXiv**: [2504.07863](https://arxiv.org/abs/2504.07863)  
**Code**: [https://github.com/mala-lab/HaMI](https://github.com/mala-lab/HaMI)  
**Area**: Hallucination Detection
**Keywords**: hallucination detection, multiple instance learning, adaptive token selection, internal representation, predictive uncertainty

## TL;DR

HaMI frames hallucination detection as a Multiple Instance Learning (MIL) problem, treating each generated sequence as a bag of token instances. By jointly optimizing token selection and hallucination detection, it adaptively identifies the most informative tokens, achieving substantial AUROC improvements over all existing methods across four QA benchmarks (up to 11.9%).

## Background & Motivation

Hallucination in LLMs is a central obstacle to their safe deployment—models may generate content that appears plausible yet is factually unfaithful or incorrect. Existing detection methods fall into two main categories:

**Uncertainty-based methods** (e.g., Semantic Entropy, Perplexity, MARS): rely on predicted probabilities or semantic consistency across multiple samples, but performance is bounded by the capability of auxiliary LLMs.

**Internal representation-based methods** (e.g., SAPLMA, HaloScope, CED): train binary classifiers on LLM hidden-state representations, but are heavily dependent on predefined token positions.

**Key Challenge**: Most internal representation methods use representations at predefined token positions (e.g., the first, last, or second-to-last token) to train detectors. However, as shown in Figure 1, the positions carrying the richest hallucination signal vary substantially with response length and the distribution of hallucinated entities. Predefined positions thus miss the critical tokens where hallucination information concentrates.

**Key Insight**: Hallucinations typically occur at only a small number of token positions within a response (e.g., incorrect entity nouns), which naturally aligns with the MIL assumption—positive bags contain only a few positive instances, while all instances in negative bags are negative. Reformulating hallucination detection as a MIL problem enables the model to adaptively learn, during training, which tokens are most indicative of hallucination.

## Method

### Overall Architecture

The HaMI framework consists of two main modules:

1. **MIL-driven Adaptive Token Selection (ATS)**: Each generated sequence is treated as a bag, with a bag-level label of hallucinated or faithful. Token selection and hallucination detection are jointly optimized via a MIL loss.
2. **Predictive Uncertainty Enhancement Module**: Multi-granularity uncertainty information is integrated into token-level internal representations to improve discriminability.

### Key Designs

1. **MIL Formulation**:

    - Positive bag $\mathcal{B}^+$ (responses containing hallucinations): only a small subset of tokens are true hallucination instances (positive instances).
    - Negative bag $\mathcal{B}^-$ (correct responses): all tokens are negative instances.
    - The hallucination detector $f_\theta$ assigns a hallucination score to each token.
    - The top-$k$ tokens with the highest hallucination scores in each bag are selected as salient tokens ($k = \lfloor 0.1 \times l \rfloor + 1$, where $l$ is the sequence length).

2. **MIL Loss**:

    - Maximizes the discriminative margin between salient tokens from positive bags and the hardest negative tokens from negative bags.
    - $\mathcal{L}_{MIL} = 1 - \|\frac{1}{k}\sum_{i^+ \in \mathcal{I}_{top-k}^+} f_\theta(h_{i^+})\|_2 + \|\frac{1}{k}\sum_{i^- \in \mathcal{I}_{top-k}^-} f_\theta(h_{i^-})\|_2$
    - Top-$k$ token scores in positive bags should be high; those in negative bags should be low.

3. **Smoothness Regularization**:

    - Exploiting the sequential nature of token generation, hallucination scores of adjacent tokens should vary smoothly.
    - $\mathcal{L}_{Smooth} = (f_\theta(h_i) - f_\theta(h_{i-1}))^2$
    - Total loss: $\mathcal{L}_{ATS} = \mathcal{L}_{MIL} + \mathcal{L}_{Smooth}$

4. **Predictive Uncertainty Enhancement**:

    - Three granularities of uncertainty:
     - Token-level: predicted probability $P^t$
     - Sentence-level: perplexity $P^s$
     - Semantic consistency-level: semantic entropy from multiple samples $P^c$
    - Enhancement formula: $h' = (1 + \lambda \cdot P_{\text{uncertainty}}) \cdot h$
    - Semantic consistency $P^c$ is used by default, with $\lambda = 1.0$.

### Loss & Training

- The detector is a two-layer MLP with hidden dimension 256.
- Four benchmark datasets: TriviaQA, SQuAD, Natural Questions, BioASQ.
- Each dataset: 2,000 QA pairs for training and 800 for testing.
- Multiple sampling: each question is presented to the LLM 6 times.
- Correctness labels are determined by GPT-4.1.
- Evaluation metric: AUROC.
- Models: LLaMA-3.1-8B, Mistral-Nemo-Instruct (12B), LLaMA-3.3-Instruct-70B.

## Key Experimental Results

### Main Results (AUROC)

**Comparison on LLaMA-3.1-8B**:

| Method | TriviaQA | SQuAD | NQ | BioASQ | Note |
|--------|----------|-------|-----|--------|------|
| Perplexity | 0.732 | 0.649 | 0.659 | 0.709 | Baseline |
| Semantic Entropy | 0.828 | 0.787 | 0.773 | 0.757 | Strong baseline |
| SAPLMA | 0.835 | 0.769 | 0.781 | 0.821 | Internal repr. |
| MARS-SE | 0.824 | 0.780 | 0.777 | 0.744 | Enhanced |
| HaMI* (Ours) | 0.854 | 0.783 | 0.788 | 0.823 | w/o uncertainty enhancement |
| HaMI (Ours) | **0.897** | **0.826** | **0.820** | **0.836** | Full model |

**Comparison on LLaMA-3.3-Instruct-70B**:

| Method | TriviaQA | SQuAD | NQ | BioASQ |
|--------|----------|-------|-----|--------|
| Semantic Entropy | 0.819 | 0.643 | 0.769 | 0.772 |
| SAPLMA | 0.842 | 0.672 | 0.817 | 0.748 |
| HaMI (Ours) | **0.891** | **0.774** | **0.846** | **0.825** |

### Ablation Study

**Adaptive Token Selection vs. Predefined Token Positions**:

| Token Position Strategy | TriviaQA | SQuAD | Avg. Gain |
|------------------------|----------|-------|-----------|
| First token | 0.849 | 0.774 | — |
| Before Last | 0.878 | 0.778 | +1.6% |
| Last token | 0.890 | 0.804 | +3.3% |
| ATS (Ours) | **0.897** | **0.826** | **+5.0%** |

**Effect of Uncertainty Enhancement**:

| Uncertainty Type | TriviaQA | SQuAD | Note |
|-----------------|----------|-------|------|
| Original (no enhancement) | 0.854 | 0.783 | HaMI* |
| Token-level $P^t$ | 0.856 | 0.782 | Marginal gain |
| Sentence-level $P^s$ | 0.871 | 0.787 | Moderate gain |
| Semantic consistency $P^c$ | **0.897** | **0.826** | Largest gain (+6.7%) |

### Key Findings

- HaMI consistently outperforms all baselines across three LLMs; the advantage is more pronounced on the 70B model (average +11.5% over SE).
- In cross-dataset generalization experiments, HaMI's performance drop on unseen datasets does not exceed 4.5%, far better than competing methods.
- Human evaluation shows that the ATS module achieves a recall of 0.84 for hallucinated tokens without any token-level supervision.
- The last token is the best predefined position choice, yet ATS outperforms it in all settings and is more robust.
- Semantic consistency enhancement ($P^c$) yields the best results; however, sentence-level perplexity ($P^s$) still surpasses all external methods requiring multiple samples (e.g., SE) while needing only a single generation pass.

## Highlights & Insights

- **Elegant MIL Formulation**: The intrinsic nature of hallucination detection—"only a few tokens in a positive sequence are hallucinated"—maps perfectly onto the MIL assumption, making this modeling choice both natural and theoretically well-motivated.
- **End-to-End Joint Optimization**: This work is the first to jointly optimize token selection and hallucination detection, avoiding the token-selection mismatch inherent in two-stage pipelines.
- **Smoothness Regularization**: Leveraging the sequential property of autoregressive generation, the inductive bias that adjacent token hallucination scores should vary continuously is simple yet effective.
- **High Practical Value**: HaMI* (without multiple sampling) already surpasses most methods that require auxiliary LLM assistance, resulting in low deployment overhead.

## Limitations & Future Work

- Semantic consistency enhancement $P^c$ requires multiple sampling passes and an external NLI model to judge semantic equivalence, increasing inference cost.
- Training relies on correctness labels from GPT-4.1, so label quality is bounded by GPT-4.1's capabilities.
- Evaluation is limited to QA tasks; applicability to other hallucination scenarios such as summarization and dialogue remains unverified.
- The choice of $k$ (top 10% + 1) is heuristic and may require task-specific tuning.
- The detector uses internal representations from a fixed layer; selecting the optimal layer requires validation-set hyperparameter search.

## Related Work & Insights

- Compared directly with SAPLMA (MLP probe + correctness labels): HaMI replaces fixed token positions with MIL, yielding a 5–8% performance improvement.
- Complementary to Semantic Entropy (multi-sample semantic consistency): HaMI can leverage its $P^c$ measure to enhance internal representations.
- Distinct from HaloScope (membership estimation scores as training signal): HaMI focuses on adaptive token-level selection rather than introducing a new training signal.
- MIL has been widely applied in medical image analysis (e.g., whole slide image classification); this work is the first to introduce it into NLP hallucination detection.
- Insight: The "bag of instances" perspective at the token level may generalize to other sequence-level tasks that require locating key positions, such as fact verification and attribution analysis.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT](beyond_token_probes_hallucination_detection_via_activation_tensors_with_act-vit.md)
- [\[ICML 2026\] Automatic Layer Selection for Hallucination Detection](../../ICML2026/hallucination/automatic_layer_selection_for_hallucination_detection.md)
- [\[ACL 2025\] Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucination in Multimodal LLMs](../../ACL2025/hallucination/mixture_of_decoding_an_attention-inspired_adaptive_decoding_strategy_to_mitigate.md)
- [\[ICLR 2026\] Cat-PO: Cross-modal Adaptive Token-rewards for Preference Optimization in Truthful Multimodal LLMs](../../ICLR2026/hallucination/cat-po_cross-modal_adaptive_token-rewards_for_preference_optimization_in_truthfu.md)
- [\[CVPR 2026\] Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations](../../CVPR2026/hallucination/beyond_global_scores_fine_grained_token_grounding_as_robust_detector_of_lvlm_hallucinations.md)

</div>

<!-- RELATED:END -->
