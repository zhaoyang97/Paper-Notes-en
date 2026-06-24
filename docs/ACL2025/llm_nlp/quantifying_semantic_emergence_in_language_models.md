---
title: >-
  [Paper Note] Quantifying Semantic Emergence in Language Models
description: >-
  [ACL 2025][LLM (Other)][Information Emergence] Proposed Information Emergence (IE), an information-theoretic quantitative metric that quantifies the ability of LLMs to extract semantics from tokens by comparing the difference between macro (sequence-level) and micro (token-level) mutual information across Transformer layers.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Information Emergence"
  - "Semantic Understanding"
  - "Mutual Information"
  - "Large Language Models"
  - "Interpretability"
date: 2026-05-08
content_hash: 1f72fdf400eabc92
---

# Quantifying Semantic Emergence in Language Models

**Conference**: ACL 2025  
**arXiv**: [2405.12617](https://arxiv.org/abs/2405.12617)  
**Code**: [github](https://github.com/Zodiark-ch/Emergence-of-LLMs)  
**Area**: LLM/NLP  
**Keywords**: Information Emergence, Semantic Understanding, Mutual Information, Large Language Models, Interpretability

## TL;DR

Proposed Information Emergence (IE), an information-theoretic quantitative metric that quantifies the ability of LLMs to extract semantics from tokens by comparing the difference between macro (sequence-level) and micro (token-level) mutual information across Transformer layers.

## Background & Motivation

### Key Challenge

**Key Challenge**: **Background**: Large language models (LLMs) are widely recognized for their excellent semantic understanding capabilities, but there is currently a lack of a **quantitative and task-agnostic** metric to measure this capacity.

Limitations of existing evaluation methods:

**Task Dependency**: Accuracy on tasks such as instruction following, search, and reasoning can only indirectly reflect semantic understanding capabilities, and dataset construction is time-consuming.

**Coarse Granularity**: Existing evaluations typically focus on text-level performance, failing to provide explanations for finer-grained token behaviors.

**Inconsistent Metrics**: Different tasks use different evaluation metrics, which can lead to contradictory conclusions.

Therefore, the authors propose a **closed-form, task-agnostic** metric, Information Emergence (IE), to deterministically quantify the ability of LLMs to extract meaningful semantics from tokens.

## Method

### Overall Architecture

The core idea stems from the concept of "emergence" in information theory: semantics is the meaningful organization presented by a set of tokens at the macro level, which is unobservable at the micro (individual token) level but observable at the macro (entire sequence) level. The authors analogize the transmission of token representations between Transformer blocks to a Markov process and quantify the difference in entropy reduction between the macro and micro levels using mutual information.

### Key Designs

1. **Markov Process Analogy**: The NTP (Next-Token Prediction) mechanism is treated as a Markov random process. For the $l$-th layer of the Transformer, the output representation $h_{l+1}^t$ of token $t$ depends on all input representations at positions $\leq t$ in the $l$-th layer. Micro variables (such as $h^0$) depend only on themselves, while macro variables (such as $h^{T-1}$) aggregate information from all preceding tokens.

2. **Definition of Information Emergence (IE)**: For the $l$-th Transformer block, IE is defined as the difference between the macro mutual information and the mean of micro mutual information:
$$E(l) = MI(h_{l+1}^{ma}, h_l^{ma}) - \frac{1}{T}\sum_{t=0}^{T-1} MI(h_{l+1}^{mi\_t}, h_l^{mi\_t})$$
$E(l) > 0$ indicates that the reduction in uncertainty (entropy reduction) of the layer over the entire sequence is greater than that of individual tokens, meaning the model successfully captures collective semantics.

3. **Computation of Micro Variables**: To ensure that micro variables depend only on themselves, each token is fed into the model individually as an input sequence, avoiding the contextual influence introduced by the autoregressive mechanism. Conversely, the macro variable is taken as the representation of the final token in the complete sequence.

### Loss & Training

A lightweight mutual information estimator (a 10-layer linear + LeakyReLU network) is utilized to approximate the KL divergence in high-dimensional continuous spaces. Based on the MINE method (Belghazi et al.), mutual information is estimated by optimizing a contrastive objective function:
- Positive samples: Representation pairs from adjacent layers in the same sequence $(h_{l+1,s}^{ma}, h_{l,s}^{ma})$
- Negative samples: Representation pairs from different sequences $(h_{l+1,s}^{ma}, h_{l,s'}^{ma})$
- The batch size is set to 300,000, with the learning rate polynomially decaying from 1e-4 to 1e-8 over 10k epochs.

## Key Experimental Results

### Main Results

**ICL Scenario (Synthetic Datasets)**:

| Dataset | Entities | Sample Size | Token Length | Shots |
|--------|--------|--------|-----------|---------|
| Country | 25 | 303,600 | 8 | 4 |
| Animal | 16 | 524,160 | 10 | 5 |
| Color | 15 | 360,360 | 10 | 5 |

**Natural Sentence Scenario**: 300,000 natural sequences (each with 8 tokens) were randomly selected from OpenOrca and OpenHermes respectively.

**Models Evaluated**: GPT2-large (812M), GPT2-XL (1.6B), GEMMA (2.51B), OpenLlama (3B).

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| ICL vs. Natural Text | Differences in IE growth patterns | Under ICL, IE only increases when a new demonstration is introduced; under natural text, it increases token-by-token |
| Model Size Growth | Rise in IE value | Positively correlated with the number of model parameters, aligning with the intuition that larger models possess stronger semantic capturing capabilities |
| ICL Shots Saturation | ~7th shot | The three ICL categories tend to saturate at the 7th demonstration |
| Human vs. LLM Text | Lower human IE | The IE values of LLM-generated texts are significantly higher than those of human-written texts (GPT-4: ~39.2 vs. Human: ~19.4) |

### Key Findings

1. **ICL Semantic Enhancement Mechanism**: ICL enhances semantic certainty via demonstrations but eventually saturates after a certain threshold. Increasing the token length within each demonstration does not alter this "stepwise ascent" pattern.

2. **Correlation Between IE and Hallucinations**: When IE stops growing and the standard deviation peaks, LLMs are more prone to generating erroneous responses (e.g., repeating mistakes). This aligns with existing research on hallucinations: LLMs find it difficult to self-correct after an error has been generated.

3. **Differentiating Human and LLM Texts**: Texts generated by different LLMs (GPT-4, Claude3, Llama3) exhibit distinct IE values and growth patterns, making it possible to distinguish the source of the text using IE—even without computing the target LLM's own Transformer representations.

## Highlights & Insights

- **Outstanding Theoretical Contribution**: For the first time, the concept of emergence from information theory is systematically applied to the quantitative measurement of semantic understanding in LLMs.
- **Practical Value**: The proposed lightweight estimator does not require access to the LLM's internal states, enabling smaller models to estimate the IE values of larger or closed-source models.
- **Cross-Domain Insights**: Findings regarding IE offer fresh analytical perspectives for various directions, including ICL mechanisms, hallucination detection, and AI-generated text detection.
- **Correlation with Emergence**: Within the range of $10^8$ to $10^{10}$ parameters, IE exhibits a sharp increase similar to that seen in downstream task performance.

## Limitations & Future Work

1. **Position Sensitivity**: It requires specific meanings at each token position (e.g., beginning/end of a sentence), which may limit interpretability when directly applied to current tasks.
2. **Substantial Sample Size Requirements**: It requires over 300,000 samples to ensure accurate estimation of the joint and marginal distributions of high-dimensional continuous representations.
3. **Limitations in Model Scale and Text Length**: Due to computing resource constraints, the method has not been validated on larger models and longer text sequences.
4. **Causality Not Established**: Only a correlation has been shown between IE and hallucinations so far, without proving a causal relationship.

## Related Work & Insights

- **Information Theory Foundations**: It draws upon the information emergence theory of Rosas et al. and the mutual information estimation methodology of MINE (Belghazi et al.).
- **Distinction from LLM Emergence**: It explicitly distinguishes between information emergence (a quantifiable phenomenon of macro-micro discrepancy) and LLM emergence (abilities that manifest in large models but are absent in smaller ones).
- **Inspirational Directions**: The token-level analysis of IE can be highly useful in understanding attention mechanisms, inter-layer information flow, and differences in semantic extraction capabilities across various architectures.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to propose a task-agnostic quantitative metric for semantic understanding, integrating the information-theoretic concept of emergence into LLM analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ Encompasses both ICL and natural sentence scenarios across multiple models, though model scales remain constrained.
- Writing Quality: ⭐⭐⭐⭐ Displays clear logic from theory to experiments with rigorous mathematical derivation, despite a high density of formulas.
- Value: ⭐⭐⭐⭐ Delivers novel tools and perspectives for LLM interpretability, hallucination detection, and AI-generated text identification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models](a_semantic-aware_layer-freezing_approach_to_computation-efficient_fine-tuning_of.md)
- [\[ACL 2025\] Semantic Exploration with Adaptive Gating for Efficient Problem Solving with Language Models](semantic_exploration_adaptive_gating.md)
- [\[ICML 2026\] Emergence of Hierarchical Emotion Organization in Large Language Models](../../ICML2026/llm_nlp/emergence_of_hierarchical_emotion_organization_in_large_language_models.md)
- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)
- [\[ACL 2025\] Argument Mining in the Age of Large Language Models](argument_mining_in_the_age_of_large_language_models.md)

</div>

<!-- RELATED:END -->
