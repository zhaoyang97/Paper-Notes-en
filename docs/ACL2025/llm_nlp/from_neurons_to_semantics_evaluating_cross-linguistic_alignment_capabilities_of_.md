---
title: >-
  [Paper Note] From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment
description: >-
  [ACL 2025][LLM (Other)][Cross-lingual alignment] A cross-lingual alignment evaluation framework, NeuronXA, is proposed based on neuron activation states. By utilizing FFN layer neuron states as translation-invariant internal representations to measure the cross-lingual alignment capability of multilingual LLMs, it achieves a Pearson correlation of 0.9556 with downstream task performance using only 100 parallel sentence pairs.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Cross-lingual alignment"
  - "neuron states"
  - "multilingual LLMs"
  - "FFN analysis"
  - "semantic retrieval"
date: 2026-05-08
content_hash: 448e0d615ce424c4
---

# From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment

**Conference**: ACL 2025  
**arXiv**: [2507.14900](https://arxiv.org/abs/2507.14900)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Cross-lingual alignment, neuron states, multilingual LLMs, FFN analysis, semantic retrieval

## TL;DR

A cross-lingual alignment evaluation framework, NeuronXA, is proposed based on neuron activation states. By utilizing FFN layer neuron states as translation-invariant internal representations to measure the cross-lingual alignment capability of multilingual LLMs, it achieves a Pearson correlation of 0.9556 with downstream task performance using only 100 parallel sentence pairs.

## Background & Motivation

Large language models (LLMs) demonstrate strong multilingual capabilities, but evaluating cross-lingual alignment remains under-investigated. Existing alignment evaluation methods mainly rely on similarity in sentence embedding spaces (such as cosine similarity). However, they suffer from a fundamental issue: neural network models (e.g., BERT, GPT) tend to produce anisotropic representation spaces, leading to representation collapse and diminishing the semantic expression capability of low-resource languages, which limits the reliability of embedding-based cross-lingual alignment evaluation.

The key inspiration of this study comes from a neuroscience finding: similar information activates overlapping neural regions. The authors hypothesize that neuron activations in FFN layers can serve as an intrinsic representation of multilingual inputs, providing a more structured and robust means of capturing cross-lingual knowledge. Prior research indicates that neurons in FFN modules encode various forms of knowledge (factual knowledge, positional information, syntactic triggers, etc.), which provides a theoretical foundation for using neuron states to estimate cross-lingual alignment.

## Method

### Overall Architecture

The NeuronXA framework consists of three core steps:
1. Neuron State Detection: Extraction of neuron activation information from FFN layers.
2. Sentence Representation Construction: Securing sentence-level neuron states via position-weighted averaging.
3. Alignment Score Calculation: Computing weak alignment ratios based on the cosine similarity matrix.

### Key Designs

1. **Neuron States Detection**: Two detection methods are proposed:

    - **NAS (Neuron Activation State)**: Binarized activation states, where values > 0 are active ($1$) and $\le 0$ are inactive ($0$). This reflects the immediate response of neurons to inputs.
    - **NAV (Neuron Activation Value)**: Employs the absolute value of neuron activation to reflect the contribution scale of neurons to the FFN layer output, serving as a more refined functional metric.

2. **Sentence Representation**: Addressing the causal self-attention mechanism in decoder-only LLMs, a position-weighted averaging strategy is utilized instead of simple averaging: $N_l = \sum_{t=1}^{T} w_t n_{lt}$, where $w_t = \frac{t}{\sum_{k=1}^{T}k}$. Tokens in later positions are assigned higher weights to mitigate the over-representation of early tokens under causal attention.

3. **NeuronXA Alignment Score**: Generates a cosine similarity matrix $C(l)$ and calculates the proportion of parallel sentences satisfying weak alignment: $\mu_{C(l)} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}(c_{ii} > \{c_{ij}, c_{ji}\}_{j \neq i})$. This checks whether each pair of parallel sentences are mutual nearest neighbors. Average pooling across layers is conducted to obtain the final alignment score.

4. **Two Alignment Evaluation Methods**:

    - **NASCA**: Calculates alignment scores based on binarized neuron activation states.
    - **NAVCA**: Calculates alignment scores based on the absolute values of neuron activation.

### Loss & Training

This paper introduces an evaluation methodology and does not involve training. Evaluation utilizes off-the-shelf pre-trained LLMs (LLaMA, Qwen, Mistral, GLM, OLMo series); the alignment evaluation can be completed using only 100 parallel sentence pairs.

## Key Experimental Results

### Main Results

#### Parallel Sentence Retrieval

| Representation Method | Direction | FLORES-200 (Head) | FLORES-200 (Long-tail) | Tatoeba (Head) | Tatoeba (Long-tail) |
|---|---|---|---|---|---|
| Embedding | En⇔xx | 83.78 | 40.95 | 16.86 | 10.12 |
| NAS | En⇔xx | **87.07** | **42.20** | **57.78** | **32.47** |

NAS consistently outperforms traditional sentence embedding on bidirectional retrieval, particularly achieving massive gains on Tatoeba ($16.86 \rightarrow 57.78$).

#### Alignment-Downstream Task Correlation

| Method | XNLI Correlation | BMLAMA-53 Correlation | Multilingual Benchmark Avg Correlation |
|---|---|---|---|
| MEXA | 0.8370 | 0.7463 | 0.8291 |
| NASCA | **0.9326** | **0.7701** | **0.8312** |
| NAVCA | 0.8937 | **0.8065** | 0.8191 |

The Pearson correlation of NASCA with XNLI zero-shot transfer performance reaches 0.9326, and with BMLAMA-53 reaches 0.7701.

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| NAS vs Embedding Retrieval | +40.92% (Tatoeba En⇔xx) | NAS representation far outperforms embedding representation |
| Directional Symmetry | NAS is almost symmetric | Embedding directional difference reaches 30.73% |
| 100 pairs vs More sentences | Stably high correlation | 100 parallel sentence pairs are sufficient |

### Key Findings

1. **Mitigation of Directional Asymmetry**: Traditional embeddings exhibit differences as high as 30.73% between En $\rightarrow$ xx and xx $\rightarrow$ En directions on Tatoeba. In contrast, the NAS representation almost entirely mitigates this asymmetry, indicating that it captures cross-lingual semantics more effectively.

2. **Layer-wise Alignment Dynamics**: Alignment scores peak in middle layers and are lowest in the preliminary and final layers. Lower layers primarily map different languages to a shared semantic space centered around high-resource languages, while higher layers project the semantic content back onto language-specific vocabularies.

3. **High-to-Low Resource Gap**: High-resource language pairs (e.g., Italian $\rightarrow$ French, NASCA = 0.8372) perform far better than low-resource language pairs (e.g., Gujarati $\rightarrow$ Banjar, NASCA = 0.2191). However, the improvement brought by the NAS representation is more pronounced for low-resource languages.

4. **Consistency Across Models**: NeuronXA evaluation remains consistent and effective across multiple models including LLaMA, Qwen, Mistral, GLM, and OLMo.

## Highlights & Insights

1. **Interdisciplinary Inspiration**: Intelligently borrows the neuroscience finding that "similar stimuli activate overlapping neural circuits", presenting a novel perspective for evaluating cross-lingual alignment in NLP.
2. **Exceptional Efficiency**: High-quality evaluation is achieved with only 100 parallel sentence pairs, significantly reducing evaluative costs.
3. **Elimination of Directional Bias**: The NAS representation achieves near-symmetrical retrieval performance, solving a prominent flaw of embedding-based methods.
4. **Beneficial for Low-Resource Languages**: The NAS representation space is smoother, mitigating the negative impact of representation collapse on low-resource languages.
5. **Insights from Layer-wise Analysis**: Unveils the inner multilingual processing mechanism of LLMs, demonstrating that middle layers are critical for semantic alignment.

## Limitations & Future Work

1. Currently, the study focuses exclusively on neurons in FFN layers, leaving the contributions of attention layers unexplored.
2. English is used solely as the pivot language; the effectiveness of utilizing other high-resource languages as pivots remains uninvestigated.
3. There is a lack of deep theoretical explanations as to why the NAS representation space is smoother.
4. The evaluation only covers models within the 7B-14B scale; the effectiveness on larger or smaller models remains to be verified.
5. The position-weighted averaging strategy is heuristic and may not be the optimal sentence representation method.

## Related Work & Insights

- **Cross-lingual Alignment**: Embedding similarity-based methods like MEXA suffer from anisotropy issues.
- **Neuron Analysis**: Prior works such as Dai et al. 2022 demonstrate that FFN neurons encode diverse types of knowledge.
- **Multilingual LLM Inner Mechanisms**: Wendler et al. 2024 discover the existence of latent languages.
- The neuron-state representation paradigm proposed in this paper can be extended to other scenarios requiring intrinsic representation, such as model interpretability and knowledge probing.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Assessing cross-lingual alignment through neuron activation states is a completely new perspective, though the methodology itself is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple models, datasets, and languages. The correlation analysis is comprehensive, including both retrieval and transfer tasks.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, natural motivation, and rich visualizations.
- **Value**: ⭐⭐⭐⭐ Provides an efficient and effective evaluation tool for cross-lingual alignment, offering practical guidance for multilingual LLM research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Style Alignment in Cross-Cultural Translation](towards_style_alignment_in_cross-cultural_translation.md)
- [\[ACL 2025\] Cross-Modal Alignment for LLM-Enhanced Spoken Language Understanding](cross-modal_alignment_for_llm-enhanced_spoken_language_understanding.md)
- [\[ACL 2025\] DeAL: Decoding-time Alignment for Large Language Models](deal_decoding_time_alignment.md)
- [\[ACL 2025\] Binary Classifier Optimization for Large Language Model Alignment](bco_binary_classifier_alignment.md)
- [\[ACL 2025\] Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models](gapo_multi_objective_alignment.md)

</div>

<!-- RELATED:END -->
