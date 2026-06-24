---
title: >-
  [Paper Note] Probing LLMs for Multilingual Discourse Generalization Through a Unified Label Set
description: >-
  [ACL2025][Multilingual & Machine Translation][Discourse relation classification] This paper proposes the first cross-framework, cross-lingual unified discourse relation label set (17 categories). Through attention probing experiments on 23 LLMs, it demonstrates that multilingual LLMs can encode cross-linguistically transferable discourse-level representations in their intermediate layers, and that multilingual training combined with model scale enhances generalization capabil…
tags:
  - "ACL2025"
  - "Multilingual & Machine Translation"
  - "Discourse relation classification"
  - "Multilingual generalization"
  - "Unified label set"
  - "LLM probing"
  - "Attention representation"
date: 2026-05-08
content_hash: b6c72f525be3cf80
---

# Probing LLMs for Multilingual Discourse Generalization Through a Unified Label Set

**Conference**: ACL2025  
**arXiv**: [2503.10515](https://arxiv.org/abs/2503.10515)  
**Code**: [mainlp/discourse_probes](https://github.com/mainlp/discourse_probes)  
**Area**: Multilingual Translation  
**Keywords**: Discourse relation classification, Multilingual generalization, Unified label set, LLM probing, Attention representation

## TL;DR

This paper proposes the first cross-framework, cross-lingual unified discourse relation label set (17 categories). Through attention probing experiments on 23 LLMs, it demonstrates that multilingual LLMs can encode cross-linguistically transferable discourse-level representations in their intermediate layers, and that multilingual training combined with model scale enhances generalization capabilities.

## Background & Motivation

**Importance of Discourse Understanding**: Many NLP tasks (reading comprehension, sentiment analysis, summarization generation, etc.) rely on the identification of inter-sentential discourse relations, but most research remains limited to sentence-level analysis.

**Framework Fragmentation**: Existing discourse relation annotations rely on specific theoretical frameworks (RST, PDTB, SDRT, DEP, etc.). Different frameworks use distinct label systems, making datasets incomparable with each other.

**Scarcity of Multilingual Resources**: Framework-specific annotated corpora are concentrated mainly in high-resource languages (English), with minimal coverage of low-resource languages, posing a severe data bottleneck for cross-lingual discourse processing.

**Theoretical Consensus on Core Relations**: Bunt & Prasad (2016) pointed out that a set of "core" discourse relations exists across different frameworks, but previous mapping schemes were limited only to English-French bilingual pairings and two-framework pairs.

**Unexplored Cross-Lingual Potential of LLMs**: While evidence suggests that LLMs can learn cross-linguistically shared grammatical abstractions (such as morphology and syntax), their discourse-level generalization capabilities have barely been systematically investigated.

**Ours Key Insight**: Constructing a unified label set + large-scale multi-model probing experiments to systematically answer for the first time: "Do LLMs encode cross-lingual, cross-framework discourse-level abstractions?"

## Method

### Overall Architecture

**Two-Stage Approach**: (1) First, a cross-framework unified label set is constructed, mapping corpus labels from 26 datasets across 4 frameworks in the DISRPT benchmark to 17 unified categories. (2) Using discourse relation classification as a testbed, attention scores from 23 LLMs are extracted as representations to train probing classifiers for evaluating discourse-level generalization.

### Key Designs

#### Key Design 1: Unified Discourse Relation Label Set

Expanding on the top-level mapping of Benamara & Taboada (2015), 5 major categories and 17 relation labels are defined:
- **temporal**: Maps sequence in RST, Temporal in PDTB, etc.
- **structuring**: list, textual-organization, disjunction, etc.
- **thematic**: Contains 6 subcategories: framing, attribution, mode, reformulation, comparison, elaboration.
- **causal-argumentative**: Contains 6 subcategories: causal, adversative, explanation, evaluation, contingency, enablement.
- **topic-management**: Contains 3 subcategories: topic-adjustment, topic-change, topic-comment.

This label set covers 4 frameworks (RST/PDTB/SDRT/DEP), 13 languages, and 26 datasets for the first time.

#### Key Design 2: Attention Score-Based Probing Method

- Inputting the entire document into a decoder-only LLM in a single forward pass to obtain the full attention matrix $\mathbf{X} \in \mathbb{R}^{L \times H \times N \times N}$.
- For token indices of two discourse units $s_1, s_2$, extracting **cross-unit attention** $\mathbf{C}$ and **intra-unit attention** $\mathbf{D}_1, \mathbf{D}_2$ respectively, and compressing them to a fixed length using max pooling.
- Concatenating $(\mathbf{D}_1, \mathbf{D}_2, \mathbf{C})$, flattening the result, and feeding it into a two-layer MLP probe (tanh + Sigmoid) for relation classification.
- Applying a sliding window strategy with a step size of half the window for ultra-long documents (>4000 tokens).

#### Key Design 3: Multi-Dimensional Train-Test Settings

Three training conditions are designed to systematically evaluate generalization:
- **mono-probe**: Trained using monolingual data only.
- **multi-lang-probe**: Trained using languages from the same language family (e.g., Germanic, Romance).
- **multi-all-probe**: Trained using a mix of all 13 languages.

#### Key Design 4: Layer-by-Layer Analysis

In addition to the all-layer combined probe, layer-by-layer probes are trained to analyze variations of discourse representations at different depths of the model, revealing at which layers cross-lingual alignment is most prominent.

## Key Experimental Results

### Main Results

#### Table 1: Model Scale and Multilingual Coverage vs. Probe Accuracy (Selected Core Results)

| Model | Parameters | DISRPT Language Coverage | Probe Accuracy |
|------|--------|----------------|------------|
| BLOOM-560m | 0.56B | 54% | ~35% |
| Llama3-3B | 3B | 23% | 49.1% |
| Phi-4 (English) | 14B | 8% | ~51% |
| Qwen2.5-14B | 14B | 77% | ~54% |
| Aya-23-35B | 35B | 85% | **58.2%** |
| Llama3-72B | 72B | 23% | ~56% |

Reference system DisCoDisCo (fine-tuned xlm-roberta-base): 47.9%

#### Table 2: Aya-23-35B Cross-Lingual Generalization (mono vs. multi-all)

| Language | mono-probe | multi-all-probe | Difference |
|------|------------|-----------------|------|
| Persian | - | +2.0% | ↑ |
| Russian | - | +1.2% | ↑ |
| Turkish | - | +6.9% | ↑ |
| Chinese | - | +2.0% | ↑ |
| French | - | +1.7% | ↑ |
| Thai | Decreased in multi-all | -3.2% | ↓ |
| Basque | Unchanged | 0% | — |

### Key Findings

1. **Probe > Fine-tuned System**: A simple attention probe on Llama3-3B (49.1%) surpasses the fine-tuned DisCoDisCo (47.9%), with the best probe (Aya-23-35B) exceeding it by 10.3 percentage points.
2. **Key Role of Multilingual Training**: The English monolingual model Phi-4 (14B) lags behind the multilingual model Qwen2.5-14B of the same scale, proving that multilingual training is more important than pure scaling.
3. **Best Alignment in Intermediate Layers**: Layer-by-layer analysis shows that discourse representations are most cross-linguistically transferable in layers 10-15, with accuracy dropping by 5-10% in higher layers.
4. **Effective Transfer within Language Families**: The Romance family multi-lang-probe improves Spanish by 6.5% and Italian by 3.1%, even though these languages only account for 1-2% of the training data.
5. **Cross-Lingual Generalization of Monolingual English Probes**: The English mono-probe achieves 48.2% on Portuguese and 49% on Spanish, proving the existence of a cross-lingual discourse "concept space" within the model.
6. **BLOOM Series Fully Lags Behind**: Despite multilingual training, BLOOM probes score lower than DisCoDisCo, which can be attributed to training data quality and tokenizer effects.

## Highlights & Insights

- **Pioneering Nature**: The first large-scale experimental study applying a unified label set to multilingual discourse relation classification.
- **Lightweight and Effective Method**: Using only attention scores (without fine-tuning), a single forward pass can encode multiple relational pairs across the entire document.
- **Linguistic Findings**: The hypothesis of an "English-aligned concept space" in the intermediate layers is validated at the discourse level for the first time.
- **Counter-Intuitive Discovery**: Aya-23-35B (35B) outperforms Qwen2.5-70B and Llama3-72B, indicating that training data quality and multilingual optimization are more important than parameters.

## Limitations & Future Work

1. **Data Imbalance**: English accounts for 53.5% in DISRPT, resulting underrepresentation of low-resource languages.
2. **Unified Labels Simplify Framework Differences**: Collapsing fine-grained relations into 17 categories inevitably loses theoretical nuances.
3. **Simple Probing Method**: Using only a two-layer MLP, leaving the upper-bound potential of stronger classifiers or fine-tuned LLMs unexplored.
4. **Unaddressed Segmentation Differences**: Different frameworks have varying discourse unit segmentation standards, which may affect the probing performance of relations such as attribution.
5. **Efficiency Bottlenecks**: The memory footprint of the attention matrix grows quadratically with the token count, limiting the maximum document length to 4000 tokens.

## Related Work & Insights

- **Comparison with Koto et al. (2021)**: The former used residual streams for framework-dependent coherence probes, whereas this work uses attention for cross-framework unified probes.
- **Comparison with Kurfalı & Östling (2021)**: The former performed single-framework multilingual probing on mBERT/XLM-R, whereas this work extends to 23 decoder-only LLMs and a unified label set.
- **Analogy to Universal Dependencies**: Just as UD unified syntactic annotation, this work takes the first step toward unifying discourse relations.
- **Insights**: The unified label set approach can be generalized to other fragmented fields (such as dialogue acts, rhetorical structures), and the layer-by-layer analysis concept can be used to locate the encoding positions of other linguistic phenomena within models.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of a unified label set and large-scale attention probing is a first in the field)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (23 models, 13 languages, with extremely comprehensive multi-dimensional analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and rich tables/figures, but mapping details are scattered in the appendix)
- Value: ⭐⭐⭐⭐ (Provides critical references for both multilingual discourse processing and LLM interpretability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs](a_case_study_of_cross-lingual_zero-shot_generalization_for_classical_languages_i.md)
- [\[ACL 2025\] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](crosslingual_pitfalls.md)
- [\[ACL 2025\] Cross-Lingual Generalization and Compression: From Language-Specific to Shared Neurons](cross_lingual_neurons_compression.md)
- [\[ACL 2025\] The Hidden Space of Safety: Understanding Preference-Tuned LLMs in Multilingual Contexts](the_hidden_space_of_safety_understanding_preference-tuned_llms_in_multilingual_c.md)
- [\[ACL 2025\] Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning](cross-lingual_representation_alignment_through_contrastive_image-caption_tuning.md)

</div>

<!-- RELATED:END -->
