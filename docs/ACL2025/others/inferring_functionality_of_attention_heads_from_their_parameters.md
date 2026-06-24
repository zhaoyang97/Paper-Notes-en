---
title: >-
  [Paper Note] Inferring Functionality of Attention Heads from their Parameters
description: >-
  [ACL 2025][Attention heads] Proposes the MAPS framework, which constructs a token mapping matrix $M$ by projecting attention head parameters into the vocabulary space. MAPS infers the functions realized by attention heads without requiring any forwarding inference or training. The mapping accuracy is validated across 20 relational operations on 6 LLMs, and an automated pipeline is developed to discover numerous previously unidentified attention head functions.
tags:
  - "ACL 2025"
  - "Attention heads"
  - "Interpretability"
  - "Parameter analysis"
  - "LLM internal mechanisms"
  - "Vocabulary projection"
date: 2026-05-08
content_hash: ea50977fe62e3bcf
---

# Inferring Functionality of Attention Heads from their Parameters

**Conference**: ACL 2025  
**arXiv**: [2412.11965](https://arxiv.org/abs/2412.11965)  
**Authors**: Amit Elhelo, Mor Geva (Tel Aviv University)  
**Code**: [github.com/amitelhelo/MAPS](https://github.com/amitelhelo/MAPS)  
**Area**: Other  
**Keywords**: Attention heads, Interpretability, Parameter analysis, LLM internal mechanisms, Vocabulary projection  

## TL;DR

Proposes the MAPS framework, which constructs a token mapping matrix $M$ by projecting attention head parameters into the vocabulary space. MAPS infers the functions realized by attention heads without requiring any forwarding inference or training. The mapping accuracy is validated across 20 relational operations on 6 LLMs, and an automated pipeline is developed to discover numerous previously unidentified attention head functions.

## Background & Motivation

### Background
Attention heads are the core building blocks of LLMs, and understanding their functions is crucial for model interpretability. Existing research primarily understands their actions by analyzing attention behavior during inference (attention patterns, output projections, causal interventions), but this approach has inherent limitations.

### Limitations of Prior Work
- **Incomplete Coverage**: Analysis relying on specific inputs may overlook attention head functionalities on other inputs, as the same head can behave differently across different contexts.
- **High Computational Cost**: Comprehensive analysis requires executing model inference over a large number of inputs, which is computationally expensive, and training data may be unavailable.
- **Interpretation Difficulties**: Analyzing activation patterns is often non-intuitive and can lead to misleading conclusions.
- **Limited to Specific Circuits**: Previous vocabulary-space projection methods were only used to study a few heads in specific circuits or single operations, lacking systematic application.

### Core Motivation
Can we directly infer the function of attention heads from their parameters, completely bypassing model inference? This paper extends the vocabulary-space interpretation method to a unified framework, MAPS, systematically addressing two types of questions: (a) given an operation, which heads in the model implement it; (b) given a head, what is its salient function.

## Method

### Core Idea: Vocabulary Projection of Attention Heads
Based on the formulation by Elhage et al., the $W_{VO}$ matrix of an attention head is projected into the vocabulary space using the embedding and unembedding matrices:

$$M = E \cdot W_{VO} \cdot U \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}$$

Each element $M[s,t]$ in matrix $M$ represents the strength score of the head mapping the source token $s$ to the target token $t$.

### Two Analysis Methods of MAPS Framework

**Approach A: Predefined Relations**

Given a dataset of token pairs $\mathcal{D}_R$ expressing relation $R$ (e.g., country-to-capital), the score of a head implementing this relation is calculated as:

$$\phi_R(M) = \frac{1}{|\mathcal{D}_R|} \sum_{(s,t) \in \mathcal{D}_R} \mathbb{1}[t \in \text{topk}(\mathbf{m}_s)]$$

That is, checking whether the target token appears in the top-k map of the corresponding row of the source token. A threshold of $\tau=15\%$ is used to classify whether a head implements a certain relation. It also supports suppression operations by considering the top-k of $-\mathbf{m}_s$.

**Approach B: Salient Operations**

1. Identify the top-k tokens with the most salient transformations using the saliency score $\sigma_t(W_{VO}) = \|e_t W_{VO}\| / \|e_t\|$.
2. Collect the top-n mapping targets for each salient token.
3. Use GPT-4o to automatically describe patterns within these mappings.

This method is more reliable than directly taking the highest-scoring mappings in $M$, since the latter is affected by token embedding norms and may bias towards a few tokens.

### Design of Relation Types
A dataset containing 20 relations across 4 categories was constructed:
- **Algorithmic**: copying, name copying, word to first/last letter, year to next year
- **Knowledge**: country to capital, country to language, object to hypernym, product to company, work to location
- **Linguistic**: antonyms, adjectives to comparative/superlative, nouns to pronouns, verbs to past tense, homophones, synonyms, compound words
- **Translation**: English to French, English to Spanish

## Key Experimental Results

### Main Results: Correlation with Inference Output

Pearson correlation coefficients between MAPS static estimation scores $\phi_R(M)$ and inference-time dynamic scores $\phi_R^*(h)$ on Llama-3.1 8B:

| Category | Relation | Out-of-context correlation | In-context correlation |
|------|------|-------------|-------------|
| Algorithmic | Copying | 0.76 | 0.73 |
| Algorithmic | Name copying | 0.95 | 0.95 |
| Algorithmic | Word to first letter | 0.90 | 0.78 |
| Knowledge | Country to capital | 0.85 | 0.85 |
| Knowledge | Country to language | 0.76 | 0.62 |
| Linguistic | Antonyms | 0.90 | 0.86 |
| Linguistic | Adjectives to comparative | 0.85 | 0.86 |
| Linguistic | Verbs to past tense | 0.91 | 0.86 |
| Translation | English to French | 0.71 | 0.68 |
| Translation | English to Spanish | 0.82 | 0.81 |

The vast majority of relations achieve a strong to extremely strong correlation of 0.71-0.95, indicating that MAPS can accurately estimate the in-inference behavior of heads.

### Ablation Study: Causal Effect Validation

Impact of removing MAPS-identified relation heads vs removing random heads on model accuracy in Pythia 12B:

| Relation | Baseline accuracy | Removing relation heads | Removing random heads | Control task - Removing relation heads |
|------|-----------|-------------|-------------|-------------------|
| Adjective to comparative | 0.91 | 0.20 | 0.82 | 0.63 |
| Copying | 1.00 | 0.68 | 1.00 | 0.88 |
| Country to capital | 0.97 | 0.00 | 0.95 | 0.90 |
| Country to language | 1.00 | 0.08 | 0.96 | 0.89 |
| Name copying | 1.00 | 0.24 | 1.00 | 0.92 |
| Word to first letter | 0.91 | 0.34 | 0.87 | 0.74 |
| Year to next year | 0.92 | 0.00 | 0.87 | 0.79 |

Across all relations, removing the heads identified by MAPS caused accuracy to drop by more than 32%, whereas removing random heads only resulted in a drop of less than 13%. This validates that the heads identified by MAPS have a causal relationship with the model's behavior.

## Highlights & Insights

- **Zero Inference Overhead**: Functionality of attention heads is inferred purely from parameters without requiring model training or inference, offering extreme computational efficiency.
- **Systematic Framework**: First work to extend the vocabulary-space projection method into a generalized framework that simultaneously supports "operation localization" and "functionality discovery", validated at scale on 6 LLMs and 20 relations.
- **Discovering New Heads**: Discovered 25 and 46 previously unidentified heads performing similar operations in GPT-2 small and medium, respectively, expanding the coverage of existing circuit analyses.
- **Architectural Insights**: Unveiled several valuable architectural biases—smaller models tend to encode more relations within a single head; in Llama-3.1's grouped-query attention, heads in the same group often implement structural or similar relations; relational heads are generally concentrated in middle and upper layers.
- **Automated Pipeline**: Combined with GPT-4o to automatically produce natural language descriptions of attention head functions, achieving 60%-96% coverage in the middle and upper layers, with 80% accuracy verified by human evaluation.

## Limitations & Future Work

- **Only Analyzing $W_{VO}$**: $W_{QK}$ matrices (responsible for attention computation and contextualization) are ignored, failing to fully characterize the selective behavior of the heads.
- **Vocabulary Space Constraints**: Only captures operations that can be expressed via token pairs, unable to capture more abstract computations such as idioms or positional features.
- **Insufficient Coverage in Early Layers**: Heads in early layers show lower interpretability in vocabulary space (20%-60%), likely because they compute generic features rather than vocabulary-level operations.
- **Omitting Bias Terms**: Biases of $W_V$ and $W_O$ are omitted, which might affect estimation precision.
- **Limited Generalization to Multi-token Entities**: Although experiments show single-token estimations generalize to multi-token inputs, minor drops in correlation still exist.
- **Automated Description Quality**: While GPT-4o's functional descriptions are 80% correct, there is still room for mischaracterizations.

## Related Work & Insights

- **Wang et al. (2023), McDougall et al. (2024)**: Used vocabulary projection to validate known head functions within specific circuits (e.g., IOI); MAPS scales this to a general framework and discovers extensive new heads.
- **Gould et al. (2024)**: Only studied the cross-model distribution of a single relation (copying); MAPS supports systematic mapping of 20 relations.
- **Voita et al. (2019), Clark et al. (2019)**: Analyzed head functions based on attention patterns; MAPS is entirely parameter-based and requires no inference.
- **Millidge & Black (2022)**: Used LLMs to interpret singular vectors of parameters, but did not consider input-output mapping relations, making it unable to estimate head functionalities.
- **Hernandez et al. (2024)**: Proven that relational operations of heads can be approximated via linear functions; MAPS further shows these relations are encoded directly in the parameter projections.
- **Merullo et al. (2412.11965)**: Found multi-functional heads in GPT-2 medium; MAPS expands and quantifies this finding through systematic analysis.

## Rating

- Novelty: ⭐⭐⭐⭐ — Systematizes the parameter-level vocabulary projection method into a generalized framework. The core approach is natural, but the scale and depth of application represent a significant breakthrough.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 6 models x 20 relations; three-fold validation consisting of correlation, causation, and generalization, along with human evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, informative charts and tables, with motivations and methods accurately and precisely articulated.
- Value: ⭐⭐⭐⭐ — Provides an efficient and practical tool for LLM interpretability with inspiring architectural insights, though its explanatory power for more complex computations remains limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] The Hidden Attention of Mamba Models](the_hidden_attention_of_mamba_models.md)
- [\[ACL 2025\] Segment-Based Attention Masking for GPTs](segment-based_attention_masking_for_gpts.md)
- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)
- [\[ACL 2025\] Unique Hard Attention: A Tale of Two Sides](unique_hard_attention_a_tale_of_two_sides.md)
- [\[NeurIPS 2025\] Normalization in Attention Dynamics](../../NeurIPS2025/others/normalization_in_attention_dynamics.md)

</div>

<!-- RELATED:END -->
