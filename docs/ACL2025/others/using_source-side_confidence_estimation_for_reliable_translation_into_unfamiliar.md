---
title: >-
  [Paper Note] Using Source-Side Confidence Estimation for Reliable Translation into Unfamiliar Languages
description: >-
  [ACL 2025][Source-Side Confidence Estimation] This paper proposes a gradient-based source-side confidence estimation method that directly detects potential mistranslations by measuring the sensitivity of the output sequence probability to the source embeddings. This approach outperforms traditional methods without requiring word alignment, and supports the construction of an interactive translation Web application for users fluent in the source language.
tags:
  - "ACL 2025"
  - "Source-Side Confidence Estimation"
  - "Gradient Attribution"
  - "Mistranslation Detection"
  - "Interactive Translation"
  - "Uncertainty"
date: 2026-05-08
content_hash: f09fdbbc41684e98
---

# Using Source-Side Confidence Estimation for Reliable Translation into Unfamiliar Languages

**Conference**: ACL 2025  
**arXiv**: [2503.23305](https://arxiv.org/abs/2503.23305)  
**Code**: Yes ([https://github.com/kennethsible/confidence-estimation](https://github.com/kennethsible/confidence-estimation))  
**Area**: NLP / Machine Translation  
**Keywords**: Source-Side Confidence Estimation, Gradient Attribution, Mistranslation Detection, Interactive Translation, Uncertainty

## TL;DR

This paper proposes a gradient-based source-side confidence estimation method that directly detects potential mistranslations by measuring the sensitivity of the output sequence probability to the source embeddings. This approach outperforms traditional methods without requiring word alignment, and supports the construction of an interactive translation Web application for users fluent in the source language.

## Background & Motivation

Confidence estimation in machine translation has a history of several decades, but most work focuses on the **target side**—assisting users fluent in the target language with post-editing. However, an equally important yet overlooked application scenario exists: users who are fluent in the source language but do not understand the target language. For example, when travelers use an MT system to express their needs in a foreign country, they need to verify if the translation is correct and should be able to improve the translation by modifying the source text when errors are detected.

Traditional source-side confidence estimation relies on projecting target-side word probabilities to the source side through word alignment, which is an indirect method limited by alignment quality. This paper proposes a direct, alignment-free gradient attribution method.

## Method

### Overall Architecture

For each source word $x_i$, uncertainty is estimated by calculating the gradient of the output sequence probability with respect to the source embedding vector. Words with high uncertainty are highlighted to prompt the user, who can click them to obtain alternative suggestions.

### Key Designs

1. **Gradient Attribution Uncertainty Estimation**: For each source word $x_i$, the uncertainty is defined as $U(x_i) = \sum |\partial P(y|x)/\partial x_i^k|$ ($L_1$ norm), which is the sum of the absolute partial derivatives of the output probability with respect to each dimension of the word embedding. The intuition is that if perturbing the source embedding has minimal effect on the output, the model is confident (robust) in the translation of that word; otherwise, it is uncertain.

2. **Subword Aggregation Strategy**: Since MT models use subword tokenization, subword-level uncertainties must be aggregated into word-level metrics. The experiments compared three strategies—sum, avg, and max—and selected sum.

3. **GPT-4o Automated Annotation Evaluation**: A few-shot chain-of-thought prompt was designed to let GPT-4o detect mistranslations (given the source sentence, MT candidate translation, and reference translation), serving as a low-cost, reproducible evaluation framework.

4. **Interactive Web Application**: Built as a PWA, the application displays the source text with uncertainty highlighting. When users click on a highlighted word, it displays $k$-NN nearest neighbor substitution suggestions (using FAISS to accelerate retrieval based on the cosine similarity of the encoder's final layer output).

## Key Experimental Results

### Main Results — Mistranslation Detection

| Method | Max F1 | AUC-PR ($10^{-2}$) | AUC-ROC ($10^8$) |
|------|--------|----------------|-----------------|
| MGIZA (Alignment Projection) | 0.12 | 1.94 | 0.73 |
| Attention (Attention Projection) | 0.10 | 0.77 | 1.00 |
| **Gradient (Ours)** | **0.19** | **8.36** | **1.31** |

### Ablation Study — Dimensionality Reduction and Subword Aggregation

| Norm | Aggregation Function | AUC-PR |
|------|---------|--------|
| $L_1$ | sum | **Best** |
| $L_2$ | sum | Second Best |
| $L_\infty$ | sum | Worse |
| $L_1$ | avg | Slightly Worse |
| $L_1$ | max | Worse |

### Key Findings

1. The gradient method outperforms MGIZA by 4.3x and Attention by 10.9x in AUC-PR (the most critical metric due to the extremely scarce positive class).
2. The $L_1$ norm and sum aggregation are the optimal configurations.
3. GPT-4o-based automated annotation correctly identifies mistranslations, providing a reproducible evaluation framework.

## Highlights & Insights

- It ingeniously transforms gradient attribution from "explaining predictions" to "estimating confidence," presenting a natural and effective shift in perspective.
- It requires no additional training or independent alignment models, relying solely on the backpropagation of the MT model itself, which leads to a clean implementation.
- The product-oriented mindset based on user scenarios is highly instructive—instead of having users modify the translation, it enables them to modify the source text.
- Substitution suggestions are based on $k$-NN in the encoder space, achieving semantically related synonym recommendations.

## Limitations & Future Work

- The gradient method requires backpropagation, resulting in higher computational costs compared to simple probability-based methods.
- Currently, it only demonstrates the English-to-German language pair, and multilingual generalization needs to be validated.
- GPT-4o model snapshots may not be permanently available, affecting the long-term reproducibility of the evaluation framework.
- Substitution suggestions are solely based on the cosine similarity of encoder embeddings, without utilizing more advanced semantic methods like masked language models.

## Related Work & Insights

- Complementary to the field of Quality Estimation (QE): while QE typically predicts translation quality scores, this work focuses on localizing specific words.
- The framework of the gradient attribution method can be extended to input sensitivity analysis in other seq2seq tasks (e.g., summarization, dialogue).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Using direct gradient attribution for source-side confidence estimation is a novel and elegant approach.
- **Experimental Thoroughness**: ⭐⭐⭐ — The validation is sufficient but limited to a single language pair on a relatively small scale.
- **Writing Quality**: ⭐⭐⭐⭐ — Exceptionally clear logic with a vivid demonstration of the product application.
- **Value**: ⭐⭐⭐⭐ — Practical application scenarios with an open-source implementation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Uniform Meaning Representation Help GPT-4 Translate from Indigenous Languages?](can_uniform_meaning_representation_help_gpt-4_translate_from_indigenous_language.md)
- [\[ACL 2025\] CiteEval: Principle-Driven Citation Evaluation for Source Attribution](citeeval_principle-driven_citation_evaluation_for_source_attribution.md)
- [\[ACL 2025\] RoToR: Towards More Reliable Responses for Order-Invariant Inputs](rotor_towards_more_reliable_responses_for_order-invariant_inputs.md)
- [\[ACL 2025\] Causal Estimation of Tokenisation Bias](causal_tokenisation_bias.md)
- [\[ACL 2025\] The Noisy Path from Source to Citation: Measuring How Scholars Engage with Past Research](the_noisy_path_from_source_to_citation_measuring_how_scholars_engage_with_past_r.md)

</div>

<!-- RELATED:END -->
