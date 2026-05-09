---
title: >-
  [Paper Note] Explaining and Mitigating Crosslingual Tokenizer Inequities
description: >-
  [NeurIPS 2025][Robotics][Token Premium] This work systematically trains approximately 7,000 monolingual tokenizers covering 97 languages, providing the first demonstration that significant token premium disparities persist across languages even after controlling for training data size, vocabulary size, and algorithm. It further identifies vocabulary size and pre-tokenization strategy as key contributing factors, and proposes two mitigation approaches: language-specific optimal vocabulary size and SuperBPE.
tags:
  - NeurIPS 2025
  - Robotics
  - Token Premium
  - Crosslingual Tokenization
  - BPE
  - Vocabulary Size
  - Pre-tokenization
  - Compression Rate
date: 2026-05-08
content_hash: bd9d1337e3b4a2c9
---

# Explaining and Mitigating Crosslingual Tokenizer Inequities

**Conference**: NeurIPS 2025
**arXiv**: [2510.21909](https://arxiv.org/abs/2510.21909)
**Code**: [MonTok](https://huggingface.co/datasets/catherinearnett/montok)
**Area**: Robotics
**Keywords**: Token Premium, Crosslingual Tokenization, BPE, Vocabulary Size, Pre-tokenization, Compression Rate

## TL;DR
This work systematically trains approximately 7,000 monolingual tokenizers covering 97 languages, providing the first demonstration that significant token premium disparities persist across languages even after controlling for training data size, vocabulary size, and algorithm. It further identifies vocabulary size and pre-tokenization strategy as key contributing factors, and proposes two mitigation approaches: language-specific optimal vocabulary size and SuperBPE.

## Background & Motivation

1. **Token Premium Problem**: Multilingual tokenizers exhibit substantial differences in the number of tokens required to encode equivalent content across languages. A higher token premium implies longer sequences, leading to greater training/inference costs and latency.
2. **Gaps in Prior Understanding**: Previous work observed this phenomenon only in multilingual tokenizers, attributing it to imbalanced per-language training data proportions. This paper is the first to demonstrate via monolingual tokenizers that **significant compression rate disparities across languages persist even when data size, vocabulary size, and algorithm are held constant**.
3. **Practical Impact**: For instance, the byte premium for Burmese reaches 3.51 and for Shan 3.94, imposing higher service costs and degraded user experience on speakers of these languages.

## Core Problem
What linguistic features and tokenizer design choices drive crosslingual token premium disparities, and can tokenizer design adjustments mitigate this inequity?

## Method

### Experimental Infrastructure
- Approximately **7,000 monolingual tokenizers** trained, covering **97 languages**
- Each language trained on **300 MB of text** (ensuring coverage of low-resource languages)
- Manipulated variables: tokenization algorithm (BPE / Unigram), vocabulary size (8,192 → 262,144), and whether training data is scaled by byte premium
- Evaluation metric: **Corpus Token Count (CTC)** — total token count computed on the FLORES-200 parallel corpus; lower values indicate better compression

### Key Finding 1: BPE vs. Unigram
BPE achieves better compression rates and smaller crosslingual variance across all vocabulary sizes. SentencePiece Unigram yields the worst compression. Scaling training data by byte premium has **no effect** on compression rate ($t(3544)=-0.615, p=0.539$).

### Key Finding 2: Explanatory Factors for Token Premium

Linear regression analysis of CTC against language and tokenizer features (vocabulary size 65,536):

| Predictor | $R^2$ |
|-----------|-------|
| Train–eval data similarity | 0.239 |
| Mean token length (vocabulary) | 0.168 |
| Whitespace proportion | 0.157 |
| **Combined model** | **0.297** |

- **Data similarity**: Lexical overlap between the training set and the FLORES test set explains the largest share of variance, though subsequent intervention experiments indicate this is not a genuine causal factor.
- **Mean token length**: The average length of tokens actually used on FLORES correlates with CTC ($R^2=0.168$), whereas the average length of all tokens in the vocabulary does not — indicating the issue lies in "frequently used tokens being insufficiently long."
- **Whitespace proportion**: Different languages encode equivalent amounts of information using whitespace in varying ways; whitespace-based pre-tokenization disadvantages low-whitespace languages.
- **Other factors**: Writing system, phoneme inventory, and character/bigram entropy have some predictive power but are not statistically significant.

### Mitigation Strategy 1: Parallel Data Training
Tokenizers for 7 high-CTC languages are trained on NLLB parallel data. Results: statistically significant but negligibly small (average CTC reduction of approximately 1%), primarily driven by small vocabulary sizes. **Conclusion: parallel data training does not effectively mitigate token premium.**

### Mitigation Strategy 2: Optimal Vocabulary Size
Core idea: fit a power-law curve $\text{CTC} = a \cdot V^b$ between CTC and vocabulary size for each language, predict the optimal vocabulary size needed to reach a target CTC, and train a tokenizer for each language at that size.

- Result: crosslingual CTC variance is **significantly reduced** when language-specific optimal vocabulary sizes are used, confirmed by a Fisher–Snedecor test.
- Optimal vocabulary sizes differ by more than 10× across languages.

### Mitigation Strategy 3: SuperBPE (Whitespace-Free Pre-tokenization)
Removes the whitespace pre-tokenization constraint, allowing merges to cross word boundaries (SuperBPE / superword tokenizer).

- Result: simultaneously reduces overall CTC and crosslingual CTC variance.
- Improvement is especially pronounced for languages with high whitespace proportions.
- This finding is consistent with whitespace proportion being identified as an important predictor of CTC.

## Key Experimental Results

| Configuration | Effect |
|---------------|--------|
| Uniformly larger vocabulary | Overall CTC decreases, but crosslingual variance unchanged ($F_{96,96}=1.125, p=0.565$) |
| Language-specific optimal vocabulary | CTC variance significantly reduced |
| Parallel data training | Marginal effect (~1% CTC reduction) |
| SuperBPE | Simultaneously reduces CTC and crosslingual disparity |

## Highlights & Insights
- Unprecedented scale of controlled experiments: ~7,000 monolingual tokenizers × 97 languages, with rigorous experimental design.
- First use of monolingual tokenizers to isolate the contribution of language-intrinsic features to token premium.
- Identification of two effective interventions: optimal vocabulary size and SuperBPE.
- All tokenizers publicly released on HuggingFace for reproducibility.

## Limitations & Future Work
- Training data is limited to 300 MB per language, below the scale of mainstream tokenizers (several GB); comparability with OLMo/Pythia is validated, but generalizability remains limited.
- The relationship between compression and downstream performance is not examined (citing Schmidt et al. 2024, who suggest the two may be unrelated).
- The optimal vocabulary size must be tuned to a target CTC; how to unify vocabulary sizes when deploying multilingual models remains an open question.
- Only BPE and Unigram are analyzed; other algorithms such as WordPiece are not covered.
- Tokenizers trained with different language-specific optimal vocabulary sizes cannot be directly combined into a multilingual tokenizer.
- CTC as a compression metric depends on the coverage and translation quality of the FLORES parallel corpus.

## Related Work & Insights
- **MYTE (Limisiewicz et al. 2024)**: Replaces long byte sequences using morphological dictionaries, requiring predefined lexical resources.
- **MAGNET (Ahia et al. 2024)**: Employs script-specific boundary prediction modules, modifying the tokenization algorithm itself.
- The proposed approach requires no modification to the tokenization algorithm; adjustments are made solely through vocabulary size and pre-tokenization strategy, making it easier to integrate into existing pipelines.

The idea of customizing optimal vocabulary size per language generalizes naturally to customizing optimal pre-tokenization rules per language. The success of SuperBPE suggests that whitespace-based pre-tokenization in mainstream tokenizers is suboptimal, particularly for analytic languages such as Chinese and Thai. The controlled experimental methodology employed here — large-scale training, systematic variable manipulation, and statistical testing — is broadly applicable to tokenizer research. Token premium affects not only computational cost but potentially also crosslingual alignment quality in models, warranting further investigation.

## Rating
- Novelty: ⭐⭐⭐⭐ — First systematic study of token premium in a monolingual setting, revealing counterintuitive but important findings.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 7,000 tokenizers, 97 languages, multidimensional ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear logic, experiment-driven narrative, and strong conclusions.
- Value: ⭐⭐⭐⭐ — Directly actionable for the multilingual NLP community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Token Taxes: Mitigating AGI's Economic Risks](../../ICLR2026/robotics/token_taxes_mitigating_agis_economic_risks.md)
- [\[NeurIPS 2025\] MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents](mip_against_agent_malicious_image_patches_hijacking_multimod.md)
- [\[NeurIPS 2025\] LLMscape](llmscape.md)
- [\[NeurIPS 2025\] Understanding Prompt Tuning and In-Context Learning via Meta-Learning](understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)
- [\[NeurIPS 2025\] Redefining Experts: Interpretable Decomposition of Language Models for Toxicity Mitigation](redefining_experts_interpretable_decomposition_of_language_models_for_toxicity_m.md)

</div>

<!-- RELATED:END -->
