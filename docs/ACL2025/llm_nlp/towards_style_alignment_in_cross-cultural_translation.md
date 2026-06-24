---
title: >-
  [Paper Note] Towards Style Alignment in Cross-Cultural Translation
description: >-
  [ACL 2025][LLM (Other)][Style Alignment] This paper defines "style alignment" as a core goal of cross-cultural translation for the first time, systematically revealing style neutralization bias and English-centric bias in LLM translation. It proposes the RASTA method, which learns cultural alignment mappings in the embedding space to retrieve style-matched few-shot examples, improving style alignment by up to 56% without degrading translation quality.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Style Alignment"
  - "Cross-Cultural Translation"
  - "RASTA"
  - "Stylistic Concepts"
  - "Retrieval-Augmented Translation"
date: 2026-05-08
content_hash: 43127bab9c3404df
---

# Towards Style Alignment in Cross-Cultural Translation

**Conference**: ACL 2025  
**arXiv**: [2507.00216](https://arxiv.org/abs/2507.00216)  
**Code**: [shreyahavaldar/style_alignment](https://github.com/shreyahavaldar/style_alignment)  
**Area**: Others  
**Keywords**: Style Alignment, Cross-Cultural Translation, RASTA, Stylistic Concepts, Retrieval-Augmented Translation

## TL;DR

This paper defines "style alignment" as a core goal of cross-cultural translation for the first time, systematically revealing style neutralization bias and English-centric bias in LLM translation. It proposes the RASTA method, which learns cultural alignment mappings in the embedding space to retrieve style-matched few-shot examples, improving style alignment by up to 56% without degrading translation quality.

## Background & Motivation

- **Key Challenge**: Successful cross-cultural communication requires the speaker's **intended style** to align with the listener's **interpreted style**, but cultural differences often lead to misalignment. LLM translation focuses solely on content accuracy, ignoring cross-cultural adaptation at the stylistic level.
- **Neutralization Bias**: Experiments show that LLM translation tends to "**neutralize**" text—compressing strong polite or impolite expressions in the original text into mid-range values. The standard deviation of style in translated text is significantly lower than that of native text (e.g., Japanese politeness: native 0.20 vs. translated 0.09).
- **English-Centric Bias**: Style alignment is lowest in translations involving **non-Western languages** (Japanese, Chinese, Brazilian Portuguese), illustrating systematic weaknesses in existing LLMs' ability to capture styles in non-English cultures.
- **Metric Blind Spots**: Popular translation quality metrics (GEMBA, CometKiwi) show **negative or non-significant** correlation with style alignment, failing to detect translation failures at the stylistic level.
- **Typical Example**: An American user calling a professor by their first name represents politeness, but this is considered impolite in Japanese culture. LLMs translate the content literally but ignore the stylistic mismatch caused by cultural differences.

## Method

### Overall Architecture

The workflow of RASTA (Retrieval-Augmented STylistic Alignment) consists of: (1) discovering centroid representations of stylistic concepts in a multilingual embedding space; (2) learning native text mappings $\mathbf{v}_{\text{native}}$ and translated text mappings $\mathbf{v}_{\text{trans}}$; (3) correcting input embeddings with the difference $\mathbf{v}_{\text{align}} = \mathbf{v}_{\text{native}} - \mathbf{v}_{\text{trans}}$ for cultural alignment; (4) using the corrected embeddings to retrieve the 5 most style-matched samples from a native text corpus in the target language as few-shot examples; (5) injecting these examples into the translation prompt to guide the LLM to generate culturally appropriate translations.

### Key Designs

**1. Style Alignment Metric $\mathcal{A}(\mathcal{L}_1, \mathcal{L}_2)$**

Mistral-7B is fine-tuned for each language individually as a style quantifier to output style scores in $[0, 1]$, covering three dimensions: politeness, intimacy, and formality. The metric is defined as the Pearson correlation coefficient between the style scores of the source text and the translated text: $\mathcal{A}(\mathcal{L}_1, \mathcal{L}_2) = r(\mathcal{C}_1(X_{\mathcal{L}_1}), \mathcal{C}_2(T(X_{\mathcal{L}_1})))$. A correlation coefficient of 1 denotes perfect style alignment, while 0 denotes complete independence. The quantifier's average test RMSE is 0.157 (politeness), 0.183 (intimacy), and 0.255 (formality), respectively.

**2. Style Concept Discovery and Mapping Learning in Embedding Space**

Centroid embeddings $\mu(\mathcal{L}, \mathcal{S})$ are calculated for texts with different stylistic levels using the BGE-M3 multilingual embedding model. Silhouette scores are utilized to verify that different styles are indeed distinguishable in the embedding space. Then, two direction vectors are learned: $\mathbf{v}_{\text{native}} = \mu(\mathcal{L}_2, \mathcal{S}) - \mu(\mathcal{L}_1, \mathcal{S})$ representing the native style shift direction across languages, and $\mathbf{v}_{\text{trans}} = \mu(\mathcal{L}_1 \to \mathcal{L}_2, \mathcal{S}) - \mu(\mathcal{L}_1, \mathcal{S})$ representing the actual shift direction introduced by translation. The difference between the two exposes the cultural and stylistic information lost during translation.

**3. Cultural Alignment Mapping and Retrieval-Augmented Translation**

The alignment direction $\mathbf{v}_{\text{align}} = \mathbf{v}_{\text{native}} - \mathbf{v}_{\text{trans}}$ is calculated, and this directional correction is applied to the input text embeddings, moving them to the embedding positions where the native texts of the target language should reside. Then, cosine similarity is used to retrieve the 5 most similar native texts in the target language training set as few-shot examples to be injected into the translation prompt. This approach requires no additional training, achieving cross-cultural style alignment solely through vector arithmetic in the embedding space with minimal computational overhead.

### Loss & Training

- The style quantifiers are fine-tuned on Mistral-7B (using QLoRA), with **each language trained individually** to prevent cross-lingual interference.
- The RASTA framework itself **requires no training**, only pre-computing embedding centroids and direction vectors.
- Three multilingual style-annotated datasets are used: Holistic Politeness (4 languages: EN/ES/JA/ZH), Multilingual Tweet Intimacy (6 languages), and GYAFC + XFORMAL (4 languages).

## Key Experimental Results

### Main Results: RASTA Style Alignment Performance (GPT-4)

| Style Dimension | Method | $\mathcal{A}$↑ | CometKiwi↑ | GEMBA↑ | $\mathcal{A}$ Gain |
|---|---|---|---|---|---|
| Politeness | Vanilla Translation | 0.53 | 0.78 | 95.18 | — |
| Politeness | + Prompt "Maintain Style" | 0.60 | 0.78 | 95.56 | +13.2% |
| Politeness | **RASTA** | **0.70** | 0.77 | 95.13 | **+32.1%** |
| Intimacy | Vanilla Translation | 0.45 | 0.72 | 94.07 | — |
| Intimacy | + Prompt "Maintain Style" | 0.53 | 0.73 | 94.96 | +17.8% |
| Intimacy | **RASTA** | **0.55** | 0.72 | 94.49 | **+22.2%** |
| Formality | Vanilla Translation | 0.48 | 0.81 | 97.46 | — |
| Formality | + Prompt "Maintain Style" | 0.64 | 0.81 | 97.60 | +33.3% |
| Formality | **RASTA** | **0.75** | 0.80 | 97.12 | **+56.3%** |

### Correlation between Translation Metrics and Style Alignment

| Translator | $\mathcal{A}$ vs GEMBA | $\mathcal{A}$ vs CometKiwi | GEMBA vs CometKiwi |
|---|---|---|---|
| Google Translate | -0.154 | -0.548 | 0.674* |
| GPT-4 | 0.243 | -0.216 | 0.702* |
| GPT-3.5 | 0.030 | -0.396* | 0.648* |
| Llama 3.2 | 0.070 | -0.171 | 0.788* |
| NLLB-1.3B | 0.030 | -0.270* | 0.889* |
| Gemma-7B | -0.369* | -0.181 | 0.287* |

*Note: \* indicates $p < 0.05$. Traditional translation metrics are highly correlated with each other, but show negative or non-significant correlation with style alignment.*

### Key Findings

1. **Severe Neutralization Bias**: The standard deviation of politeness in translated text is only 45-50% of that in native text (ES/JA/ZH: [0.17, 0.09, 0.13] vs. [0.23, 0.20, 0.20]), with extreme styles almost entirely disappearing.
2. **RASTA Mitigates English-Centric Bias**: The style alignment of Japanese and Chinese translations jumps from the lowest ranks to near-average levels, narrowing the cross-lingual performance gap from 0.35 to 0.12.
3. **RASTA Restores Stylistic Variance**: The standard deviation of translated text increases by 36% on average ([0.14, 0.10, 0.10] $\to$ [0.18, 0.13, 0.15]), matching the native text distribution more closely.
4. **Human Evaluation Verification**: Bilingual annotators prefer RASTA translations over prompted translations in 61% (politeness) and 63% (formality) of cases.
5. **No Significant Loss in Translation Quality**: CometKiwi decreases by at most 1.3%, while GEMBA remains largely unchanged.

## Highlights & Insights

- **Groundbreaking Problem Definition**: This work systematically defines "style alignment" as a core goal of cross-cultural translation for the first time, deviating from traditional translation evaluation paradigms that only focus on content.
- **Practical Value of Neutralization Bias**: The discovery that LLM translation "flattens" emotional extremes has significant implications for high-emotion scenarios such as medical care and education.
- **An Elegant Zero-Training Solution**: RASTA achieves style alignment solely through vector arithmetic in the embedding space without extra training. The computational overhead is minimal, and the methodology is inspired by word vector arithmetic.
- **Unveiling Metric Blind Spots**: It proves that current mainstream translation metrics fail to capture translation quality along the stylistic dimension, calling on the community to re-evaluate existing evaluation frameworks.

## Related Work & Insights

- Complementary to Hershcovich et al. (2022)'s survey on cross-cultural NLP—this work provides a concrete solution for style alignment.
- The conceptual vector manipulation in the embedding space is inspired by the word vector arithmetic concept of Mikolov et al.
- Differing from traditional style transfer (which forces an output style): RASTA preserves the input style and corrects the cultural mapping.
- Differing from cultural translation methods based on entity replacement (Yao et al., 2024): RASTA modifies style instead of content.

## Limitations & Future Work

- Covers only three style dimensions: politeness, intimacy, and formality, lacking verification on dimensions such as humor, sarcasm, or authority.
- The RMSE of the style quantifiers (0.157-0.255) introduces noise to measurements, affecting evaluation accuracy.
- RASTA relies on a target language corpus with styled native annotations of sufficient scale, raising feasibility concerns for low-resource languages.
- Only verified on high-resource languages, lacking exploration in low-resource language scenarios.
- Style and content are deeply coupled, making perfect style alignment potentially unattainable.
- Experiments were conducted using a single prompt only; translation results are sensitive to prompt phrasing.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The introduction of the style alignment concept and the discovery of neutralization bias are both pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Verified across 6 LLMs, 3 styles, and multiple language pairs, including human preference evaluations. However, it lacks comparison with culture-aware translation methods.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The motivation is clear, the methodology derivation is rigorous, and the cross-cultural communication examples are vivid and intuitive.
- **Value**: ⭐⭐⭐⭐ — The method is plug-and-play and open-source for reproducibility, but its dependence on style-annotated data limits generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cross-Modal Alignment for LLM-Enhanced Spoken Language Understanding](cross-modal_alignment_for_llm-enhanced_spoken_language_understanding.md)
- [\[ACL 2025\] From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment](from_neurons_to_semantics_evaluating_cross-linguistic_alignment_capabilities_of_.md)
- [\[CVPR 2025\] Chat-based Person Retrieval via Dialogue-Refined Cross-Modal Alignment](../../CVPR2025/llm_nlp/chat-based_person_retrieval_via_dialogue-refined_cross-modal_alignment.md)
- [\[ACL 2025\] Cultural Learning-Based Culture Adaptation of Language Models](cultural_learning-based_culture_adaptation_of_language_models.md)
- [\[ACL 2025\] An Empirical Study of Iterative Refinements for Non-Autoregressive Translation](an_empirical_study_of_iterative_refinements_for_non-autoregressive_translation.md)

</div>

<!-- RELATED:END -->
