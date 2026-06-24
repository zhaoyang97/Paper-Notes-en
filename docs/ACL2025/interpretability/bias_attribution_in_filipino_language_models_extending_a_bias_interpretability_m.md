---
title: >-
  [Paper Note] Bias Attribution in Filipino Language Models: Extending a Bias Interpretability Metric for Application on Agglutinative Languages
description: >-
  [ACL 2025][Interpretability][Bias attribution] Extends an information-theoretic bias attribution score metric to agglutinative languages (Filipino) by averaging subword scores to handle complex morphemic structures. Analysis on four multilingual PLMs reveals that bias in Filipino models is driven by entity-type topical words (people/objects/relationships), contrasting sharply with action-type topical words (crime/sexual activity) in English.
tags:
  - "ACL 2025"
  - "Interpretability"
  - "Bias attribution"
  - "Filipino"
  - "agglutinative languages"
  - "token attribution"
date: 2026-05-08
content_hash: 3bb895707a0eaf02
---

# Bias Attribution in Filipino Language Models: Extending a Bias Interpretability Metric for Application on Agglutinative Languages

**Conference**: ACL 2025  
**arXiv**: [2506.07249](https://arxiv.org/abs/2506.07249)  
**Code**: [https://github.com/gamboalance/bias_attribution_filipino](https://github.com/gamboalance/bias_attribution_filipino)  
**Area**: Interpretability  
**Keywords**: Bias attribution, Filipino, agglutinative languages, interpretability, token attribution

## TL;DR
Extends an information-theoretic bias attribution score metric to agglutinative languages (Filipino) by averaging subword scores to handle complex morphemic structures. Analysis on four multilingual PLMs reveals that bias in Filipino models is driven by entity-type topical words (people/objects/relationships), contrasting sharply with action-type topical words (crime/sexual activity) in English.

## Background & Motivation
**Background**: Research on LLM bias increasingly focuses on multilingual and non-English scenarios, but most efforts remain at the bias evaluation stage, rarely addressing bias interpretability—specifically, which tokens drive the biased behavior.

**Limitations of Prior Work**: Gamboa & Lee (2024) proposed an information-theory-based bias attribution score metric, but it is only applicable to English. For agglutinative languages (e.g., Filipino), a single word contains multiple morphemes/tokens, and directly applying attribution methods yields multiple inconsistent scores.

**Key Challenge**: Agglutinative languages exhibit complex morphemic structures (e.g., nakikipagtalik = na+ki+ki+pag+talik). PLM tokenizers split them into multiple tokens, each receiving a different attribution score—how to aggregate these into word-level bias contributions?

**Goal**: To adapt bias attribution methods to agglutinative languages and reveal the semantic patterns of bias in Filipino PLMs.

**Key Insight**: Averaging the attribution scores of tokenizer-split tokens to obtain the attribution score of the entire word, and then analyzing the thematic categories of bias-contributing tokens through semantic tagging.

**Core Idea**: Averaging subword attribution scores adapts the method to agglutinative languages, leading to the discovery that bias in Filipino models is driven by entities ("people/objects/relationships") rather than actions as observed in English.

## Method

### Overall Architecture
Filipino CrowS-Pairs dataset → 4 multilingual PLMs → compute token attribution score b(u) → subword aggregation → semantic tagging analysis

### Key Designs

1. **Bias Attribution Score b(u)**:

    - **Function**: Quantifies the contribution of each unmodified token to the model's biased decisions.
    - **Mechanism**: For each sentence pair in CrowS-Pairs, the shared token is masked in both contexts (more biased / less biased) respectively; then, the difference in JSD (Jensen-Shannon Divergence) between the model's predicted probability distribution for that token and the ground truth distribution is compared: $b(u) = \sqrt{JSD(P_{u,more} \| G_u)} - \sqrt{JSD(P_{u,less} \| G_u)}$
    - Negative score → the token drives the model toward bias; Positive score → drives the model away from bias.

2. **Agglutinative Language Adaptation (Core Contribution)**:

    - **Function**: When a word is split into multiple subwords by the tokenizer, the mean of the subword attribution scores is taken.
    - $b(u) = \frac{1}{n}\sum_{i=1}^{n} b(t_i)$
    - **Design Motivation**: A word in an agglutinative language (e.g., nakikipagtalik) may be split into 5 tokens, requiring aggregation into a word-level bias contribution to ensure interpretability.

3. **Semantic Analysis**:

    - Translation to English using googletrans → semantic tagging using pymusas → statistical profiling of the semantic category distribution of bias-contributing tokens.
    - Filtering out low-frequency words that appear fewer than 10 times.

## Key Experimental Results

### Main Results (Model Bias Score, Ideal 50%)

| Model | Training Languages | Gender Bias | Sexual Orientation Bias | Overall |
|------|---------|---------|-----------|------|
| GPT-2 | Global | 53.43 | 68.49 | 58.82 |
| RoBERTa-Tagalog | Tagalog | 53.43 | 73.97 | 60.78 |
| SEA-LION-3B | English + Southeast Asian | **74.81** | 67.12 | **72.06** |
| SeaLLMs-v3-7B | English + Southeast Asian | 51.14 | 52.06 | 51.47 |

### Semantic Analysis: Thematic Categories of Bias-Driving Tokens

| Semantic Category | Filipino Models | English Models |
|---------|------------|---------|
| Relationships (friends/lovers) | 50-60% bias contribution | Lower |
| People/objects | High proportion | Lower |
| Crime/sexual activity | Lower | **Main driver** |
| Helping behavior | Lower | High proportion |

### Key Findings
- **Filipino bias is driven by entities, while English bias is driven by actions**—reflecting cultural differences in the expression of bias.
- SEA-LION-3B exhibits the most severe bias (72.06%), while SeaLLMs-v3-7B is closest to fairness (51.47%)—larger models are not necessarily more biased.
- "Relationship"-related words (kaibigan/kasintahan, etc.) are the highest bias-contributing semantic categories across all four models.
- RoBERTa-Tagalog, trained purely on Tagalog, exhibits the highest sexual orientation bias (73.97%), possibly due to cultural bias in the training data.

## Highlights & Insights
- **Qualitative differences in cross-lingual bias**: Rather than simply differing in "degree of bias," they differ in "bias mechanisms"—Filipino manifests bias through entity associations, whereas English does so through action associations.
- **Agglutinative language adaptation is minimalist yet effective**: The subword mean aggregation method is simple and general, making it directly applicable to other agglutinative languages (e.g., Turkish, Japanese, Korean).
- **First application of bias interpretability methods to a non-English language**: Elevating research from simply "quantifying the amount of bias" to "understanding where bias originates."

## Limitations & Future Work
- The Filipino CrowS-Pairs dataset contains only 204 pairs, which is relatively small.
- It only covers two types of bias: gender and sexual orientation.
- Subword mean aggregation might be overly simplified—different morphemes may write unequal contributions to the overall meaning of the word.
- Semantic analysis relies on machine translation (which may introduce noise).
- Intervention methods to mitigate bias were not explored.

## Related Work & Insights
- **vs Gamboa & Lee (2024)**: Their evaluation was limited to English; this work extends it to Filipino and uncovers differences in cross-lingual bias mechanisms.
- **vs Multilingual Bias Evaluation Studies**: Most prior works only quantify the degree of bias, whereas this work delves into token-level causal analysis.
- The methodology can be directly expanded to bias analysis in other agglutinative languages.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of bias attribution methods to a non-English agglutinative language, revealing substantial cross-cultural differences.
- Experimental Thoroughness: ⭐⭐⭐ Analyzed 4 models but on a relatively small dataset (204 pairs) with only two types of bias.
- Writing Quality: ⭐⭐⭐⭐ Clear description of methodology with adequate linguistic background.
- Value: ⭐⭐⭐⭐ Pioneering significance for research in multilingual bias interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Missingness Bias Calibration in Feature Attribution Explanations](../../ICLR2026/interpretability/missingness_bias_calibration_in_feature_attribution_explanations.md)
- [\[AAAI 2026\] Hypothesis Generation via LLM-Automated Language Bias for ILP](../../AAAI2026/interpretability/hypothesis_generation_via_llm-automated_language_bias_for_ilp.md)
- [\[ACL 2025\] EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations](expert_an_explainable_image_captioning_evaluation_metric_with_structured_explana.md)
- [\[ACL 2025\] Mechanistic Interpretability of Emotion Inference in Large Language Models](mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)
- [\[ACL 2025\] Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability](normalized_aopc_faithfulness_metrics.md)

</div>

<!-- RELATED:END -->
