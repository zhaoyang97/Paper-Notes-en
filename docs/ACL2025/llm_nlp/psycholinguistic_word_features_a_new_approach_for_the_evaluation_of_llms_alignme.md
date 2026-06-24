---
title: >-
  [Paper Note] Psycholinguistic Word Features: A New Approach for the Evaluation of LLMs Alignment with Humans
description: >-
  [ACL 2025][LLM (Other)][Psycholinguistics] This paper systematically proposes the use of psycholinguistic word norms (Glasgow: 5,553 words $\times$ 7 features + Lancaster: 39,707 words $\times$ 6 sensory modalities, totaling 13 lexical features) to evaluate the alignment between LLMs and humans. The study finds that while GPT-4o shows a relatively high correlation on Glasgow emotional/conceptual features, all models perform extremely poorly on Lancaster sensorimotor features…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Psycholinguistics"
  - "LLM Evaluation"
  - "Human Alignment"
  - "Word Features"
  - "Embodied Cognition"
date: 2026-05-08
content_hash: ce693e5fe740760f
---

# Psycholinguistic Word Features: A New Approach for the Evaluation of LLMs Alignment with Humans

**Conference**: ACL 2025  
**arXiv**: [2506.22439](https://arxiv.org/abs/2506.22439)  
**Code**: [https://zenodo.org/records/14866800](https://zenodo.org/records/14866800)  
**Area**: LLM NLP / LLM Evaluation  
**Keywords**: Psycholinguistics, LLM Evaluation, Human Alignment, Word Features, Embodied Cognition

## TL;DR

This paper systematically proposes the use of psycholinguistic word norms (Glasgow: 5,553 words $\times$ 7 features + Lancaster: 39,707 words $\times$ 6 sensory modalities, totaling 13 lexical features) to evaluate the alignment between LLMs and humans. The study finds that while GPT-4o shows a relatively high correlation on Glasgow emotional/conceptual features, all models perform extremely poorly on Lancaster sensorimotor features, quantitatively revealing the fundamental limitation of LLMs lacking embodied cognition.

## Background & Motivation

**Background**: Current LLM evaluations predominantly focus on task performance (objectively measurable abilities such as mathematics, reasoning, and question answering). However, since LLM-generated texts are read and interacted with by humans, alignment evaluation (affect, perception, and preference) is equally critical—and even more challenging to execute effectively.

**Limitations of Prior Work**: (1) Human evaluation is unscalable, requiring tens of thousands of evaluations per model; (2) LLM-as-Judge is limited by the biases of the evaluator models themselves; (3) Chatbot Arena relies on community crowdsourcing with uncontrolled prompts and participants, offering rankings rather than in-depth analyses; (4) Existing benchmarks focus on task completion, neglecting deeper semantic properties of language such as arousal, concreteness, and sensory associations.

**Key Insight**: The field of psycholinguistics has accumulated decades of human-rating data—specifically Likert ratings for tens of thousands of words across dimensions such as arousal, valence, concreteness, and sensory strength. These datasets are readily available, large-scale (Lancaster covers nearly 40,000 words), and heavily validated by past research, making them ideal for cost-free reuse to evaluate LLM-human semantic alignment.

**Design Motivation**: Embodied cognition theory in cognitive science posits that linguistic meaning arises not only from word co-occurrences (the learning mechanism of LLMs) but also from sensory experiences, bodily interactions, and emotional engagements. The symbol grounding problem states that pure text cannot fully acquire language—at least approximately 1% of words (around 400 words) must be grounded through real-world experience. Therefore, evaluating the alignment of LLMs on sensory dimensions can quantitatively reveal the ceiling of "text-only learning."

## Method

### Overall Architecture

Reuse existing psycholinguistic word rating datasets as the benchmark $\rightarrow$ prompt the LLM using the **exact same instructions** as in human experiments $\rightarrow$ obtain numerical ratings from the LLM for each word $\rightarrow$ compute 4 correlation coefficients between LLM scores and mean human scores $\rightarrow$ measure the degree of alignment. The entire workflow is fully automated and scalable, requiring no new human annotations.

### Key Designs

1. **Complementary Dual-Dataset Design**: The Glasgow norms (5,553 words $\times$ 7 features: arousal, valence, dominance, concreteness, imageability, familiarity, gender) and the Lancaster norms (39,707 words $\times$ 6 sensory modalities: touch, hearing, smell, taste, vision, interoception) are selected. Glasgow contains emotional/conceptual features known to align relatively well (validating the methodology), while Lancaster focuses on sensorimotor aspects (expected to align poorly, testing the embodied cognition hypothesis). Together, they form a comprehensive 13-dimensional evaluation space.

2. **Logprob-based Probability-Weighted Scoring**: Instead of utilizing a single output number for LLM ratings, the output probabilities for each value on the rating scale (e.g., 0-5 or 1-9) are extracted to compute a probability-weighted average. For example, on a 1-9 scale, if a word has $P(7)=0.6$, $P(8)=0.3$, and $P(6)=0.1$, the estimated score is $7 \times 0.6 + 8 \times 0.3 + 6 \times 0.1 = 7.2$. This method leverages probability distribution information, which is more stable than directly taking the argmax, and has been demonstrated by prior research to yield superior performance.

3. **Four-Dimensional Correlation Metric Framework**: Both Pearson and Spearman correlation coefficients are calculated, evaluated on both raw values and rounded integer values respectively (totaling 4 metrics). Pearson emphasizes alignment in outlier regions (e.g., high-concreteness words), while Spearman handles the entire distribution with equal weight. For skewed distributions (e.g., taste/smell ratings highly clustered on the lower end), the two diverge significantly; reporting both prevents misleading conclusions drawn from a single metric.

## Key Experimental Results

Eight LLMs were evaluated: GPT-4o and GPT-4o-mini (closed-source); LLaMA-3.2-3B, LLaMA-3.1-8B, LLaMA-3.2-11B (multimodal), Gemma-2-9B, Yi-1.5-9B, and Occiglot-7B (open-source).

### Glasgow Norms Alignment (7 Affective/Conceptual Features)

| Lexical Feature | Scale | GPT-4o (Pearson) | GPT-4o-mini | LLaMA-3.1-8B | LLaMA-3.2-3B | Gemma-2-9B |
|---|---|---|---|---|---|---|
| Arousal | 1-9 | ~0.75 | ~0.70 | ~0.60 | ~0.45 | ~0.55 |
| Valence | 1-9 | ~0.80 | ~0.75 | ~0.65 | ~0.50 | ~0.60 |
| Concreteness | 1-7 | ~0.80 | ~0.75 | ~0.55 | ~0.40 | ~0.55 |
| Imageability | 1-7 | ~0.75 | ~0.70 | ~0.55 | ~0.40 | ~0.50 |
| Familiarity | 1-7 | ~0.70 | ~0.65 | ~0.50 | ~0.35 | ~0.50 |
| Gender | 1-7 | ~0.50 | ~0.45 | ~0.35 | ~0.30 | ~0.55 |
| Dominance | 1-9 | ~0.55 | ~0.50 | ~0.40 | ~0.30 | ~0.40 |

*Note: Specific correlation coefficients are read from the paper's radar charts, where ~ indicates approximate values.*

- **Typical Case**: "bicycle" (a concrete word) has a human rating of 6.81, for which GPT-4o outputs 7.00 (even more extreme), whereas LLaMA-3.2-3B only outputs 4.73. "bid" (an abstract word) has a human rating of 3.42, for which GPT-4o outputs 2.96, and LLaMA-3.2-3B outputs 4.50—indicating that small models lack discriminative power.

### Lancaster Norms Alignment (6 Sensorimotor Features)

| Sensory Modality | Scale | GPT-4o (Pearson) | Spearman Discrepancy | Trend in Other Models | Typical Case |
|---|---|---|---|---|---|
| Gustatory | 0-5 | ~0.40 | Pearson >> Spearman | Extremely low (0.1-0.3) | "Lemon" Human 4.45 / GPT-4o 4.49 / Gemma-2-9B **0.01** |
| Olfactory | 0-5 | ~0.45 | Pearson > Spearman | Low (0.15-0.35) | Skewed distribution leads to metric discrepancies |
| Haptic | 0-5 | ~0.35 | Small | Low (0.1-0.3) | — |
| Auditory | 0-5 | ~0.40 | Small | Low (0.15-0.3) | — |
| Visual | 0-5 | ~0.35 | Small | Low (0.1-0.3) | Multimodal LLaMA-3.2-11B **shows no advantage** |
| Interoceptive | 0-5 | ~0.30 | Small | Low (0.1-0.25) | — |

*Note: The ideal alignment range is 0.8–1.0; all Lancaster features fall far below this threshold.*

### Summary of Analysis Dimensions

| Analysis Dimension | Key Findings |
|---|---|
| Model Scale | Alignment gradually improves from LLaMA 3B to 8B to 11B, but it is not a decisive factor. |
| Multimodality | LLaMA-3.2-11B, GPT-4o, and GPT-4o-mini are multimodal, yet their visual feature alignment is not superior to text-only models. |
| Pearson vs. Spearman | Discrepancies between the two are large on skewed distributions such as taste and smell; Pearson performs better due to its weighting of outliers. |
| Model Family | The GPT-4o family is the best overall; open-source models are competitive on specific features (e.g., Gemma-2 on gender). |
| Direct Output vs. Logprob | Logprob probability-weighted estimation consistently outperforms direct argmax output. |

### Key Findings
- Alignment on the Glasgow norms (emotional/conceptual features) is acceptable but still shows a notable gap from the ideal range of 0.8–1.0.
- Alignment on the Lancaster norms (sensorimotor features) is extremely poor, confirming the fundamental limitation of LLMs lacking embodied cognition.
- Multimodal models do not demonstrate any expected advantages in visual-related features.
- For the same word (e.g., "bicycle"), LLaMA-3.2-3B offers concreteness ratings with almost no differentiation (4.73 vs 4.50), whereas GPT-4o is capable of outputting even more extreme differentiations than humans (7 vs 2.96).

## Highlights & Insights

- **Interdisciplinary Methodological Innovation**: Rather than designing new tasks or training new models, this paper proposes a fresh perspective on LLM evaluation—leveraging decades of human-rating datasets accumulated in psycholinguistics at zero cost. This paradigm of "evaluating AI with existing humanities and social science data" is highly generalizable.
- **Tightly Bridging Theory and Experiment**: The alignment gap between Glasgow and Lancaster is tightly linked to the symbol grounding problem and embodied cognition theory. LLMs align well on arousal/valence (which can be learned from textual co-occurrence) but poorly on taste/smell (which requires physical embodiment), perfectly validating predictions from cognitive science.
- **Logprob Probability-Weighted Technique**: When LLMs perform Likert scale ratings, computing a weighted average from logprobs (instead of directly taking the argmax) significantly improves consistency with human judgements. This technique possesses high practical value for any scenario requiring numerical ratings from LLMs.
- **Prudent Evaluation Metric Design**: Reporting 4 coefficients (Pearson and Spearman on raw/rounded data) uncovers the pitfalls of relying on a single metric (for instance, Pearson performs much better than Spearman on smell/taste due to highly skewed distributions).
- **Counter-intuitive Finding**: Multimodal models (LLaMA-3.2-11B, GPT-4o) do not perform better than text-only models in visual feature alignment, indicating that current multimodal training fails to effectively transfer sensorimotor grounding capabilities.

## Limitations & Future Work

- The study only utilizes two English datasets, lacking multilingual coverage; translation-based tests might introduce cultural and semantic biases.
- The evaluation cohort of 8 LLMs lacks the latest flagship models (e.g., Claude, Gemini Pro, Qwen), which limits representativeness.
- The evaluation relies solely on correlation coefficients without exploring absolute error metrics like MAE or RMSE, or distribution-level alignment measures.
- Only overall correlations are reported, missing finer-grained subgroup analyses (e.g., alignment variations between high- vs. low-frequency words, or concrete vs. abstract words).
- The root causes of alignment disparities (such as the proportion of sensorimotor descriptions in training corpora) remain unanalyzed, and mitigation strategies (e.g., post-training on synthetic sensory experience descriptions) are left unexplored.

## Related Work & Insights

- **Trott (2024); Martínez et al. (2025)**: These works proved that GPT-4 correlates highly with humans on valence and concreteness, yet they primarily focus on "using LLMs to generate psycholinguistic data" to assist research. Conversely, this paper reverses the perspective: "evaluating LLMs with psycholinguistic data," returning the authority of evaluation to the humanities.
- **Ivanova et al. (2024)**: Proposed that logprob probability-weighted estimation is closer to human judgment than prompt-based evaluation; this paper directly adopts this technique to acquire LLM scores.
- **Embodied Cognition Theory (Barsalou 2008; Borghi et al. 2024)**: Posits that cognition is grounded in bodily interactions with the environment. This study leverages Lancaster sensory data to provide quantifiable, LLM-based evidence for this theory.
- **Inspirations for Future Directions**: If the sensorimotor alignment gap stems from a lack of physical experience, can multimodal training (incorporating video, audio, or tactile sensor data) bridge this gap? This paper establishes a measurable evaluation target. Furthermore, is post-training on synthetic sensory description corpora effective? These unexplored paths warrant deeper investigation.

## Rating

- Novelty: ⭐⭐⭐⭐ The interdisciplinary perspective is innovative, systematizing the use of psycholinguistic rating data for LLM alignment evaluation for the first time. However, the methodology itself (calculating correlation) is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐ Covers 13 features $\times$ 8 models, supported by concrete case studies (bicycle/bid/lemon), though it lacks highly precise numerical tables and subgroup analyses.
- Writing Quality: ⭐⭐⭐⭐ Clarifies the underlying motivation, creating a cohesive feedback loop between theory (embodied cognition/symbol grounding) and experimental conclusions.
- Value: ⭐⭐⭐⭐ Blazes a new path for LLM evaluation and provides a reusable benchmarking methodology, accelerating the participation of the psycholinguistics community in AI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] KoGEM: Polishing Every Facet of the GEM: Testing Linguistic Competence of LLMs and Humans in Korean](polishing_every_facet_of_the_gem.md)
- [\[ACL 2025\] Comparing Moral Values in Western English-speaking Societies and LLMs with Word Associations](moral_values_western.md)
- [\[ACL 2025\] How Humans and LLMs Organize Conceptual Knowledge: Exploring Subordinate Categories in Italian](conceptual_knowledge_org.md)
- [\[ACL 2025\] Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching](self-tuning_instructing_llms_to_effectively_acquire_new_knowledge_through_self-t.md)
- [\[ACL 2025\] SkillVerse: Assessing and Enhancing LLMs with Tree Evaluation](skillverse_tree_eval.md)

</div>

<!-- RELATED:END -->
