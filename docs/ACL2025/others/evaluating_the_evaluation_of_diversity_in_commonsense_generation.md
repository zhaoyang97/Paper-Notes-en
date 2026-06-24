---
title: >-
  [Paper Note] Evaluating the Evaluation of Diversity in Commonsense Generation
description: >-
  [ACL 2025][Diversity Evaluation] A systematic meta-evaluation of 12 diversity evaluation metrics in Generative Commonsense Reasoning (GCR) tasks reveals that form-based (n-gram) metrics severely overestimate diversity on low-quality generations, while content-based (sentence embedding) metrics align better with human judgments. Consequently, content-level metrics such as VS-Embed or Chamfer Distance are recommended for future GCR research.
tags:
  - "ACL 2025"
  - "Diversity Evaluation"
  - "Commonsense Generation"
  - "Meta-evaluation"
  - "Content-level Metrics"
  - "LLM Annotation"
date: 2026-05-08
content_hash: 3bdd93599ee52ebf
---

# Evaluating the Evaluation of Diversity in Commonsense Generation

**Conference**: ACL 2025  
**arXiv**: [2506.00514](https://arxiv.org/abs/2506.00514)  
**Code**: Yes ([https://github.com/LivNLP/Evaluating-Diversity-Metrics](https://github.com/LivNLP/Evaluating-Diversity-Metrics))  
**Area**: Others  
**Keywords**: Diversity Evaluation, Commonsense Generation, Meta-evaluation, Content-level Metrics, LLM Annotation

## TL;DR

A systematic meta-evaluation of 12 diversity evaluation metrics in Generative Commonsense Reasoning (GCR) tasks reveals that form-based (n-gram) metrics severely overestimate diversity on low-quality generations, while content-based (sentence embedding) metrics align better with human judgments. Consequently, content-level metrics such as VS-Embed or Chamfer Distance are recommended for future GCR research.

## Background & Motivation

Generative Commonsense Reasoning (GCR) requires models to generate sentences that are both commonsense-compliant and diverse for a given concept set. While mature metrics (such as BLEU and ROUGE) exist to evaluate generation quality, **the choice of metrics to evaluate diversity remains chaotic**—various papers randomly adopt metrics like self-BLEU, Distinct, and Vendi Score, yet no systematic validation has been conducted to verify whether these metrics truly capture meaningful diversity.

The paper reveals the severity of this issue with an intuitive example: given the concept set {walk, dog, take, park, couple}, Set-1 contains semantically diverse sentences, while Set-2 merely repeats near-synonymous sentences. However, self-BLEU-3 assigns a higher diversity score to Set-2—because the paraphrasing in Set-2 reduces n-gram overlap, even though it offers no semantic novelty.

Core Problem: **Which diversity metrics are best suited for evaluating commonsense generation? Under what conditions?**

## Method

### Overall Architecture

The authors design a comprehensive meta-evaluation methodology:

1. **Create a diversity-annotated dataset**: Use an LLM (GPT-4o) to assign diversity scores to pairs of sentence sets.
2. **Construct sentence sets of varying quality**: Generate high- and low-quality candidate sets through operations such as paraphrasing and shuffling.
3. **Calculate the accuracy of target metrics**: Treat each metric as an "annotator" and perform pairwise comparisons with LLM preference judgments.

### Key Designs

1. **LLM as a Diversity Annotator**:

    - Why not use humans? Tevet & Berant (2021) found that crowdsourced annotators have very low agreement on diversity judgments and are highly costly.
    - Use GPT-4o for scoring, coupled with a carefully designed few shot prompt (8 human high-agreement examples).
    - Adopt a 1-5 rating scale instead of direct preference selection (to avoid the LLM's order sensitivity—during preliminary experiments, direct selection resulted in 87% choosing Set-2).
    - Each pair is evaluated 5 times and averaged, with the temperature set to 1.0 to obtain statistically stable scores.
    - Human verification: 5 linguistically trained human annotators achieved an agreement rate of **80.6%** with GPT-4o on 70 pairs.

2. **High-Quality Candidate Set Construction**:

    - **Default**: Three LLMs (GPT-4-turbo, Llama3.1-8B, Qwen2.5-14B) are used to generate 4 sentences following the original CommonGen instructions.
    - **Para-1/2/3**: Apply increasing levels of paraphrasing and substitution to the sentences in Default, expecting diversity to decrease progressively.
        - Para-1: {A, A*, B, C} — 1 paraphrase
        - Para-2: {A, A*, B, B*} — 2 paraphrases
        - Para-3: {A, A*, A**, B} — 3 paraphrases (homologous)

3. **Low-Quality Candidate Set Construction**:

    - **Nonsensical**: Instruct the LLM to generate grammatically correct but non-sensical sentences.
    - **NounShuff**: Shuffle only the positions of nouns and pronouns in the sentences.
    - **RndShuff**: Completely and randomly shuffle the word order of all tokens.

4. **12 Diversity Metrics**:

    - Form-level (6 types): self-BLEU-3/4, VS-ngram-0.5/1/∞, Distinct-4, Entropy-2
    - Content-level (6 types): self-CosSim, Chamfer Distance, VS-Embed-0.5/1/∞ (all based on SimCSE sentence embeddings)

### Evaluation Method

For each pair of candidate sets sharing the same input concept, a prediction is considered correct if both the LLM and the target metric agree on which set is more diverse. Accuracy = number of agreed pairs / total pairs. Vague pairs with LLM score differences < 0.5 are filtered out.

## Key Experimental Results

### Main Results: Metric Accuracy on CommonGen

| Metric Type | Metric Name | GPT-4-turbo | Qwen2.5 | Llama3.1 |
|----------|--------|-------------|---------|----------|
| **Form-level** | self-BLEU-3 | 48.4 | 50.7 | 52.7 |
| | VS-ngram-∞ | 47.5 | 58.9 | 56.5 |
| | Distinct-4 | 64.0 | 69.0 | 61.7 |
| | Entropy-2 | 62.9 | 74.0 | 62.5 |
| **Content-level** | Chamfer | 80.6 | 78.9 | 71.9 |
| | self-CosSim | 76.9 | 80.0 | 71.9 |
| | **VS-Embed-0.5** | **80.7** | **80.8** | **73.2** |
| | VS-Embed-1 | 79.3 | 81.1 | 73.1 |

### Ablation Study: Decoupled Performance under High/Low Quality Generation

| Metric | High Quality (GPT-4t) | Low Quality (GPT-4t) | High Quality (Qwen) | Low Quality (Qwen) |
|------|---------------|---------------|-------------|-------------|
| self-BLEU-3 | 73.5 | **27.6** | 68.4 | **35.3** |
| self-BLEU-4 | 72.0 | 30.0 | 67.1 | 38.7 |
| Distinct-4 | 61.7 | 65.9 | 58.6 | 79.4 |
| Chamfer | **80.2** | **80.8** | 67.5 | **88.9** |
| VS-Embed-0.5 | **80.2** | **81.1** | **72.3** | **88.2** |

### Key Findings

1. **Content-level metrics consistently outperform form-level metrics under all conditions**: VS-Embed-0.5 and Chamfer achieve the highest accuracy across all three datasets: CommonGen, ComVE, and DimonGen.

2. **Form-level metrics fail completely on low-quality generations**: The accuracy of self-BLEU drops to ~28% (worse than the 50% random baseline!) on low-quality sets, because random shuffling of word order reduces n-gram overlap, which is incorrectly judged as "more diverse".

3. **All metrics assign higher diversity scores to low-quality generations**—random, nonsensical sentences appear more "diverse". This exposes a fundamental limitation: **diversity should not be evaluated separately from quality**.

4. **Cohen's Kappa analysis** shows: Content-level metrics are highly consistent with each other (Kappa > 0.8), while form-level vs. content-level metrics show negative Kappa (inverse correlation!) on low-quality sets, indicating that the two types of metrics measure fundamentally different things.

5. **Distribution visualization**: The Default/Paraphrased distributions of self-BLEU-3 heavily overlap (making them indistinguishable), whereas the distribution of Chamfer is well-separated.

## Highlights & Insights

- **Meta-research on "evaluating evaluation metrics"**: Rather than proposing a new method or model, this work rigorously validates whether existing tools are reliable—such studies are crucial for normalizing community standards.
- **Methodology of LLM-as-Annotator**: Using GPT-4o for large-scale diversity annotation, combined with few-shot calibration and human verification (80.6% agreement), demonstrates a viable path for using LLMs in subjective evaluation tasks.
- **Strong critique of form-level metrics**: self-BLEU is completely unreliable in mixed high/low quality scenarios (accuracy < 30%), yet many existing papers continue to use it. This finding should shape future evaluation practices.
- **Pragmatic advice**: Recommending the use of VS-Embed-0.5 or Chamfer Distance, based on SimCSE embeddings, which maintains a manageable computational cost.

## Limitations & Future Work

1. Evaluated only in English—form-level metrics might behave differently in morphologically rich languages.
2. Used only GPT-4o as the annotating LLM—other LLMs may exhibit different preferences.
3. Content-level metrics depend on the quality of SimCSE embeddings—using different embedding models might lead to different conclusions.
4. All metrics yield high diversity scores for low-quality generations—future work needs to develop joint metrics that simultaneously consider quality and diversity.
5. Only three GCR datasets and three generator LLMs were considered; the coverage could be further expanded.

## Related Work & Insights

- **Tevet & Berant (2021)**: Human evaluation study of NLG diversity. However, it relies on crowdsourced annotations (low consistency), requires hyperparameter tuning (temperature) to control diversity, and lacks commonsense constraints.
- **Zhang et al. (2024)**: Pioneering work on LLM evaluation of GCR diversity, reporting that LLMs have moderate agreement with humans. This work performs a larger-scale, more systematic evaluation on top of that.
- **Friedman & Dieng (2023)**: Proposed Vendi Score, a diversity metric based on the eigenvalues of a kernel matrix. This paper confirms the superiority of its embedding-based version (VS-Embed).
- **Insight**: Similar meta-evaluations should be performed for other NLG evaluation dimensions (e.g., creativity, informativeness). The community might currently be using unreliable metrics to measure dimensions they shouldn't.

## Rating

- **Novelty**: ⭐⭐⭐ — The meta-evaluation methodology is clear but not entirely new; the core contribution lies in the empirical findings rather than methodological innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 12 metrics $\times$ 3 datasets $\times$ 3 generator LLMs $\times$ high/low quality analysis $\times$ distribution visualization $\times$ Cohen's Kappa. The experimental design is exceptionally rigorous.
- **Writing Quality**: ⭐⭐⭐⭐ — The example in Figure 1 intuitively illustrates the problem. The tables are informative, though slightly redundant, and the overall structure is highly lucid.
- **Value**: ⭐⭐⭐⭐ — Directly impacts GCR evaluation practices: self-BLEU should be explicitly deprecated in favor of content-level metrics. The insight that "diversity should not be evaluated in isolation from quality" has profound implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] What Matters in Evaluating Book-Length Stories? A Systematic Study of Long Story Evaluation](what_matters_in_evaluating_book-length_stories_a_systematic_study_of_long_story_.md)
- [\[ACL 2025\] Spotting Out-of-Character Behavior: Atomic-Level Evaluation of Persona Fidelity in Open-Ended Generation](spotting_out-of-character_behavior_atomic-level_evaluation_of_persona_fidelity_i.md)
- [\[ACL 2025\] Evaluating Design Decisions for Dual Encoder-based Entity Disambiguation](evaluating_design_decisions_for_dual_encoder-based_entity_disambiguation.md)
- [\[ACL 2025\] RMoA: Optimizing Mixture-of-Agents through Diversity Maximization and Residual Compensation](rmoa_optimizing_mixture-of-agents_through_diversity_maximization_and_residual_co.md)
- [\[ACL 2025\] Minimal Pair-Based Evaluation of Code-Switching](minimal_pair-based_evaluation_of_code-switching.md)

</div>

<!-- RELATED:END -->
