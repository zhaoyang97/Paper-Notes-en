---
title: >-
  [Paper Note] Autalic: A Dataset for Anti-Autistic Ableist Language In Context
description: >-
  [ACL 2025][Anti-autistic ableism] This paper proposes Autalic, the first dataset dedicated to detecting anti-autistic ableist language in context. It contains 2,400 Reddit sentences annotated with context by experts from neurodivergent backgrounds. Experiments reveal that current LLMs (including DeepSeek, Llama3, Gemma2, and Mistral) exhibit severe disagreement with human judgment when identifying anti-autistic ableism (with an average Cohen's Kappa of only 0.091)…
tags:
  - "ACL 2025"
  - "Anti-autistic ableism"
  - "ableist language"
  - "contextual annotation"
  - "neurodiversity"
  - "Reddit dataset"
  - "LLM bias"
date: 2026-05-08
content_hash: c64a11126baba5cf
---

# Autalic: A Dataset for Anti-Autistic Ableist Language In Context

**Conference**: ACL 2025  
**arXiv**: [2410.16520](https://arxiv.org/abs/2410.16520)  
**Code**: [https://nrizvi.github.io/AUTALIC.html](https://nrizvi.github.io/AUTALIC.html)  
**Authors**: Naba Rizvi, Harper Strickland, Daniel Gitelman, Tristan Cooper, Alexis Morales-Flores, Michael Golden, Aekta Kallepalli, Akshat Alurkar, Haaset Owens, Saleha Ahmedi, Isha Khirwadkar, Imani Munyaka, Nedjma Ousidhoum
**Institution**: UC San Diego, Cardiff University
**Area**: Social Bias & Fairness / Hate Speech Detection  
**Keywords**: Anti-autistic ableism, ableist language, contextual annotation, neurodiversity, Reddit dataset, LLM bias

## TL;DR

This paper proposes Autalic, the first dataset dedicated to detecting anti-autistic ableist language in context. It contains 2,400 Reddit sentences annotated with context by experts from neurodivergent backgrounds. Experiments reveal that current LLMs (including DeepSeek, Llama3, Gemma2, and Mistral) exhibit severe disagreement with human judgment when identifying anti-autistic ableism (with an average Cohen's Kappa of only 0.091), highlighting the difficulty of this task.

## Background & Motivation

**Background**: The medical model of autism defines it as a "disease" or "deficit." While widely used in technical research, this framework conflicts with the concept of "neurodiversity," which advocates that all neurological types are valid forms of human diversity. Anti-autistic ableist language poses a significant challenge for NLP research due to its subtlety and context-dependency.

**Limitations of Prior Work**:
   - Prior toxicity detection datasets primarily focus on hate speech and offensive language, with almost none dedicated to the autistic community.
   - Out of 23 LLM bias evaluation datasets, only 3 concern disability, and none target autism.
   - Toxicity classifiers exhibit strong negative bias toward disability, tending to flag any disability-related text as toxic.
   - LLMs have been found to implicitly propagate ableist stereotypes.

**Core Motivation**: To construct an annotated dataset centered on the perspective of the autistic community, and to evaluate the capabilities of current NLP tools on this task.

## Method

### Overall Architecture

Data Collection (Reddit) → Annotator Training → Expert Annotation → Baseline Evaluation (Traditional Models + LLMs)

### Data Collection

1. **Data Source**: Reddit (text-oriented, fewer API restrictions)
2. **Search Keywords**: Including "autis*", "ASD", "aspergers", "disabilit*", etc.
3. **Collection Strategy**: For each target sentence, both preceding and succeeding context sentences are collected.
4. **Final Scale**: 2,400 target sentences + 2,014 preceding contexts + 2,400 succeeding contexts
5. **Sources**: 192 different subreddits, with main sources including r/Aspergers (116), r/Autism (88), r/AmITheAsshole (39), etc.
6. **Data Cleaning**: Exact-word search filtering to resolve abbreviation ambiguity, excluding non-English posts, and removing posts containing media.

### Annotation Design

#### Annotator Selection and Training
- Recruited 9 senior undergraduate student volunteers, divided into 3 groups (3 members per group) to annotate 800 instances each.
- Diverse backgrounds of annotators: at least 3 self-identified as neurodivergent, and 4 belonged to gender minorities.
- Provided comprehensive training: history of ableism (including Nazi eugenics) → limitations of the medical model → concepts of neurodiversity → contemporary examples of discrimination → explanation of annotation examples.
- Provided a glossary as a dynamic reference resource.

#### Annotation Labels

| Label | Meaning | Count |
|------|------|------|
| 1 (Ableist) | Contains anti-autistic ableist sentiments | 1,023 |
| 0 (Not Ableist) | Positive/neutral/in-group discussions | 5,582 |
| -1 (Needs More Context) | Classification cannot be determined | 595 |

**Note**: Annotators could refer to the context when labeling target sentences to determine intent (e.g., whether it is an in-group discussion, irony, etc.).

#### Annotation Results
- Final labels determined by majority vote: 242 (10%) ableist, 2,160 (90%) non-ableist.
- Fleiss' Kappa = 0.25 (the low agreement highlights the difficulty of the task).
- Completion time was significantly negatively correlated with agreement ($R = -0.644$, $p = 0.0096$)—annotators who completed the task immediately after training showed higher agreement.

### Importance of Context

The paper illustrates the critical role of context in judgment through detailed case studies:
- For instance, "it's good that at least there's no link between the two" is ambiguous when viewed in isolation.
- When context is provided, it is revealed that the author is discussing the false vaccine-autism link (anti-autistic stigmatization).
- Annotators were allowed to revise previous annotations as their understanding evolved.

## Experiments

### Experimental Setup

- **Traditional Baselines**: Logistic Regression (BoW), BERT (pre-trained + fine-tuned)
- **LLMs**: Gemma2, Mistral, Llama3, DeepSeek (all with $< 10\text{B}$ parameters)
- **Prompting Types**: Three terminologies—PFL (person-first: "people with autism"), IFL (identity-first: "autistic people"), and AA (conceptual: "anti-autistic")
- **Prompting Approaches**: Simple zero-shot vs. ICL (In-Context Learning, with examples drawn from the annotation training)

### Main Results

| Model | Mode | PFL F1 | IFL F1 | AA F1 |
|------|------|--------|--------|-------|
| LR (BoW) | Pre-trained | 0.20 | — | — |
| BERT | Pre-trained | 0.43 | — | — |
| BERT | Fine-tuned | **0.90** | — | — |
| Gemma2 | Zero-shot | 0.23 | 0.19 | 0.33 |
| Mistral | Zero-shot | 0.28 | 0.27 | 0.34 |
| Llama3 | Zero-shot | 0.09 | 0.10 | 0.15 |
| DeepSeek | Zero-shot | **0.58** | **0.57** | **0.59** |
| Gemma2 | ICL | 0.25 | 0.24 | 0.34 |
| Mistral | ICL | 0.31 | 0.24 | 0.34 |
| Llama3 | ICL | 0.14 | 0.14 | 0.11 |
| DeepSeek | ICL | 0.55 | 0.56 | 0.55 |

### Key Findings

1. **Severe disagreement between LLMs and humans**: The average Cohen's Kappa across all LLMs is only 0.091 ($\text{SD}=0.110$), which is far below a reliable agreement level.
2. **DeepSeek performs best but remains unreliable**: DeepSeek achieves the best and most consistent performance (unaffected by phrasing variations), but its agreement with humans is still only around 0.11.
3. **Sensitivity to terminology**:
    - The F1 score of Llama3 shifts by up to **67.49%** from PFL to AA, indicating that the model fails to understand that different descriptions point to the same phenomenon.
    - Agreement improves after ICL (e.g., Llama3's variance drops from 67.49% to 17.40%), but its absolute performance remains low.
4. **Mixed effects of ICL**: Llama3 (+22.96%) and Gemma2 (+12.68%) show substantial improvements, while DeepSeek's performance drops slightly.
5. **Fine-tuned BERT significantly outperforms all LLMs** ($\text{F1}=0.90$), although it initially suffers from a high false-positive rate.

### Error Analysis

An analysis of the top 10% sentences with consistent human annotations but high LLM disagreement reveals that LLMs severely **over-classify** ableism:
- Llama3 labeled 42 sentences as ableist, whereas the ground truth human labels were all 0 (non-ableist).
- Out of these 42 sentences, **29 are in-group community discussions**—using LLMs for content moderation would result in severe censorship of the community.
- 34 sentences contain words with negative connotations (such as "burden", "threat"), but they are not used in an anti-autistic context.
- Example: A sentence quotes an organization's viewpoint, and the author explicitly expresses disagreement with that viewpoint. Annotators correctly labeled it as "non-ableist", but the LLM misclassified it solely due to the presence of negative words.

### Inconsistency Analysis

Among the 100 sentences with high disagreement, the following were observed:
1. 48 sentences used medical model terminology or stereotypes (the terminology itself is controversial).
2. The remaining sentences required additional information beyond the provided context.

## Highlights & Insights

1. **First Dataset**: Autalic is the first annotated dataset targeting anti-autistic ableist language, filling a crucial gap in NLP fairness research.
2. **Centering the Autistic Perspective**: The annotators include neurodivergent individuals, and the training covers medical model critique and neurodiversity education—departing from the "critical distance" annotation paradigm employed by mainstream datasets.
3. **Quantification of Contextual Importance**: Through case studies and statistical analysis, the paper clearly demonstrates that out-of-context classification inevitably leads to massive misclassifications.
4. **Risks of Using LLMs for Content Moderation**: The over-classification tendency of LLMs risks silencing discussions within the autistic community, serving as a critical warning for content moderation strategies.
5. **Preserving Individual Annotations**: The release of all individual annotations (rather than just aggregated labels) supports subsequent research on annotation disagreement.
6. **Relationship Between Training and Completion Time**: Quantitative evidence shows that timely annotation (doing it immediately after training) has a major impact on agreement.

## Limitations & Future Work

1. **Data Selection Bias**: Reliance on keyword search and specific social media threads may miss implicit ableist expressions.
2. **Western-Centric Perspective**: It only reflects anti-autistic ableism in Western, English-speaking contexts; manifestations of discrimination in different cultures may vary significantly.
3. **Relatively Small Data Scale** (2,400 sentences), limiting the training of deep learning models.
4. **Computational Resource Constraints**: LLMs were not fine-tuned; only open-source models with $< 10\text{B}$ parameters were used, leaving larger models unevaluated.
5. **Broader Scope of Search Word "r*tard"** may introduce content not directly related to autism.
6. The data is primarily from 2023 and may not reflect earlier or more recent linguistic evolution.

## Related Work & Insights

- **Hate Speech Detection**: Waseem & Hovy (2016) → Founta (2018) → but mostly overlooking the disability dimension.
- **Disability Bias**: Toxicity detectors exhibit a negative bias against disability-related content (Narayanan Venkit et al., 2023) → LLMs propagate implicit bias (Gadiraju et al., 2023) → only 3 out of 23 bias datasets involve disability.
- **Annotation Disagreement Studies**: Plank et al. (2014) → Pavlick & Kwiatkowski (2019) → Leonardelli et al. (2021) → Autalic preserves all individual annotations to support disagreement studies.
- **Autism and AI**: Mainstream research mostly adopts the medical model (diagnosis/treatment) → Bottema-Beutel et al. (2021) critique the deficit framework → Autalic shifts toward a neurodiverse perspective.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — First anti-autistic ableist language dataset, filling an important research gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — A comprehensive combination of 4 LLMs + traditional baselines $\times$ 3 prompting terminologies $\times$ zero-shot/ICL.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Thorough ethical considerations, in-depth background elaboration, and intuitive case studies.
- **Value**: ⭐⭐⭐⭐ — Publicly available dataset, presenting crucial warnings for content moderation strategies.
- **Limitations**: Small data scale, English-only, and inability to fine-tune LLMs for deeper comparison.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Inducing Lexicons of In-Group Language with Socio-Temporal Context](inducing_lexicons_of_in-group_language_with_socio-temporal_context.md)
- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)
- [\[ACL 2025\] Task-Informed Anti-Curriculum by Masking Improves Downstream Performance on Text](task-informed_anti-curriculum_by_masking_improves_downstream_performance_on_text.md)
- [\[ACL 2025\] The Time Scale of Redundancy between Prosody and Linguistic Context](the_time_scale_of_redundancy_between_prosody_and_linguistic_context.md)
- [\[ACL 2025\] Contextual Experience Replay for Self-Improvement of Language Agents](contextual_experience_replay_for_self-improvement_of_language_agents.md)

</div>

<!-- RELATED:END -->
