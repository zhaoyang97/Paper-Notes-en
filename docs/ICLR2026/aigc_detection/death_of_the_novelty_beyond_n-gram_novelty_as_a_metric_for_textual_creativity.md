---
title: >-
  [Paper Note] Death of the Novel(ty): Beyond n-Gram Novelty as a Metric for Textual Creativity
description: >-
  [ICLR 2026][AIGC Detection][textual creativity] Through close reading annotations of 8,618 expressions by 26 professional writers, this paper demonstrates that n-gram novelty is insufficient for measuring textual creativity — approximately 91% of expressions with high n-gram novelty are not perceived as creative, and in open-source LLMs, high n-gram novelty negatively correlates with pragmaticality.
tags:
  - ICLR 2026
  - AIGC Detection
  - textual creativity
  - n-gram novelty
  - LLM evaluation
  - close reading
  - pragmaticality
date: 2026-05-08
content_hash: 6ad0a74cc07dc4ed
---

# Death of the Novel(ty): Beyond n-Gram Novelty as a Metric for Textual Creativity

**Conference**: ICLR 2026
**arXiv**: [2509.22641](https://arxiv.org/abs/2509.22641)
**Code**: [github.com/asaakyan/ngram-creativity](https://github.com/asaakyan/ngram-creativity)
**Area**: AIGC Detection
**Keywords**: textual creativity, n-gram novelty, LLM evaluation, close reading, pragmaticality

## TL;DR

Through close reading annotations of 8,618 expressions by 26 professional writers, this paper demonstrates that n-gram novelty is insufficient for measuring textual creativity — approximately 91% of expressions with high n-gram novelty are not perceived as creative, and in open-source LLMs, high n-gram novelty negatively correlates with pragmaticality.

## Background & Motivation

**N-gram novelty is widely used for creativity evaluation**: Tools such as infini-gram have made it feasible to compute n-gram novelty over trillion-token corpora, and metrics like the Creativity Index rely heavily on n-gram novelty to measure textual creativity.

**The standard psychological definition of creativity requires dual attributes**: creativity = novelty + appropriateness. N-gram novelty alone fails to capture the complete definition of creativity.

**Two sub-dimensions of appropriateness**: The paper decomposes "appropriateness" into "sensicality" (whether an expression is semantically coherent in isolation) and "pragmaticality" (whether an expression is contextually natural and reasonable).

**Growing concerns about creativity under LLM-assisted writing**: Prior research suggests that LLM-assisted writing may lead to collective creativity decline, homogenization, and the proliferation of AI slop.

**Limitations of existing evaluation approaches**: Automated n-gram metrics are overly coarse, expert manual evaluation is difficult to scale, and whether LLM-as-a-Judge can substitute for expert judgment remains unclear.

**Core research questions**: What is the true relationship between n-gram novelty and creativity as judged by human experts? Can LLMs replicate expert close reading creativity judgments?

## Method

### Overall Architecture

The study comprises three main components: (1) data collection — close reading annotations by professional writers; (2) statistical modeling — hierarchical logistic regression to analyze the relationship between n-gram novelty and creativity; (3) LLM-as-a-Judge evaluation — testing the ability of frontier LLMs to replicate expert judgments.

### Key Designs

**Dataset construction**:
- 50 human-written passages (~400 words each) sampled from New Yorker fiction
- 25 LLM-generated passages each from OLMo (7B) and OLMo-2 (32B), prompted via summaries of human texts
- An additional 5 passages each from GPT-5 and Claude 4.1 for exploratory analysis
- 26 professional writers from top MFA programs performed annotations; each batch of 10 passages was annotated by 3 annotators
- Annotation items: sensicality, pragmaticality, perceived novelty, and free highlighting of creative expressions

**N-gram novelty measurement**:
- Infini-gram's ∞-probability is used to compute expression-level perplexity as a proxy for n-gram novelty
- Reference corpora: OLMo and OLMo-2 respective training corpora (2.6T / 4.2T tokens)

**Statistical modeling**:
- Hierarchical/multilevel logistic regression (GLMM) with random intercepts controlling for inter-annotator variance and passage-level topic effects
- Outcome variable: creativity (sensical + pragmatic + novel simultaneously)
- Predictors: log-normalized perplexity, generation source (human/OLMo/OLMo-2), and their interactions

### LLM-as-a-Judge Evaluation

- **Task definition**: Given a passage, models extract perceived novel or non-pragmatic expressions; F1 is computed using a matching threshold of normalized indel similarity ≥ 90%
- **Models evaluated**: Zero-shot/few-shot testing of GPT-5, Claude 4.1/4.5, Gemini 2.5 Pro/3 Pro; fine-tuning of OLMo2 7B, Qwen3 8B, Llama-3.1 8B, GPT-4.1, and Gemini 2.5 Pro
- **Out-of-distribution validation**: Style Mimic and LMArena datasets are used to validate alignment between LLM-J novelty scores and expert/crowdsourced preferences

## Key Experimental Results

### Main Results

| Metric | Value |
|--------|-------|
| Total annotated expressions | 8,618 (2,783 unique expressions) |
| Perceived novel expressions | 589 unique expressions |
| Non-pragmatic expressions | 722 unique expressions |
| Creative highlights | 241 novel expressions |
| Inter-annotator agreement $\kappa_\text{free}$ (novelty) | 0.78 (sd=0.11) |
| Inter-annotator agreement $\kappa_\text{free}$ (pragmaticality) | 0.72 (sd=0.12) |

**Core finding**: N-gram perplexity is significantly positively correlated with creativity (OR ≈ 1.96/SD, p < 0.001); however, approximately 91% of expressions in the highest n-gram novelty quartile (n=3,928) were not judged as creative, and approximately 25% of creative expressions fall below the mean perplexity.

### Ablation Study

**Creativity by generation source (EMM contrasts vs. Human)**:

| Model | Odds Ratio | 95% CI | p-value |
|-------|-----------|--------|---------|
| Claude 4.1 | 0.521 | [0.289, 0.939] | 0.024 |
| GPT-5 | 0.511 | [0.279, 0.938] | 0.024 |
| OLMo | 0.500 | [0.370, 0.676] | <0.001 |
| OLMo-2 | 0.588 | [0.439, 0.788] | <0.001 |

Expressions from all LLMs are significantly less likely to be judged as creative compared to human-written text.

**LLM-as-a-Judge performance**:
- Novel expression detection F1: reasoning models ≈ 41.3 (few-shot GPT-5), random baseline 9.6
- Non-pragmatic expression detection F1: reasoning models ≈ 13.5, random baseline 2.3
- Non-pragmatic expression detection is substantially harder than novelty detection

### Key Findings

1. **In open-source LLMs, n-gram novelty negatively correlates with pragmaticality**: OLMo β=−0.17 (p=0.027), OLMo-2 β=−0.48 (p<0.001); no such effect is observed in human writing (β=0.01, p=0.92)
2. **AI detector scores do not predict creativity**: Likelihood scores from the Pangram AI detector show no significant association with expression-level creativity or pragmaticality
3. **Writing quality reward models predict creativity**: Reward model scores are significantly positively associated with both creativity (OR=1.30, p<0.001) and pragmaticality (OR≈1.33, p<0.001)
4. **LLM-J novelty scores outperform the Creativity Index**: On the Style Mimic dataset, LLM-J novelty (β=0.63, p=0.014) better predicts expert preference than the Creativity Index (β=0.51, p=0.038)

## Highlights & Insights

- **Operationalizing the psychological definition of creativity**: Decomposing creativity into three annotatable dimensions — sensicality, pragmaticality, and perceived novelty — lays the groundwork for automated evaluation
- **A counterintuitive finding that undermines the authority of n-gram novelty**: 91% of high n-gram novelty expressions are not perceived as creative, serving as an important warning for all downstream work relying on n-gram-based metrics
- **A novelty–pragmaticality trade-off in LLM writing**: The more LLMs attempt to generate novel text, the more likely they are to produce contextually inappropriate expressions — a trade-off absent in human writing
- **The intrinsic value of the close reading annotation paradigm**: The study contributes an expression-level, fine-grained annotation dataset for textual creativity research

## Limitations & Future Work

- The exploratory study on frontier closed-source models (GPT-5/Claude 4.1) is limited in scale (only 5 passages each), resulting in insufficient statistical power
- The study focuses exclusively on fiction; applicability to other genres (poetry, technical writing, journalism) remains unknown
- Using perplexity as a proxy for n-gram novelty may introduce measurement noise, particularly for closed-source models whose training data are inaccessible
- LLM-as-a-Judge performs poorly on non-pragmatic expression detection (F1 < 20), leaving substantial room for improvement in automated evaluation
- All annotators are English-background MFA writers; cross-lingual and cross-cultural generalizability is unverified

## Related Work & Insights

- **Lu et al. (2025) Creativity Index**: This paper directly challenges the core assumption of that work, arguing that n-gram novelty cannot be equated with creativity
- **Chakrabarty et al. (2025) AI-Slop study**: Complementary relationship — this paper analyzes writing quality from both positive (creativity) and negative (non-pragmaticality) perspectives
- **McCoy et al. (2023)**: Report that GPT-2 exhibits degraded coherence at high n-gram novelty, consistent with this paper's findings on open-source LLMs
- **Implication**: The three-dimensional framework (sensicality + pragmaticality + novelty) proposed here could be applied to AIGC detection — not merely to detect "whether AI-generated," but also to assess "which dimension of generation quality is problematic"

## Rating

- ⭐ Novelty: 4.5/5 — Operationalizing the psychological definition of creativity and validating it at scale offers a distinctive perspective
- ⭐ Experimental Thoroughness: 4/5 — Main experimental design is rigorous with substantial annotation volume, though frontier model experiments are limited in scale
- ⭐ Writing Quality: 4.5/5 — Clear structure, rigorous statistical modeling, and professional exposition
- ⭐ Value: 4/5 — Offers direct practical guidance for creativity evaluation and AI writing quality assessment

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] CLAWS: Creativity Detection for LLM-Generated Solutions Using Attention Window of Sections](../../NeurIPS2025/aigc_detection/clawscreativity_detection_for_llm-generated_solutions_using_attention_window_of_.md)
- [\[ACL 2026\] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](../../ACL2026/aigc_detection/beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)
- [\[ICLR 2026\] PoliCon: Evaluating LLMs on Achieving Diverse Political Consensus Objectives](policon_evaluating_llms_on_achieving_diverse_political_consensus_objectives.md)
- [\[ICLR 2026\] Is Your Paper Being Reviewed by an LLM? Benchmarking AI Text Detection in Peer Review](is_your_paper_being_reviewed_by_an_llm_benchmarking_ai_text_detection_in_peer_re.md)
- [\[ICLR 2026\] Calibrating Verbalized Confidence with Self-Generated Distractors](calibrating_verbalized_confidence_with_self-generated_distractors.md)

<!-- RELATED:END -->
