---
title: >-
  [Paper Note] Investigating Counterfactual Unfairness in LLMs towards Identities through Humor
description: >-
  [ACL 2026][Image Generation][Counterfactual Fairness] This paper systematically investigates counterfactual unfairness in LLMs through humor scenarios—observing behavioral changes after swapping speaker/listener identities. Results reveal that jokes told by privileged-group speakers are refused at a rate as high as 67.5%, are judged as malicious with 64.7% higher probability, and receive social harm scores up to 1.5 points (on a 5-point scale), demonstrating that models have internalized fixed social privilege hierarchies rather than performing genuine social reasoning.
tags:
  - ACL 2026
  - Image Generation
  - Counterfactual Fairness
  - Humor Bias
  - Identity Bias
  - LLM Refusal Behavior
  - Social Privilege Hierarchy
date: 2026-05-08
content_hash: bed4b504d06ac88b
---

# Investigating Counterfactual Unfairness in LLMs towards Identities through Humor

**Conference**: ACL 2026
**arXiv**: [2604.18729](https://arxiv.org/abs/2604.18729)
**Code**: [GitHub](https://github.com/) (Code and Dataset available)
**Area**: AI Fairness / LLM Bias
**Keywords**: Counterfactual Fairness, Humor Bias, Identity Bias, LLM Refusal Behavior, Social Privilege Hierarchy

## TL;DR

This paper systematically investigates counterfactual unfairness in LLMs through humor scenarios—observing behavioral changes after swapping speaker/listener identities. Results reveal that jokes told by privileged-group speakers are refused at a rate as high as 67.5%, are judged as malicious with 64.7% higher probability, and receive social harm scores up to 1.5 points (on a 5-point scale), demonstrating that models have internalized fixed social privilege hierarchies rather than performing genuine social reasoning.

## Background & Motivation

**State of the Field**: LLMs are increasingly deployed in high-stakes domains (hiring, education, law), where encoded social and cultural biases can cause serious societal harm. Existing bias research focuses primarily on decontextualized bias detection, neglecting representational bias in conversational roles and social interactions.

**Limitations of Prior Work**: (1) Humor naturally occupies an ambiguous zone of social perception—what is acceptable depends on the identities of both speaker and listener; (2) existing computational humor research ignores conversational context and does not examine how meaning shifts when interactional roles change; (3) systematic methods for quantifying behavioral asymmetry in LLMs under identity swaps are lacking.

**Root Cause**: The safety alignment and bias mitigation of LLMs is bidirectional—models not only impose stricter scrutiny on privileged groups but also implicitly frame marginalized groups as vulnerable through over-protection, both of which reinforce fixed social hierarchies.

**Paper Goals**: To systematically expose identity-related bias in LLMs across humor scenarios through three complementary tasks: generation refusal, intent inference, and impact prediction.

**Starting Point**: Applying the principle of counterfactual fairness—changing only sensitive attributes (speaker/listener identity) while holding all other factors constant—to observe whether model outputs change. Humor serves as a particularly sensitive probe, as it forces models to make judgments in ambiguous territory.

**Core Idea**: LLMs encode fixed social privilege hierarchies rather than performing genuine social reasoning—they use identity as a proxy signal for harm, systematically refusing jokes that "punch down" while permitting those that "punch up," thereby creating bidirectional representational harm.

## Method

### Overall Architecture

Three tasks track bias manifestation across the full pipeline: Task 1 (humor generation refusal under speaker–target conditions) tests whether models differentially refuse based on identity configuration; Task 2 (speaker intent inference) tests whether models attribute different intentions to fixed joke content depending on identity; Task 3 (relational/social impact prediction) tests whether impact assessments are asymmetric across identities. The study covers 33 identities, 10 categories, and 5 state-of-the-art models.

### Key Designs

1. **Asymmetric Refusal Rate (ARR) and Speaker Effect (SE) Metrics**:

    - Function: Quantify directional bias in model behavior under identity swaps.
    - Mechanism: $\text{ARR} = |\text{RR}(A \to B) - \text{RR}(B \to A)|$ detects non-commutative safety policies. $\text{SE}(A \to B) = \text{RR}(A \to B) - \text{RR}(B)$ isolates the amplifying or dampening effect of speaker identity on target protection. A positive SE indicates that the specified speaker increases the refusal rate.
    - Design Motivation: Binary refusal rates are insufficient—it is necessary to separately capture "directional asymmetry" and "speaker-independent effect." ARR detects hierarchy; SE detects amplification.

2. **Multi-Granularity Refusal Type Analysis**:

    - Function: Reveal that model bias exists not only in *whether* to refuse, but in *how* to refuse.
    - Mechanism: Refusals are categorized into direct refusal, explicit substitution, and implicit substitution. GPT-4o directly refuses 62.7% of White→Black requests but only 25.0% of the reverse, while offering alternative jokes in 47.5% of the latter cases.
    - Design Motivation: The "severity" of refusal itself expresses hierarchical judgment—direct refusal implies absolute prohibition, while offering alternatives implies conditional permissibility.

3. **Identity-Agnostic vs. Identity-Specific Denigrating Humor Datasets**:

    - Function: Disentangle the effect of identity markers alone (even when jokes contain no identity content) from the effect of identity-laden content.
    - Mechanism: The identity-agnostic dataset (400 jokes, four styles) demonstrates that speaker–listener identity configurations influence model judgments even when jokes contain no identity markers. The identity-specific dataset (737 denigrating jokes) is rigorously filtered to ensure applicability across arbitrary identity pairings.
    - Design Motivation: If bias appeared only in identity-related jokes, it could be attributed to content. Its appearance in identity-agnostic jokes proves that models use identity itself as a judgment signal.

### Loss & Training

No training is involved. Five state-of-the-art models are evaluated (Claude 3.5 Haiku, GPT-4o, DeepSeek-Reasoner, Gemini 2.5 Flash-Lite, Grok 4) with a total of 48,400+ requests. GPT-4o is used as an automatic judge to assess refusal behavior.

## Key Experimental Results

### Main Results

**Highest ARR (%) per Category**

| Category | Identity Pair | Claude | GPT | DeepSeek | Gemini | Grok |
|----------|--------------|--------|-----|----------|--------|------|
| Wealth | poor, wealthy | 67.5 | 58.8 | 61.3 | 27.5 | 3.8 |
| Health | disabled, able-bodied | 50.0 | 47.5 | 63.8 | 48.8 | 10.0 |
| Race | Black, White | 16.3 | 43.8 | 43.8 | 33.8 | 25.0 |

### Ablation Study

**ARR Comparison: Grok 4 vs. Other Models**

| Model | Mean ARR | Notes |
|-------|---------|-------|
| Claude 3.5 Haiku | 38.1 | Strictest safety alignment |
| DeepSeek-Reasoner | 38.5 | High asymmetry |
| GPT-4o | 33.3 | Moderate |
| Gemini 2.5 | 27.1 | Relatively lenient |
| Grok 4 | 6.3 | Low scrutiny, yet racial ARR persists |

### Key Findings

- All models consistently refuse "punching down" (privileged→marginalized) jokes while permitting "punching up," mapping onto cultural conceptions of power hierarchy.
- Models implicitly encode contestable privilege judgments: Chinese is treated as more marginalized than American; janitor as more marginalized than software engineer.
- Grok 4 (trained with reduced censorship) shows significantly lower ARR overall, yet a 25% ARR persists on the racial dimension—suggesting that bias originates partly from training data rather than safety alignment alone.
- Negative SE effects reveal "protective discrimination"—speakers identified as blind or janitor are granted greater creative latitude.

## Highlights & Insights

- Using humor as a lens for probing bias is an inspired research design—the social sensitivity of humor forces models to expose their internalized social assumptions, which might otherwise be masked by safety filters in more direct tasks.
- The ARR and SE metrics are elegantly designed, separating two orthogonal dimensions of bias: directional asymmetry and speaker amplification effects.
- Identifying the bidirectionality of bias is a significant contribution—both over-protection and over-scrutiny are manifestations of bias; models are performing identity lookup rather than social reasoning.

## Limitations & Future Work

- Evaluation is limited to English-language scenarios; cross-cultural humor norms vary substantially.
- The selection of identity categories is grounded in a Western social framework.
- Minimal-pair experiments cover only a limited range of cognitive marker types.
- Further exploration of defensive and mitigation strategies is needed.

## Related Work & Insights

- **vs. BBQ/BOLD**: Those benchmarks test surface-level bias (lexical/sentiment); this paper tests deep bias embedded in social reasoning.
- **vs. Holistic Bias**: That work focuses on English descriptors; this paper addresses morphological realization across multiple languages.
- **vs. Safety alignment studies**: Prior work examines the overall level of refusal rates; this paper focuses on the directional asymmetry of refusal rates.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Highly creative framework for probing bias through humor; three-task design traces bias across the full pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 48,400+ requests, 5 models, 33 identities, 10 categories, multiple metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ Findings are thought-provoking and argumentation is compelling.
- Value: ⭐⭐⭐⭐⭐ Important implications for AI safety and fairness research.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] When Identities Collapse: A Stress-Test Benchmark for Multi-Subject Personalization](../../CVPR2026/image_generation/when_identities_collapse_a_stress-test_benchmark_for_multi-subject_personalizati.md)
- [\[ACL 2026\] CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment](codial_interpretable_task-oriented_dialogue_systems_through_dialogue_flow_alignm.md)
- [\[ACL 2026\] Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions](large_language_models_are_bad_dice_players_llms_struggle_to_generate_random_numb.md)
- [\[ICLR 2026\] DoFlow: Flow-based Generative Models for Interventional and Counterfactual Forecasting](../../ICLR2026/image_generation/doflow_flow-based_generative_models_for_interventional_and_counterfactual_foreca.md)
- [\[AAAI 2026\] LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs](../../AAAI2026/image_generation/longllada_unlocking_long_context_capabilities_in_diffusion_llms.md)

<!-- RELATED:END -->
