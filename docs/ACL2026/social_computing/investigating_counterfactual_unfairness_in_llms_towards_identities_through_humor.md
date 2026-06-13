---
title: >-
  [Paper Note] Investigating Counterfactual Unfairness in LLMs towards Identities through Humor
description: >-
  [ACL 2026][Social Computing][Counterfactual Fairness] This paper systematically investigates counterfactual unfairness in LLMs through humor scenarios by observing changes in model behavior after swapping speaker/listene…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Counterfactual Fairness"
  - "Humor Bias"
  - "Identity Bias"
  - "LLM Refusal Behavior"
  - "Social Privilege Hierarchy"
date: 2026-05-08
content_hash: 3868719f5697b1a1
---

# Investigating Counterfactual Unfairness in LLMs towards Identities through Humor

**Conference**: ACL 2026  
**arXiv**: [2604.18729](https://arxiv.org/abs/2604.18729)  
**Code**: [GitHub](https://github.com/) (Code and Dataset provided)  
**Area**: AI Fairness / LLM Bias  
**Keywords**: Counterfactual Fairness, Humor Bias, Identity Bias, LLM Refusal Behavior, Social Privilege Hierarchy

## TL;DR

This paper systematically investigates counterfactual unfairness in LLMs through humor scenarios by observing changes in model behavior after swapping speaker/listener identities. The study finds that jokes told by privileged groups have a refusal rate as high as 67.5%, are 64.7% more likely to be judged as malicious, and receive social harm scores up to 1.5 points higher (on a 5-point scale), revealing that models internalize fixed social privilege hierarchies rather than performing genuine social reasoning.

## Background & Motivation

**Background**: LLMs are increasingly deployed in high-stakes domains (recruitment, education, law), where encoded social and cultural biases can lead to severe social harm. Existing bias research primarily focuses on decontextualized bias detection, overlooking representational biases in conversational roles and social interactions.

**Limitations of Prior Work**: (1) Humor naturally involves a gray area of social perception—what is acceptable depends on the identities of the speaker and the listener; (2) Existing computational humor research ignores conversational context and fails to examine "how meaning shifts when interaction roles change"; (3) There is a lack of systematic methods to quantify the asymmetry of LLM behavior under identity swapping.

**Key Challenge**: The safety alignment and bias protection of LLMs are bidirectional—models not only impose stricter scrutiny on privileged groups but also implicitly characterize marginalized groups as fragile through over-protection, both of which reinforce fixed social hierarchies.

**Goal**: Systematically reveal identity-related biases in LLMs within humor scenarios through three complementary tasks: refusal generation, intent inference, and impact prediction.

**Key Insight**: Utilize the principle of counterfactual fairness—changing only sensitive attributes (speaker/listener identity) while keeping other factors constant to observe whether model outputs change. Humor serves as a particularly sensitive probe because it forces models to make judgments in ambiguous areas.

**Core Idea**: Ours argues that LLMs encode fixed social privilege hierarchies rather than genuine social reasoning—they use identity as a proxy signal for harm, systematically refusing "punching down" jokes while permitting "punching up" jokes, thereby creating bidirectional representational harm.

## Method

### Overall Architecture

Three tasks track the performance of bias throughout the pipeline: Task 1 (Humor generation refusal conditioned on speaker-target) tests whether models differentiate refusal based on identity configurations; Task 2 (Speaker intent inference) tests whether models attribute different intentions based on identity given fixed joke content; Task 3 (Relationship/social impact prediction) tests if impact assessments are asymmetric due to identity. The study covers 33 identities, 10 categories, and 5 SOTA models.

### Key Designs

1.  **Asymmetric Refusal Rate (ARR) and Speaker Effect (SE) Metrics**:
    - **Function**: Quantify the directional bias of model behavior under identity swapping.
    - **Mechanism**: $ARR = |RR(A \rightarrow B) - RR(B \rightarrow A)|$ detects non-commutative safety policies. $SE(A \rightarrow B) = RR(A \rightarrow B) - RR(B)$ isolates the amplification/weakening effect of speaker identity on target protection. A positive SE indicates that a specific speaker increases the refusal rate.
    - **Design Motivation**: Binary refusal rates are insufficient—it is necessary to separate the dimensions of "directional asymmetry" and "speaker-independent effects." ARR detects hierarchies, while SE detects amplification effects.

2.  **Multi-granular Refusal Type Analysis**:
    - **Function**: Reveals that models are biased not only in *whether* they refuse but also in *how* they refuse.
    - **Mechanism**: Refusals are subdivided into direct refusal, explicit substitution, and implicit substitution. It was found that GPT-4o directly refuses 62.7% of White $\rightarrow$ Black requests, but for the reverse, it directly refuses only 25.0% and provides alternative jokes 47.5% of the time.
    - **Design Motivation**: The "severity" of a refusal is itself an expression of hierarchical judgment—direct refusal implies "absolutely not," while providing an alternative implies "permitted but requires modification."

3.  **Identity-Agnostic vs. Identity-Specific Deprecating Humor Datasets**:
    - **Function**: Isolate the effect of the identity markers themselves (even if the joke involves no identity) versus the effect of identity content.
    - **Mechanism**: The identity-agnostic dataset (400 jokes, four styles) proves that even when jokes contain no identity markers, speaker-listener configurations still influence model judgment. The identity-specific dataset (737 deprecating jokes) is strictly filtered to ensure usability across any identity pair.
    - **Design Motivation**: If bias only appeared in identity-related jokes, it could be attributed to content. However, bias appearing in identity-agnostic jokes proves that models use identity itself as a judgment signal.

### Loss & Training

No training involved. Evaluated 5 SOTA models (Claude 3.5 Haiku, GPT-4o, DeepSeek-Reasoner, Gemini 2.5 Flash-Lite, Grok 4), totaling 48,400+ requests. GPT-4o was used as an automated judge to evaluate refusal behavior.

## Key Experimental Results

### Main Results

**Highest ARR (%) per Category**

| Category | Identity Pair | Claude | GPT | DeepSeek | Gemini | Grok |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Wealth | poor, wealthy | 67.5 | 58.8 | 61.3 | 27.5 | 3.8 |
| Health | disabled, able-bodied | 50.0 | 47.5 | 63.8 | 48.8 | 10.0 |
| Race | Black, White | 16.3 | 43.8 | 43.8 | 33.8 | 25.0 |

### Ablation Study

**ARR Comparison: Grok 4 vs. Other Models**

| Model | Average ARR | Description |
| :--- | :--- | :--- |
| Claude 3.5 Haiku | 38.1 | Strictest safety alignment |
| DeepSeek-Reasoner | 38.5 | High asymmetry |
| GPT-4o | 33.3 | Moderate |
| Gemini 2.5 | 27.1 | Relatively mild |
| Grok 4 | 6.3 | Low scrutiny but racial ARR persists |

### Key Findings

- All models consistently refuse "punching down" (privileged $\rightarrow$ marginalized) jokes while allowing "punching up"—this maps to power hierarchies in cultural concepts.
- Models implicitly encode controversial privilege judgments: Chinese is perceived as more disadvantaged than American; janitor as more disadvantaged than software engineer.
- Grok 4 (trained with low scrutiny) shows significantly lower ARR, but a 25% ARR still exists in the racial dimension—suggesting bias partially stems from training data rather than just safety alignment.
- Negative SE effects reveal "protective discrimination"—blind and janitor speakers receive greater creative freedom.

## Highlights & Insights

- "Viewing bias through humor" is a brilliant research design—the social sensitivity of humor forces models to expose their internalized social assumptions, which might be hidden by safety filters in more direct tasks.
- The design of ARR and SE metrics is sophisticated—separating directional asymmetry and speaker amplification as two orthogonal dimensions of bias.
- Identifying the bidirectionality of bias is a major contribution—over-protection and over-scrutiny are both forms of bias; models are performing identity lookup rather than social reasoning.

## Limitations & Future Work

- Evaluation is limited to English contexts; cross-cultural humor norms vary significantly.
- Selection of identity categories is based on Western social frameworks.
- Minimal pair experiments only cover a limited type of cognitive markers.
- More exploration of defense and mitigation strategies is required.

## Related Work & Insights

- **vs. BBQ/BOLD**: These test surface bias (vocabulary/sentiment), while ours tests deep bias in social reasoning.
- **vs. Holistic Bias**: Focuses on English descriptors, while ours focuses on identity configurations in interaction.
- **vs. Safety alignment studies**: These focus on the overall level of refusal rates, while ours focuses on the directional asymmetry of refusal rates.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The framework design of probing bias through humor is highly creative, with three tasks tracking the full pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 48,400+ requests, 5 models, 33 identities, 10 categories, multiple metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ Thought-provoking findings and powerful argumentation.
- Value: ⭐⭐⭐⭐⭐ Significant implications for AI safety and fairness research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs](to_lie_or_not_to_lie_investigating_the_biased_spread_of_global_lies_by_llms.md)
- [\[ACL 2026\] Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation](dynamics_of_cognitive_heterogeneity_investigating_behavioral_biases_in_multi-sta.md)
- [\[ICLR 2026\] GRADIEND: Feature Learning within Neural Networks Exemplified through Biases](../../ICLR2026/social_computing/gradiend_feature_learning_within_neural_networks_exemplified_through_biases.md)
- [\[ICLR 2026\] Tracing and Reversing Edits in LLMs](../../ICLR2026/social_computing/tracing_and_reversing_edits_in_llms.md)
- [\[ACL 2026\] mdok-style at SemEval-2026 Task 9: Finetuning LLMs for Multilingual Polarization Detection](mdok-style_at_semeval-2026_task_9_finetuning_llms_for_multilingual_polarization_.md)

</div>

<!-- RELATED:END -->
