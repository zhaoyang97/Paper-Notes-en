---
title: >-
  [Paper Note] Investigating Counterfactual Unfairness in LLMs towards Identities through Humor
description: >-
  [ACL 2026][Social Computing][Counterfactual Fairness] This paper systematically investigates counterfactual unfairness in LLMs within humor scenarios by observing changes in model behavior after swapping speaker/listener identities. Results show that jokes told by privileged groups have a refusal rate as high as 67.5%, are 64.7% more likely to be judged as malicious, and receive social harm scores up to 1.5 (on a 5-point scale). This reveals that models internalize fixed soci…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Counterfactual Fairness"
  - "Humor Bias"
  - "Identity Bias"
  - "LLM Refusal Behavior"
  - "Social Privilege Hierarchy"
date: 2026-05-08
content_hash: 8959cea603bc9363
---

# Investigating Counterfactual Unfairness in LLMs towards Identities through Humor

**Conference**: ACL 2026  
**arXiv**: [2604.18729](https://arxiv.org/abs/2604.18729)  
**Code**: [GitHub](https://github.com/) (Code and Dataset provided)  
**Area**: AI Fairness / LLM Bias  
**Keywords**: Counterfactual Fairness, Humor Bias, Identity Bias, LLM Refusal Behavior, Social Privilege Hierarchy

## TL;DR

This paper systematically investigates counterfactual unfairness in LLMs within humor scenarios by observing changes in model behavior after swapping speaker/listener identities. Results show that jokes told by privileged groups have a refusal rate as high as 67.5%, are 64.7% more likely to be judged as malicious, and receive social harm scores up to 1.5 (on a 5-point scale). This reveals that models internalize fixed social privilege hierarchies rather than performing genuine social reasoning.

## Background & Motivation

**Background**: LLMs are increasingly deployed in high-risk domains (recruitment, education, law), where encoded social and cultural biases can lead to significant social harm. Existing bias research primarily focuses on decontextualized bias detection, overlooking representational biases in conversational roles and social interactions.

**Limitations of Prior Work**: (1) Humor naturally involves ambiguous social perceptions—acceptability depends on the identities of the speaker and listener; (2) Existing computational humor research ignores conversational context and fails to examine how meaning shifts when interaction roles change; (3) There is a lack of systematic methods to quantify behavioral asymmetries in LLMs under identity swapping.

**Key Challenge**: LLM safety alignment and bias protection are bidirectional—models not only apply stricter scrutiny to privileged groups but also implicitly characterize marginalized groups as fragile through over-protection, both of which reinforce fixed social hierarchies.

**Goal**: Systematically reveal identity-related biases in LLMs across humor scenarios via three complementary tasks: refusal generation, intent inference, and impact prediction.

**Key Insight**: Utilize counterfactual fairness principles—varying only sensitive attributes (speaker/listener identity) while keeping other factors constant—to observe changes in model output. Humor serves as a particularly sensitive probe as it forces models to make judgments in ambiguous spaces.

**Core Idea**: LLMs encode fixed social privilege hierarchies rather than genuine social reasoning. They use identity as a proxy for harm, systematically refusing "punching down" jokes while allowing "punching up," thereby creating bidirectional representational harms.

## Method

### Overall Architecture

Three tasks track bias performance across the pipeline: Task 1 (Speaker-Target conditioned humor generation refusal) tests differential refusal based on identity configurations; Task 2 (Speaker intent inference) tests attributional intent differences given fixed joke content; Task 3 (Relationship/Social impact prediction) tests whether impact assessment is asymmetric due to identity. The study covers 33 identities, 10 categories, and 5 SOTA models.

### Key Designs

**1. Asymmetric Refusal Rate (ARR) and Speaker Effect (SE): Decoupling directional bias under identity swapping into two orthogonal dimensions**

Binary refusal rates alone cannot distinguish whether a model is equally strict for everyone or only for specific directions. The authors define Asymmetric Refusal Rate $\text{ARR} = |\text{RR}(A{\to}B) - \text{RR}(B{\to}A)|$, where $\text{RR}(A{\to}B)$ is the refusal rate when Group A tells a joke to Group B. A high ARR indicates the model employs asymmetric safety strategies when swapping roles, directly exposing internalized hierarchies. To isolate the specific impact of the speaker, the Speaker Effect $\text{SE}(A{\to}B) = \text{RR}(A{\to}B) - \text{RR}(B)$ is introduced, calculating the difference between the refusal rate when A is the speaker versus when no speaker is specified (targeting B only). ARR detects hierarchical direction, while SE detects speaker amplification.

**2. Multi-granular Refusal Type Analysis: Bias hidden in "How" models refuse, not just "If"**

Treating refusal as a binary 0/1 variable misses the intensity of model attitudes. The authors subdivide refusal into direct refusal, explicit substitution, and implicit substitution. The severity of refusal is itself a hierarchical stance: direct refusal implies "absolute prohibition," while providing an alternative joke implies "acceptable with modification." For example, GPT-4o's requests for White→Black result in $62.7\%$ direct refusal, whereas the reverse Black→White results in only $25.0\%$ direct refusal but $47.5\%$ explicit substitutions.

**3. Identity-Agnostic vs. Identity-Specific Derogatory Humor Datasets: Isolating effects of identity markers vs. content**

To ensure bias isn't solely attributed to joke content, two datasets were created: an identity-specific set of $737$ derogatory jokes filtered to fit any identity pair, and an identity-agnostic set of $400$ jokes without any identity markers. Evidence from the latter showed that even when jokes do not mention identity, simply changing the speaker-listener configuration alters model judgments. This proves models use identity as a signal for harm rather than performing social reasoning on the humor content.

### Loss & Training

No training involved. Evaluated 5 SOTA models (Claude 3.5 Haiku, GPT-4o, DeepSeek-Reasoner, Gemini 2.5 Flash-Lite, Grok 4) with 48,400+ total requests. GPT-4o served as an automated judge to evaluate refusal behavior.

## Key Experimental Results

### Main Results

**Highest ARR per Category (%)**

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

- Models consistently refuse "punching down" (privileged → marginalized) jokes while permitting "punching up," mapping to cultural power hierarchies.
- Models implicitly encode controversial privilege judgments: Chinese is perceived as more vulnerable than American; janitor as more vulnerable than software engineer.
- Grok 4 (trained with low scrutiny) shows significantly reduced ARR, but a 25% ARR remains in the racial dimension, suggesting bias stems partially from training data rather than just safety alignment.
- Negative SE effects reveal "protective discrimination"—identities like "blind" or "janitor" as speakers receive greater creative freedom.

## Highlights & Insights

- Using humor to observe bias is an excellent research design; the social sensitivity of humor forces models to expose internalized social assumptions that might be masked by safety filters in direct tasks.
- The design of ARR and SE metrics is sophisticated, separating directional asymmetry from speaker amplification effects as orthogonal dimensions of bias.
- Identifying the bidirectionality of bias is a major contribution—over-protection and over-scrutiny are both forms of bias. Models appear to perform identity lookups rather than social reasoning.

## Limitations & Future Work

- Evaluation is limited to English, whereas cross-cultural humor norms vary significantly.
- Selection of identity categories is based on Western social frameworks.
- Minimal pair experiments cover only limited types of cognitive markers.
- Requires further exploration of defense and mitigation strategies.

## Related Work & Insights

- **vs. BBQ/BOLD**: These test surface bias (vocabulary/sentiment); Ours tests deep bias in social reasoning.
- **vs. Holistic Bias**: Focuses on English descriptors; Ours focuses on morphological implementation across interaction roles.
- **vs. Safety alignment studies**: These focus on overall refusal levels; Ours focuses on directional asymmetry in refusal rates.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Highly creative framework using humor as a probe; three tasks track the entire pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 48,400+ requests, 5 models, 33 identities, 10 categories, multiple metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ Thought-provoking findings with robust argumentation.
- Value: ⭐⭐⭐⭐⭐ Significant implications for AI safety and fairness research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs](to_lie_or_not_to_lie_investigating_the_biased_spread_of_global_lies_by_llms.md)
- [\[ACL 2025\] Explicit vs. Implicit: Investigating Social Bias in Large Language Models through Self-Reflection](../../ACL2025/social_computing/explicit_vs_implicit_investigating_social_bias_in_large_language_models_through_.md)
- [\[ACL 2026\] Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation](dynamics_of_cognitive_heterogeneity_investigating_behavioral_biases_in_multi-sta.md)
- [\[ICLR 2026\] GRADIEND: Feature Learning within Neural Networks Exemplified through Biases](../../ICLR2026/social_computing/gradiend_feature_learning_within_neural_networks_exemplified_through_biases.md)
- [\[ICLR 2026\] Tracing and Reversing Edits in LLMs](../../ICLR2026/social_computing/tracing_and_reversing_edits_in_llms.md)

</div>

<!-- RELATED:END -->
