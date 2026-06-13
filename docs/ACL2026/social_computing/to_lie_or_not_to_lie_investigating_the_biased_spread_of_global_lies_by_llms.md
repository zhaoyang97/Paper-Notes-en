---
title: >-
  [Paper Note] To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs
description: >-
  [ACL 2026][Social Computing][Misinformation] This paper proposes GlobalLies—a multilingual parallel dataset containing 440 misinformation generation templates and 6,867 entities (8 languages…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Misinformation"
  - "Multilingual Safety"
  - "Global Bias"
  - "Safety Classifiers"
  - "Retrieval-Augmented Fact-checking"
date: 2026-05-08
content_hash: 645a59bc054152c5
---

# To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs

**Conference**: ACL 2026  
**arXiv**: [2604.06552](https://arxiv.org/abs/2604.06552)  
**Code**: [GitHub](https://github.com/zohaib-khan5040/globallies)  
**Area**: AI Safety / Misinformation Generation  
**Keywords**: Misinformation, Multilingual Safety, Global Bias, Safety Classifiers, Retrieval-Augmented Fact-checking

## TL;DR

This paper proposes GlobalLies—a multilingual parallel dataset containing 440 misinformation generation templates and 6,867 entities (8 languages, 195 countries). It reveals systematic national and linguistic biases in LLM misinformation dissemination: misinformation generation rates for low-HDI countries are significantly higher (statistically correlated $\rho=-0.355$, $p=5\times10^{-7}$), compliance rates for low-resource languages are over 30% higher than English, and existing safety classifiers and RAG defenses provide uneven protection.

## Background & Motivation

**Background**: The powerful writing capabilities of LLMs have lowered the threshold for malicious actors to produce and spread misinformation at scale. Existing research focuses on LLM compliance regarding misinformation in medical and US political domains, but is primarily limited to English and Western contexts.

**Limitations of Prior Work**: (1) LLM safety alignment is highly uneven across languages—strongest in English, while low-resource languages are nearly unprotected; (2) For the same false claim, models selectively refuse or comply based on the country or person involved (e.g., refusing for UK politicians but complying for Lebanese politicians); (3) Existing safety classifiers (e.g., Llama Guard) lack effective categories for misinformation detection and exhibit massive cross-lingual performance gaps.

**Key Challenge**: LLMs are powerful dual-use technologies—assisting legitimate writing while being susceptible to malicious exploitation for large-scale misinformation. Current safety measures provide unequal protection across linguistic and regional dimensions, creating structural inequality in global information security.

**Goal**: Systematically investigate the global bias patterns of LLM misinformation generation, evaluate the effectiveness of existing safety defenses, and provide data resources for developing fairer mitigation strategies.

**Key Insight**: Construct a global-scale multilingual parallel dataset to precisely measure bias in LLM behavior through controlled variables (same content, different languages/countries).

**Core Idea**: The propensity of LLMs to spread misinformation is not random but systematically correlates with the target country's Human Development Index (HDI) and the resource level of the prompt language—Western countries and high-resource languages receive better protection.

## Method

### Overall Architecture

The research proceeds in three layers: (1) **Data Construction**—collecting real false claims → templating → multilingual translation → entity collection to form a fully parallel dataset; (2) **Misinformation Generation Analysis**—evaluating 8 core regions with human annotation (RQ1), then expanding to 195 countries using templates + entities with LLM-based evaluation (Global Analysis); (3) **Safety Defense Evaluation**—testing the effectiveness of safety classifiers and RAG fact-checking pipelines (RQ2).

### Key Designs

1. **GlobalLies Dataset Construction**:

    - Function: Provides the first global-scale multilingual parallel testbed for misinformation generation.
    - Mechanism: Collects verified false claims from trusted fact-checking sources in 8 regions, manually rewritten into misinformation generation prompts by native speakers, then translated into 8 languages (Arabic, English, Persian, French, Igbo, Nepali, Turkish, Urdu) to create a fully parallel corpus. By templating, countries/entities in prompts are replaced with placeholders, paired with 6,867 country-specific entities from Wikidata, expandable to 195 countries.
    - Design Motivation: The fully parallel design allows for precise control of variables—differences in LLM behavior for the same false claim across different languages/countries are attributable solely to linguistic or regional factors.

2. **Global Misinformation Spread Analysis**:

    - Function: Quantifies national and linguistic biases in LLM misinformation generation.
    - Mechanism: Generates 669,280 responses (Llama-3.3-70B) for 195 countries × 440 templates × 8 languages, using LLM judges for compliance/refusal (judgment accuracy 89.9%). Analyzes the correlation between misinformation generation rate and country HDI.
    - Design Motivation: Large-scale analysis reveals systematic patterns rather than anecdotal observations.

3. **Safety Defense Effectiveness Evaluation**:

    - Function: Tests the performance of existing safety measures in global misinformation scenarios.
    - Mechanism: (a) Safety Classifiers—testing Llama Guard 1/2/3 on 669,280 prompts to measure the proportion of misinformation prompts flagged as "unsafe"; (b) RAG Fact-checking—retrieving top-5 credible documents to let models judge if prompt content is evidence-supported, comparing misinformation rates with and without RAG.
    - Design Motivation: Evaluates whether safety barriers in real-world deployments effectively prevent global misinformation generation.

### Loss & Training

This work is analytical and evaluative and does not involve model training. The core metric is the Misinformation Generation Rate, the proportion of cases where the model complies and generates a false article.

## Key Experimental Results

### Main Results

**Misinformation Generation Rate (Human Annotation, 8 Core Regions)**

| Model | English (US) | English (Pakistan) | Nepali (US) | Urdu (Global) |
|------|---------|------------|------------|----------|
| Llama-3.3-70B | 0.68 | 0.90+ | 0.96-1.00 | 0.88+ |
| GPT-4o | ~0.70 | 0.85+ | - | 1.00 |

**Safety Classifier Detection Rate (Proportion of misinformation flagged as unsafe)**

| Guard Model | English | Arabic | Igbo | Urdu |
|-----------|------|--------|-------|--------|
| Llama Guard 1 | 4.2% | 5.5% | 1.4% | 0.7% |
| Llama Guard 2 | 6.1% | 5.0% | 2.4% | 10.2% |
| Llama Guard 3 | 42.6% | 46.7% | **9.1%** | 50.3% |

### Ablation Study

**Impact of RAG on Misinformation Generation**

| Setting | Misinformation Generation Rate | Factual Information Generation Rate |
|------|-------------|-------------|
| No RAG (0-shot) | ~80%+ | ~100% |
| With RAG | Decrease up to 53% | Also significantly decreased (Over-caution) |

### Key Findings

- Misinformation generation rate is significantly negatively correlated with country HDI ($\rho=-0.355$, $p=5\times10^{-7}$); low-HDI countries are more likely to have misinformation spread about them.
- Simply changing the prompt language can shift compliance rates by over 30% (e.g., Llama for Nigeria: English 0.69 vs. Nepali 1.00).
- Adding the "Defamation" category to Llama Guard 3 improved detection rates from <10% to 30-50%, yet Igbo remained at only 9.1%.
- RAG effectively reduces misinformation generation (up to 53%), but also leads to over-refusal of factual requests—an "over-skepticism" issue.
- Fact-checking accuracy is highest in language-native regions (e.g., Arabic for Arab countries) and drops significantly in cross-cultural scenarios.

## Highlights & Insights

- The fully parallel multilingual design is a key methodological innovation—making measurements of language and country bias causally rigorous.
- The discussion "Should LLMs write news?" is sharp and pragmatic—proposing a factual verification-based policy framework.
- The "over-skepticism" issue in RAG reveals a dilemma: enhancing safety while sacrificing utility.

## Limitations & Future Work

- Focuses only on textual misinformation, not multi-modal (image/video) false content.
- Template-country combinations may occasionally produce factually correct statements (dual labeling shows ~4% might be true).
- Did not analyze the difference in persuasiveness of LLM-generated false articles (binary compliance/refusal only).
- Large-scale experiments used Llama-3.3-70B as the primary generation model (GPT-4o limited by cost).

## Related Work & Insights

- **vs Vykopal et al.**: Previous work tested LLM compliance on only 20 narratives in English; GlobalLies expands to 440 templates × 8 languages × 195 countries.
- **vs Hussain et al.**: Previous work focused on 109 prompts in the medical domain; GlobalLies covers politics, economy, public health, religion, etc.
- **vs Monolingual Studies**: Previous Arabic/Chinese/Kazakh studies were independent; GlobalLies achieves cross-lingually comparable parallel evaluation for the first time.

## Rating

- Novelty: ⭐⭐⭐⭐ First global-scale multilingual parallel misinformation evaluation, discovering systematic bias.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 669K+ generations + human annotation + safety classifiers + RAG evaluation + statistical correlation analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured research questions with intuitive and powerful data visualization.
- Value: ⭐⭐⭐⭐⭐ Highlights global inequality in AI safety, with direct implications for policy-making.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Investigating Counterfactual Unfairness in LLMs towards Identities through Humor](investigating_counterfactual_unfairness_in_llms_towards_identities_through_humor.md)
- [\[ACL 2026\] Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation](dynamics_of_cognitive_heterogeneity_investigating_behavioral_biases_in_multi-sta.md)
- [\[ICLR 2026\] Tracing and Reversing Edits in LLMs](../../ICLR2026/social_computing/tracing_and_reversing_edits_in_llms.md)
- [\[ACL 2026\] mdok-style at SemEval-2026 Task 9: Finetuning LLMs for Multilingual Polarization Detection](mdok-style_at_semeval-2026_task_9_finetuning_llms_for_multilingual_polarization_.md)
- [\[ICML 2026\] FLIPS: Instance-Fingerprinting for LLMs via Pseudo-Random Sequences](../../ICML2026/social_computing/flips_instance-fingerprinting_for_llms_via_pseudo-random_sequences.md)

</div>

<!-- RELATED:END -->
