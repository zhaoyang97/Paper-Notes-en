---
title: >-
  [Paper Note] To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs
description: >-
  [ACL 2026][Social Computing][Misinformation] This paper proposes GlobalLies—a multilingual parallel dataset containing 440 misinformation generation templates and 6,867 entities (spanning 8 languages and 195 countries). It reveals systematic national and linguistic biases in LLM misinformation propagation: misinformation generation rates are significantly higher for low HDI countries (statistical correlation $\rho=-0.355$, $p=5\times10^{-7}$)…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Misinformation"
  - "Multilingual Safety"
  - "Global Bias"
  - "Safety Classifiers"
  - "Retrieval-Augmented Fact-checking"
date: 2026-05-08
content_hash: ee89b00d7f7efa50
---

# To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs

**Conference**: ACL 2026  
**arXiv**: [2604.06552](https://arxiv.org/abs/2604.06552)  
**Code**: [GitHub](https://github.com/zohaib-khan5040/globallies)  
**Area**: AI Safety / Misinformation Generation  
**Keywords**: Misinformation, Multilingual Safety, Global Bias, Safety Classifiers, Retrieval-Augmented Fact-checking

## TL;DR

This paper proposes GlobalLies—a multilingual parallel dataset containing 440 misinformation generation templates and 6,867 entities (spanning 8 languages and 195 countries). It reveals systematic national and linguistic biases in LLM misinformation propagation: misinformation generation rates are significantly higher for low HDI countries (statistical correlation $\rho=-0.355$, $p=5\times10^{-7}$), compliance rates for low-resource languages are over 30% higher than for English, and existing safety classifiers and RAG safeguards provide uneven protection.

## Background & Motivation

**Background**: The powerful writing capabilities of LLMs have lowered the threshold for malicious actors to produce and disseminate misinformation at scale. Existing research has focused on LLM misinformation compliance in medical and US political domains, but remains largely limited to English and Western contexts.

**Limitations of Prior Work**: (1) LLM safety alignment is highly uneven across languages—strongest in English, while low-resource languages are nearly unprotected. (2) Models selectively refuse or comply with the same false claim based on the involved country/person (e.g., refusing a prompt regarding a UK politician but complying for a Lebanese politician). (3) Existing safety classifiers (such as Llama Guard) lack effective categories for misinformation detection and exhibit massive performance gaps across languages.

**Key Challenge**: LLMs are powerful dual-use technologies—they can assist in legitimate writing but can also be maliciously exploited for large-scale misinformation propagation. Existing safety measures provide unequal protection across linguistic and regional dimensions, resulting in structural inequalities in global information security.

**Goal**: To systematically investigate global bias patterns in LLM misinformation generation, evaluate the effectiveness of existing safety protections, and provide data resources for developing more equitable mitigation strategies.

**Key Insight**: By constructing a global-scale multilingual parallel dataset, the authors precisely measure LLM behavioral biases through controlled variables (identical content across different languages/countries).

**Core Idea**: The propensity of LLMs to spread misinformation is not random; rather, it is systematically correlated with the Human Development Index (HDI) of the target country and the resource level of the prompt language—Western countries and high-resource languages receive significantly better protection.

## Method

### Overall Architecture

This paper seeks to answer not simply "whether LLMs lie," but "who they are more willing to lie about." To this end, the authors established a complete pipeline from data to measurement: first, they rewrote verified false claims from the real world into controllable generation prompts, translated them into 8 languages, and templated them for 195 countries to create the GlobalLies corpus. This corpus was then used to drive large-scale LLM generations, where compliance/refusal rates were statistically analyzed against national HDI. Finally, the authors stress-tested two existing safety lines—safety classifiers and RAG-based fact-checking—on the same corpus to determine if they provide consistent protection for low-resource languages and low-HDI countries.

### Key Designs

**1. GlobalLies Parallel Corpus: Isolating Bias Using "Controlled Variables"**

If misinformation prompts are chosen arbitrarily, observed differences in compliance rates could stem from content, language, or country, making it impossible to isolate the cause. The authors decoupled these dimensions: they collected verified false claims from trusted fact-checking sources across 8 regions, manually rewrote them into misinformation generation prompts by native speakers, and translated them into 8 languages (Arabic, English, Persian, French, Igbo, Nepali, Turkish, and Urdu). By replacing country/person names with placeholders, they created 440 templates paired with 6,867 country-specific entities from Wikidata, enabling expansion to 195 countries. Because the phrasing of the same false claim is identical across all languages and countries, any variation in model behavior can be attributed solely to the language or country variables—giving the conclusions (such as "low-HDI countries are prone to more misinformation") a level of rigor approaching causal inference.

**2. Global Propagation Analysis: Turning Cases into Systemic Patterns via 670k Generations**

Using the parallel corpus, the authors generated 669,280 responses across 195 countries × 440 templates × 8 languages using Llama-3.3-70B. An LLM-based discriminator was used to label whether the model complied by generating a false article or refused (human validation showed 89.9% classification accuracy). The core metric is the **Misinformation Generation Rate**, defined as the proportion of requests where the model complied with producing a false article. Correlation analysis between this rate and the target country's HDI yielded a significant negative correlation of $\rho=-0.355$ ($p=5\times10^{-7}$). This scale of statistics elevates anecdotal observations into a documented systemic structural bias.

**3. Safety Safeguard Stress Testing: Measuring "Protection Uniformity"**

After identifying the bias, the authors evaluated whether existing defenses could mitigate it. Using the same 669,280 prompts, they conducted two types of assessments. First, they tested safety classifiers (Llama Guard 1/2/3) to see the proportion of misinformation prompts flagged as "unsafe" across different languages. Second, they evaluated RAG-based fact-checking: retrieving the top-5 credible documents and tasking the model to judge if the prompt was supported by evidence. By comparing misinformation generation rates with and without RAG, they measured the effectiveness of "verify before answering." Testing both defenses on the same parallel corpus allowed the authors to align linguistic/regional protection gaps with the previously observed generation biases—revealing side effects like "excessive skepticism," where RAG causes models to reject even factual requests.

### Loss & Training

This work is an analysis and evaluation study and does not involve training models; hence, there is no loss function. The core metric throughout is the Misinformation Generation Rate (the proportion of compliant false article generations). For safety classifiers, the metric is the detection hit rate (flagging as unsafe), while the RAG evaluation compares the change in generation rates before and after activating RAG.

## Key Experimental Results

### Main Results

**Misinformation Generation Rate (Human Annotated, 8 Core Regions)**

| Model | English (USA) | English (Pakistan) | Nepali (USA) | Urdu (Global) |
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
| With RAG | Decreased by up to 53% | Also decreased significantly (Excessive caution) |

### Key Findings

- The misinformation generation rate is significantly negatively correlated with a country's HDI ($\rho=-0.355$, $p=5\times10^{-7}$); lower HDI countries face a higher probability of misinformation propagation.
- Merely changing the prompt language can shift compliance rates by over 30% (e.g., Llama for Nigeria: English 0.69 vs. Nepali 1.00).
- Adding a "Defamation" category to Llama Guard 3 increased detection rates from <10% to 30-50%, yet Igbo remained at only 9.1%.
- RAG effectively reduces misinformation generation (up to 53%) but simultaneously leads to the over-rejection of factual requests—an "excessive skepticism" problem.
- Fact-checking accuracy is highest in the native regions of the language (e.g., Arabic for Arab countries) and drops significantly in cross-cultural scenarios.

## Highlights & Insights

- The fully parallel multilingual design is a key methodological innovation, enabling rigorous causal inference for language and country bias measurements.
- The discussion on "Should LLMs write the news?" is sharp and pragmatic, proposing a policy framework based on factual verification.
- The "excessive skepticism" problem in RAG reveals a dilemma: enhancing safety often comes at the cost of utility.

## Limitations & Future Work

- Focuses only on textual misinformation, excluding multimodal (image/video) false content.
- Template-country combinations occasionally produce factually correct claims (dual annotation shows ~4% may be true).
- Does not analyze the differences in the persuasiveness of LLM-generated false articles (only binary compliance/refusal classification).
- Large-scale experiments primarily used Llama-3.3-70B as the generator (GPT-4o was limited due to cost).

## Related Work & Insights

- **vs. Vykopal et al.**: Previous work tested LLM compliance on only 20 narratives in English; GlobalLies expands this to 440 templates × 8 languages × 195 countries.
- **vs. Hussain et al.**: Previous work focused on 109 prompts in the medical domain; GlobalLies covers politics, economics, public health, religion, and more.
- **vs. Monolingual Studies**: Previous studies on Arabic, Chinese, or Kazakh were independent; GlobalLies achieves the first cross-lingually comparable parallel evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐ First global-scale multilingual parallel misinformation evaluation, identifying systemic biases.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 669K+ generations + human annotation + safety classifier + RAG evaluation + statistical correlation analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Research questions are logically progressive; data visualization is intuitive and compelling.
- Value: ⭐⭐⭐⭐⭐ Highlights global inequality in AI safety, with direct implications for policy-making.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Investigating Counterfactual Unfairness in LLMs towards Identities through Humor](investigating_counterfactual_unfairness_in_llms_towards_identities_through_humor.md)
- [\[ACL 2025\] Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection](../../ACL2025/social_computing/llm_label_propagation.md)
- [\[ACL 2026\] Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation](dynamics_of_cognitive_heterogeneity_investigating_behavioral_biases_in_multi-sta.md)
- [\[ICLR 2026\] Tracing and Reversing Edits in LLMs](../../ICLR2026/social_computing/tracing_and_reversing_edits_in_llms.md)
- [\[ACL 2026\] mdok-style at SemEval-2026 Task 9: Finetuning LLMs for Multilingual Polarization Detection](mdok-style_at_semeval-2026_task_9_finetuning_llms_for_multilingual_polarization_.md)

</div>

<!-- RELATED:END -->
