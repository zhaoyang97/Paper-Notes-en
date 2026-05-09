---
title: >-
  [Paper Note] JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks
description: >-
  [ICLR 2026][LLM Alignment][Fake News Generation] This paper introduces JailNewsBench, the first multilingual and multi-regional benchmark for evaluating LLM robustness against fake news generation under jailbreak attacks. Covering 34 regions, 22 languages, and approximately 300,000 instances, the benchmark reveals attack success rates as high as 86.3% and exposes a systematic safety imbalance in which English- and U.S.-topic defenses are significantly weaker than those for other regions.
tags:
  - ICLR 2026
  - LLM Alignment
  - Fake News Generation
  - Jailbreak Attacks
  - Multilingual Safety
  - LLM Safety Evaluation
  - Regional Safety Imbalance
date: 2026-05-08
content_hash: a7feec390a78c242
---

# JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks

**Conference**: ICLR 2026
**arXiv**: [2603.01291](https://arxiv.org/abs/2603.01291)
**Code**: [https://github.com/kanekomasahiro/jail_news_bench](https://github.com/kanekomasahiro/jail_news_bench)
**Area**: Alignment & RLHF
**Keywords**: Fake News Generation, Jailbreak Attacks, Multilingual Safety, LLM Safety Evaluation, Regional Safety Imbalance

## TL;DR
This paper introduces JailNewsBench, the first multilingual and multi-regional benchmark for evaluating LLM robustness against fake news generation under jailbreak attacks. Covering 34 regions, 22 languages, and approximately 300,000 instances, the benchmark reveals attack success rates as high as 86.3% and exposes a systematic safety imbalance in which English- and U.S.-topic defenses are significantly weaker than those for other regions.

## Background & Motivation
Fake news poses severe threats to social trust and decision-making, affecting politics, economics, public health, and international relations. Because fake news inherently reflects region-specific political, social, and cultural contexts and is expressed in particular languages, assessing LLM safety risks demands a multilingual and multi-regional perspective.

Malicious actors can exploit jailbreak attacks to circumvent safety guardrails and induce LLMs to generate fake news. However, no existing benchmark systematically evaluates LLM robustness across diverse languages and regions. Current safety datasets such as HarmBench and TrustLLM primarily target toxicity and social bias, offering very limited coverage of fake news.

**Key Challenge**: LLM safety alignment is predominantly trained on English and general harmful content, whereas fake news is highly regionalized and language-specific, resulting in systematic blind spots in safety guardrails for non-English regions and languages.

**Key Insight**: Construct the first cross-lingual and cross-regional fake news jailbreak benchmark to systematically expose language- and region-level imbalances in LLM safety defenses.

## Method

### Overall Architecture
JailNewsBench is a benchmark rather than a methodological contribution. The overall pipeline consists of: (1) construction of multi-regional and multilingual fake news topics → (2) application of multiple jailbreak attack strategies → (3) multi-dimensional LLM-as-Judge evaluation → (4) cross-lingual and cross-regional safety analysis.

### Key Designs
1. **Multi-Regional and Multilingual Coverage**: The benchmark spans 34 regions and 22 languages. Fake news topics are tailored to the political, social, and cultural context of each region—for example, election manipulation in the United States and nuclear wastewater controversy in Japan. Each topic is region-specific rather than a simple translation. The design motivation is that the harm caused by fake news is highly dependent on geopolitical and cultural context.

2. **Five Jailbreak Attack Strategies**: The benchmark includes role play, system override, research front, negative prompting, context overload, and explicit requests. These strategies span a spectrum from simple to complex attack paradigms, ensuring comprehensive evaluation coverage.

3. **Eight-Dimensional LLM-as-Judge Evaluation**: Eight sub-metrics are used to assess the harmfulness of generated fake news: faithfulness, verifiability, adherence, scope, scale, formality, subjectivity, and agitativeness. Each dimension is scored from 0 to 4, with GPT-4o serving as the judge.

4. **Large-Scale Instance Set**: Approximately 300,000 instances (34 regions × 22 languages × 5 attacks × 9 models), far exceeding the scale of prior safety benchmarks.

### Loss & Training
As an evaluation benchmark, no training is involved. Evaluation metrics include Attack Success Rate (ASR), Influentness Rate (IFL), and average harmfulness score (avg_score). The evaluation pipeline proceeds as follows: (1) generate prompts based on topics and attack strategies → (2) obtain responses from target LLMs → (3) have GPT-4o score responses across eight dimensions. Evaluation scripts are open-sourced and support OpenAI, Anthropic, Gemini APIs, and local vLLM models, enabling comprehensive evaluation of any model with a single command. It is worth noting that results may vary slightly across data splits (train/val/test); care should be taken when comparing average scores against the numbers reported in the paper.

## Key Experimental Results

### Main Results

| Model | Metric (ASR) | Max ASR | Max Harm Score | English ASR vs. Others |
|-------|-------------|---------|----------------|------------------------|
| 9 LLMs | ASR | 86.3% | 3.5/5 | English/U.S. defense significantly weaker |
| GPT series | ASR | High | Medium–High | English regions relatively weaker |
| Claude series | ASR | Moderate | Moderate | Relatively balanced |
| Llama series | ASR | High | Medium–High | Stronger defense on non-English |

### Ablation Study

| Configuration | Key Metric | Remarks |
|--------------|-----------|---------|
| Different attack strategies | Large ASR variation | Role play and context overload are most effective |
| Different languages | Significant ASR differences | Low-resource languages show stronger defense (less training data → more conservative safety rules) |
| Different regions | Varying harm scores | English- and U.S.-related topics are most exploitable |
| Fake news vs. toxicity | Defense comparison | Fake news defenses are significantly weaker than toxicity defenses |

### Key Findings
- The maximum attack success rate reaches 86.3% with a maximum harmfulness score of 3.5/5, indicating that LLM defenses against fake news are far from adequate.
- Defenses for English and U.S.-related topics are significantly weaker than those for other regions—the American-centric perspective dominant in alignment training data may paradoxically introduce exploitable vulnerabilities.
- Fake news is severely underrepresented in existing safety datasets, resulting in defenses far weaker than those for toxicity and social bias.
- Multilingual LLMs tend to exhibit stronger safety behavior in non-English languages, likely because uneven safety training data distributions cause models to adopt a more conservative stance toward less frequently observed languages.

## Highlights & Insights
- Fills a critical gap in fake news generation safety evaluation as the first systematic cross-lingual and cross-regional study.
- Reveals a counterintuitive finding: English/U.S. defenses are the weakest, challenging the assumption that more training data implies better safety.
- The eight-dimensional evaluation framework provides a fine-grained quantitative tool for assessing the harmfulness of fake news.
- The 300,000-instance scale ensures statistical reliability.
- The dataset and evaluation scripts are open-sourced (HuggingFace: MasahiroKaneko/JailNewsBench), supporting single-command evaluation of any model.
- Five distinct jailbreak attack strategies (role play, system override, etc.) provide comprehensive attack surface coverage.
- Analysis reveals that fake news is severely neglected in existing safety datasets, with important implications for the construction of safety training data.
- Large cross-model and cross-language variations in safety behavior suggest that current safety RLHF generalizes poorly across languages.

## Limitations & Future Work
- LLM-as-Judge evaluation may introduce biases, particularly regarding evaluation quality and consistency for non-English languages.
- Only single-turn attacks are evaluated; multi-turn progressive jailbreaking (e.g., methods such as SEMA) may pose greater risks.
- The selected fake news topics may not fully capture the sensitive issues of every region, necessitating continuous updates.
- Attack strategies are relatively fixed; adaptive attacks with dynamic adjustment based on model feedback are not covered.
- Only textual fake news is considered; evaluation of multimodal fake news (e.g., image–text or video combinations) is an important future direction.
- The temporal validity of the benchmark is limited, as fake news topics evolve with current events, making periodic dataset updates essential.

## Related Work & Insights
- **vs. HarmBench/TrustLLM**: These general safety benchmarks do not focus on fake news and are primarily English-centric.
- **vs. SafetyBench**: SafetyBench covers diverse harmful categories but lacks multilingual and regional dimensions.
- **vs. Red-Teaming Methods**: This work is an evaluation benchmark rather than an attack method, but the safety imbalances it reveals offer guidance for the design of red-teaming strategies.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First multilingual and multi-regional fake news jailbreak benchmark, filling an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 34 regions × 22 languages × 5 attacks × 9 models; large-scale and comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and impactful findings.
- **Value**: ⭐⭐⭐⭐ Direct reference value for LLM safety research and policy-making.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks](sema_simple_yet_effective_learning_for_multi-turn_jailbreak_attacks.md)
- [\[ICLR 2026\] CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation](cage_a_framework_for_culturally_adaptive_red-teaming_benchmark_generation.md)
- [\[ICLR 2026\] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)
- [\[AAAI 2026\] AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)
- [\[ICLR 2026\] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)

<!-- RELATED:END -->
