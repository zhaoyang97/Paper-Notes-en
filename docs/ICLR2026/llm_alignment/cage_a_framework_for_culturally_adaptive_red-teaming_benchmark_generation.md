---
title: >-
  [Paper Note] CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation
description: >-
  [ICLR 2026][LLM Alignment][red teaming] This paper proposes the CAGE framework, which decouples the adversarial structure of red-teaming prompts from their cultural content via a construct termed the Semantic Mold. CAGE…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "red teaming"
  - "cultural adaptation"
  - "semantic mold"
  - "multilingual safety"
  - "benchmark generation"
date: 2026-05-08
content_hash: 9c97bd94db047312
---

# CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation

**Conference**: ICLR 2026
**arXiv**: [2602.20170](https://arxiv.org/abs/2602.20170)
**Code**: [https://github.com/selectstar-ai/CAGE-paper](https://github.com/selectstar-ai/CAGE-paper)
**Area**: LLM Alignment
**Keywords**: red teaming, cultural adaptation, semantic mold, multilingual safety, benchmark generation

## TL;DR
This paper proposes the CAGE framework, which decouples the adversarial structure of red-teaming prompts from their cultural content via a construct termed the Semantic Mold. CAGE systematically adapts English red-teaming benchmarks to diverse cultural contexts, yielding culturally grounded prompts that achieve substantially higher attack success rates (ASR) than direct translation.

## Background & Motivation
**Background**: LLM safety evaluation predominantly relies on English red-teaming benchmarks (e.g., AdvBench, HarmBench), with cross-lingual evaluation typically achieved through direct translation. However, stereotypes, social norms, and legal frameworks vary considerably across cultures.

**Limitations of Prior Work**: Direct translation fails to capture cultural specificity—flag burning is protected speech in the United States but constitutes a criminal offense in South Korea; certain racial slurs carry meaning in English contexts but have no equivalent in Korean. Template-based generation approaches (e.g., KoBBQ) offer limited semantic diversity, while constructing native datasets from scratch (e.g., KorNAT) incurs prohibitive costs.

**Key Challenge**: A fundamental trade-off exists between cultural fidelity and scalability—high-fidelity datasets require manual authoring at small scale, while machine translation enables scale but sacrifices fidelity. No prior work achieves both simultaneously.

**Goal**: How can one preserve the adversarial structure of English red-teaming prompts while injecting culture-specific content from the target culture?

**Key Insight**: The paper treats a prompt's "adversarial intent" (what harmful act to elicit) and its "cultural content" (which specific entities or scenarios are invoked) as two separable dimensions.

**Core Idea**: The Semantic Mold decomposes a prompt into a slot-tagged structure that preserves the attack framework, which is then filled with culturally legitimate content from the target culture to achieve cultural grounding.

## Method

### Overall Architecture
A three-stage pipeline: (1) **Seed Collection**—prompts are gathered from six English red-teaming datasets, filtered via consensus voting among six frontier LLMs, and mapped onto a three-tier risk taxonomy; (2) **Refine-with-Slot**—prompts are rewritten into slot-tagged Semantic Molds; (3) **Translate-with-Context**—slots are populated with target-culture content to produce localized prompts.

### Key Designs

1. **Three-Tier Risk Taxonomy**:

    - **Function**: Defines a hierarchical classification of 5 risk domains → 12 categories → 53 types.
    - **Content**: Toxic Contents, Unfair Representation, Misinformation Harms, Info & Safety Harms, Malicious Use.
    - **Design Motivation**: Refines existing taxonomies (e.g., Weidinger et al.) to enable category-specific slot schema customization.

2. **Semantic Mold**:

    - **Function**: Defines the minimal semantic elements required to express a harmful scenario for each risk category.
    - **Mechanism**: Each category is assigned Required Slots and Optional Slots—e.g., Illegal Activities requires `[Act]`, `[Target]`, `[Method/Approach]`; Bias and Hate requires `[Target/Group]`, `[Attribute]`. Seed prompts are rewritten by an LLM into slot-tagged form (e.g., "There are rumors that [Fake Event] in [Time]"), preserving rhetorical structure while abstracting specific content.
    - **Design Motivation**: The Mold specifies *what semantic elements should be present* rather than *how sentences should be worded*, balancing linguistic diversity with adversarial fidelity in generated prompts.

3. **Cultural Content Repository Construction**:

    - **Function**: Collects localized slot-filling content for the target culture (e.g., South Korea).
    - **Dual-track strategy**: (a) *Taxonomy-Driven*—objective categorical content sourced from legislation, case law, and enforcement regulations; (b) *Trend-Driven*—trending topics and keywords automatically scraped from news portals and online communities.
    - **Quality Control**: Content is filtered via binary pass/fail classification rather than manual item-by-item authoring.

4. **Multi-Model Consensus Voting + Human Validation**:

    - **Function**: Ensures accurate classification of seed prompts.
    - Six frontier models (GPT-4.1, Claude 3.5/4, Gemini 2.5 Pro, Llama 3.3, Qwen 2.5) independently classify each prompt; only unanimous results are retained, followed by human verification.

## Key Experimental Results

### Main Results: KorSET vs. Translation Baseline ASR

| Category | Attack Method | Llama-3.1-8B | Qwen2.5-7B | gemma2-9B | EXAONE-3.5-7.8B | gemma3-12B |
|----------|--------------|-------------|------------|-----------|----------------|------------|
| Toxic Language | Direct | 32.8 | 11.9 | 27.2 | 27.0 | 13.5 |
| | GPTFuzzer | 35.3 | 39.3 | 28.8 | 41.8 | 39.5 |
| Misinformation | Direct | 48.8 | 21.2 | 20.9 | 13.9 | 12.3 |
| | GPTFuzzer | 47.4 | 56.3 | 56.3 | 50.4 | 42.6 |
| Malicious Use | Direct | 34.7 | 10.3 | 5.8 | 9.5 | 9.2 |
| | AutoDAN | 50.4 | 27.2 | 37.3 | 36.1 | 27.5 |

### CAGE vs. Direct Translation
Culturally grounded prompts generated by CAGE achieve substantially higher ASR than directly translated English benchmarks (see Tables 4–5 in the paper), validating the necessity of cultural adaptation. EXAONE, a Korean-optimized model, remains vulnerable on KorSET, demonstrating that linguistic competence does not imply safety competence.

### Key Findings
- CAGE generates 7,161 prompts for Korean (KorSET) spanning 12 categories and 53 types.
- Direct translation prompts typically yield ASR 15–30 percentage points lower than CAGE prompts, confirming the blind spots of "culturally naive" benchmarks.
- GPTFuzzer performs strongest on CAGE prompts; GCG is less effective in Korean, likely because gradient-based optimization degrades on non-English tokens.
- The framework is successfully transferred to Khmer (an extremely low-resource language), demonstrating cross-cultural scalability.

## Highlights & Insights
- **Core Insight of the Semantic Mold**: Decomposing red-teaming prompts into orthogonal dimensions of "attack structure" and "cultural content" not only enables scaling to arbitrary cultures but also allows researchers to precisely control variables—comparing the effect of different cultural fillings under an identical attack structure.
- **Quantitative Evidence of "Cultural Naivety"**: The paper provides the first systematic demonstration that directly translated safety benchmarks underestimate model vulnerability in non-English contexts, with direct implications for safety evaluation policy in global LLM deployment.
- **Low-Resource Language Extension**: Successful application to Khmer confirms that the framework does not depend on abundant resources in the target language.

## Limitations & Future Work
- The quality of the cultural content repository remains dependent on the availability of information sources for the target culture; extremely low-resource cultures may lack legal texts and news data.
- Slot schemas in the Semantic Mold are defined by human experts, inevitably introducing subjectivity.
- Validation is currently limited to Korean and Khmer; experiments across additional languages and cultures are needed.
- Generated prompts carry misuse risks; the paper focuses primarily on benchmark construction and provides limited discussion of usage restrictions.

## Related Work & Insights
- **vs. XSafety / PolyGuardPrompts (direct translation)**: Direct translation loses cultural context and yields lower ASR. CAGE preserves adversarial structure while substituting cultural content via the Semantic Mold.
- **vs. KoBBQ / MBBQ (template-based adaptation)**: Template-based methods are constrained by predefined entity lists and offer insufficient expressive diversity. CAGE's Mold defines semantics rather than syntax, producing more natural and diverse prompts.
- **vs. Align Once (MLC)**: MLC addresses multilingual safety from the training side, while CAGE addresses it from the evaluation side. The two are complementary—CAGE can be used to evaluate whether MLC-aligned models remain safe under culturally grounded scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The Semantic Mold concept is concise and powerful; this is the first systematic framework for cross-cultural red-teaming benchmark generation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Experiments span 5 models × 5 attack methods × 12 risk categories at considerable scale, though validation across more languages is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ The framework is described clearly and the taxonomy is thoroughly elaborated.
- **Value**: ⭐⭐⭐⭐ Addresses an important gap in cross-cultural safety evaluation with direct policy implications for global LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks](jailnewsbench_multi-lingual_and_regional_benchmark_for_fake_news_generation_unde.md)
- [\[ACL 2026\] STAR-Teaming: A Strategy-Response Multiplex Network Approach to Automated LLM Red Teaming](../../ACL2026/llm_alignment/star-teaming_a_strategy-response_multiplex_network_approach_to_automated_llm_red.md)
- [\[NeurIPS 2025\] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models](../../NeurIPS2025/llm_alignment/jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)
- [\[NeurIPS 2025\] PolyJuice Makes It Real: Black-Box, Universal Red Teaming for Synthetic Image Detectors](../../NeurIPS2025/llm_alignment/polyjuice_makes_it_real_black-box_universal_red_teaming_for_synthetic_image_dete.md)
- [\[ICLR 2026\] From Utterance to Vividity: Training Expressive Subtitle Translation LLM via Adaptive Local Preference Optimization](from_utterance_to_vividity_training_expressive_subtitle_translation_llm_via_adap.md)

</div>

<!-- RELATED:END -->
