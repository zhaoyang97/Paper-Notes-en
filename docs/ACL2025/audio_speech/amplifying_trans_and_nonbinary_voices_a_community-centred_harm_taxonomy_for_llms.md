---
title: >-
  [Paper Note] Amplifying Trans and Nonbinary Voices: A Community-Centred Harm Taxonomy for LLMs
description: >-
  [ACL 2025][Audio & Speech][Transgender] This paper adopts a community-centred research methodology, building a specialized harm taxonomy for LLM outputs affecting Trans and Nonbinary (TNB) individuals through deep collaboration with the TNB community, thereby uncovering unique harm categories unaddressed by existing LLM safety evaluations.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Transgender"
  - "Non-binary"
  - "Harm Taxonomy"
  - "Community-centred"
  - "LLM Bias"
date: 2026-05-08
content_hash: 823cd88905196c37
---

# Amplifying Trans and Nonbinary Voices: A Community-Centred Harm Taxonomy for LLMs

**Conference**: ACL 2025  
**Code**: None  
**Area**: Audio & Speech  
**Keywords**: Transgender, Non-binary, Harm Taxonomy, Community-centred, LLM Bias

## TL;DR
This paper adopts a community-centred research methodology, building a specialized harm taxonomy for LLM outputs affecting Trans and Nonbinary (TNB) individuals through deep collaboration with the TNB community, thereby uncovering unique harm categories unaddressed by existing LLM safety evaluations.

## Background & Motivation

**Background**: Safety evaluation and red-teaming of large language models have received widespread attention in recent years. Existing safety classification frameworks (such as OpenAI's usage policies, Anthropic's model card) cover general harmful categories such as hate speech, discrimination, and violence.

**Limitations of Prior Work**: (1) Existing harm classification frameworks severely under-represent the unique types of harm faced by the TNB community—such as misgendering, erasure of identity journey, and misleading medical information; (2) Safety evaluations are typically designed by technical researchers without direct participation of the affected groups, leading to blind spots in the taxonomy; (3) The TNB community is a high-risk group vulnerable to LLM biases, yet systematic evaluation tools centered on this community are almost non-existent.

**Key Challenge**: LLM safety evaluation systems are defined "top-down" by developers rather than co-constructed "bottom-up" by the affected communities. This leads to the systematic neglect of specific harm categories faced by marginalized groups.

**Goal**: To build an LLM harm taxonomy reflecting the genuine experiences and needs of the TNB community through deep collaboration with its members.

**Key Insight**: Adopting the Participatory Action Research methodology, empowering TNB community members as co-researchers throughout the process of defining, collecting examples for, and validating the harm taxonomy.

**Core Idea**: To establish a community-centred TNB harm taxonomy through community interviews, focus groups, and collaborative coding, incorporating multiple harm categories omitted in existing frameworks to provide more inclusive evaluation dimensions for LLM safety assessments.

## Method

### Overall Architecture
The study follows a three-stage participatory process: (1) Exploration stage — understanding the negative experiences of TNB individuals when interacting with LLMs through semi-structured interviews; (2) Construction stage — coding these experiences into a systematic harm taxonomy using focus groups and thematic analysis; (3) Validation stage — validating the coverage and accuracy of the taxonomy within a larger TNB community.

### Key Designs

1. **Community-Centred Data Collection**:

    - Function: Extracting LLM harm patterns from first-hand experiences of the TNB community.
    - Mechanism: Recruiting TNB participants from diverse backgrounds (race, age, transition phase) to collect their interaction experiences with various LLMs (ChatGPT, Claude, Gemini, etc.) through in-depth interviews. Recording harm scenarios, emotional impacts, and desired improvement paths. Utilizing a "double-coding" method where each case is co-coded by a TNB participant and a researcher.
    - Design Motivation: Only the affected community can accurately identify and characterize behavioral patterns that might seem negligible to "outsiders" but are actually highly harmful.

2. **Multi-level Harm Taxonomy**:

    - Function: Providing a structured evaluation framework for TNB-related harms.
    - Mechanism: Inducing harm categories from interview data using Grounded Theory. The system encompasses multiple levels: (1) Representational Harm — including misgendering, stereotyping, identity erasure, and overpathologization; (2) Allocative Harm — such as withholding reasonable medical information or over-censoring TNB-related queries; (3) Participatory Harm — where system design prevents expressing non-binary identities or forces binary categorization. Each category is appended with real LLM output examples and impact assessments.
    - Design Motivation: Drawing on Crawford's AI harm classification framework but adapting it with TNB-specific refinement and expansion.

3. **Intersectionality Considerations**:

    - Function: Uncovering compounded harms when TNB identity intersects with other marginalized identities.
    - Mechanism: Analyzing how LLM harm amplifies when TNB identity intersects with factors like race, disability, and socioeconomic status. For example, TNB individuals of color face dual stereotyping on gender and race, and non-English TNB users suffer from more severe misgendering patterns. Intersectionality is sustained as an analytical dimension throughout the taxonomy.
    - Design Motivation: Harm is not single-dimensional; neglecting intersectionality would lead evaluation frameworks to underestimate the true degree of victimization in certain subgroups.

### Loss & Training
This work centers on building a harm taxonomy and does not involve model training. Consistency reviews by community members and Cohen's Kappa metrics are used for validation.

## Key Experimental Results

### Main Results

| LLM | Misgendering Frequency | Stereotyping | Over-censoring | Identity Erasure | Total Harm Rate |
|-----|-----------|---------|---------|---------|---------|
| ChatGPT-4 | 12.3% | 23.5% | 18.7% | 8.2% | 62.7% |
| Claude-3 | 8.1% | 19.2% | 22.4% | 6.5% | 56.2% |
| Gemini | 15.6% | 27.8% | 14.3% | 11.2% | 68.9% |
| Llama-3 | 18.2% | 31.4% | 9.8% | 14.7% | 74.1% |

### Coverage Analysis

| Harm Category | OpenAI Coverage | Anthropic Coverage | Ours Coverage |
|---------|-----------|-------------|------------|
| Hate Speech | ✓ | ✓ | ✓ |
| Misgendering | ✗ | Partial | ✓ |
| Identity Journey Erasure | ✗ | ✗ | ✓ |
| Overpathologization | ✗ | ✗ | ✓ |
| Forced Binary Classification of Non-binary Identity | ✗ | ✗ | ✓ |
| Misleading Medical Information | Partial | Partial | ✓ |

### Key Findings
- All mainstream LLMs exhibit a TNB harm rate exceeding 50%, demonstrating that current safety training is severely insufficient in protecting the TNB community.
- Misgendering and the reinforcement of stereotypes are the most common harm types, neither of which is explicitly defined in existing safety taxonomies.
- Over-censorship constitutes a "well-intentioned but harmful" behavior where models are overly sensitive to TNB topics, refusing to answer legitimate medical and legal questions.
- Open-source models (e.g., Llama-3) show higher harm rates than proprietary ones, which may correlate with resources invested in safety alignment.

## Highlights & Insights
- The community-centred methodology ensures the taxonomy is grounded — every harm category originates from real experiences rather than researcher speculation. This approach can be extended to AI harm assessments for other marginalized groups (e.g., disabled people, indigenous populations).
- Uncovering the counter-intuitive phenomenon of "over-censorship as a harm" is highly significant — showing that the safety mechanisms themselves can inflict harm on specific groups, which bears profound implications for safety evaluation design.
- Incorporating the analytical dimension of intersectionality makes the evaluation more comprehensive, avoiding the oversimplification of treating marginalized groups as monoliths.

## Limitations & Future Work
- The participant sample size and diversity might be insufficient to cover all experiences of TNB subgroups.
- The taxonomy is primarily based on LLM interactions within English-speaking contexts; harm patterns in non-English socio-cultural and linguistic environments may differ.
- The taxonomy has not yet been operationalized into automated evaluation tools (such as benchmarks or classifiers).
- Cross-cultural differences may affect the applicability of the taxonomy, as gender identity issues in other contexts (e.g., the Chinese context) involve different societal backgrounds.

## Related Work & Insights
- **vs SafetyBench**: SafetyBench covers general safety scenarios, whereas ours focuses on TNB-specific harms, making them complementary.
- **vs BOLD (Bias in Open-ended Language Generation)**: BOLD evaluates general biases, whereas ours drills down into fine-grained harms specific to the TNB community.
- **vs HarmBench**: HarmBench focuses on safety evaluation under malicious misuse, whereas ours addresses unintentional harms during everyday interactions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The community-centred harm taxonomy framework has methodological innovation, and the identified new harm types fill the evaluation gap.
- Experimental Thoroughness: ⭐⭐⭐ The depth of qualitative research is commendable, but the scale of quantitative evaluation is limited.
- Writing Quality: ⭐⭐⭐ Cannot be fully assessed (full paper not reviewed).
- Value: ⭐⭐⭐⭐ Holds significant practical guidance for AI fairness and safety assessment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Soundwave: Less is More for Speech-Text Alignment in LLMs](soundwave_less_is_more_for_speech-text_alignment_in_llms.md)
- [\[ICML 2025\] Teaching Physical Awareness to LLMs through Sounds](../../ICML2025/audio_speech/teaching_physical_awareness_to_llms_through_sounds.md)
- [\[NeurIPS 2025\] Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation](../../NeurIPS2025/audio_speech/can_llms_outshine_conventional_recommenders_a_comparative_evaluation.md)
- [\[ACL 2026\] Phun-Bench: Evaluating LLMs on Phonological Understanding in Chinese](../../ACL2026/audio_speech/phun-bench_evaluating_llms_on_phonological_understanding_in_chinese.md)
- [\[ICLR 2026\] Can Speech LLMs Think while Listening?](../../ICLR2026/audio_speech/can_speech_llms_think_while_listening.md)

</div>

<!-- RELATED:END -->
