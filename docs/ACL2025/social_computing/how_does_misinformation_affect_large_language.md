---
title: >-
  [Paper Note] How does Misinformation Affect Large Language Model Behaviors and Preferences?
description: >-
  [ACL 2025][Social Computing][misinformation] This study constructs MisBench (10.34 million entries of misinformation), the largest misinformation evaluation benchmark to date. It systematically analyzes LLM behaviors and preferences toward misinformation across the dimensions of knowledge conflict types and text styles, and proposes the RtD method to enhance misinformation detection by integrating external knowledge sources.
tags:
  - "ACL 2025"
  - "Social Computing"
  - "misinformation"
  - "knowledge conflict"
  - "MisBench"
  - "LLM robustness"
  - "Reconstruct to Discriminate"
date: 2026-05-08
content_hash: 9566b35c0ec8b2fe
---

# How does Misinformation Affect Large Language Model Behaviors and Preferences?

**Conference**: ACL 2025  
**arXiv**: [2505.21608](https://arxiv.org/abs/2505.21608)  
**Code**: [https://github.com/GKNL/MisBench](https://github.com/GKNL/MisBench)  
**Area**: Social Computing  
**Keywords**: misinformation, knowledge conflict, MisBench, LLM robustness, Reconstruct to Discriminate

## TL;DR
This study constructs MisBench (10.34 million entries of misinformation), the largest misinformation evaluation benchmark to date. It systematically analyzes LLM behaviors and preferences toward misinformation across the dimensions of knowledge conflict types and text styles, and proposes the RtD method to enhance misinformation detection by integrating external knowledge sources.

## Background & Motivation

**Background**: LLMs excel in knowledge-intensive tasks but remain vulnerable to inaccurate, outdated, or fabricated knowledge. Although multiple misinformation benchmarks exist, they are limited in scale and coverage.

**Limitations of Prior Work**: Prior studies have demonstrated the vulnerability of LLMs to misinformation but lack fine-grained analysis—specifically, in what aspects and to what extent are LLMs misled by misinformation? How do the effects of misinformation vary across different types, sources, and styles?

**Key Challenge**: There is a lack of a sufficiently large-scale and multi-dimensional benchmark to fully understand the interaction mechanisms between LLMs and misinformation.

**Goal**: (1) Construct a large-scale, multi-dimensional misinformation benchmark. (2) Systematically analyze LLM response patterns to different types and styles of misinformation. (3) Propose methods to improve misinformation detection in LLMs.

**Key Insight**: Constructing misinformation from two orthogonal dimensions: knowledge conflicts (factual, temporal, and semantic conflicts) and text styles (6 writing styles), using Wikidata one-hop and multi-hop relations to ensure broad coverage.

**Core Idea**: Constructing a benchmark of 10.34 million instances of misinformation covering 3 conflict types × 6 text styles, revealing the differentiated vulnerability of LLMs to various forms of misinformation.

## Method

### Overall Architecture
(1) Extract one-hop and multi-hop claims from Wikidata → (2) Construct three types of knowledge conflict claims → (3) Generate ground-truth evidence and misinformation texts using LLaMA-3-70B → (4) Style into 6 variants → (5) Perform quality control → (6) Evaluate LLM + propose the RtD method.

### Key Designs

1. **Construction of Three Knowledge Conflicts**:

    - **Factual Conflict**: Replacing the object in relation triples with an entity of the same category, such as $(s, r, o) \to (s, r, o')$.
    - **Temporal Conflict**: Adding a future timestamp to turn the claim into outdated information $(s, r, o', T_s, T_e)$.
    - **Semantic Conflict**: Preserving the subject name but replacing its description to point to a different semantic entity $(s, r, o', d_s^*, d_{o'})$.
    - Design Motivation: These three conflicts simulate the three major sources of misinformation in the real world: factual errors, outdated information, and entity ambiguity.

2. **Six Text Stylizations**:

    - Function: Transforming each piece of misinformation into six styles: Wikipedia entries, news reports, scientific literature, blogs, technical language, and confident language.
    - Mechanism: LLMs tend to over-rely on LLM-generated evidence based on text similarity and relevance, and stylistic differences affect LLM judgment.
    - Design Motivation: The harmfulness of real-world misinformation is influenced by its presentation—formal/objective versus narrative/subjective misinformation has different impacts on LLMs.

3. **Reconstruct to Discriminate (RtD)**:

    - Function: Utilizing the ability of LLMs to detect contextual inconsistencies, combined with reconstructing evidence text for key entities from external knowledge sources, to identify misinformation.
    - Mechanism: Reconstructing evidence text for key subject entities from external sources (e.g., Wikipedia) and comparing it with the given context to determine whether it is misinformation.
    - Design Motivation: Leveraging the inherent "contextual inconsistency detection" capability of LLMs (detecting contradictions even when the answer is unknown) and bridging knowledge gaps using external knowledge.

### Data Statistics
- 431,113 claims/QA pairs, 10,346,712 misinformation evidence texts, 82 one-hop relations, 148 multi-hop relations

## Key Experimental Results

### Main Results (Success Rate, higher is better = stronger misinformation detection capability)

| Model | Factual Conflict (Memorized) | Factual Conflict (Unknown) | Temporal Conflict (Memorized) | Semantic Conflict (Memorized) |
|------|:---:|:---:|:---:|:---:|
| GPT-4o | High | Medium | High | Medium |
| Claude 3.5 Haiku | 67.15 | 60.33 | 85.04 | 62.96 |
| DeepSeek-V2.5 | 34.56 | 26.42 | 55.61 | 43.78 |
| Gemma2-9B | Low | Low | Medium | Low |

### Ablation Study

| Finding | Explanation |
|------|------|
| Temporal conflicts are the easiest to detect | LLMs are sensitive to timestamp changes |
| Semantic conflicts are the hardest to detect | Entity ambiguity is the most deceptive |
| Formal and objective style (one-hop) is more hazardous | Formal language is more deceptive in single-hop tasks |
| Narrative and subjective style (multi-hop) is more hazardous | Subjective narratives are more misleading in multi-hop reasoning |
| Significant improvement with RtD | Qwen2.5-14B +6%, Gemma2-9B +20.6% |

### Key Findings
- **LLMs possess inherent misinformation identification capabilities**: Even without knowing the topic, they can identify misinformation through contextual inconsistency detection.
- **Factual Conflict > Semantic Conflict**: LLMs exhibit some resistance to direct factual contradictions but are particularly vulnerable to semantic conflicts involving "same name, different meaning".
- **Stylistic impact varies with task complexity**: Formal and objective styles are more hazardous in simple tasks, whereas narrative and subjective styles are more deceptive in complex tasks.

## Highlights & Insights
- **Unprecedented Scale**: 10.34 million entries of misinformation, which is 18 times larger than the previous largest benchmark ConflictBank (550K), while simultaneously covering multi-cause, multi-hop, and multi-stylistic aspects.
- **The cross-analysis of style × conflict type is insightful**: It reveals the crucial conclusion that "the deceptiveness of misinformation depends on the combination of both content and format."
- **Simple and Effective RtD Method**: Leverages the inherent capabilities of LLMs + external knowledge reconstruction, requiring no additional training.

## Limitations & Future Work
- Misinformation is generated by an LLM (LLaMA-3-70B), potentially containing LLM-specific stylistic cues that make it easier for other LLMs to detect.
- Claims based on Wikidata may be biased toward specific knowledge domains.
- RtD relies heavily on the availability and accuracy of external knowledge sources.

## Related Work & Insights
- **vs ConflictBank**: ConflictBank also considers multi-cause and multi-stylistic factors but only scale is 550K; MisBench is 18 times larger and incorporates multi-hop reasoning.
- **vs LLMFake**: LLMFake contains only 1,032 entries, whereas MisBench is four orders of magnitude larger.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-dimensional construction method and the stylistic analysis perspective are highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Represents the largest scale, covers both open-source and proprietary models, and provides multi-dimensional analyses.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with deep analysis.
- Value: ⭐⭐⭐⭐⭐ Establishes a standard benchmark for researching LLM robustness against misinformation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models](a_survey_on_proactive_defense_strategies_against_misinformation_in_large_languag.md)
- [\[ACL 2025\] Exploring the Impact of Instruction-Tuning on LLMs' Susceptibility to Misinformation](exploring_the_impact_of_instruction-tuning_on_llms_susceptibility_to_misinformat.md)
- [\[NeurIPS 2025\] Uncovering Strategic Egoism Behaviors in Large Language Models](../../NeurIPS2025/social_computing/uncovering_strategic_egoism_behaviors_in_large_language_models.md)
- [\[ACL 2025\] Translate With Care: Addressing Gender Bias, Neutrality, and Reasoning in Large Language Model Translations](translate_with_care_addressing_gender_bias_neutrality_and_reasoning_in_large_lan.md)
- [\[NeurIPS 2025\] Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector](../../NeurIPS2025/social_computing/any_large_language_model_can_be_a_reliable_judge_debiasing_w.md)

</div>

<!-- RELATED:END -->
