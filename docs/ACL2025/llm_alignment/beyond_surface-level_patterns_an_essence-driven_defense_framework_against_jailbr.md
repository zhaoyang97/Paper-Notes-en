---
title: >-
  [Paper Note] Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs
description: >-
  [ACL 2025][LLM Alignment][Jailbreak Defense] This paper proposes EDDF, a jailbreak defense framework based on "attack essence" rather than surface-level patterns. It offline extracts essential strategies of known attacks to store in a vector database, and online performs essence abstraction, retrieval, and fine-grained judgment on new queries. This reduces the attack success rate by at least 20% with a false positive rate of only 2.18%.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Jailbreak Defense"
  - "Attack Essence"
  - "Vector Retrieval"
  - "Input Filtering"
  - "Plug-and-Play"
date: 2026-05-08
content_hash: ad2b79661d3debf2
---

# Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs

**Conference**: ACL 2025  
**arXiv**: [2502.19041](https://arxiv.org/abs/2502.19041)  
**Code**: [https://github.com/ShiyuXiang77/EDDF](https://github.com/ShiyuXiang77/EDDF)  
**Area**: LLM Alignment / AI Safety  
**Keywords**: Jailbreak Defense, Attack Essence, Vector Retrieval, Input Filtering, Plug-and-Play

## TL;DR
This paper proposes EDDF, a jailbreak defense framework based on "attack essence" rather than surface-level patterns. It offline extracts essential strategies of known attacks to store in a vector database, and online performs essence abstraction, retrieval, and fine-grained judgment on new queries. This reduces the attack success rate by at least 20% with a false positive rate of only 2.18%.

## Background & Motivation
**Background**: Alignment-trained LLMs remain vulnerable to jailbreak attacks. Existing defense methods include safety alignment (training-time), inference guidance (prompt engineering), and input/output filtering.

**Limitations of Prior Work**: Existing methods focus on the **surface-level patterns** of attacks (such as specific templates or encoding methods), failing when the surface form of the attack prompt changes (while the core strategy remains the same). For example, intention analysis methods show an ASR degradation from 12% to 25% under complex variant attacks.

**Key Challenge**: Attackers can easily generate different surface variants of the same strategy (jailbreak multiplication), but the core "attack essence"—the combination of strategies that hide malicious intent—remains unchanged.

**Goal**: Extract the deep strategic essence of attacks rather than surface features, enabling the defense to generalize to unseen attack variants.

**Key Insight**: A two-stage approach—offline construction of an attack essence vector database, and online identification of new attacks using essence retrieval combined with fine-grained LLM judgment.

**Core Idea**: Shift from analyzing "surface-level patterns" to analyzing "the essence of attack strategies", using vector retrieval to achieve generalized defense against unknown variants.

## Method

### Overall Architecture
- **Offline**: Known attack prompts $\rightarrow$ LLM extracts attack strategies $\rightarrow$ Logical combination into attack essence $\rightarrow$ Quality verification $\rightarrow$ Embedding stored in the Essence Vector Database (EVD).
- **Online**: User query $\rightarrow$ LLM extracts query essence $\rightarrow$ Vector retrieval of Top-K similar attack essences $\rightarrow$ LLM fine-grained judgment (safe/unsafe).

### Key Designs

1. **Attack Essence Extraction**:

    - Function: Extract strategy combinations from jailbreak prompts and generate natural language descriptions.
    - Example: "Role-play + ignore ethical rules + formatted output" $\rightarrow$ Essence: "Assigning an unethical persona, ignoring ethical guidelines, using gamified language, and outputting harmful content in a templated format."
    - Quality Assurance: 4 verification steps (non-refusal, strategic correctness, logical consistency of essence, abstract rather than concrete description).

2. **Online Essence Retrieval**:

    - The user query's essence is also extracted $\rightarrow$ Embedded $\rightarrow$ Cosine similarity retrieval of Top-K in EVD.
    - Similarity threshold $\tau$ filtering: If it exceeds the threshold, it enters fine-grained judgment; otherwise, it is directly classified.

3. **Fine-Grained Judgment**:

    - Provide the original query + query essence + similar jailbreak prompts + similar attack essences together to the LLM.
    - The LLM makes the final judgment—since high semantic similarity of essence does not necessarily equate to malice (e.g., the essence of "How to kill a Python process" is "direct technical question").

## Key Experimental Results

### Main Results (Qwen-plus as the target model)

| Method | Original Attack ASR↓ | Variant Attack ASR↓ | FPR↓ |
|------|-------------|-------------|------|
| Llama3-Guard | 55.00 | 42.40 | 8.30 |
| Intention Analysis | 12.58 | 25.41 | **34.89** |
| Self-Reminder | 16.37 | 36.59 | 12.46 |
| Defense Prompt | 9.93 | 60.51 | 19.75 |
| **EDDF** | **5.82** | **5.71** | 2.18 |

### Ablation Study

| Configuration | ASR | FPR |
|------|-----|-----|
| Full EDDF | 5.71 | 2.18 |
| w/o Fine-Grained Judgment | 35.41 (+29.7%) | 36.29 (+34.1%) |
| w/o Essence Storage | 15.24 (+9.5%) | 10.80 (+8.6%) |
| w/o User Essence | 21.66 (+16.0%) | 9.40 (+7.2%) |

### Key Findings
- EDDF achieves an ASR of only 5.71% against variant attacks, which is at least 20 percentage points lower than the second-best method, verifying the effectiveness of "essence generalization".
- Fine-grained judgement is the most critical component—without it, the ASR surges from 5.71% to 35.41%.
- The FPR is only 2.18%, indicating very few false positives for benign queries and showcasing high practicality.
- Other methods (especially Intention Analysis and Defense Prompt) degrade severely under variant attacks, confirming the limitations of focusing on surface-level patterns.

## Highlights & Insights
- **Precise abstraction level of "attack essence"**: Instead of concretely describing malicious behaviors, it abstracts combinations of strategies—enabling both generalization and distinctiveness.
- **Plug-and-play input filtering**: Requires no retraining of the target model, requiring only the maintenance of a vector database.
- **Low false positive rate for benign queries**: Avoids over-defensiveness (such as Intention Analysis's 34.89% FPR) through fine-grained judgment.

## Limitations & Future Work
- Relies on LLMs for essence extraction and judgment, making it potentially susceptible to manipulation.
- The vector database needs continuous updates to cover new attack strategies.
- The online phase requires multiple LLM calls (essence extraction + retrieval + judgment), resulting in higher latency.
- Only text attacks were tested; multimodal jailbreak scenarios were not considered.
- The threshold $\tau$ settings may need adjustment for different target models.

## Related Work & Insights
- **vs Intention Analysis**: Direct query intention analysis is easily deceived (FPR 34.89%), whereas EDDF is more reliable through essence retrieval combined with fine-grained judgment.
- **vs PPL Filtering**: PPL filtering is rigid with high false positive rates, while EDDF is flexible and has a low FPR.
- **vs Safety Alignment**: Alignment is a training-time defense, while EDDF is an inference-time plug-and-play defense—making them complementary.
- Insight: Safety defenses need to upgrade from "feature matching" to "strategy understanding".

## Rating
- Novelty: ⭐⭐⭐⭐ The abstract thinking of "attack essence" is novel, and the architecture design of offline database + online retrieval is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete evaluations across multiple attack methods, defense baselines, and ablation studies, with convincing variant attack tests.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, and the method's motivation is fully elaborated.
- Value: ⭐⭐⭐⭐⭐ Highly practical, reducing ASR by over 20% in a plug-and-play manner, offering direct value for secure LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs](jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)
- [\[ACL 2025\] AGD: Adversarial Game Defense Against Jailbreak Attacks in Large Language Models](agd_adversarial_game_defense_against_jailbreak_attacks_in_large_language_models.md)
- [\[NeurIPS 2025\] SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism](../../NeurIPS2025/llm_alignment/safeptr_token-level_jailbreak_defense_in_multimodal_llms_via_prune-then-restore_.md)
- [\[AAAI 2026\] AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)
- [\[ACL 2025\] Beyond the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models](beyond_the_tip_of_efficiency_uncovering_the_submerged_threats_of_jailbreak_attac.md)

</div>

<!-- RELATED:END -->
