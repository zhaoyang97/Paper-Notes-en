---
title: >-
  [Paper Note] Do LLMs Capture Embodied Cognition and Cultural Variation? Cross-Linguistic Evidence from Demonstratives
description: >-
  [ACL 2026][Interpretability][Demonstratives] The authors utilize demonstratives (e.g., "this/that" and "这/那") as probes to construct a bilingual Chinese-English dataset (80 items/language × 4 cues × 4 perspectives × 5 sc…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Demonstratives"
  - "Embodied Cognition"
  - "Cross-Cultural"
  - "Symmetry Index"
  - "proximal/distal"
  - "self/other perspective"
date: 2026-05-08
content_hash: 6e68e628bf95a1f9
---

# Do LLMs Capture Embodied Cognition and Cultural Variation? Cross-Linguistic Evidence from Demonstratives

**Conference**: ACL 2026  
**arXiv**: [2604.25423](https://arxiv.org/abs/2604.25423)  
**Code**: To be confirmed (not directly provided in the paper)  
**Area**: Cross-Cultural / LLM Evaluation / Embodied Cognition  
**Keywords**: Demonstratives, Embodied Cognition, Cross-Cultural, Symmetry Index, proximal/distal, self/other perspective

## TL;DR
The authors utilize demonstratives (e.g., "this/that" and "这/那") as probes to construct a bilingual Chinese-English dataset (80 items/language × 4 cues × 4 perspectives × 5 scenarios). By establishing a human baseline from 6,400 responses from 320 native speakers, they found that English speakers excel at proximal–distal distinctions but struggle with the "other" perspective, while Chinese speakers show the opposite trend. Conversely, five SOTA LLMs failed to distinguish proximal–distal reliably and lacked cross-cultural variation, consistently reverting to English-centric reasoning or "All of the above" safety fallbacks.

## Background & Motivation

**Background**: While LLMs have advanced significantly in textual tasks, there is ongoing debate regarding their mastery of grounded cognition. Most evaluations rely on knowledge/reasoning benchmarks, lacking probes for physical spatial sense, perspective-taking, and cultural pragmatics.

**Limitations of Prior Work**: Grounded knowledge is rarely explicitly stated in text (e.g., writers seldom note "I am facing a table with a cup near me"), making it difficult for LLMs to learn from pure text. Existing benchmarks focus on reasoning or multimodal inputs rather than isolating spatial deixis—a universal phenomenon carried by specific tokens.

**Key Challenge**: (i) Demonstratives are universal and acquired early (age 2–3), representing highly embodied linguistic phenomena; (ii) their interpretation depends heavily on physical position, interlocutor perspective, and cultural norms, signals which are nearly absent in pure text; (iii) different cultures exhibit distinct preferences for proximal-distal vs. self-other perspectives, providing an ideal scenario to test if LLMs truly capture cultural variation.

**Goal**: (1) Design controlled experiments using demonstratives to detect if LLMs master embodied spatial grounding; (2) compare LLM cross-linguistic behavior against Chinese-English pragmatic differences; (3) establish human baselines to verify asymmetries in proximal-distal vs. perspective-taking across languages.

**Key Insight**: Demonstratives serve as effective probes because they are (a) universal yet subtly different across languages, (b) context-implicit and rarely documented (hard probes), and (c) quantifiable through multiple-choice tasks and the Symmetry Index.

**Core Idea**: A sophisticated experimental design—featuring pair-to-pair multiple choice, 4 cue conditions, 5 scenarios, and logical trap options ("All of the above")—is used to force LLMs to reveal whether they understand the mutual exclusivity of proximal-distal and whether their behavior shifts across languages.

## Method

### Overall Architecture

The experiment consists of three modules: (1) Dataset construction (160 items, pair-to-pair design); (2) Human baseline collection (320 native speakers, 6,400 responses); (3) LLM evaluation (5 models × 10 runs = 4,800 instances).

#### Dataset Structure

- **Per Item**: A four-choice question. It begins with a scene description (two characters facing each other) + designated speaker (red) + simple instruction (blue, with a target `{fruit}`) + follow-up: "Done! Are there any items left on the place?" (eliminating the referred object to ask what remains).
- **4 Options**:
    1. proximal (objects near the speaker)
    2. distal (objects near the interlocutor)
    3. middle (distractor item)
    4. All of the above (logical trap—rational agents avoid this as 1 and 2 are mutually exclusive).
- **Pair-to-pair**: 4 items per scenario (proximal/distal × self/other) to avoid single-item randomness.
- **4 Cue Conditions**:
    1. only demonstratives ("take this/that {cup}", **main experiment**)
    2. only pronouns ("take my/your {cup}")
    3. demo + reinforcing pronoun ("take this {cup} of mine")
    4. demo + inconsistent pronoun ("take that {cup} of mine").
- **5 Scenarios**: Various verbs used (eat, hide, take, etc.), totaling 80 items/language × 2 languages = 160 items.

#### Human Baseline

Four independent surveys (one per cue), each with 40 participants (gender-balanced) × 20 items × 2 languages = 320 native speakers × 20 = 6,400 responses. Chinese data via Credamo, English via Prolific.

#### LLM Evaluation

Five SOTA models: GPT-5.1, Claude-Sonnet-4.5, Gemini-2.5-Pro (closed-source) + DeepSeek-V3.1, Qwen3-Max (open-source). Zero-shot prompt refinement: "Please reply [The answer is: 1, 2, 3, 4], and give a brief reason." Average of 10 runs per model, with standard deviations of 0.02–0.08.

### Key Designs

1. **"Are there any items left" questioning + logical trap**:
    - **Function**: Transitions from asking "who is referred to" to "what remains after removal," embedding "All of the above" as a logical violation of the proximal/distal mutual exclusivity.
    - **Mechanism**: Direct questions are too easy to guess. By asking for the remainder and including "All of the above," the design tests spatial understanding. Humans select option 4 only ~0.5% of the time.
    - **Design Motivation**: It quantifies the understanding of mutual exclusivity. Gemini-2.5-Pro selected option 4 in 60% of self-distal cases, and Qwen3-Max reached 84%, exposing a "safety fallback" behavior rather than spatial grounding.

2. **Pair-to-pair × 4 cue × 4 perspective Control**:
    - **Function**: Intersects proximity, perspective, and pronoun reinforcement, requiring disentangled capabilities from humans and LLMs.
    - **Mechanism**: (i) SI measures symmetry across 4 items per scenario; (ii) cue conditions isolate demonstrative understanding from pronoun dependency; (iii) bilingual comparison reveals cultural differences.
    - **Design Motivation**: Prevents perspective from confounding proximity accuracy. This isolates fine-grained features, such as English speakers' accuracy in distal contexts vs. their collapse during perspective switching.

3. **Symmetry Index (SI)**:
    - **Function**: Uses a scalar to quantify the symmetry of matched response distributions, with a 0.1 threshold.
    - **Mechanism**: $$\mathrm{SI} = \frac{|A_1 - B_2| + |B_1 - A_2|}{A_1 + A_2 + B_1 + B_2}$$, where $A_1, A_2$ and $B_1, B_2$ are counts of response categories. Low SI (<0.1) indicates high symmetry; high SI indicates collapse.
    - **Design Motivation**: Traditional accuracy is unsuitable for open-ended linguistic experiments without a single ground truth. SI allows for direct quantification of how participants mirror distributions (e.g., Self-Proximal vs. Self-Distal).

### Loss & Training
No models were trained. Evaluation used Rao-Scott adjusted chi-square tests and Jensen-Shannon divergence (JSD) to compare model and human distributions.

## Key Experimental Results

### Main Results: Human vs 5 LLMs (only-demonstrative, selected conditions)

| Condition | Category | Human-en | Human-zh | GPT-5.1-en | GPT-5.1-zh | Gemini-2.5-Pro-en | Gemini-2.5-Pro-zh | Qwen3-Max-en | Qwen3-Max-zh |
|------|------|----------|----------|------------|------------|--------------------|--------------------|---------------|---------------|
| Self-Proximal | (2,3) | **76.5%** | **84.5%** | 72.0% | 80.0% | 62.0% | 48.0% | 20.0% | 12.0% |
| Self-Proximal | (4) "All" | 1.0% | 0% | 20.0% | 18.0% | 36.0% | 18.0% | **80.0%** | **54.0%** |
| Self-Distal | (1,3) | **81.5%** | 52.0% | 30.0% | 24.0% | 10.0% | 10.0% | 0% | 0% |
| Self-Distal | (2,3) | 18.5% | **44.5%** | 0% | 0% | 0% | 0% | 12.0% | 2.0% |
| Self-Distal | (4) | 0% | 0% | 40.0% | 46.0% | 42.0% | 28.0% | **84.0%** | 74.0% |
| Other-Proximal | (1,3) | **62.5%** | **86.0%** | 60.0% | 70.0% | 36.0% | 36.0% | 0% | 0% |
| Other-Distal | (2,3) | **64.0%** | 53.0% | 30.0% | 48.0% | 16.0% | 10.0% | 2.0% | 2.0% |
| Other-Distal | (4) | 0% | 0% | 46.0% | 32.0% | 56.0% | 46.0% | **80.0%** | 64.0% |

### Ablation Study: Human Symmetry Index (Baseline)

| Comparison | English SI | Chinese SI |
|------|-----------|-----------|
| Self-Proximal vs Self-Distal | **0.0309 ✓** | 0.3472 ✗ |
| Other-Proximal vs Other-Distal | **0.0254 ✓** | 0.3541 ✗ |
| Self-Proximal vs Other-Proximal | 0.1731 ✗ | **0.0131 ✓** |
| Self-Distal vs Other-Distal | 0.1646 ✗ | **0.0077 ✓** |

✓ indicates SI<0.1. Findings suggest **English speakers** are symmetric in proximal-distal contrasts within the same perspective but collapse across perspectives. **Chinese speakers** are the opposite: symmetric across perspectives but vague in distal interpretation.

### Key Findings
- **LLMs fail to grasp mutual exclusivity**: While humans virtually never select "All of the above," Qwen3-Max hits 84% (en) on self-distal, and Gemini hits 36% on self-proximal-en. This "safe fallback" proves models lack spatial grounding for demonstratives.
- **Absence of cultural variation in LLMs**: Human differences between English and Chinese self-distal responses are 29.5 percentage points; LLM differences are only 0–6 points. LLMs apply an English-centric reasoning framework regardless of input language.
- **Performance on self-perspective**: LLMs occasionally approximate humans in self-proximal conditions (chi-square p > 0.05), likely due to high coverage of these common contexts in training data.
- **Pronouns aid LLMs**: Claude-Sonnet-4.5 improves significantly with reinforcing pronouns (80% vs 34%). This indicates reliance on explicit lexical cues rather than spatial grounding.
- **Inconsistent cues reveal shallow pattern matching**: In conflict conditions, models follow pronouns but still choose invalid logical options, suggesting their "understanding" of pronouns is also pattern-based.

## Highlights & Insights
- **Logical traps as diagnostic tools**: The high frequency of "All of the above" selections in LLMs versus humans provides a clear metric for grounding failure.
- **Demonstratives as hard probes**: They offer a universal benchmark that is context-dependent, linguistically simple, yet culturally distinct.
- **SI over Accuracy**: Symmetry Index is more informative than accuracy for tasks without a unique ground truth, quantifying internal consistency.
- **Evidence for English-centric bias**: The lack of difference between LLM behavior in Chinese and English, despite vast human differences, confirms that multilingual models prioritize English reasoning patterns.
- **Individual variation challenges**: Humans show multi-modal distributions (e.g., in Chinese distal), while LLMs collapse to a single "expert" answer, highlighting the limitations of current training paradigms.

## Limitations & Future Work
- **Text-only limitation**: Complete spatial grounding requires multimodal input (visual/3D).
- **Scale**: The dataset size (160 items) is limited to maintain controlled design.
- **Language coverage**: Only English and Chinese were tested; others (e.g., Japanese or Spanish) have 3-way demonstrative systems.
- **Attribution**: It remains unclear if English-centricity stems from training data distribution, RLHF bias, or tokenization.

## Related Work & Insights
- Comparison to **Kauf et al. (2023)**: Both use linguistic probes for grounding, but this work focuses on spatial deixis.
- Comparison to **Xu et al. (2025)**: Complements findings that LLMs capture non-sensorimotor features better than sensorimotor ones.
- **Hall's (1976) Theory**: Data supports high/low-context theories—Chinese is flexible/vague while English is precise/rigid.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Unique use of demonstratives and logical traps.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid human baseline and model variance checks, though limited to two languages.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear exposition of the SI and experimental logic.
- **Value**: ⭐⭐⭐⭐⭐ Crucial challenge to the "multilingual = multicultural" narrative.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)
- [\[ACL 2026\] Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations](do_llms_know_tool_irrelevance_demystifying_structural_alignment_bias_in_tool_inv.md)
- [\[ACL 2026\] Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining](crosscoding_through_time_tracking_emergence_consolidation_of_linguistic_represen.md)
- [\[AAAI 2026\] PragWorld: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics](../../AAAI2026/interpretability/pragworld_a_benchmark_evaluating_llms_local_world_model_under_minimal_linguistic.md)
- [\[ACL 2026\] Jacobian Scopes: Token-Level Causal Attributions in LLMs](jacobian_scopes_token-level_causal_attributions_in_llms.md)

</div>

<!-- RELATED:END -->
