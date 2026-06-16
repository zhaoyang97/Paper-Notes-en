---
title: >-
  [Paper Note] Do LLMs Capture Embodied Cognition and Cultural Variation? Cross-Linguistic Evidence from Demonstratives
description: >-
  [ACL 2026][Interpretability][proximal/distal] Academic paper note for Do LLMs Capture Embodied Cognition and Cultural Variation? Cross-Linguistic Evidence from Demonstratives.
tags:
  - ACL 2026
  - Interpretability
  - proximal/distal
date: 2026-05-08
content_hash: e9ac49aac754fb20
---
# Do LLMs Capture Embodied Cognition and Cultural Variation? Cross-Linguistic Evidence from Demonstratives

**Conference**: ACL 2026  
**arXiv**: [2604.25423](https://arxiv.org/abs/2604.25423)  
**Code**: TBD (Not directly provided in the paper)  
**Area**: Cross-cultural / LLM Evaluation / Embodied Cognition  
**Keywords**: Demonstratives, Embodied Cognition, Cross-cultural, Symmetry Index, Proximal/Distal, Self/Other Perspective

## TL;DR
The authors use demonstratives like "this/that" and "这/那" as probes to construct a cross-linguistic English-Chinese dataset (80 items/language × 4 cues × 4 perspectives × 5 scenarios). By establishing a human baseline of 6,400 responses from 320 native speakers, the study finds that English speakers excel at proximal–distal distinctions but struggle with the "other" perspective, whereas Chinese speakers show the opposite pattern. In contrast, five SOTA LLMs failed to provide stable proximal–distal distinctions and lacked cross-linguistic variation, generally reverting to English-centric reasoning or "All of the above" safety fallbacks.

## Background & Motivation

**Background**: While LLMs have achieved rapid progress in textual tasks, the academic community continues to debate whether they truly possess grounded cognition (embodied knowledge). Most evaluations still rely on knowledge/reasoning benchmarks, lacking probes specifically designed for "physical spatial awareness," "perspective switching," and "cultural pragmatics."

**Limitations of Prior Work**: Grounded knowledge is rarely explicitly stated in text (e.g., a speaker in a novel rarely writes "I am facing the table, and there is a cup near me"), making it difficult for LLMs to learn from pure text. Furthermore, existing benchmarks often focus on reasoning or multimodal inputs, failing to isolate "spatial deixis"—a universal phenomenon carried by only a few words.

**Key Challenge**: (i) Demonstratives are universal across humanity (acquired at age 2-3) and constitute one of the most embodied linguistic phenomena. (ii) However, the interpretation of demonstratives highly depends on the speaker's physical position, the interlocutor's perspective, and cultural norms; such grounding signals are nearly absent in pure text. (iii) Different languages and cultures have distinct preferences for "proximal-distal" and "self-other" distinctions, providing an ideal scenario to test whether LLMs have genuinely learned cultural variation.

**Goal**: (1) Design controlled experiments using demonstratives to detect whether LLMs master embodied spatial grounding. (2) Compare the cross-linguistic behavior of LLMs to see if they reflect pragmatic and cultural differences between English and Chinese. (3) Establish human baselines to verify the asymmetry of proximal-distal vs. perspective-taking in both languages.

**Key Insight**: Choosing demonstratives as probes offers three advantages: (a) they are universal phenomena with subtle cross-linguistic variations; (b) they are usually implicit and not explicitly written (making them a "hard probe" for LLMs); and (c) the task design is quantifiable (MCQ + Symmetry Index).

**Core Idea**: An elaborate experimental design using "pair-to-pair MCQ + 4 cue conditions + 5 scenarios + reverse logic options (All of the above)" is used to force LLMs to reveal whether they truly understand the mutual exclusivity of proximal-distal terms and to compare whether their behavior changes across languages.

## Method

### Overall Architecture

This study does not train models but uses controlled demonstrative experiments to transform "whether the model has grounded spatial meaning" into observable selection behavior. The dataset consists of 160 items (80 per language). Each item is a four-option multiple-choice question: it starts by describing a scene with two characters sitting opposite each other, specifies a speaker (marked in red), gives a blue instruction (the target object is in curly braces, e.g., `{fruit}`), and then asks an inverse question: "Done! Are there any items left on the place?" (i.e., remove the referred object first, then ask what remains). The four options are: proximal (near speaker), distal (near interlocutor), middle (distractor), and the logic trap "All of the above." The questions span 4 cue conditions (Pure Demonstrative / Pure Pronoun / Demo+Reinforcing Pronoun / Demo+Conflicting Pronoun) and 5 scenarios (verbs like eat, hide, take). The human baseline comprises four independent surveys (one per cue, 40 people × 20 questions × 2 languages), involving 320 native speakers and 6,400 responses (using Credamo for Chinese and Prolific for English). Five SOTA models (GPT-5.1, Claude-Sonnet-4.5, Gemini-2.5-Pro, DeepSeek-V3.1, Qwen3-Max) were evaluated using zero-shot prompt refinement, with averages taken over 10 runs (4,800 instances, SD 0.02–0.08).

### Key Designs

**1. "Are there any items left" inverse questioning + logic trap options: Exposing grounding through behavior.**

Explicitly asking "which one to take" is too easy for models to guess. Thus, the task asks what remains after the referred item is removed and includes proximal, distal, and middle items together with a counter-logical "All of the above." Since proximal and distal terms are naturally mutually exclusive in physical space, a subject who understands them will never select "All of the above" (human selection rate ~0.5%). This transforms linguistic understanding into observable behavior: the frequency with which a model selects "All of the above" quantifies its grasp of mutual exclusivity. In practice, Gemini-2.5-Pro selected option 4 up to 60% of the time in self-distal conditions, and Qwen3-Max reached 84%. This "safety fallback" is direct evidence that models fail to ground spatial meanings.

**2. Pair-to-pair × 4 cue × 4 perspective cross-control: Disentangling confounding dimensions.**

Evaluating single dimensions (like proximal accuracy) can be confounded by perspective. The experiment crosses proximity (proximal/distal), perspective (self/other), and pronoun reinforcement (none/present/conflicting). Scenarios are generated in pairs (2 proximity × 2 perspective) to avoid randomness, quantified by a Symmetry Index. Cue conditions range from pure demonstratives to pure pronouns and reinforcing/conflicting pronouns, allowing the separation of "demonstrative understanding" and "pronoun dependency." This design allows the isolation of fine-grained cross-cultural traits, such as "English speakers are accurate on distal items but collapse during perspective-taking."

**3. Symmetry Index (SI) for quantifying response distribution: A substitute for accuracy.**

In open referential experiments without a single ground truth, traditional accuracy fails. The authors adapt symmetry analysis from Robinson (1987) to define:

$$\mathrm{SI} = \frac{|A_1 - B_2| + |B_1 - A_2|}{A_1 + A_2 + B_1 + B_2}$$

where $A_1, A_2$ and $B_1, B_2$ are response counts under compared conditions. With a threshold of 0.1: a low SI (<0.1) indicates high symmetry (e.g., Self-Proximal and Self-Distal are mirror images, showing stability), while a high SI indicates behavioral collapse. The SI more intuitively characterizes the balance of multiple response types than chi-square tests, quantifying the complementary cross-linguistic pattern: "English speakers are symmetric within perspectives (proximal-distal), while Chinese speakers are symmetric across perspectives."

### Loss & Training

The study focuses on evaluation rather than training. The homogeneity of LLM and human distributions was tested using Rao-Scott adjusted chi-square tests, and distribution distances were quantified using Jensen-Shannon divergence (JSD).

## Key Experimental Results

### Main Results: Human vs. 5 LLMs Response Distributions (Only-demonstrative, selected conditions)

| Condition | Category | Human-en | Human-zh | GPT-5.1-en | GPT-5.1-zh | Gemini-2.5-Pro-en | Gemini-2.5-Pro-zh | Qwen3-Max-en | Qwen3-Max-zh |
|-----------|----------|----------|----------|------------|------------|--------------------|--------------------|---------------|---------------|
| Self-Proximal | (2,3) | **76.5%** | **84.5%** | 72.0% | 80.0% | 62.0% | 48.0% | 20.0% | 12.0% |
| Self-Proximal | (4) "All"| 1.0% | 0% | 20.0% | 18.0% | 36.0% | 18.0% | **80.0%** | **54.0%** |
| Self-Distal | (1,3) | **81.5%** | 52.0% | 30.0% | 24.0% | 10.0% | 10.0% | 0% | 0% |
| Self-Distal | (2,3) | 18.5% | **44.5%** | 0% | 0% | 0% | 0% | 12.0% | 2.0% |
| Self-Distal | (4) | 0% | 0% | 40.0% | 46.0% | 42.0% | 28.0% | **84.0%** | 74.0% |
| Other-Proximal| (1,3) | **62.5%** | **86.0%** | 60.0% | 70.0% | 36.0% | 36.0% | 0% | 0% |
| Other-Distal | (2,3) | **64.0%** | 53.0% | 30.0% | 48.0% | 16.0% | 10.0% | 2.0% | 2.0% |
| Other-Distal | (4) | 0% | 0% | 46.0% | 32.0% | 56.0% | 46.0% | **80.0%** | 64.0% |

### Ablation Study: Human Symmetry Index (Human Baseline)

| Comparison | English SI | Chinese SI |
|------------|------------|------------|
| Self-Proximal vs Self-Distal | **0.0309 ✓** | 0.3472 ✗ |
| Other-Proximal vs Other-Distal | **0.0254 ✓** | 0.3541 ✗ |
| Self-Proximal vs Other-Proximal | 0.1731 ✗ | **0.0131 ✓** |
| Self-Distal vs Other-Distal | 0.1646 ✗ | **0.0077 ✓** |

✓ = SI < 0.1 (High Symmetry), ✗ = SI > 0.1 (Asymmetry). This reveals a perfect contrast: **English speakers** are symmetric in proximal-distal contrasts within the same perspective but collapse across perspectives; **Chinese speakers** are the opposite, showing cross-perspective symmetry but blurred distal interpretation.

### Key Findings
- **LLMs fail to understand proximal-distal mutual exclusivity**: Humans rarely choose "All of the above" (~0.5%), but Qwen3-Max chose it 84% (en) / 74% (zh) of the time for self-distal. This "safety fallback" proves models treat demonstratives as vague puzzle pieces rather than spatially exclusive categories.
- **LLMs lack cross-linguistic cultural variation**: Humans show a 29.5 percentage point difference between English and Chinese in the Self-Distal condition for option (1,3), while Gemini-2.5-Pro's difference was only 0–6 percentage points. This suggests LLMs do not use Chinese pragmatic habits but apply an English-centric reasoning framework to all languages.
- **LLMs occasionally mimic humans in self-perspective**: The only condition where p > 0.05 (chi-square) was self-perspective proximal, likely because it is the most common context in training data.
- **Pronouns significantly aid LLMs**: In the reinforcing pronoun condition, Claude-Sonnet-4.5 reached 80% for (1,3) in English self-distal, close to the human 89.5% (vs. only 34% in the only-demo condition). This suggests LLMs rely on explicit lexical cues rather than true spatial grounding.
- **Conflicting pronouns reveal cue dominance**: In inconsistent pronoun conditions, both humans and models follow the pronoun, but LLMs still frequently choose illogical options like (4), showing their "pronoun understanding" is also shallow pattern matching.
- **Invariance across prompt strategies**: Zero-shot, CoT, and few-shot strategies yielded differences within noise levels, indicating that the lack of grounding is a fundamental capacity issue, not a prompting one.

## Highlights & Insights
- **"Logic trap options" are elegant diagnostics for grounding**: Including "All of the above" in mutually exclusive choices exposes model failures. This design can be transferred to any dimension requiring uniqueness constraints (e.g., temporal causality, magnitude).
- **The genius of demonstratives as probes**: They are universal, carried by minimal tokens, highly dependent on physical context absent from training corpora, and possess clear cross-linguistic cultural differences.
- **Symmetry Index is superior to accuracy for No-GT experiments**: SI quantifies internal consistency across paired conditions, which is more informative than pseudo-accuracy in ambiguous referential contexts.
- **Quantitative evidence for "LLMs as English-centric reasoners"**: The lack of difference between LLM behavior in English vs. Chinese, compared to the 30% gap in humans, directly refutes the optimistic narrative that multilingual LLMs truly understand multilingual cultures.
- **Pronoun > Demonstrative sensitivity**: LLMs' reliance on explicit lexical cues explains why they collapse when cues are weak (e.g., pure demonstratives requiring perspective inference).

## Limitations & Future Work
- **Text-only input limitation**: Demonstratives intrinsically require multimodal grounding; the study suggests future work using 3D simulations.
- **Small dataset (160 items)**: Scale was sacrificed for controlled design, limiting statistical power.
- **Language coverage**: Only English and Chinese were tested; languages with three-way distinctions (Spanish, Japanese) were not covered.
- **Lack of "Why" analysis**: It is unclear if English-centrism stems from training data proportions, RLHF reward biases, or tokenization issues.
- **Individual variation**: Humans often show multi-modal distributions (50/50 splits), but LLMs collapse to a single "expert" answer.

## Related Work & Insights
- **vs. Traditional Grounding Benchmarks**: While others require visual/3D input, this study exposes failures using pure text, offering a low-cost replication method.
- **vs. Kauf et al. (2023)**: Similar minimal-pair linguistic probe approach but focused on more embodied spatial phenomena.
- **vs. Hall (1976) Theory**: The data provides demonstrative-level support for high-context (Chinese) vs. low-context (English) theories—Chinese is flexible/ambiguous, English is precise/rigid.
- **Insight**: Multilingual benchmarks should include "culturally divergent" tasks to check if models reason according to the specific culture rather than just translating English reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Uses demonstratives to probe both embodied cognition and cultural variation; excellent logic trap design)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Well-controlled; 320 participants; but limited to two languages)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic and intuitive visualizations)
- Value: ⭐⭐⭐⭐⭐ (Provides a crucial refutation of "Multilingual = Multicultural")

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations](do_llms_know_tool_irrelevance_demystifying_structural_alignment_bias_in_tool_inv.md)
- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)
- [\[AAAI 2026\] PragWorld: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics](../../AAAI2026/interpretability/pragworld_a_benchmark_evaluating_llms_local_world_model_under_minimal_linguistic.md)
- [\[ACL 2026\] Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining](crosscoding_through_time_tracking_emergence_consolidation_of_linguistic_represen.md)
- [\[ACL 2025\] Llama See, Llama Do: A Mechanistic Perspective on Contextual Entrainment and Distraction in LLMs](../../ACL2025/interpretability/llama_see_llama_do_entrainment.md)

</div>

<!-- RELATED:END -->
