---
title: >-
  [Paper Note] Language Models Don't Know What You Want: Evaluating Personalization in Deep Research Needs Real Users
description: >-
  [ACL 2026][LLM Evaluation][Personalized Deep Research] The authors construct the first open-source personalized Deep Research (DR) system, MyScholarQA (featuring a profile → action → report triad)…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Personalized Deep Research"
  - "LLM-as-Judge"
  - "User Study"
  - "Interpretable Agents"
  - "User-Centered Eval"
date: 2026-05-08
content_hash: 901e5e51b4b2ab95
---

# Language Models Don't Know What You Want: Evaluating Personalization in Deep Research Needs Real Users

**Conference**: ACL 2026  
**arXiv**: [2603.16120](https://arxiv.org/abs/2603.16120)  
**Code**: https://github.com/allenai/personalized-scholarqa-eval  
**Area**: LLM Evaluation / Personalization / Deep Research / Human-Computer Interaction  
**Keywords**: Personalized Deep Research, LLM-as-Judge, User Study, Interpretable Agents, User-Centered Eval

## TL;DR
The authors construct the first open-source personalized Deep Research (DR) system, MyScholarQA (featuring a profile → action → report triad), which outperforms other DR baselines across 16 offline metrics. however, 90-minute interviews with 21 real researchers reveal 9 types of personalization failure modes that offline evaluations fail to detect entirely. Furthermore, four mainstream LLM judges are unable to predict user satisfaction accurately, sounding an alarm for the practice of replacing real users with LLM judges.

## Background & Motivation

**Background**: Deep Research (DR) tools using LLM retrieval and synthesis to write multi-section reports with citations have become essential for researchers. However, most DR systems lack personalization; a Diffusion researcher and an NLP researcher asking "what is Attention" receive nearly identical answers. A few DR systems (e.g., OpenAI/Gemini DR) ask clarifying questions, but users must re-explain themselves for every new query.

**Limitations of Prior Work**: Among 31 personalization-related papers at ACL'25, **all 31** conducted offline evaluations (18 used synthetic user datasets + 17 used LLM judges), while only 2 conducted real-user studies. This "offline evaluation hegemony" assumes LLM judges can substitute for user judgment—a premise the authors question as a potential "systemic illusion."

**Key Challenge**: (1) Personalization is not a verifiable objective attribute; it is fundamentally about "whether this specific user finds it useful," yet an LLM judge lacks a "self." (2) DR reports take 5 minutes to generate, requiring a persistent user model rather than repeated prompting. (3) Metrics that correlate with real-user satisfaction may not align with those scoring high in offline evaluations.

**Goal**: (1) Build a deployable, controllable, and interpretable personalized DR system as a vehicle for research. (2) Verify its superiority across 16 offline metrics. (3) Use this system as a "technology probe" for real users to expose failure modes missed by LLM judges. (4) Distill a methodology and design lessons for "user-essential" evaluation.

**Key Insight**: Borrowing from Brusilovsky’s 1980s "adaptive hypermedia," the authors construct a persistent user model (inferred from user-selected papers), transform it into an editable list of actions for each query, and drive a multi-step LLM retrieval-writing pipeline. Every step allows user toggles/edits to turn "where and how personalization happens" into observable experimental variables.

**Core Idea**: Run two sets of evaluations on a functional personalized DR system—offline metrics and 21 real-user interviews—using the discrepancies between them to prove that LLM judges cannot replace real users.

## Method

### Overall Architecture
MyScholarQA (MYSQA) consists of three steps: (1) **Infer Profile**—From 5 papers $D$ selected by the user, the LLM infers $n_1{=}25$ sentence-level inferences $P=\{I_1,\dots,I_{n_1}\}$ covering five aspects: knowledge, research style, writing style, audience, and positions. Each inference $I$ explicitly cites passages from $D$. Users can edit/disable these to get $P^*$. (2) **Propose Actions**—Upon receiving a query $q$, the system generates $n_2{=}16$ actions $A=A_{\text{gen}}\cup A_{\text{person}}$ (four categories: content, style, specificity, research ideas; generic actions look only at $q$, while personalized actions consider both $q$ and $P^*$). Users select/edit these to get $A^*$. (3) **Synthesize Report**—Based on the ScholarQA pipeline (Semantic Scholar retrieval + clustering + multi-section generation), $A^*$ is inserted into the prompts with two modifications: retrieval generates multiple sets of search terms, and generation executes all $A^*$ while highlighting corresponding report segments with specific colors per action. Claude-4 Sonnet serves as the backbone LLM. All steps are open-sourced with an online UI demo.

### Key Designs

1. **Persistent User Model from Papers to 5-Dimension Profile**:
    - **Function**: Transitions the "who the user is" from transient clarifying questions to a persistent profile across queries, with each inference citing specific paper segments as evidence.
    - **Mechanism**: Users upload 5 papers $D$. The LLM is prompted to produce 5 sentence-level inferences for each of the five aspects: knowledge, research style, writing style, audience, and positions. Each $I$ explicitly cites $D$ snippets with explanations. The profile is editable, and individual items can be disabled. Gemini-2.5 Pro was selected as the profile backbone for its optimal performance (97.1% inference accuracy, 97.4% citation relevance, 3.73/5 specificity).
    - **Design Motivation**: Since DR reports take 5 minutes to generate, it is too costly to ask users to re-input preferences (the follow-up Q paradigm of OpenAI DR). A persistent, editable profile directly satisfies user expectations (e.g., "I want it to understand me once and follow through").

2. **Dual-Track Action Proposals (Generic + Personalized)**:
    - **Function**: Explicitly lists the system's personalization intent as an action list for user control prior to report generation.
    - **Mechanism**: Two prompts generate $A_{\text{gen}}$ (conditioned on $q$) and $A_{\text{person}}$ (conditioned on $q+P^*$), which are then merged. Categories include content, style, specificity, and research ideas. Soft prompts (e.g., "skip basic terms for experts") are used instead of hard rules to observe "how personalization occurs." LLM judges gave $A_{\text{person}}$ a win rate of 91-95% and uniqueness of 60-72%, proving actions vary significantly based on the profile.
    - **Design Motivation**: Generating reports directly makes personalization a "black box." Providing an action list first acts as a form of query clarification and allows for fine-grained analysis of which action types cause user dissatisfaction.

3. **Color-Highlighted Personalized Report Generation + Multi-step Prompt Refinement**:
    - **Function**: Embeds $A^*$ into the SCHOLARQA pipeline and uses per-action color highlighting in the report to show which text serves which action, making personalization visible and auditable.
    - **Mechanism**: Minimalist changes to two SCHOLARQA prompts: (i) `q → search terms` is modified to "produce multiple search terms based on $A^*$"; (ii) `section generation` is modified to "execute $A^*$ step-by-step and highlight corresponding text snippets with one color per action." Action adherence evaluates whether an action is followed anywhere in the report.
    - **Design Motivation**: Highlighting is a low-cost transparency method for users to "see personalization at a glance." It also serves as a probe to observe neglected or misused actions (e.g., "why does this action have no color?" reveals IGNORE-type failures).

### Loss & Training
Ours does not involve training new models; MYSQA relies entirely on prompt engineering and multi-LLM chains. Backbone LLMs: Gemini-2.5 Pro / Claude-4 Sonnet (thinking) / o3 / DS-r1 for profiling; Gemini-2.5 Flash / GPT-4.1 / Claude-4 Sonnet / DS-V3 for actions; Claude-4 Sonnet for reporting by default. Profile temperature is 1.0, max tokens 40,960, retrieved via Semantic Scholar API. MYSQA uses 200 DR queries from ScholarQA-CS2 for a synthetic benchmark, with user expertise (low/medium/high) simulated using author papers from CS-PaperSum categorized by GRIT-LM embedding cosine similarity.

## Key Experimental Results

### Main Results
**Profiles** (4 LLMs × 4 metrics, 0-100% / specificity 1-5):

| LLM | Inf. Acc | Cit. Rel. | Cat. Acc. | Specificity |
|-----|----------|-----------|-----------|-------------|
| Gemini-2.5 Pro | **97.1** | **97.4** | 99.4 | 3.73 |
| Claude-4 Sonnet | 92.5 | 97.4 | 99.1 | 4.12 |
| OpenAI o3 | 88.6 | 91.8 | **99.8** | **4.20** |
| DeepSeek-R1 | 77.8 | 80.7 | 97.2 | 3.56 |

**Reports** (vs. 5 DR baselines, 5 metrics):

| System | Ans. Cov ↑ | Ans. Prec ↑ | Cit. Prec ↑ | Cit. Rec ↑ | Action Adh ↑ |
|------|------------|-------------|-------------|------------|--------------|
| **Ours (MYSQA)** | **91.4** | 89.9 | **91.8** | **81.4** | 83.2 |
| ScholarQA (Base) | 88.9 | 89.1 | 90.5 | 76.9 | 81.3 |
| OpenScholar | 77.2 | **97.4** | 82.5 | 60.4 | 82.5 |
| STORM | 72.0 | 92.2 | 73.3 | 64.7 | 74.4 |
| Sonar DR | 81.0 | 82.9 | 64.3 | 46.3 | 75.0 |
| o3 DR | 89.1 | 90.2 | 79.2 | 56.7 | **93.8** |

MYSQA achieved the best results in 3 out of 5 metrics and second in one, outperforming all baselines and consistently exceeding the base ScholarQA.

### Ablation Study (9 Failure Modes Missed by Offline Metrics)

| Output | Failure Type | Description | Frequency |
|------|----------|------|----------|
| Profile | DOMAIN | Misused domain terminology | 27.6% |
| Profile | OVERCLAIM | Generalizing local paper findings to the entire user | 17.9% |
| Profile | CONVENTION | Mistaking general domain practices as a "user" trait | 12.8% |
| Profile | CONTRAST | Distorting user stance with incorrect comparisons | 12.2% |
| Action | NARROW | Action is too narrow, lacking full coverage | 43.8% |
| Action | OFFTOPIC | Action deviates from query intent | 23.6% |
| Report | UNINFORM | Content is too vague or lacks detail | 38.0% |
| Report | PRESENT | Mismatched presentation style or format | 25.3% |
| Report | IGNORE | Neglected implicit or explicit requirements in actions | 22.8% |

**LLM Judge Prediction of User Satisfaction**: 4 LLM judges (Gemini-2.5 Flash / GPT-4.1 / Claude-4 Sonnet / DS-V3) performed binary classification on these 9 failure types to predict "if the user would be satisfied," given identical context + 6-shot examples + definitions. Results: **No LLM significantly outperformed the majority-class baseline on any failure type** (α=0.05 Binomial test + Bonferroni correction). Removing few-shots, removing definitions, or adding zero/3-shot prompts did not improve performance.

### Key Findings
- **MYSQA Overall Usability at 73%**: 21 real users had a total satisfaction of 73% for profile/action/report; however, the remaining 27% corresponded almost entirely to zero-hits on offline metrics.
- **DOMAIN + NARROW + UNINFORM are the most common pain points**: Profiles are easily swayed by general terms, actions are often too narrow, and reports are often too vague—forming the primary axis of "Personalization Hallucination."
- **Inconsistency between Human and LLM Judge Preferences**: LLM judges gave personalized actions a 91-95% win rate, but humans only felt personalized actions were better than generic ones ~60% of the time, suggesting LLM judges systematically over-prefer "specialized-looking" answers.
- **Humans want more control**: Users expressed desires to add new actions (U3, U16), weight emphasis (U4), and use paper filters, multi-turn dialogues, or long-term memory, validating the "persistent + editable user model" approach.
- **The Over-trust Trap**: When an action lacked highlighting, most users assumed "that information does not exist" rather than "the system failed to execute the action," representing a hidden risk in personalization.
- **LLM Judges as "Necessary but Insufficient"**: The authors explicitly propose that offline evaluations should be treated as preliminary screenings rather than final verdicts.

## Highlights & Insights
- Reconceptualizes "DR personalization" as an adaptive hypermedia problem in HCI, introducing persistent user models and controllable action lists to turn the invisible dimension of personalization into observable entities.
- The statistic that 0 out of 31 personalization papers at ACL'25 used human evaluation is a powerful quantification of the current state of the community.
- The experimental design—running offline and online evaluations on the same system and using binary satisfaction classification to challenge LLM judges—provides empirical evidence against replacing humans with LLM judges.
- Some of the 9 failure types are known NLP issues (DOMAIN ≈ factuality, PRESENT ≈ style transfer), but others were only discovered by humans (TRUST, UNIMPORT), highlighting the irreplaceable epistemic value of user studies in NLP.

## Limitations & Future Work
- MYSQA takes ~5 minutes for a report and ~3 minutes for a profile; while faster than OpenAI DR, it is still too slow. Model distillation or pre-computation are potential engineering directions.
- The study was limited to 21 interviews in the DR setting; whether the conclusion regarding LLM judges holds for other personalized tasks (dialogue, recommendation, RAG) remains to be verified.
- LLM judge performance is susceptible to prompt engineering; the authors acknowledge that specialized reward models might better approximate user judgment.
- User models are based only on published papers, which limits applicability for those without publications (students, industry researchers).
- Personalization may reinforce filter bubbles and carries risks of identity bias (e.g., "odd grammar → non-native speaker → implies lower research quality"), requiring more robust safeguards.

## Related Work & Insights
- **vs. Co-STORM / OpenAI DR**: These rely on lightweight personalization via clarifying questions during queries; MYSQA uses persistent profiles + editable actions, which user studies show are preferred.
- **vs. OpenScholar / STORM / Perplexity Sonar / o3 DR**: While MYSQA outperforms these in outcome quality, its more significant contribution is revealing that "high scores ≠ good personalization."
- **vs. LaMP / Persona-DB / Step-back Profiling**: These are offline personalization methods; this paper challenges the entire paradigm through a user study.
- **vs. Liang et al. 2025 (Personalized DR benchmark)**: They use synthetic data and LLM judges without releasing a system or conducting human studies; Ours directly counters the blind spots of that paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐ First open-source personalized DR system + first large-scale user study challenging LLM-judge hegemony.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Combined 16 offline metrics × 5 baselines × 4 LLM judges × 21 humans × 90-minute interviews × 9 failure modes.
- Writing Quality: ⭐⭐⭐⭐ Seamless integration of system, experiments, user studies, and lessons; Figures 5 and 6 are particularly compelling.
- Value: ⭐⭐⭐⭐⭐ Provides a significant methodological critique for the personalization NLP community while offering a deployable personalization architecture.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation](teaching_language_models_to_forecast_research_success_through_comparative_idea_e.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ACL 2026\] Evaluating Temporal Consistency in Multi-Turn Language Models](evaluating_temporal_consistency_in_multi-turn_language_models.md)
- [\[ACL 2026\] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)

</div>

<!-- RELATED:END -->
