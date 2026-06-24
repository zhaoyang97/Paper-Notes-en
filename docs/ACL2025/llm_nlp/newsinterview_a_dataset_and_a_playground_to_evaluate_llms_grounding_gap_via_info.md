---
title: >-
  [Paper Note] NewsInterview: a Dataset and a Playground to Evaluate LLMs' Grounding Gap via Informational Interviews
description: >-
  [ACL2025][LLM (Other)][Dialogue strategy] The authors constructed a dataset of 40,000 news interview dialogues and discovered that LLMs lack acknowledgement (by over 50%) and topic-switching capabilities (by 30%) in interview scenarios. Additionally, they designed a simulated game environment with persuasion mechanisms (NewsInterview), demonstrating that even the best LLM (gpt-4o) can only extract 50.4% of the target information items.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Dialogue strategy"
  - "news interview"
  - "grounding gap"
  - "information extraction"
  - "simulated game environment"
date: 2026-05-08
content_hash: 9565c82ea62f3065
---

# NewsInterview: a Dataset and a Playground to Evaluate LLMs' Grounding Gap via Informational Interviews

**Conference**: ACL2025  
**arXiv**: [2411.13779](https://arxiv.org/abs/2411.13779)  
**Code**: [GitHub](https://github.com/alex2awesome/news-interview-question-generation)  
**Area**: LLM/NLP  
**Keywords**: Dialogue strategy, news interview, grounding gap, information extraction, simulated game environment

## TL;DR

The authors constructed a dataset of 40,000 news interview dialogues and discovered that LLMs lack acknowledgement (by over 50%) and topic-switching capabilities (by 30%) in interview scenarios. Additionally, they designed a simulated game environment with persuasion mechanisms (NewsInterview), demonstrating that even the best LLM (gpt-4o) can only extract 50.4% of the target information items.

## Background & Motivation

**Background**: LLMs perform exceptionally well in dialogue generation, but they exhibit insufficient grounding behaviors in scenarios requiring strategic communication (such as educational tutoring, psychological counseling, and conflict mediation).

**Limitations of Prior Work**: Existing dialogue datasets are either crowdsourced (which makes them unnatural) or limited by privacy constraints (educational/therapy dialogues are difficult to collect at scale, typically sized around 100 to 1,000 instances).

**Key Challenge**: While LLMs can understand dialogue context, they cannot plan across multiple turns like human journalists do (e.g., executing "empathic acknowledgement $\rightarrow$ strategic questioning $\rightarrow$ information guidance"). Current training objectives lack long-term strategic reward signals.

**Goal**: (1) Quantify the grounding gap of LLMs in interview scenarios; (2) build a simulated environment that can be used to train strategic dialogue agents.

**Key Insight**: News interviewing is selected as the study scenario. In interviews, journalists must handle anxious or uncooperative sources, which inherently requires grounding communication, and a vast number of public transcripts are readily available.

**Core Idea**: Expose the gap through counterfactual analysis (asking LLMs to predict the $t$-th turn question given the first $t-1$ turns), and then quantify the long-term utility of grounding as an optimizable reward signal using a game environment with personality and persuasion mechanisms.

## Method

### Overall Architecture

**Phase 1: Dataset Construction**
- Collected 487,310 interview transcripts from NPR and CNN.
- Filtered the transcripts using Llama-3.1-70b (excluding multi-party dialogues, game shows, etc.) to retain 45,848 one-on-one informational interviews.
- Identified interviewer/interviewee roles via question mark counting (human validation shows >98% accuracy).
- Each interview averages 7.5 turns, with sources averaging 551 words and interviewers averaging 270 words.

**Phase 2: Counterfactual Analysis**
- Given the first $t-1$ turns of real dialogue, prompted the LLM (Llama-3.1-70b) to generate the $t$-th turn question.
- Implemented four prompting strategies: Baseline, Chain-of-Thought, Outline (including interview outline), and Outline-CoT.
- Used GPT-4o to evaluate six dimensions of similarity (Exact Match, Information, Motivation, Style, Discourse, Context).
- Developed a discourse role classification schema: Follow-up, Outline-Level, Acknowledgement, Opinion, Broadening, Verification, and Challenge.

**Phase 3: NewsInterview Game Environment**
- **Interviewer**: Given high-level goals (similar to outlines prepared by journalists before interviews), questioning at each turn based on dialogue history and goals.
- **Source**: Possesses a set of information items + personality + persuasion response mechanism.
- **Game Loop**: Interviewer asks a question $\rightarrow$ Source retrieves relevant information items $\rightarrow$ Evaluate persuasion level $\rightarrow$ Randomly decide the amount of returned information based on a Beta distribution $\rightarrow$ Generate a reply consistent with the personality.
- **Reward**: The total number of unique list of extracted information items at the end of the interview.

### Key Designs

**Source Personality System** (8 personalities, sourced from journalism textbook Sedorkin 2015):
- Anxious, Avoidant, Adversarial, Defensive, Straightforward, Poor Explainer, Dominating, Clueless.
- Different personalities correspond to different persuasion thresholds and Beta distribution parameterizations.

**Persuasion Mechanism**:
- `getPersuasionLevel(C)`: LLM evaluates the persuasion level on a scale of 1-5 based on the complete dialogue history.
- `getItemsToReturn(r, p)`: Samples based on a Beta distribution; higher persuasion levels tilt the distribution to the left (returning more information).
- Different personalities show distinct response modes to persuasion (e.g., Anxious is persuaded by "fair treatment" promises, while Adversarial is the hardest to handle).

**Validation Experiment**:
- 5 participants (including 2 professional journalists) evaluated persuasion levels turn-by-turn. The correlation with the LLM source's evaluation is $r=0.43$ ($p<0.0001$), rising to $r=0.68$ after removing the Adversarial personality.

## Key Experimental Results

### Counterfactual Analysis: LLM vs Human Journalists

| Dimension | Baseline-LLM | CoT | LLM w. Outline | Human Journalist |
|:--|:--|:--|:--|:--|
| Exact Match | 3.9% | 4.5% | 3.7% | 8.2% |
| Information Alignment | 4.4% | 3.6% | 3.8% | 17.5% |
| Motivation Alignment | 4.7% | 5.2% | 4.1% | 35.4% |
| Style Alignment | 11.9% | 12.8% | 9.6% | 40.2% |
| Discourse Alignment | 36.2% | 37.0% | 36.2% | 54.5% |
| Context Alignment | 53.0% | 56.9% | 46.6% | 60.3% |

- LLMs are close to humans in context understanding, but show massive gaps in information, motivation, and style dimensions.

### Discourse Role Distribution
- Humans: Acknowledgement accounts for ~9%, LLMs: close to 0% (**reduction of 50%+**).
- LLMs over-rely on Follow-up and Opinion/Broadening questions (increasing over time) and lack Outline-Level questions (**reduction of 30%**).

### Game Environment: Information Extraction Rate

| Model | Complete Game | No Persuasion | No Info Hiding |
|:--|:--|:--|:--|
| gpt-4o | **50.4%** | 49.8% | 84.2% |
| gpt-4o-mini | 49.3% | 47.5% | 84.7% |
| Llama-3.1-70b | 42.6% | 45.5% | 80.1% |
| Llama-3.1-8b | 42.4% | 48.3% | 74.9% |

- Information hiding is the primary barrier (extraction rate surges from ~50% to over ~80% upon removal).
- The removal of the persuasion mechanism has a smaller impact, indicating that LLMs themselves are not good at persuading.
- The Adversarial personality is the hardest to extract information from, yet its persuasion level is relatively easier to improve — indicating that LLMs are heavily disrupted by adversarial sources.

## Highlights & Insights

1. **Outstanding Dataset Scale and Quality**: The 40,000 real news interviews far exceed similar studies (typically 100 to 1,000 records), with quality guaranteed through rigorous filtering.
2. **Progressive Insights**: (1) LLMs lack acknowledgement $\rightarrow$ (2) lack multi-turn strategic planning $\rightarrow$ (3) the root cause is the absence of long-term reward signals.
3. **Exquisite Game Environment Design**: The combination of personality systems, persuasion mechanisms, and Beta-distribution sampling ensures the simulated environment is both realistic (human validation $r=0.43\text{--}0.68$) and trainable.
4. **Asymmetric Findings**: The asymmetric behavior between the source LLM and interviewer LLM is intriguing: LLMs simulate human behavior reasonably as sources, but suffer from a severe lack of strategic ability as interviewers.

## Limitations & Future Work

1. In the game environment, evaluating the persuasion of the source relies on the LLM's self-assessment, whereas LLMs have limited ability to judge their own level of persuasion.
2. The dataset is based on US news organizations (NPR/CNN), introducing cultural and linguistic biases toward English contexts.
3. Only Llama-3.1-70b was evaluated in the counterfactual analysis, lacking coverage of other models.
4. Reduced correlation under the Adversarial personality suggests that the simulation of extreme scenarios still needs improvement.
5. Only the realism of the environment has been validated so far; it has not yet been used for training actual dialogue agents (left for future work).

## Related Work & Insights

- **Relationship to Shaikh et al. (2024a)**: The former identified LLMs' lack of grounding language in emotional dialogues, whereas this work extends this finding to strategic dialogue scenarios and further quantifies it.
- **Relationship to Dialogue Tracking**: LLMs perform reasonably well on context understanding (60.3% vs. 53.0% for humans). The limitation lies in strategy rather than comprehension.
- **Insights**: (1) The NewsInterview game environment can be leveraged for RL training of strategic dialogue agents; (2) the design approach of the persuasion mechanism can be transferred to other scenarios requiring long-term grounding, such as tutoring and therapy.

## Rating

- Novelty: ⭐⭐⭐⭐ — Combining interview scenarios with a game environment is highly novel; the designs of source personality and persuasion mechanisms are creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Large-scale dataset + multi-dimensional discourse analysis + human validation + ablation studies, highly solid.
- Writing Quality: ⭐⭐⭐⭐ — The narrative from problem analysis to environment design is clear; Algorithm 1 is concise.
- Value: ⭐⭐⭐⭐ — Both the dataset and the game environment possess long-term reuse utility, serving as infrastructure for training strategic dialogue agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs](autogui_scaling_gui_grounding_with_automatic.md)
- [\[ACL 2025\] Mind the (Belief) Gap: Group Identity in the World of LLMs](mind_the_belief_gap_group_identity_in_the_world_of_llms.md)
- [\[ACL 2025\] Palm: A Culturally Inclusive and Linguistically Diverse Dataset for Arabic LLMs](palm_a_culturally_inclusive_and_linguistically_diverse_dataset_for_arabic_llms.md)
- [\[ACL 2025\] ASPERA: A Simulated Environment to Evaluate Planning for Complex Action Execution](aspera_a_simulated_environment_to_evaluate_planning_for_complex_action_execution.md)
- [\[ICLR 2026\] VERIFY: A Novel Multi-Domain Dataset Grounding LTL in Contextual Natural Language via Provable Intermediate Logic](../../ICLR2026/llm_nlp/verify_a_novel_multi-domain_dataset_grounding_ltl_in_contextual_natural_language.md)

</div>

<!-- RELATED:END -->
