---
title: >-
  [Paper Note] You need to MIMIC to get FAME: Solving Meeting Transcript Scarcity with Multi-Agent Conversations
description: >-
  [ACL 2025][Meeting Summarization] This paper proposes the MIMIC framework to generate synthetic meeting transcripts through multi-agent debate simulations, constructing the FAME dataset consisting of 800 meetings (500 English + 300 German), and designing a psychological-behavior-based evaluation framework for conversational realism.
tags:
  - "ACL 2025"
  - "Meeting Summarization"
  - "Synthetic Data"
  - "Multi-Agent"
  - "Meeting Transcripts"
  - "Psychological Behavior Modeling"
date: 2026-05-08
content_hash: 6275885848d5f0b5
---

# You need to MIMIC to get FAME: Solving Meeting Transcript Scarcity with Multi-Agent Conversations

**Conference**: ACL 2025  
**arXiv**: [2502.13001](https://arxiv.org/abs/2502.13001)  
**Code**: Yes (GitHub)  
**Area**: Other  
**Keywords**: Meeting Summarization, Synthetic Data, Multi-Agent, Meeting Transcripts, Psychological Behavior Modeling

## TL;DR

This paper proposes the MIMIC framework to generate synthetic meeting transcripts through multi-agent debate simulations, constructing the FAME dataset consisting of 800 meetings (500 English + 300 German), and designing a psychological-behavior-based evaluation framework for conversational realism.

## Background & Motivation

Meeting summarization research has long been constrained by the scarcity of high-quality training data. The primary reasons include:

**Privacy and Intellectual Property Restrictions**: Real-world meetings often contain highly sensitive or proprietary information.

**Prohibitive Annotation Costs**: Manual transcript annotation and summarization are extremely labor-intensive and expensive.

**Homogeneous Scenarios in Existing Datasets**: Current benchmarks like AMI (simulated business), ICSI (academic), and MeetingBank (municipal meetings) only cover a narrow range of scenarios.

**Scarcity of Non-English Resources**: Non-English datasets, such as FREDSum (French political debates), are highly fragmented and scarce.

Existing synthetic generation approaches also exhibit notable limitations: single-model-generated dialogue lacks genuine knowledge interaction and argumentative dynamics; crowdsourced role-playing is costly and difficult to scale; and automated heuristic methods (such as noise injection or sentence swapping) yield unnatural and disjointed conversations.

## Method

### Overall Architecture

MIMIC (Multi-agent IMItation of Conversations) borrows concepts from movie production workflows, structuring the pipeline into three stages and seven steps:
- **Pre-Production**: Content Brainstorming $\rightarrow$ Character Casting $\rightarrow$ Scriptwriting
- **Production**: Scene-by-Scene Shooting $\rightarrow$ Quality Control
- **Post-Production**: FX Injection $\rightarrow$ Linguistic Polishing

### Key Designs

1. **Content Brainstorming (Stage 1)**: Given a source knowledge document, the LLM extracts hierarchical topics and subtopics, then drafts an abstract target summary covering the discussion points (mimicking the style of 5 human-written QMSum summaries to maintain consistency).

2. **Character Casting (Stage 2)**:

    - Define participant profiles: functional roles (e.g., project manager, technical specialist), background, areas of expertise, and unique perspectives.
    - Define speaking styles: tone, vocabulary complexity, catchphrases, and filler words.
    - Distribute source knowledge segments based on expertise to create knowledge asymmetry, thereby fostering interdependence among participants.
    - Assign psychological behavioral roles (such as evaluator-critic, blocker, etc., based on the Benne & Sheats classification).
    - Utilize the LLM to inspect and resolve conflicting traits (e.g., being both "proactive" and "blocking").

3. **Scene Shooting (Stage 4)**:

    - Each participant is modeled as an independent LLM instance, taking turns to speak.
    - **Non-Omniscient Design**: Each participant only has access to their own profile, assigned knowledge snippets, summaries of prior scenes, and the conversation history of the current scene.
    - At the end of each turn, the current speaker designates the next speaker.
    - Voting mechanism to conclude a scene: The scene ends when more than 50% of the participants agree, or automatically after a system reminder at 50 turns.

4. **Quality Control (Stage 5)**: A director model reviews each scene across three dimensions—topic alignment, dialogue naturalness, and coherence/factual accuracy. A maximum of 3 reshoot rounds is permitted.

5. **FX Injection (Stage 6)**: Multi-modal-like real-world occurrences (e.g., phone interruptions, technical difficulties, off-topic questions) are injected with a 25% probability.

6. **Linguistic Polishing (Stage 7)**: A two-step refinement process is applied to eliminate redundant phrases and overly formal language, as well as to inject colloquial markers (e.g., hesitations, self-corrections).

### Loss & Training

MIMIC itself does not involve model training. It utilizes GPT-4o as the backbone LLM, leveraging its 128k-token context window and role-playing capabilities.

## Key Experimental Results

### Dataset Statistics Comparison

| Dataset | Meetings | Avg. Speakers | Avg. Turns | Avg. Words | Vocabulary | Interruptions | Language |
|--------|--------|-----------|---------|---------|--------|------|------|
| AMI | 137 | 4.0 | 513.5 | 4937.5 | 9,388 | No | EN |
| ICSI | 44 | 6.2 | 757.5 | 9889.4 | 9,164 | No | EN |
| QMSum | 232 | 7.2 | 521.0 | 7303.4 | 20,505 | No | Both |
| FAME-EN | 500 | 5.1 | 405.0 | 6223.4 | 10,347 | Yes (~0.5) | Both |
| FAME-GER | 300 | 5.0 | 393.3 | 6272.4 | 9,589 | Yes (~0.5) | Both |

### Realism Evaluation (Human Evaluation, 5-Point Scale)

| Dimension | FAME Score | Description |
|------|---------|------|
| Naturalness | 4.5/5 | Close to the spontaneity of real-world meetings |
| Colloquial Features | 3/5 | Preserves speaker-related challenges |
| Info-Density Difficulty | 4/5 | Introduces richer information-oriented difficulties |

### LLM Summarization Evaluation

| Model | Performance |
|------|------|
| GPT-4o | Consistently exhibits context-handling issues |
| Gemini 1.5 Pro | Similar issues |
| DeepSeek-R1 | Similar issues |
| Llama 3.3 70B | Similar issues |

### Key Findings

1. The average meeting length, number of turns, and participant count of FAME closely match real-world corpora (such as AMI).
2. The text overlap rate is extremely low (0.081 for English, 0.096 for German), indicating that participants paraphrase rather than copy the source knowledge.
3. Ablation studies demonstrate that MIMIC generates high-quality transcripts across different backbone LLMs.
4. Up to 4 behavior transitions are allowed within 3000+ participant profiles, successfully simulating role dynamics in real-world meetings.
5. Synthesized behavioral patterns closely align with patterns identified in 100 crowdsourced meeting experience reports.

## Highlights & Insights

- **The movie-production metaphor is consistently utilized**, making the complex multi-stage pipeline intuitive and easy to understand.
- **The Non-Omniscient Design is a crucial innovation**: Each agent possesses only private memory, avoiding the artificiality of a single omniscient model.
- **Psychological behavioral role allocation** (based on the Benne & Sheats classification) provides a solid theoretical foundation for agent interactions.
- The voting mechanism allows the meeting length to be determined dynamically by the participants rather than being artificially fixed.
- Provides the first parallel English-German bilingual meeting corpus.

## Limitations & Future Work

1. The source knowledge is restricted to Wikipedia, which limits topic diversity.
2. Synthetic speech features (e.g., filler word patterns) might still not be fully natural.
3. The high operation cost of GPT-4o poses a financial barrier for large-scale generation.
4. Target summaries are synthetically generated by the LLM instead of human-written, which might lead to quality bias.
5. The feasibility on low-resource or minor languages has not been evaluated.

## Related Work & Insights

- Google NotebookLM and Nvidia PDF-to-Podcast support only two-party conversations and lack multi-participant dynamics.
- Shares technical and theoretical foundations with multi-agent debate frameworks such as Liang et al. (2024) and Du et al. (2024).
- Employs self-refinement (Self-Refinement, Madaan et al., 2023) in the quality control phase.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Multi-stage movie-production pipeline + psychological behavior modeling + non-omniscient design
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Human evaluation, LLM evaluation, ablation studies, and comparison with real-world data
- **Writing Quality**: ⭐⭐⭐⭐ Vivid metaphors with a highly clear structure
- **Value**: ⭐⭐⭐⭐ Addresses the widely acknowledged data scarcity bottleneck in the meeting summarization domain

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)
- [\[ACL 2025\] Hanging in the Balance: Pivotal Moments in Crisis Counseling Conversations](hanging_in_the_balance_pivotal_moments_in_crisis_counseling_conversations.md)
- [\[ICCV 2025\] I Am Big, You Are Little; I Am Right, You Are Wrong](../../ICCV2025/others/i_am_big_you_are_little_i_am_right_you_are_wrong.md)
- [\[ACL 2025\] USDC: A Dataset of User Stance and Dogmatism in Long Conversations](usdc_a_dataset_of_underlineuser_underlinestance_and_underlinedogmatism_in_long_u.md)
- [\[AAAI 2026\] Local Guidance for Configuration-Based Multi-Agent Pathfinding](../../AAAI2026/others/local_guidance_for_configuration-based_multi-agent_pathfinding.md)

</div>

<!-- RELATED:END -->
