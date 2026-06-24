---
title: >-
  [Paper Note] AI4Reading: Chinese Audiobook Interpretation System Based on Multi-Agent Collaboration
description: >-
  [ACL 2025][Audio & Speech][Audiobook Interpretation] This paper proposes AI4Reading, a Chinese audiobook interpretation system based on the collaboration of 11 specialized LLM Agents. It automatically generates interpretation scripts through phases of thematic analysis, case expansion, editorial refining, colloquial rewriting, and integration/revision, and then synthesizes audio using TTS. The generated interpretation scripts outperform the professional human interpretation p…
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Audiobook Interpretation"
  - "Multi-Agent Collaboration"
  - "LLM"
  - "Speech Synthesis"
  - "MetaGPT"
date: 2026-05-08
content_hash: a1071935f0bb1f3d
---

# AI4Reading: Chinese Audiobook Interpretation System Based on Multi-Agent Collaboration

**Conference**: ACL 2025  
**arXiv**: [2512.23300](https://arxiv.org/abs/2512.23300)  
**Code**: [https://github.com/9624219/AI4reading](https://github.com/9624219/AI4reading)  
**Area**: Speech / LLM Agent  
**Keywords**: Audiobook Interpretation, Multi-Agent Collaboration, LLM, Speech Synthesis, MetaGPT

## TL;DR
This paper proposes AI4Reading, a Chinese audiobook interpretation system based on the collaboration of 11 specialized LLM Agents. It automatically generates interpretation scripts through phases of thematic analysis, case expansion, editorial refining, colloquial rewriting, and integration/revision, and then synthesizes audio using TTS. The generated interpretation scripts outperform the professional human interpretation platform, FanDeng Reading, in terms of quality (conciseness, completeness, accuracy, and coherence).

## Background & Motivation
**Background**: In the audiobook market, "interpreted" audiobooks (such as FanDeng Reading) are gaining popularity. Unlike unabridged or summarized versions, interpretation requires a creative transformation of the original book—restating core viewpoints in more accessible language and supplementing them with cases and analyses.

**Limitations of Prior Work**: Human creation of interpreted audiobooks is extremely time-consuming and labor-intensive, requiring collaboration among authors, editors, and announcers. This limits the scale of interpreted content production and restricts it to specific languages.

**Key Challenge**: Direct generation of interpretation scripts using a single LLM (even with CoT or RAG) yields poor results. LLMs tend to generate summaries rather than interpretations, and the content is often too brief, lacking in-depth analysis and real-case supplements. An interpretation task is inherently a multi-dimensional, complex creative task that requires heterogeneous steps such as theme extraction, case analysis, logical argumentation, and colloquial rewriting.

**Goal**: How to automatically generate high-quality audiobook interpretation scripts that simultaneously satisfy three objectives: accurate preservation of content, enhanced comprehensibility, and a logical narrative structure.

**Key Insight**: Drawing inspiration from human team collaboration models (Thematic Researcher + Case Analyst + Editor + Announcer + Proofreader), this work designs a multi-agent collaborative framework where each agent is responsible for a clear, specific subtask.

**Core Idea**: An 11-agent specialized team is simulated to mimic the collaboration workflow of human publishing teams. It progresses step-by-step from theme extraction to case expansion, editorial integration, and colloquial rewriting, eventually generating interpretation scripts that are more accurate and coherent than human efforts.

## Method

### Overall Architecture
The system consists of two main modules: **Interpretation Script Generation** and **Audio Generation**.

The script generation module consists of 4 phases:
- Input: Book chapter content
- Phase 1 - Theme and Case Identification (TCI): Extracts core themes and relevant cases.
- Phase 2 - Preliminary Interpretation (PI): Expands cases, constructs arguments, and forms a first draft.
- Phase 3 - Oral Rewriting (OR): Converts written drafts into colloquial expressions.
- Phase 4 - Reconstruction and Revision (RR): Integrates paragraphs into a coherent full text.
- Output: Complete interpretation script $\rightarrow$ TTS Audio

### Key Designs

1. **Theme and Case Identification (TCI) - 3 Agents**:

    - **Function**: The Topic Analyst (TA) extracts up to 3 core themes and preliminary cases from chapters; Proofreader-1 (PR-1) reviews the rationality of the theme-case pairs and sends them back to the TA if they are unreasonable; Case Analyst-1 (CA-1) supplements richer background information and key details.
    - **Mechanism**: $Agent_{TA}(S) \rightarrow (T, C)$, which extracts a theme set $T$ and case set $C$. Then, PR-1 validates and categorizes them into valid/invalid pairs; invalid pairs trigger regeneration by the TA, while CA-1 performs information enrichment for valid pairs.
    - **Design Motivation**: This mimics the human cognitive process of reading and summarizing—first identifying key points and then supplementing details—while ensuring quality through a proofreading loop.

2. **Preliminary Interpretation (PI) - 4 Agents**:

    - **Function**: CA-2 adds personal anecdotes and real-life cases to make the content more relatable; CA-3 constructs logical arguments to demonstrate how cases support the themes; Editor-1 (ED-1) integrates all analytical materials into a coherent initial draft; PR-2 reviews the draft across two dimensions: completeness and logical flow.
    - **Mechanism**: $Agent_{ED-1}(t_i, c'_i, l_i, a_i) \rightarrow d_i$, where the editor synthesizes the theme, cases, arguments, and expanded materials into a preliminary draft. After PR-2 evaluates and provides feedback, ED-1 iteratively revises the draft until it passes review or reaches the maximum iteration limit $I_{max}$.
    - **Design Motivation**: Interpretation is not merely summarization; it requires supplementing cases to help understand theoretical concepts. The editor-proofreader iterative loop is leveraged to guarantee draft quality.

3. **Oral Rewriting (OR) - 2 Agents**:

    - **Function**: The Narrator (NR) simplifies complex sentence structures and incorporates colloquial vocabulary and conversational markers; PR-3 evaluates conversational naturalness and fluency.
    - **Mechanism**: $Agent_{NR}(d_i) \rightarrow o_i$, which converts the written preliminary draft into a spoken script, followed by iterative optimization after PR-3 review.
    - **Design Motivation**: Since audiobooks are an auditory medium, translating written text to oral speech is a critical step that requires a specialized agent.

4. **Reconstruction and Revision (RR) - 2 Agents**:

    - **Function**: Editor-2 (ED-2) incrementally merges the independent spoken scripts $\{o_1, ..., o_n\}$ of each theme into a coherent comprehensive text; PR-4 performs the final review of the merged text.
    - **Mechanism**: $M_i = Agent_{ED-2}(M_{i-1}, o_i)$, which merges paragraph-by-paragraph to ensure logical coherence and natural transitions.
    - **Design Motivation**: Scripts generated independently for each theme may lack cohesion, necessitating a global perspective for integration.

### Audio Generation
Fish-Speech TTS is utilized to convert the final script into audio, with transitional sound effects added between chapters to enhance the auditory experience.

## Key Experimental Results

### Experimental Setup
- Base LLM: DeepSeek-V3, temperature=1.3, max_token=8192, $I_{max}$=3
- Implementation framework: MetaGPT
- Baseline: FanDeng Reading (FanDeng)—a leading knowledge service platform in China, narrated by Fan Deng himself.
- Data: 5 books (psychology, personal growth, business management), 10 chapters; 10 interpretation segments were randomly selected.
- 7 human evaluators (undergraduates), with 2 excluded due to excessively short evaluation times.

### Main Results

| Dimension | Metric | AI4Reading | FanDeng | Comparison |
|------|------|-----------|---------|------|
| Audio Quality | Naturalness (Nat.) | 4.1 | **4.9** | -0.8 |
| Audio Quality | Concentration (Conc.) | 3.4 | **4.2** | -0.8 |
| Audio Quality | Comprehension (Compn.) | 3.1 | 3.3 | -0.2 |
| Text Quality | Simplification (Simp.) | **4.6** | 4.4 | +0.2 |
| Text Quality | Completeness (Compt.) | **4.0** | 3.8 | +0.2 |
| Text Quality | Accuracy (Acc.) | **4.3** | 4.2 | +0.1 |
| Text Quality | Coherence (Coh.) | **4.4** | 4.1 | +0.3 |

### Ablation Study

| Comparison | Description |
|--------|------|
| Single LLM (CoT) vs. Multi-Agent | Single LLMs tend to generate summaries instead of interpretations, showing insufficient content volume and lacking depth in cases. |
| Complete 11-Agent System | Outperforms human interpretation across all text quality metrics. |
| Audio Quality Gap | TTS still falls short of professional announcers in naturalness and engagement. |

### Key Findings
- AI4Reading outperforms FanDeng Reading (professional human interpretation) in all four dimensions of text quality, especially in coherence (+0.3).
- Audio quality still lags behind professional announcers, with the main gap lying in naturalness (4.1 vs 4.9).
- A single LLM cannot complete the interpretation task and degrades into summarization, confirming the necessity of the multi-agent design.
- The system generates segments averaging 4 minutes and 59 seconds vs. FanDeng's 4 minutes and 33 seconds, indicating slightly richer AI-generated content.

## Highlights & Insights
- **Multi-Agent division of labor mimicking human teams**: The role separation of the 11 agents (analysts, editors, announcers, proofreaders) precisely simulates the collaborative workflow of the publishing industry, where each agent has a clear responsibility and precise prompt. This division of labor is transferable to any content creation task requiring multi-step collaboration.
- **Iterative quality control**: Each phase incorporates a specialized Proofreader agent to perform an evaluation-feedback-revision loop. This "write-review-revise" model is an effective paradigm for ensuring LLM output quality.
- **Insight that Interpretation $\neq$ Summarization**: The paper clearly points out that a single LLM performing interpretation degrades into summarization, necessitating steps like case expansion and colloquial rewriting to achieve genuine "interpretation." This observation is highly instructive for similar tasks.

## Limitations & Future Work
- The evaluation scale is extremely small (only 5 valid evaluators, 10 segments), making statistical significance questionable.
- Only books on psychology, business, and personal growth were tested; literary and fiction works were not evaluated.
- The gap in TTS quality compared to professional narrating is significant, serving as the main bottleneck of the system.
- API calling costs and latency based on DeepSeek-V3 are not reported.
- Comparison with other multi-agent systems (e.g., AutoGen, CrewAI) is lacking.
- Whether 11 agents is the optimal configuration remains unverified through ablation, and redundancy may exist.

## Related Work & Insights
- **vs. MetaGPT**: AI4Reading is implemented based on the MetaGPT framework, but the agent roles are deeply customized for the audiobook interpretation scenario.
- **vs. Document Summarization/Simplification**: The interpretation task is far more complex than summarization and simplification, requiring content expansion, case supplementation, and colloquial rewriting, rather than just compression or simplification.
- **vs. LongWriter**: While LongWriter focuses on long-form text generation, AI4Reading focuses on multi-stage collaborative generation of interpreted content.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The first work to apply a multi-agent LLM system to audiobook interpretation, offering a highly valuable task definition.
- **Experimental Thoroughness**: ⭐⭐⭐ The evaluation scale is too small (5 people), lacking ablation studies for agent ablation and comparisons with more Baselines.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Detailed system design descriptions and clear mathematical formulations, though the experimental section is weak.
- **Value**: ⭐⭐⭐⭐ The system design concepts are highly referenceable, though the insufficient evaluation limits the credibility of some conclusions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sensorium Arc: AI Agent System for Oceanic Data Exploration and Interactive Eco-Art](../../NeurIPS2025/audio_speech/sensorium_arc_ai_agent_system_for_oceanic_data_exploration_and_interactive_eco-a.md)
- [\[AAAI 2026\] DeepDebater: A Superpersuasive Autonomous Policy Debating System](../../AAAI2026/audio_speech/a_superpersuasive_autonomous_policy_debating_system.md)
- [\[ACL 2026\] MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus](../../ACL2026/audio_speech/mcga_a_multi-task_classical_chinese_literary_genre_audio_corpus.md)
- [\[NeurIPS 2025\] From Generation to Attribution: Music AI Agent Architectures for the Post-Streaming Era](../../NeurIPS2025/audio_speech/from_generation_to_attribution_music_ai_agent_architectures_for_the_post-streami.md)
- [\[ACL 2026\] Phun-Bench: Evaluating LLMs on Phonological Understanding in Chinese](../../ACL2026/audio_speech/phun-bench_evaluating_llms_on_phonological_understanding_in_chinese.md)

</div>

<!-- RELATED:END -->
