---
title: >-
  [Paper Note] PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding
description: >-
  [ACL2026 Findings][Audio & Speech][Long-form Audio Understanding] PlanRAG-Audio reformulates long-form audio understanding as a process of "planning which modalities and time segments to query, then retrieving evidence from a structured audio database." This reduces the LLM input for a 60-minute audio from approximately 115k tokens to about 1k tokens, while significantly improving performance in speaker counting, event ordering, and speaker-constrained QA.
tags:
  - "ACL2026 Findings"
  - "Audio & Speech"
  - "Long-form Audio Understanding"
  - "Retrieval Planning"
  - "Structured Audio Database"
  - "SQL Retrieval"
  - "Multimodal Audio Reasoning"
date: 2026-05-08
content_hash: e34c6440c2691f2a
---

# PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding

**Conference**: ACL2026 Findings  
**arXiv**: [2605.20414](https://arxiv.org/abs/2605.20414)  
**Code**: Data and code are stated to be released upon acceptance  
**Area**: Long-form Audio Understanding / Audio RAG  
**Keywords**: Long-form Audio Understanding, Retrieval Planning, Structured Audio Database, SQL Retrieval, Multimodal Audio Reasoning

## TL;DR
PlanRAG-Audio reformulates long-form audio understanding as a process of "planning which modalities and time segments to query, then retrieving evidence from a structured audio database." This reduces the LLM input for a 60-minute audio from approximately 115k tokens to about 1k tokens, while significantly improving performance in speaker counting, event ordering, and speaker-constrained QA.

## Background & Motivation
**Background**: Large audio-language models can process speech content, speakers, emotions, and non-speech events, but long audio rapidly reaches token and memory bottlenecks. For instance, a one-hour lecture corresponds to ~12k text tokens but can exceed 100k speech tokens. Text RAG has proven that "taking only relevant evidence" alleviates long-context issues, but audio RAG must also handle multimodality and temporal alignment.

**Limitations of Prior Work**: Many long-form audio methods perform ASR transcription followed by NLP, which ignores intonation, speakers, emotions, and background events. Feeding an entire audio segment directly to long-context models is costly, produces unstable output formats, and provides insufficient support for non-text tasks like speaker diarization, emotion recognition, and sound event detection.

**Key Challenge**: The difficulty of long-form audio lies not only in input length but also in questions spanning multiple heterogeneous cues. A single query might require knowing a specific speaker's words, the emotion of a particular period, the sequence of background events, and a specific output format simultaneously. Without explicit planning, models neither know which streams to examine nor easily lose critical evidence amidst irrelevant information.

**Goal**: The authors aim to build a reproducible, zero-shot, task-agnostic long-form audio understanding framework where the model first generates a structured retrieval plan, then retrieves relevant segments from an audio database using deterministic SQL, and finally generates an answer using compact evidence.

**Key Insight**: The paper organizes audio preprocessing results into a time-aligned database. Instead of consuming the entire audio, the LLM generates a constrained retrieval plan specifying streams, filters, fusion, return fields, and an answer schema.

**Core Idea**: Externalize long-form audio reasoning as database queries using planned structured retrieval, allowing the LLM to process only a small amount of cross-modal evidence relevant to the question.

## Method

### Overall Architecture
PlanRAG-Audio addresses the issue that "feeding one hour of audio into an LLM is long, expensive, and unstable." Its key is not replacing the audio encoder with a stronger one, but reformulating "listening to the whole audio before answering" into "offline indexing + online retrieval planning + deterministic execution + compact generation." This ensures the model only processes a small amount of cross-modal evidence relevant to the question.

In the first stage, the system performs speaker diarization, ASR, emotion recognition, and sound event detection on the raw audio, organizing the results into a time-aligned audio database $D(a)$. In the second stage, a planning LLM generates a retrieval plan $\Theta(q)$ based on the user question, determining which streams to query, which filters to use, how to fuse multiple streams, which fields to return, and the schema for the final answer. In the third stage, a rule-based SQL generator compiles the plan into a merged SQL query, executes it against the database, and returns segments $R(q,a)$. In the fourth stage, a generation LLM produces the answer based only on these retrieved segments and the output schema. Thus, the task length is decoupled from the LLM input length: the input for a 60-minute audio is reduced from ~115k tokens to ~1k tokens.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A0["Raw Audio a"] --> DB
    Q["User Question q"] --> PLAN["Constrained Retrieval Planning<br/>Planning LLM outputs fixed schema plan Θ(q)"]
    subgraph DB["Structured Audio Database D(a)"]
        direction TB
        P1["Speaker Diarization / ASR<br/>Emotion Recognition / Sound Event Detection"] --> P2["Time-aligned multimodal records<br/>transcript / speaker / emotion / event"]
    end
    DB --> SQL["Deterministic SQL Compilation & Temporal Fusion<br/>stream→CTE, filter→where, temporal join τ=2.5s"]
    PLAN --> SQL
    SQL --> R["Retrieved Segments R(q,a)<br/>~1k tokens"]
    R --> GEN["Generation LLM follows answer_schema (Scaffolding)"]
```

### Key Designs

**1. Structured Audio Database: Decomposing long audio into time-aligned, queryable multimodal records**

Long-form audio questions often require aligning "who said it," "what was said," "what was the emotion," and "what happened in the background" to the same time segment. Models directly consuming long contexts must infer these relationships implicitly within a long string of tokens, which is unstable and prone to losing cues. PlanRAG-Audio performs perception before indexing: speaker diarization creates homogeneous time segments used as shared boundaries for transcript, speaker, and emotion streams; the sound event stream is generated independently with a sliding window, recording its own start/end and label-score JSONB. Once information becomes structured records with timestamps, cross-modal matching can be performed accurately using temporal joins rather than forcing the LLM to guess which emotion corresponds to which sentence.

**2. Constrained Retrieval Planning: Thinking explicitly about "what to query" before retrieval**

The difficulty of complex audio questions lies in "which streams to use and what criteria to filter by." blindly feeding the entire database content to the model is wasteful and risks incorrect formatting. PlanRAG-Audio requires the planning LLM to output a retrieval plan with a fixed schema containing five categories of fields: streams, filters, fusion, output return_fields, and answer_schema. For a speaker-constrained MCQA, the plan selects transcription and speaker streams, specifies text or speaker filters, and constrains the answer_schema to output only A/B/C/D. Framing the planning with a fixed schema reduces invalid plans, makes subsequent SQL compilation deterministic, and mitigates the common "correct answer but unparseable output" issue in long-context models.

**3. Deterministic SQL Compilation and Temporal Fusion: High-level planning by LLM, precise execution by Database**

Generative retrieval is inherently unstable; letting an LLM directly "retrieve and align" is error-prone. PlanRAG-Audio compiles the plan into executable SQL: each stream is compiled into an independent CTE, filters into where conditions, and the final SELECT projects fields according to the output contract and performs a temporal join based on the fusion strategy. The temporal fusion in the appendix uses a nearest-midpoint distance match with a default tolerance window of $\tau=2.5$ seconds. Thus, the LLM is only responsible for high-level planning, while cross-modal alignment is transitioned from error-prone prompt reasoning to verifiable query logic. Scalability is also high, as adding a new modality only requires compiling one additional stream.

### A Full Example: Speaker-constrained MCQA for a 60-minute Lecture

Consider the question: "Regarding what a specific speaker said in the lecture, which option is correct?" If the entire audio is fed directly to Gemini, the input is ~115.2k tokens, which is expensive and often leads to unparseable formats. Using PlanRAG-Audio: in the offline phase, the one-hour audio is first indexed into a database. Diarization segments time periods for each speaker, and transcript/emotion are aligned to these boundaries, while SED records background events via sliding windows. In the online phase, the planning LLM generates a retrieval plan after seeing the question—selecting transcription and speaker streams, filtering for the target speaker, and limiting the answer_schema to A/B/C/D. The SQL generator compiles this into a query with CTEs and temporal joins, retrieving only a few segments related to that speaker, totaling ~0.9k tokens. The generation LLM answers based on this compact evidence; if the speaker has no relevant speech in the database, the structured result is empty, and the model chooses to abstain (in experiments, abstention for this setting rose from 0.54% to 94.90%), rather than hallucinating an answer.

### Loss & Training
PlanRAG-Audio does not train an end-to-end model but combines off-the-shelf perception modules and LLMs in a zero-shot setting. Main configurations include OWSM-CTC v4 medium for ASR, Pyannote community-1 for diarization, Odyssey 2024 SER baseline for emotion recognition, and BEATs iter3+ AS2M finetuned for SED. Qwen3-4B-Instruct serves as the primary generation model. Long-context baselines include Gemini 2.5 Flash and Voxtral-Mini-3B-2507. The authors explicitly avoid task-specific prompt engineering or manual SQL, instead relying on the unified planning schema.

## Key Experimental Results

### Main Results

| Experiment Item | Model / Setting | Value | Description |
|-----------------|-----------------|-------|-------------|
| 60-min MCQA Input Length | Gemini Direct Audio | 115.2k tokens | High processing cost for direct long-context |
| 60-min MCQA Input Length | Gemini + PlanRAG-Audio | 0.9k tokens | Input approx. constant after retrieval |
| 60-min MCQA Input Length | Qwen + PlanRAG-Audio | 1.2k tokens | Small models can handle retrieved evidence |
| Gemini Diarization Parse Failure | 10 to 540 minutes | 17.92% Unparseable | Output stability is a major issue in long contexts |

### Ablation Study

| Task | w/o PlanRAG-Audio | w/ PlanRAG-Audio | Key Change |
|------|-------------------|--------------------|------------|
| Gemini Speaker Count | 14.20% | 69.40% | Explicit speaker timestamps turn counting into structured reasoning |
| Gemini Event Order | Spearman 0.30 | Spearman 0.68 | Event ordering is more stable after timestamp retrieval |
| Qwen Speaker Count | 35.16% | 36.66% | Slight improvement, suggesting Qwen has some ability to use structured evidence |
| Qwen Event Order | Spearman 0.11 | Spearman 0.34 | Externalizing temporal structure yields significant gains |
| Gemini Speaker-constrained MCQA | QA 68.13%, Abst. 0.54% | QA 70.96%, Abst. 94.90% | Retrieval planning significantly improves abstention for unanswerable scenarios |
| Qwen Speaker-constrained MCQA | Direct baseline not reported | QA 67.59%, Abst. 82.20% | Small models with structured evidence can handle speaker constraints |

### Key Findings
- The main gain of PlanRAG-Audio comes from "selective retrieval" rather than a stronger generation model. Without planning, Qwen degrades under long audio even when the whole database content is provided; with planning, performance remains stable regardless of duration.
- In absolute results for OWSM + Qwen, the MCQA parseable accuracy without PlanRAG dropped from 66.24 at 10 minutes to 30.69 at 300 minutes, becoming unreportable at 540 minutes. With PlanRAG, it remained at 65.67, 67.23, 65.09, 63.87, and 56.70 from 10 to 540 minutes.
- Semantic retrieval is not necessarily superior to keyword retrieval. In the appendix, for 30-min MCQA, keyword search achieved 67.23 vs. 60.40 for vector search; at 540 minutes, keyword was 56.07 vs. 57.39 for vector, indicating that retrieval planning is more critical than retriever expressiveness.
- Preprocessing costs increase approximately linearly with audio length. It is suitable for offline indexing where multiple queries are reused, but introduces extra overhead for real-time one-off queries.

## Highlights & Insights
- The paper shifts long-form audio understanding from a model context window problem to an information systems problem. Normalizing perception results into a database and using an LLM to plan queries is a clear engineering abstraction.
- The `answer_schema` is a subtle but important design. Long-context models fail not just due to wrong answers, but also due to unparseable outputs; constraining format during the planning phase reduces these errors.
- Results show that speaker-constrained abstention is a strength of PlanRAG-Audio. When a question is unanswerable, one needs to know that "the specified speaker did not provide relevant evidence," which is exactly where structured retrieval is more reliable than full-text input.
- The paper avoids hiding complexity in a black-box model, using simple keyword retrieval for experiments to highlight the contribution of the planning mechanism.

## Limitations & Future Work
- The authors acknowledge that Gemini evaluations are affected by API limitations, including long-context instability and format failures, which may impact the precise judgment of long-context baselines.
- While the appendix shows vector search did not provide stable gains, stronger hybrid retrieval, learned rankers, or query rewriting might still improve recall.
- The framework is dependent on upstream perception modules. Errors in ASR, diarization, emotion recognition, and SED propagate directly into the database; PlanRAG-Audio itself does not optimize these modules.
- Preprocessing is reusable but not free. It is ideal for scenarios like meeting recordings, podcast archives, and other multi-query domains, but for low-latency real-time voice assistants, incremental indexing and streaming planning are required.
- Risks are inherited from pre-trained components, such as ASR bias regarding accents/languages, misidentified emotions, and speaker identification errors.

## Related Work & Insights
- **vs ASR-first long audio QA**: Traditional methods perform NLP on text after ASR, losing speakers, emotions, and non-speech events; PlanRAG-Audio stores these as parallel streams.
- **vs direct long-context LALM**: Gemini/Voxtral are expensive and format-unstable for long audio; PlanRAG-Audio compresses input to ~1k tokens and decouples task length from LLM input length.
- **vs text PlanRAG / Plan*RAG**: Text retrieval planning focuses on document selection and reasoning steps; PlanRAG-Audio additionally handles temporal alignment, speaker constraints, and acoustic events.
- **Insight**: For long-duration multimodal data like video, sensor logs, or robot trajectories, one could first build a structured event database and then let the LLM perform query planning instead of feeding raw sequences directly into the context.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically migrating planning RAG to long-form audio and using SQL/database abstractions for cross-modal temporal alignment is a solid approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers QA, MCQA, summarization, SD, ER, SED, and advanced composite tasks; further error analysis of upstream modules is possible.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and intuitive examples; some absolute results are in the appendix, requiring cross-referencing for the main text's relative results.
- Value: ⭐⭐⭐⭐⭐ High practical value for real-world scenarios like long meetings, podcasts, classroom recordings, and customer service logs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Comprehensive Benchmarking of Long-Form Speech Generation in Diverse Scenarios](comprehensive_benchmarking_of_long-form_speech_generation_in_diverse_scenarios.md)
- [\[ACL 2026\] MARQUIS: A Three-Stage Pipeline for Video Retrieval-Augmented Generation](marquis_a_three-stage_pipeline_for_video_retrieval-augmented_generation.md)
- [\[CVPR 2026\] AudioStory: Generating Long-Form Narrative Audio with Large Language Models](../../CVPR2026/audio_speech/audiostory_generating_long-form_narrative_audio_with_large_language_models.md)
- [\[ACL 2025\] WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models](../../ACL2025/audio_speech/wavrag_audio-integrated_retrieval_augmented_generation_for_spoken_dialogue_model.md)
- [\[ICLR 2026\] YuE: Scaling Open Foundation Models for Long-Form Music Generation](../../ICLR2026/audio_speech/yue_scaling_open_foundation_models_for_long-form_music_generation.md)

</div>

<!-- RELATED:END -->
