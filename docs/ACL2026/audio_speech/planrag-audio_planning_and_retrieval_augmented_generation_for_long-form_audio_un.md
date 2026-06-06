---
title: >-
  [Paper Note] PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding
description: >-
  [ACL2026][Audio & Speech][Long-form Audio Understanding] PlanRAG-Audio reframes long-form audio understanding as a problem of "planning which modalities and time segments to query…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "Long-form Audio Understanding"
  - "Retrieval Planning"
  - "Structured Audio Database"
  - "SQL Retrieval"
  - "Multimodal Audio Reasoning"
date: 2026-05-08
content_hash: d4349597337eed05
---

# PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding

**Conference**: ACL2026  
**arXiv**: [2605.20414](https://arxiv.org/abs/2605.20414)  
**Code**: The paper states that data and code will be made public after acceptance  
**Area**: Long-form Audio Understanding / Audio RAG  
**Keywords**: Long-form Audio Understanding, Retrieval Planning, Structured Audio Database, SQL Retrieval, Multimodal Audio Reasoning

## TL;DR
PlanRAG-Audio reframes long-form audio understanding as a problem of "planning which modalities and time segments to query, then retrieving evidence from a structured audio database." This approach reduces the LLM input for 60 minutes of audio from approximately 115k tokens to about 1k tokens, while significantly improving speaker counting, event ordering, and speaker-constrained QA.

## Background & Motivation
**Background**: Large audio-language models can process speech content, speakers, emotions, and non-speech events, but long audio quickly leads to token and memory bottlenecks. For instance, a one-hour lecture corresponds to roughly 12k text tokens but may exceed 100k speech tokens. While text RAG has proven that "retrieving only relevant evidence" alleviates long-context issues, audio RAG must additionally handle multi-modality and temporal alignment.

**Limitations of Prior Work**: Many long-audio methods first perform ASR to text and then apply NLP, which ignores intonation, speakers, emotions, and background events. Directly feeding the entire audio to a long-context model is costly, results in unstable output formats, and lacks support for non-text tasks such as speaker diarization, emotion recognition, and sound event detection (SED).

**Key Challenge**: The difficulty of long-form audio lies not only in the input length but also in the fact that queries often span multiple heterogeneous cues. A single question might require knowing a specific speaker's words, the emotion during a certain period, the sequence of background events, and a specific output format. Without explicit planning, models struggle to identify which streams to observe and easily lose key evidence among irrelevant information.

**Goal**: The authors aim to construct a reproducible, zero-shot, task-agnostic long-form audio understanding framework. The model first generates a structured retrieval plan, then uses deterministic SQL to retrieve relevant segments from an audio database, and finally generates an answer using compact evidence.

**Key Insight**: The paper organizes audio pre-processing results into a time-aligned database. Instead of consuming the entire audio directly, the LLM generates a constrained retrieval plan specifying streams, filters, fusion, return fields, and the answer schema.

**Core Idea**: Externalize long-form audio reasoning as database queries using planned structured retrieval, allowing the LLM to process only a small amount of cross-modal evidence relevant to the question.

## Method
The core of PlanRAG-Audio is to split "listening to the entire audio to answer" into "offline perception and database construction + online retrieval planning + SQL execution + answer generation." The key to this design is not a stronger audio encoder, but converting information such as time, speakers, emotions, and events into searchable structured records.

### Overall Architecture
In the first stage, the system performs speaker diarization, ASR, emotion recognition, and sound event detection on the raw audio to build an audio database $D(a)$. In the second stage, the planning LLM generates a retrieval plan $\Theta(q)$ based on the user's question, deciding which streams to query, what filters to apply, how to fuse multiple streams, which fields to return, and the final answer schema. In the third stage, a rule-based SQL generator compiles the plan into a merged SQL query, executes it on the database, and returns segments $R(q,a)$. In the fourth stage, the generation LLM produces the answer based on the retrieved segments and the output schema.

### Key Designs
1.  **Structured Audio Database**:
    - **Function**: Deconstructs long-form audio into retrievable, time-aligned multimodal records.
    - **Mechanism**: Speaker diarization identifies homogeneous speaker segments, acting as shared boundaries for transcript, speaker, and emotion streams; the sound event stream is generated independently using a sliding window, recording its own start/end and label-score JSONB.
    - **Design Motivation**: Long-form audio questions often require aligning "who said it," "what was said," "what was the emotion," and "what happened in the background." By tabularizing the data, cross-modal matching can be completed via temporal joins rather than implicit inference by the LLM in a long context.

2.  **Constrained Retrieval Planning**:
    - **Function**: Allows the system to explicitly decide necessary information before retrieval, avoiding blindly stuffing the entire database into the model.
    - **Mechanism**: The retrieval plan contains five categories of fields: `streams`, `filters`, `fusion`, `output return_fields`, and `answer_schema`. For example, a speaker-constrained MCQA would select transcription and speaker streams, specify text or speaker filters, and require the final answer to be restricted to A/B/C/D.
    - **Design Motivation**: The key to complex audio problems lies in "what to query." Constraining planning with a fixed schema reduces invalid plans and formatting errors, making subsequent SQL compilation deterministic and controllable.

3.  **Deterministic SQL Compilation and Temporal Fusion**:
    - **Function**: Converts LLM plans into executable retrieval, reducing the instability of generative retrieval.
    - **Mechanism**: Each stream is compiled into an independent CTE, filters are compiled into `WHERE` clauses, and the final `SELECT` projects fields according to the output contract, applying fusion strategies for temporal joins. In the appendix, temporal fusion uses nearest midpoint matching with a default tolerance window of $\tau=2.5$ seconds.
    - **Design Motivation**: The LLM handles high-level planning while the database performs precise execution. This allows for scalability to new modalities while moving cross-modal alignment from prompt reasoning to inspectable query logic.

### Loss & Training
PlanRAG-Audio does not train an end-to-end model but combines off-the-shelf perception modules and LLMs in a zero-shot setting. The primary configuration includes OWSM-CTC v4 medium for ASR, Pyannote community-1 for diarization, Odyssey 2024 SER baseline for emotion recognition, and BEATs iter3+ AS2M finetuned for SED. Qwen3-4B-Instruct serves as the main generation model. Long-context baselines include Gemini 2.5 Flash and Voxtral-Mini-3B-2507. The authors explicitly avoid task-specific prompt engineering or handwritten SQL, relying instead on a unified planning schema.

## Key Experimental Results

### Main Results
| Experiment Item | Model / Setting | Value | Description |
|-----------------|-----------------|-------|-------------|
| 60-min MCQA input length | Gemini direct audio | 115.2k tokens | High cost for direct long-context processing |
| 60-min MCQA input length | Gemini + PlanRAG-Audio | 0.9k tokens | Post-retrieval input is nearly constant |
| 60-min MCQA input length | Qwen + PlanRAG-Audio | 1.2k tokens | Small models can handle retrieved evidence |
| Gemini diarization parse failure | 10 to 540 mins | 17.92% unparsable | Format stability is a significant issue in long contexts |

### Ablation Study
| Task | Without PlanRAG-Audio | With PlanRAG-Audio | Key Change |
|------|-----------------------|--------------------|------------|
| Gemini speaker count | 14.20% | 69.40% | Explicit speaker segments turn counting into structured reasoning |
| Gemini event order | Spearman 0.30 | Spearman 0.68 | Retrieved event timestamps make sorting more stable |
| Qwen speaker count | 35.16% | 36.66% | Minor gain, indicating Qwen has some inherent ability to use structured evidence |
| Qwen event order | Spearman 0.11 | Spearman 0.34 | Externalizing temporal structure brings clear benefits |
| Gemini speaker-constrained MCQA | QA 68.13%, Abst. 0.54% | QA 70.96%, Abst. 94.90% | Retrieval planning greatly improves abstention in unanswerable scenarios |
| Qwen speaker-constrained MCQA | Direct baseline not reported | QA 67.59%, Abst. 82.20% | Small models can handle speaker constraints with structured evidence |

### Key Findings
- The primary benefit of PlanRAG-Audio comes from "selective retrieval" rather than a stronger generative model. Without planning, Qwen's performance degrades in long audio when given the full database; with planning, performance remains stable regardless of duration.
- In absolute results for OWSM + Qwen, MCQA parseable accuracy without PlanRAG dropped from 66.24 (10 mins) to 30.69 (300 mins), and was unreportable at 540 mins; with PlanRAG, it maintained 65.67, 67.23, 65.09, 63.87, and 56.70 from 10 to 540 mins.
- Semantic retrieval is not necessarily superior to keyword retrieval. In the appendix, for 30-min MCQA, keyword search achieved 67.23 vs. vector search at 60.40; at 540 mins, keyword was 56.07 vs. vector 57.39, suggesting that retrieval planning is more critical than the retriever's expressiveness.
- Pre-processing costs grow approximately linearly with audio length. It is suitable for offline indexing where queries are reused multiple times, though it introduces extra overhead for real-time one-off queries.

## Highlights & Insights
- The paper transforms long-form audio understanding from a model context window problem into an information systems problem. Normalizing perception results into a database and using LLM for query planning provides a clear engineering abstraction.
- The `answer_schema` is a crucial but often overlooked design. Long-context models fail not just because of incorrect answers, but because outputs are unparsable; constraining the output format during the planning phase reduces such errors.
- Results show that speaker-constrained abstention is a strength of PlanRAG-Audio. When a question is unanswerable, it is necessary to know that "the specified speaker provided no relevant evidence," which is where structured retrieval is more reliable than full-text input.
- Rather than hiding complexity in a black-box model, the paper uses simple keyword retrieval for experiments, highlighting the contribution of planning.

## Limitations & Future Work
- The authors acknowledge that Gemini evaluations are affected by API limitations, including long-context instability and format failures, which may impact the precise assessment of long-context baselines.
- While current results suggest vector search does not provide stable gains over keyword retrieval, stronger hybrid retrieval, learned rankers, or query rewriting could still improve recall.
- The framework is dependent on upstream perception modules. Errors in ASR, diarization, emotion recognition, and SED propagate directly into the database; PlanRAG-Audio itself does not optimize these modules.
- Pre-processing is reusable but not free. It is ideal for scenarios like meeting recordings, podcast archives, or classroom recordings with multiple queries; for low-latency real-time voice assistants, incremental database construction and streaming planning are required.
- Risks are inherited from pre-trained components, such as ASR bias against accents/languages, emotion misidentification, and speaker recognition errors.

## Related Work & Insights
- **vs ASR-first long audio QA**: Traditional methods convert audio to text for NLP, losing speakers, emotions, and non-speech events; PlanRAG-Audio preserves this information as parallel streams.
- **vs direct long-context LALM**: Gemini/Voxtral are expensive and unstable for long audio; PlanRAG-Audio compresses input to ~1k tokens by retrieval, decoupling task length from LLM input length.
- **vs text PlanRAG / Plan*RAG**: Text planning retrieval mainly handles document selection and reasoning steps; PlanRAG-Audio additionally manages temporal alignment, speaker constraints, and acoustic events.
- **Inspiration**: For long-duration multimodal data such as video, sensor logs, or robot trajectories, one can first build a structured event database and let the LLM perform query planning instead of stuffing raw sequences directly into the context.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Systematically migrating planning RAG to long audio and using SQL/database abstractions for cross-modal temporal alignment is a solid approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers QA, MCQA, summarization, SD, ER, SED, and high-level combined tasks; error analysis of upstream modules has room for deeper exploration.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and intuitive examples; some absolute results are in the appendix, requiring cross-referencing for full context.
- **Value**: ⭐⭐⭐⭐⭐ Highly valuable for practical long-form audio retrieval and QA scenarios like long meetings, podcasts, lectures, and customer service recordings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Comprehensive Benchmarking of Long-Form Speech Generation in Diverse Scenarios](comprehensive_benchmarking_of_long-form_speech_generation_in_diverse_scenarios.md)
- [\[ACL 2026\] MARQUIS: A Three-Stage Pipeline for Video Retrieval-Augmented Generation](marquis_a_three-stage_pipeline_for_video_retrieval-augmented_generation.md)
- [\[ACL 2026\] Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval](omni-embed-audio_leveraging_multimodal_llms_for_robust_audio-text_retrieval.md)
- [\[ICCV 2025\] Latent Swap Joint Diffusion for 2D Long-Form Latent Generation](../../ICCV2025/audio_speech/latent_swap_joint_diffusion_for_2d_long-form_latent_generation.md)
- [\[ACL 2026\] Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation](anchored_cyclic_generation_a_novel_paradigm_for_long-sequence_symbolic_music_gen.md)

</div>

<!-- RELATED:END -->
