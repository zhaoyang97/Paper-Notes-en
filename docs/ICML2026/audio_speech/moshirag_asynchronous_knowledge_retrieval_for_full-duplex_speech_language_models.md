---
title: >-
  [Paper Note] MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models
description: >-
  [ICML 2026][Audio & Speech][full-duplex] MoshiRAG introduces a special ⟨ret⟩ trigger token into Moshi, a full-duplex speech model, enabling the model to asynchronously call an LLM/search backend for reference documents w…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "full-duplex"
  - "speech LM"
  - "RAG"
  - "Moshi"
  - "asynchronous retrieval"
  - "keyword delay"
date: 2026-05-08
content_hash: e2cbb28b4ab55770
---

# MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models

**Conference**: ICML 2026  
**arXiv**: [2604.12928](https://arxiv.org/abs/2604.12928)  
**Code**: https://github.com/kyutai-labs/moshi-rag (available)  
**Area**: Dialogue Systems / Full-Duplex Speech / Retrieval-Augmented  
**Keywords**: full-duplex, speech LM, RAG, Moshi, asynchronous retrieval, keyword delay

## TL;DR
MoshiRAG introduces a special ⟨ret⟩ trigger token into Moshi, a full-duplex speech model, enabling the model to asynchronously call an LLM/search backend for reference documents while speaking. By leveraging the natural "keyword delay" (the interval from speaking onset to keyword appearance), retrieval latency under 2 seconds is completely masked. This elevates the factuality of the speech model to the level of GPT-4o Audio on LlamaQ/WebQ/TriviaQA/HaluEval, while preserving full-duplex real-time interaction.

## Background & Motivation
**Background**: Modern speech dialogue is shifting from cascaded ASR-Dialogue-TTS pipelines to end-to-end speech LMs. Full-duplex models (Moshi, dGSLM successors) can "listen and speak simultaneously," closely resembling human conversation; turn-based models (GLM-4-Voice, Freeze-Omni, etc.) can only alternate between listening and speaking.

**Limitations of Prior Work**: (1) Native speech LM training data is much less than that for text LMs, resulting in much lower factuality for models of similar size; (2) Simply scaling up model size can improve factuality, but full-duplex requires real-time inference, so parameter count cannot be arbitrarily increased; (3) Existing RAG work is almost entirely based on turn-based settings, as traditional RAG introduces a synchronous wait period, which conflicts with "listen while speaking."

**Key Challenge**: Factuality requires external knowledge → needs RAG → RAG introduces latency → breaks full-duplex. One must either sacrifice factuality or real-time performance.

**Goal**: (1) Enable Moshi to autonomously determine "when external knowledge is needed"; (2) Trigger retrieval without interrupting speech flow; (3) Allow hot-swappable backends without retraining.

**Key Insight**: The authors observe an overlooked temporal structure—the "keyword delay" between speaking onset (TTFAT) and keyword appearance (KD), which exceeds 3 seconds for many models. If retrieval can be completed within this gap (target ≤2 seconds), the answer can be fetched "while delivering a polished opening."

**Core Idea**: Use a ⟨ret⟩ trigger token + asynchronous backend + inject reference embedding after the lead segment, transforming RAG's synchronous blocking into a full-duplex "speak while retrieving" paradigm.

## Method

### Overall Architecture
MoshiRAG consists of three components: (1) The frontend is a RAG-aware full-duplex model based on Moshi 7B, taking user speech tokens and its own previous text/speech tokens as input, and outputting a special ⟨ret⟩ trigger token; (2) A 1B streaming ASR (0.5s latency) transcribes user speech for retrieval; (3) An asynchronous retrieval backend, which can be LLM-based (Gemma 3 27B reading context to provide references) or search-based (Tavily search engine), communicating with the frontend via text. When Moshi predicts ⟨ret⟩, the system sends the current dialogue transcription to the backend, while the frontend continues generating a "lead segment" (knowledge-independent opening) to maintain speech flow. Once the backend returns a reference document, it is projected into an embedding via a single-layer reference text encoder and frame-wise added to the temporal Transformer input, allowing Moshi to "catch" the retrieved knowledge in the body segment.

### Key Designs

1. **⟨ret⟩ Trigger Token + Asynchronous Backend**:

    - **Function**: Allows Moshi to autonomously determine "whether external knowledge is needed" and call the backend without interrupting speech generation.
    - **Mechanism**: A special ⟨ret⟩ token is added to Moshi's RQ-Transformer output vocabulary. In training data, the slot before the first text token of the lead segment in RAG-enabled turns is replaced with ⟨ret⟩, located via TTS forced alignment. During inference, once ⟨ret⟩ is predicted, the system packages the ASR transcription of the user and the model's own transcription and sends it to the backend, which can be LLM-based or Tavily search. The frontend continues without waiting for the result; the entire call is asynchronous.
    - **Design Motivation**: Direct synchronous RAG calls would interrupt "listen while speaking." By letting the model explicitly signal "I want to retrieve," control over "when to retrieve" is handed to the model, achieving decoupling of "frontend real-time, backend slow thinking."

2. **Delay-Aware Data Synthesis Leveraging Keyword Delay**:

    - **Function**: Teaches the model during training that "the opening is sufficient to cover retrieval latency," preventing pauses or discontinuities during inference.
    - **Mechanism**: Three Gemma 3 27B role LLMs (user/Moshi/reference) synthesize 474k QA-type and 5.5k expert-domain dialogues, totaling ~1.9M. Each RAG-enabled turn is structured into (lead, body, tail): lead is a generic opening like "Let me check that for you…", body is the core answer generated after receiving the reference, tail is the closing. During training, retrieval delay is simulated as $d'\sim\mathcal{U}(1.0, d_{\text{lead}}-1.0)$ (80% probability) or $d'\sim\mathcal{U}(0, d_{\text{lead}})$ (20% fallback), ensuring at least 1s buffer before the body starts.
    - **Design Motivation**: Keyword delay is the physical basis of this approach—without abundant samples where the opening covers 2s retrieval in the dataset, the model would forcibly insert the reference during inference, causing speech misalignment. Explicit lead/body/tail annotation and $d'$ sampling force the model to learn this temporal constraint.

3. **Reference Embedding Streaming Injection**:

    - **Function**: "Welds" variable-length retrieved text into Moshi's 12.5 Hz backbone at minimal cost.
    - **Mechanism**: The reference is first compressed 4× in length by a pretrained ARC-Encoder, then passed through a trainable linear projection to obtain $h_i^{\text{ref}}=\text{proj}(\text{emb}_i^{\text{ref}})$. Starting $d/f_r$ steps after ⟨ret⟩, $h_i^{\text{ref}}$ is added frame-wise to the temporal Transformer input $h_i$: $h_i'=h_i+h_{i-(i_{\text{ret}}+d/f_r)}^{\text{ref}}$, continuing for $l$ steps. During training, the entire reference is dropped out with 0.2 probability, making the model robust to missing references.
    - **Design Motivation**: Directly prepending the reference would fill up the context and break the 12.5 Hz streaming property. Frame-wise addition after length compression ensures the reference text does not crowd out speech tokens and aligns with audio frame timing, providing the lightest interface for embedding textual knowledge into speech generation.

### Loss & Training
The base loss follows Moshi's original next-token prediction for text/speech tokens. The reference text encoder is frozen; the linear projection and dropout vector are trainable. Learning rate is $2\times 10^{-6}$, batch=32, 100k updates. Input is cleaned with a simple VAD using an 80ms window and $-65$ dBFS threshold.

## Key Experimental Results

### Main Results

| Model | LlamaQ | WebQ | TriviaQA | HaluEval | TTFAT(s) | KD(s) | E2EKD(s) |
|-------|--------|------|----------|----------|----------|-------|----------|
| GPT-4o Audio | 88.4 | 81.0 | 90.6 | 68.7 | — | 5.5 | — |
| GLM-4-Voice 9B | 64.7 | 32.2 | 39.1 | 21.2 | 0.3 | 4.2 | 4.4 |
| Freeze-Omni 7B | 72.0 | 44.7 | 53.9 | 14.0 | — | — | — |
| **MoshiRAG (resp.)** | Close to GPT-4o Audio | Significantly outperforms public speech LMs | Same as left | Same as left | Smaller | Backend ≤ 2s | Within conversational range |

Table 1 in the paper uses color shading to show that MoshiRAG's reference (retrieval quality) and resp (final answer) are both clearly higher than other public speech LMs across four QA benchmarks, and approach strong non-full-duplex baselines. Meanwhile, FLOPs/sec is much lower than the larger-parameter GLM-4-Voice.

### Ablation Study

| Configuration | Key Observation | Description |
|---------------|----------------|-------------|
| Search backend (Tavily) vs LLM backend (Gemma 3 27B) | Backend can be hot-swapped without retraining; LLM backend usually achieves higher accuracy, but search backend can access real-time web info | Validates "modular, replaceable" goal |
| Different reference encoders (including ARC-Encoder/Qwen, etc.) | ARC-Encoder achieves the best quality-latency tradeoff at 4× compression (Appendix B.1) | Length compression is key to fully utilizing the 12.5 Hz backbone |
| Without lead/body/tail structure | Retrieval results arrive misaligned, answers exhibit pauses/unnatural continuation | Demonstrates that structured data is core to latency masking |
| Mathematical reasoning (out-of-domain) | Solves simple math problems via "speech → LLM tool call" | Shows the framework generalizes to tool use, not just QA |

### Key Findings
- The E2EKD time window (typically >3s) is the physical basis for converting synchronous RAG to asynchronous RAG; as long as the backend is kept within ~2s, the process is transparent to the user.
- The "factuality of speech LMs" problem is solved not by scaling parameters, but by external knowledge sources. This approach allows the 7B Moshi to match or surpass several 9B+ peers without modifying the backbone.
- Early form of full-duplex + tool use: the model effectively treats the LLM/search engine as an external brain, hinting at future voice agent architectures.

## Highlights & Insights
- Redefines "keyword delay" from a criticized latency metric into a "usable time budget"—this perspective shift is the soul of the work, exploiting a gap everyone sees but no one uses.
- The ⟨ret⟩ "model-initiated tool call" design elegantly transfers the mature function calling paradigm from LLMs to speech models.
- Reference embedding is added frame-wise to the temporal Transformer input, rather than "prompt concatenation" as in text LLMs, fully respecting the 12.5 Hz streaming constraint—a hardware-aware design.
- Data synthesis uses three-role LLMs (user/Moshi/reference) with strict information access separation to prevent leakage, serving as a good paradigm for synthetic dialogue data.

## Limitations & Future Work
- Training relies entirely on synthetic dialogue and multi-channel TTS-generated speech, which still differs from real human conversations in disfluency, accent, and noise distribution.
- The ⟨ret⟩ trigger is a "hard decision," lacking explicit confidence/cost mechanisms; when the backend is unavailable or fails, only dropout-trained robustness serves as a fallback.
- Evaluation is still focused on single-turn QA, with limited assessment of true multi-turn and strategic knowledge citation in dialogue (e.g., clarification before retrieval, error correction and re-retrieval).
- The model only covers English Moshi, still far from the goal of a multilingual voice assistant.

## Related Work & Insights
- **vs StreamRAG / KAME**: StreamRAG is limited to non-full-duplex settings; KAME supports full-duplex but uses fixed-interval LLM calls, wasting compute. MoshiRAG is an event-driven, on-demand RAG, offering better efficiency and experience.
- **vs Moshi**: Directly inherits Moshi's RQ-Transformer + dual-channel; adding just one token and a reference projection layer yields significant factuality gains, making it the most economical extension of the original Moshi.
- **vs Chain-of-Thought for audio**: CoT improves reasoning; MoshiRAG improves external knowledge access. The two are fully orthogonal and can be combined in the future.

## Rating
- Novelty: ⭐⭐⭐⭐ First full-duplex + RAG system, with the key innovation being the new perspective of "leveraging keyword delay"; most technical components reuse existing modules.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers four QA benchmarks, two backend types, multiple reference encoders, and demonstrates tool-use reasoning; lacks real human multi-turn dialogue benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Explains delay-related terms (TTFAT/KD/E2EKD/Retrieval delay) very clearly, with intuitive timing diagrams; exemplary in articulating engineering motivation.
- Value: ⭐⭐⭐⭐⭐ Directly opens the door to tool-use for voice agents; Moshi + Tavily can be directly reused in industry, offering immense value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models](../../ACL2026/audio_speech/mtr-duplexbench_towards_a_comprehensive_evaluation_of_multi-round_conversations_.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](../../ACL2026/audio_speech/closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[ICLR 2026\] EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models](../../ICLR2026/audio_speech/echomind_an_interrelated_multi-level_benchmark_for_evaluating_empathetic_speech_.md)
- [\[ACL 2026\] Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models](../../ACL2026/audio_speech/do_we_need_distinct_representations_for_every_speech_token_unveiling_and_exploit.md)

</div>

<!-- RELATED:END -->
