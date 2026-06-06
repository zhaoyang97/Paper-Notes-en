---
title: >-
  [Paper Note] MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models
description: >-
  [ICML 2026][Audio & Speech][full-duplex] MoshiRAG incorporates a special ⟨ret⟩ trigger token into the Moshi full-duplex speech model…
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
content_hash: 96d43ca051c89e48
---

# MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models

**Conference**: ICML 2026  
**arXiv**: [2604.12928](https://arxiv.org/abs/2604.12928)  
**Code**: https://github.com/kyutai-labs/moshi-rag (Yes)  
**Area**: Dialogue Systems / Full-Duplex Speech / Retrieval-Augmentation  
**Keywords**: full-duplex, speech LM, RAG, Moshi, asynchronous retrieval, keyword delay

## TL;DR
MoshiRAG incorporates a special ⟨ret⟩ trigger token into the Moshi full-duplex speech model, allowing the model to asynchronously invoke an LLM or search backend to retrieve reference documents while speaking. By leveraging the natural "keyword delay" between the start of an utterance and the appearance of the keyword, it completely hides retrieval latencies under 2 seconds. This improves the factuality of the speech model to the level of GPT-4o Audio on benchmarks like LlamaQ, WebQ, TriviaQA, and HaluEval while maintaining real-time full-duplex capabilities.

## Background & Motivation
**Background**: Modern speech dialogue is evolving from cascaded ASR-Dialogue-TTS towards end-to-end speech LMs. Full-duplex models (successors to Moshi and dGSLM) can "listen and speak simultaneously," closely mimicking human interaction, whereas turn-based models (e.g., GLM-4-Voice, Freeze-Omni) are restricted to alternating turns.

**Limitations of Prior Work**: (1) Native speech LMs have significantly less training data than text LMs, leading to lower factuality compared to similarly sized text models. (2) While scaling models improves factuality, full-duplex systems require real-time inference, limiting parameter counts. (3) Existing RAG efforts are mostly turn-based because traditional RAG pipelines introduce synchronous wait times, which conflict with the "listen-while-speaking" nature.

**Key Challenge**: Factuality requires external knowledge → requires RAG → RAG introduces latency → breaks full-duplex interaction. One must currently sacrifice either factuality or real-time performance.

**Goal**: (1) Enable Moshi to autonomously decide "when external knowledge is needed"; (2) Trigger retrieval without interrupting the speech stream; (3) Ensure the backend is hot-swappable without retraining.

**Key Insight**: The authors observe a neglected temporal structure—there is a significant "keyword delay" (KD) between the start of speech (TTFAT) and the appearance of the key answer. For many models, this segment exceeds 3 seconds. If retrieval can be completed within this gap (target $\le 2$ seconds), the answer can be fetched while the model is still speaking a natural "lead-in" sentence.

**Core Idea**: Use a ⟨ret⟩ trigger token combined with an asynchronous backend and reference embedding injection after the lead-in phase to transform synchronous RAG into a full-duplex-compatible "retrieval-while-speaking" mechanism.

## Method

### Overall Architecture
MoshiRAG consists of three components: (1) A RAG-aware full-duplex front-end based on Moshi 7B, which inputs user speech tokens and its own previous tokens to output a special ⟨ret⟩ trigger token; (2) A 1B streaming ASR (0.5s latency) dedicated to transcribing user speech for retrieval; (3) An asynchronous retrieval backend, which can be LLM-based (e.g., Gemma 3 27B reading context for reference text) or search-based (e.g., Tavily search engine). When Moshi predicts ⟨ret⟩, the system sends the current dialogue transcript to the backend. The front-end continues generating a "lead" segment (knowledge-independent opening) to maintain the speech flow. Once the backend returns reference documents, they are projected into embeddings via a single-layer reference text encoder and superimposed frame-wise onto the temporal Transformer input, allowing Moshi to incorporate the knowledge in the "body" segment.

### Key Designs

1.  **⟨ret⟩ trigger token + asynchronous backend**:
    - **Function**: Allows Moshi to autonomously judge if a response requires external knowledge and invoke the backend without interrupting speech generation.
    - **Mechanism**: A special ⟨ret⟩ token is added to the Moshi RQ-Transformer output vocabulary. In training data, the position immediately preceding the first text token of the "lead" segment in RAG-enabled turns is replaced with ⟨ret⟩, leveraging TTS forced alignment for positioning. During inference, once ⟨ret⟩ is predicted, the user transcript from ASR and the model's own transcript are sent to the backend. The front-end continues running asynchronously.
    - **Design Motivation**: Synchronous RAG calls break the "listen-while-speaking" flow. Letting the model explicitly signal the intent to retrieve grants it control over "when to search" and decouples real-time front-end performance from the "slow-thinking" backend.

2.  **Latency-aware data synthesis leveraging keyword delay**:
    - **Function**: Trains the model to learn that lead-in phrases are sufficient to cover retrieval latency, preventing pauses or disjointedness during inference.
    - **Mechanism**: Three Gemma 3 27B LLM roles (User, Moshi, Reference) are used to synthesize approximately 1.9M dialogue turns (474k QA and 5.5k expert domain). Each RAG-enabled turn is structured into (lead, body, tail) segments: "lead" is a generic opening like "Let me check that for you..."; "body" is the core answer after receiving the reference; "tail" is the closing. During training, retrieval latency $d'$ is simulated using $d'\sim\mathcal{U}(1.0, d_{\text{lead}}-1.0)$ (80% probability) or $d'\sim\mathcal{U}(0, d_{\text{lead}})$ (20% fallback) to ensure at least a 1-second buffer before the body begins.
    - **Design Motivation**: Keyword delay is the physical basis of this solution. Without training samples where lead-ins cover a 2-second retrieval delay, the model would attempt to integrate references prematurely, causing audio misalignment. Explicit lead/body/tail labeling and $d'$ sampling force the model to learn this temporal constraint.

3.  **Streaming injection of reference embeddings**:
    - **Function**: Integrates variable-length retrieved text into the 12.5 Hz Moshi backbone with minimal overhead.
    - **Mechanism**: Reference text is first compressed by a factor of 4 using a pre-trained ARC-Encoder, then projected via a trainable linear layer to obtain $h_i^{\text{ref}}=\text{proj}(\text{emb}_i^{\text{ref}})$. Starting at $d/f_r$ steps after ⟨ret⟩, $h_i^{\text{ref}}$ is added frame-wise to the temporal Transformer input $h_i$: $h_i'=h_i+h_{i-(i_{\text{ret}}+d/f_r)}^{\text{ref}}$ for $l$ steps. During training, entire references are dropped with 0.2 probability to ensure robustness when no reference is returned.
    - **Design Motivation**: Prepending reference text would consume the context window and break the 12.5 Hz streaming property. Additive injection after length compression allows text knowledge to align temporally with audio frames without crowding speech tokens, providing a lightweight interface between text and speech generation.

### Loss & Training
The base loss follows Moshi's original text/speech token next-token prediction. The reference text encoder is frozen, while the linear projection and dropout vectors are trainable. Training uses a learning rate of $2\times 10^{-6}$, batch size of 32, and 100k updates. Input audio undergoes simple VAD silence removal with an 80ms window and $-65$ dBFS threshold.

## Key Experimental Results

### Main Results

| Model | LlamaQ | WebQ | TriviaQA | HaluEval | TTFAT(s) | KD(s) | E2EKD(s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4o Audio | 88.4 | 81.0 | 90.6 | 68.7 | — | 5.5 | — |
| GLM-4-Voice 9B | 64.7 | 32.2 | 39.1 | 21.2 | 0.3 | 4.2 | 4.4 |
| Freeze-Omni 7B | 72.0 | 44.7 | 53.9 | 14.0 | — | — | — |
| **MoshiRAG (resp.)** | Near GPT-4o | Leads open models | Same | Same | Small | Backend $\le 2$s | Managed |

Table 1 in the paper shows that MoshiRAG's ref (retrieval quality) and resp (final answer) across four QA benchmarks significantly outperform other open speech LMs and approach strong non-full-duplex baselines, while maintaining much lower FLOPs/sec than larger models like GLM-4-Voice.

### Ablation Study

| Configuration | Key Observation |
| :--- | :--- |
| Search Backend (Tavily) vs LLM Backend (Gemma 3 27B) | Backend is hot-swappable; LLM typically yields higher accuracy, but search provides real-time web info. |
| Different reference encoders (ARC-Encoder vs Qwen, etc.) | ARC-Encoder offers the best trade-off between quality and latency at $4\times$ compression. |
| No lead/body/tail structure | Misalignment occurs when retrieval results arrive; response includes unnatural pauses or transitions. |
| Mathematical reasoning (Out-of-domain) | Resolves simple math via "Speech → LLM tool call," showing framework generalizability. |

### Key Findings
- E2EKD (End-to-End Keyword Delay), an often ignored time window (typically $>3$ seconds), provides the physical justification for transforming synchronous RAG into asynchronous RAG. As long as the backend response is under ~2 seconds, the process is transparent to the user.
- Improving speech LM factuality via external knowledge rather than scaling parameters allows the 7B Moshi to match or exceed 9B+ tier models without modifying the core backbone.
- This represents an early form of full-duplex tool use, where the model treats the LLM/search engine as an external brain, hinting at future voice agent architectures.

## Highlights & Insights
- Redefining "keyword delay" from a criticized latency metric into a "usable time budget" is the core insight of this work—identifying a gap that everyone sees but no one utilizes.
- The ⟨ret⟩ "model-initiated tool call" design elegantly migrates the mature function-calling paradigm from the LLM world into the speech domain.
- Adding reference embeddings frame-wise to the temporal Transformer input, rather than "prompt concatenation," is a hardware-aware design that respects 12.5 Hz streaming constraints.
- Using a three-role LLM (user/Moshi/reference) setup for data synthesis with strictly partitioned information access is a robust paradigm for generating synthetic dialogue data.

## Limitations & Future Work
- Training relies heavily on synthetic dialogue and multi-channel TTS, which still differs from real human dialogue in terms of disfluency, accents, and noise distribution.
- The ⟨ret⟩ trigger is a "hard decision" without an explicit confidence or cost mechanism; it relies on dropout-trained robustness if the backend is unavailable.
- Evaluation is concentrated on single-turn QA; evaluation of strategic knowledge citation in complex multi-turn scenarios (e.g., clarifying before searching) remains limited.
- The model language is currently limited to English Moshi, leaving a gap for multilingual voice assistants.

## Related Work & Insights
- **vs StreamRAG / KAME**: StreamRAG is limited to non-full-duplex settings. KAME supports full-duplex but uses fixed-interval LLM calls, wasting compute. MoshiRAG is an "on-demand" event-driven RAG system, optimizing both efficiency and experience.
- **vs Moshi**: Directly inherits Moshi's RQ-Transformer + dual-channel architecture. Adding only one token and a reference projection layer yields massive factuality gains, representing a highly efficient extension of the original Moshi.
- **vs Chain-of-Thought for audio**: CoT focuses on improving reasoning, while MoshiRAG improves knowledge access; these are orthogonal and can be combined.

## Rating
- Novelty: ⭐⭐⭐⭐ First full-duplex RAG system; the "utilizing keyword delay" perspective is highly original; technical components leverage proven modules.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers four QA benchmarks, two backend types, multiple reference encoders, and tool-use reasoning; lacks real human multi-turn benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear explanation of latency-related terminology (TTFAT/KD/E2EKD/Retrieval delay); temporal diagrams are intuitive.
- Value: ⭐⭐⭐⭐⭐ Opens the door for full-duplex voice agent tool use; the Moshi + Tavily approach is directly applicable in industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning](the_silent_thought_modeling_internal_cognition_in_full-duplex_spoken_dialogue_mo.md)
- [\[ACL 2026\] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models](../../ACL2026/audio_speech/mtr-duplexbench_towards_a_comprehensive_evaluation_of_multi-round_conversations_.md)
- [\[ACL 2026\] Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner](../../ACL2026/audio_speech/full-duplex-bench-v2_a_multi-turn_evaluation_framework_for_duplex_dialogue_syste.md)
- [\[ACL 2026\] How Tokenization Limits Phonological Knowledge Representation in Language Models and How to Improve Them](../../ACL2026/audio_speech/how_tokenization_limits_phonological_knowledge_representation_in_language_models.md)
- [\[ICML 2026\] Towards Understanding Modality Interaction in Multimodal Language Models via Partial Information Decomposition](towards_understanding_modality_interaction_in_multimodal_language_models_via_par.md)

</div>

<!-- RELATED:END -->
