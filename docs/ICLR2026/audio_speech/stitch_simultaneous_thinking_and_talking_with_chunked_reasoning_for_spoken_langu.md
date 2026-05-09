---
title: >-
  [Paper Note] Stitch: Simultaneous Thinking and Talking with Chunked Reasoning for Spoken Language Models
description: >-
  [ICLR 2026][Audio & Speech][spoken language models] Stitch enables "thinking while speaking" in spoken language models (SLMs) by interleaving silent reasoning tokens with speech tokens in chunks, exploiting idle compute during audio playback for reasoning. Stitch-S achieves first-chunk latency identical to the no-reasoning baseline while improving math reasoning accuracy by approximately 15 percentage points.
tags:
  - ICLR 2026
  - "Audio & Speech"
  - spoken language models
  - chain-of-thought
  - simultaneous thinking and talking
  - chunked reasoning
  - latency optimization
date: 2026-05-08
content_hash: 1335d066285a9444
---

# Stitch: Simultaneous Thinking and Talking with Chunked Reasoning for Spoken Language Models

**Conference**: ICLR 2026
**arXiv**: [2507.15375](https://arxiv.org/abs/2507.15375)
**Code**: [https://d223302.github.io/STITCH](https://d223302.github.io/STITCH)
**Area**: Audio & Speech
**Keywords**: spoken language models, chain-of-thought, simultaneous thinking and talking, chunked reasoning, latency optimization

## TL;DR
Stitch enables "thinking while speaking" in spoken language models (SLMs) by interleaving silent reasoning tokens with speech tokens in chunks, exploiting idle compute during audio playback for reasoning. Stitch-S achieves first-chunk latency identical to the no-reasoning baseline while improving math reasoning accuracy by approximately 15 percentage points.

## Background & Motivation

**Background**: Current mainstream SLMs (e.g., GLM-4-Voice, Qwen-2.5-Omni) follow an interleaved decoding pipeline: the model first generates text tokens (as a transcript of the forthcoming speech) and then generates speech tokens (synthesized into waveforms by a vocoder), alternating between the two. This interleaved text–speech design enables streaming speech output, but the model performs no additional internal reasoning before speaking — it directly verbalizes whatever answer it generates.

**Limitations of Prior Work**: Humans typically reason silently before articulating a refined answer to a complex question, yielding two benefits: (1) higher accuracy and (2) more concise expression. Incorporating analogous "unspoken chain-of-thought (CoT)" into SLMs is a natural objective, yet it introduces latency challenges.

The most straightforward approach, Think Before Speaking (TBS), generates a complete textual reasoning chain $\mathbf{z}$ followed by the spoken response $\mathbf{y}$. Experiments confirm that TBS substantially improves math reasoning quality (avg. 79.1% vs. 63.0% without reasoning), but reasoning length is unbounded (up to 360 reasoning tokens on GSM8K), resulting in uncontrollable first-chunk latency.

**Key Challenge**: The key observation motivating Stitch is that synthesizing $N_{speech}=26$ speech tokens produces approximately 2 seconds of audio, whereas generating 39 text+speech tokens at 80 tokens per second on an A100 takes only ~0.49 seconds, leaving ~1.5 seconds of idle playback time. Stitch repurposes this idle time to generate reasoning tokens for the next chunk, realizing "thinking while speaking." The theoretical upper bound is $80 \times 2 - 39 = 121$ reasoning tokens per chunk; the paper sets $N_{reason}=100$ in practice.

## Method

### Overall Architecture
Stitch introduces a third token type into the SLM output sequence — reasoning tokens (silent reasoning) — interleaved with the existing text and speech tokens. Reasoning tokens are delimited by special markers [SOPR]/[EOPR] and are never synthesized into speech. The generation pipeline proceeds as follows: user speech input → backbone autoregressively outputs reasoning/text/speech chunks → speech decoder synthesizes audio → next reasoning chunk is generated concurrently during playback.

### Key Designs

1. **Stitch-R (Reasoning-first)**: Generation order is [reasoning chunk → text chunk → speech chunk → reasoning chunk → text chunk → speech chunk → …]. First-chunk latency equals the generation time of $N_{reason}+N_{text}+N_{speech}$ tokens. Although substantially shorter than TBS's full-reasoning latency, the first reasoning chunk still incurs a wait.
2. **Stitch-S (Speaking-first)**: Generation order is [text chunk → speech chunk → reasoning chunk → text chunk → speech chunk → reasoning chunk → …]. The model speaks its first utterance without any reasoning, and begins thinking during first-chunk audio playback. First-chunk latency is only $N_{text}+N_{speech}$, identical to the original no-reasoning baseline.
3. **Training Data Construction**: Starting from TBS training data $(\mathbf{x},\mathbf{z},\mathbf{y})$, the full reasoning chain $\mathbf{z}$ is segmented into chunks of $N_{reason}=100$ tokens and interleaved with text–speech pairs. Samples where the number of reasoning chunks exceeds the number of text chunks (i.e., "thinking slower than speaking") are discarded. Data sources include general dialogue (VoiceAssistant400K), math reasoning (Tulu-3), and knowledge QA (NQ/TriviaQA); reasoning CoTs are generated by GPT-4o.
4. **Latency Guarantee (Mathematical Derivation)**: At 80 tps on an A100, with $N_{text}+N_{speech}=39$ and audio chunk duration $t_{chunk}\approx2$ s, the model can generate $80 \times 2 - 39 = 121$ reasoning tokens within the playback window. Hence $N_{reason}=100$ fits comfortably within the time budget. On slower hardware, $N_{reason}$ should be reduced accordingly.

### Loss & Training
Standard cross-entropy language modeling loss is applied. The GLM-4-Voice-9B backbone is fully fine-tuned with the speech encoder and decoder frozen. The training set contains approximately 400K samples covering general dialogue (VoiceAssistant400K), math reasoning (Tulu-3 series), and knowledge QA (NQ, TriviaQA). Reasoning CoTs are generated by GPT-4o and speech is synthesized by GPT-4o-mini-TTS. Inference is conducted on A100-80G using vLLM. Samples where the number of reasoning chunks exceeds the number of text chunks are discarded during training to ensure that reasoning does not delay speech output.

## Key Experimental Results

### Main Results (Average over 5 Math Reasoning Datasets)

| Method | AddSub | MultiArith | SinglEq | SVAMP | GSM8K | Avg. | Latency Type |
|--------|--------|-----------|---------|-------|-------|------|-------------|
| GLM-4-Voice | 59.4 | 62.0 | 71.0 | 44.0 | 29.0 | 53.1 | $N_t+N_s$ |
| No reasoning | 66.1 | 70.7 | 78.0 | 64.4 | 35.7 | 63.0 | $N_t+N_s$ |
| TBS | 79.8 | 85.6 | 89.9 | 75.3 | 64.9 | 79.1 | $N_{full}+N_t+N_s$ |
| Stitch-R | 78.9 | 88.5 | 93.6 | 73.8 | 58.7 | 78.7 | $N_r+N_t+N_s$ |
| **Stitch-S** | **81.7** | **87.9** | **91.7** | 72.2 | 56.7 | **78.0** | $N_t+N_s$ |

Stitch-S achieves first-chunk latency identical to the no-reasoning baseline (both $N_{text}+N_{speech}$) while surpassing it by 15 percentage points in average math reasoning accuracy.

### Non-Reasoning Tasks

Stitch-S matches the baseline on non-reasoning tasks, demonstrating that reasoning capability is not gained at the expense of conversational quality.

| Method | Llama Q | TriviaQA | WebQ | AlpacaEval | Avg. |
|--------|---------|---------|------|-----------|------|
| GLM-4-Voice | 74.3 | 47.1 | 51.0 | 48.6 | 55.2 |
| TBS | 74.3 | 51.5 | 52.2 | 56.3 | 58.6 |
| Stitch-S | 72.0 | 49.3 | 49.0 | 56.1 | 56.6 |

### Ablation Study

| Configuration | Math Avg. | Notes |
|--------------|---------|-------|
| Full model (Stitch-S) | 78.0 | Zero-latency reasoning |
| Mix reasoning (no reasoning at inference) | 67.4 | Trained with reasoning but not used at inference; still +4.4% |
| Mix reasoning (with reasoning at inference) | 77.5 | Comparable to TBS |
| Stitch-S | 78.0 | Same latency, better performance |

### Key Findings
- Stitch-S achieves reasoning performance close to TBS (**78.0 vs. 79.1**) with **zero additional latency**, outperforming the no-reasoning baseline by 15 percentage points.
- On non-reasoning tasks, Stitch-S is on par with the baseline (56.6 vs. 55.2), confirming that reasoning ability is not acquired at the cost of conversational quality.
- Exposure to reasoning data during training improves performance even when reasoning is not used at inference (Mix reasoning w/o reasoning: 67.4 vs. No reasoning: 63.0).
- TBS reasoning tokens can reach 360 on GSM8K (~4.5 seconds of overhead), whereas Stitch caps reasoning at 100 tokens per chunk with no additional wait.
- The performance gap between Stitch-R and Stitch-S is marginal (78.7 vs. 78.0), yet Stitch-S offers substantially lower first-chunk latency.

## Highlights & Insights
- This work is the first to introduce the concept of "silent reasoning" into SLMs, drawing an analogy to the human cognitive process of "thinking before speaking."
- Stitch-S represents an elegant system design: it exploits the audio playback window as a "free" computation slot to achieve zero-latency reasoning — a window that exists in any SLM yet has never previously been utilized.
- The mathematical derivation of first-chunk latency is rigorous, yielding a precise upper bound of 121 for $N_{reason}$ on an A100; the practical choice of 100 preserves a safety margin.
- The design progression TBS → Stitch-R → Stitch-S reflects an engineering philosophy that advances from "feasible" to "efficient" to "zero-overhead."
- The method for converting TBS training data into Stitch training data is remarkably simple: segment the reasoning chain into chunks and interleave them with text–speech pairs.

## Limitations & Future Work
- Experiments are conducted solely on GLM-4-Voice (9B); validation on larger models or thinker–talker architectures remains absent.
- $N_{reason}=100$ is a fixed value; adaptive reasoning budget allocation may be more effective — simple questions require little reasoning, while complex ones may benefit from more.
- Reasoning quality is bounded by the quality of GPT-4o-generated CoTs in the training data; the diversity and accuracy of CoT data directly constrain performance ceilings.
- Significant improvements are demonstrated only on math reasoning; code generation, logical reasoning, scientific QA, and other tasks remain unexplored.
- On weaker GPUs where reasoning token generation is slower than audio playback, the latency advantage of Stitch would diminish.
- Current evaluation measures accuracy via text tokens only; systematic assessment of speech quality (e.g., MOS, naturalness) is absent.

## Related Work & Insights
- **vs. Qwen-2.5-Omni's thinker–talker**: The thinker–talker paradigm uses two specialized models (a thinker for text generation and a talker for speech synthesis), whereas Stitch interleaves reasoning within a single model. The two paradigms are complementary.
- **vs. text-domain CoT (e.g., o1)**: In text-domain CoT, latency equals the total number of reasoning plus response tokens. In SLMs, however, audio playback duration >> token generation time, creating a unique "free computation window" that Stitch exploits.
- **Broader Insight**: Any "fast-generate, slow-consume" scenario could similarly exploit idle consumption-side time for additional computation (e.g., video rendering, streaming 3D model transmission).

## Rating
- Novelty: ⭐⭐⭐⭐ First to introduce silent reasoning into SLMs; the zero-latency design of Stitch-S is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five math + four non-reasoning datasets with detailed latency analysis; however, validation is limited to a single SLM.
- Writing Quality: ⭐⭐⭐⭐⭐ Timeline diagrams and comparison figures are clear; the methodology is presented in a well-structured progression; the TBS → Stitch-R → Stitch-S design evolution is elegant.
- Value: ⭐⭐⭐⭐ Practically oriented; directly integrable into existing SLM inference pipelines.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [\[ICLR 2026\] EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning](emotionthinker_prosody-aware_reinforcement_learning_for_explainable_speech_emoti.md)
- [\[NeurIPS 2025\] AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound](../../NeurIPS2025/audio_speech/audsemthinker_enhancing_audio-language_models_through_reasoning_over_semantics_o.md)
- [\[ICLR 2026\] EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models](echomind_an_interrelated_multi-level_benchmark_for_evaluating_empathetic_speech_.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](../../AAAI2026/audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)

<!-- RELATED:END -->
