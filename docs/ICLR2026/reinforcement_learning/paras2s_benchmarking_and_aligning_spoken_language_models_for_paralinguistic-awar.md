---
title: >-
  [Paper Note] ParaS2S: Benchmarking and Aligning Spoken Language Models for Paralinguistic-Aware Speech-to-Speech Interaction
description: >-
  [ICLR 2026][Reinforcement Learning][speech-to-speech] This paper proposes the ParaS2S framework, which comprises ParaS2SBench — a benchmark for evaluating paralinguistic-aware (emotion/sarcasm/age/gender) speech-to-speech interaction — and ParaS2SAlign, a GRPO-based RL alignment framework that enables S2S models to learn style-adaptive response generation with minimal labeled data.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - speech-to-speech
  - paralinguistic awareness
  - benchmark
  - GRPO
  - reward model
date: 2026-05-08
content_hash: 1b665520f7bfe437
---

# ParaS2S: Benchmarking and Aligning Spoken Language Models for Paralinguistic-Aware Speech-to-Speech Interaction

**Conference**: ICLR 2026
**arXiv**: [2511.08723](https://arxiv.org/abs/2511.08723)
**Code**: [Project Page](https://paras2sbench.github.io/)
**Area**: Spoken Dialogue / Reinforcement Learning
**Keywords**: speech-to-speech, paralinguistic awareness, benchmark, GRPO, reward model

## TL;DR

This paper proposes the ParaS2S framework, which comprises ParaS2SBench — a benchmark for evaluating paralinguistic-aware (emotion/sarcasm/age/gender) speech-to-speech interaction — and ParaS2SAlign, a GRPO-based RL alignment framework that enables S2S models to learn style-adaptive response generation with minimal labeled data.

## Background & Motivation

Speech conveys not only linguistic content but also rich **paralinguistic cues** — emotion, intonation, speaker attributes, etc. — that collectively shape true intent and guide appropriate responses. Existing S2S models suffer from several fundamental issues:

**The "Tone-deaf" Problem**: Current S2S models (including Qwen2.5-Omni, GPT-4o Voice mode, and GLM-4-Voice) produce nearly identical responses regardless of speaking style. Experiments show these models score around 3 out of 5, comparable to a pipeline baseline that completely ignores speaking style.

**Lack of Evaluation Benchmarks**: Existing benchmarks mostly focus on speech-to-text comprehension (e.g., VoiceBench) or evaluate text response quality, with no benchmark directly assessing the appropriateness of S2S model output speech in terms of both content and style.

**Data Scarcity Bottleneck**: Constructing paralinguistic-aware S2S training data requires style annotations and expressive recordings, making it prohibitively expensive — the primary barrier to developing such models.

Core Research Question: Inspired by DeepSeek-R1 — where reasoning ability can emerge through RL without SFT demonstrations — **can paralinguistic-aware dialogue capability similarly emerge through RL with minimal supervision**?

## Method

### Overall Architecture

ParaS2S consists of two major components:

**ParaS2SBench** (Benchmark):
- Automatically generates high-quality spoken test queries
- Covers 4 paralinguistic dimensions: emotion, sarcasm, age, and gender
- Each query is paired with two contrastive speaking styles
- Directly evaluates the appropriateness of input-output speech pairs

**ParaS2SAlign** (Alignment Framework):
- Stage 1: SFT warm-up
- Stage 2: Distilled reward model
- Stage 3: GRPO post-training

### Key Designs

1. **Scenario-Controlled Query Generation**: Each test query adheres to three design principles:

    - **Neutral textual content**: e.g., "I just bumped into my ex." carries no inherent emotional implication
    - **Contrastive speaking styles**: the same text is paired with two contrasting styles (e.g., surprised vs. sad)
    - **Paralinguistic relevance**: the speaking style genuinely influences what constitutes an appropriate response

   → Mechanism: Only when a model must infer the speaker's state from audio signals rather than text can paralinguistic awareness be truly tested.
   → Design Motivation: Prevents models from "guessing" the correct response based solely on textual content.

   Data quality is ensured through triple filtering: neutrality test, plausibility test, and paralinguistic relevance test — all automated via ChatGPT and confirmed by human reviewers.

2. **Multi-Model Evaluation Pipeline**: When evaluating output speech, content (transcribed via Whisper-v3) and speaking style (analyzed via AudioReasoner) are extracted separately, then jointly scored by ChatGPT 4.1:

    $f_{\text{gpt}} = GPT(c_i, s_i, c_o, s_o, r)$

   → Mechanism: Speech evaluation is decomposed into two sub-problems — content understanding and style understanding — before a holistic judgment is made.
   → Design Motivation: Direct speech quality evaluation is difficult; using text as an intermediary enables automated, scalable assessment.

3. **Three-Stage Alignment Framework**:

   **Stage 1 — SFT Warm-up**: High-quality demonstrations are constructed from 10k speech prompts, with expressive responses synthesized via gpt-4o-mini-tts. SFT training equips the model with basic paralinguistic awareness. This step is necessary — the base model entirely lacks this capability and cannot sample meaningful responses for RL.

   **Stage 2 — Distilled Reward Model**: The SFT model generates 32 responses per query for 10k queries (320k pairs total), which are scored by the evaluation pipeline. A reward model $\phi$ is then trained to approximate pipeline scores, resolving the pipeline's slow evaluation speed.

   **Stage 3 — GRPO Post-Training**: Using 100k unlabeled speech prompts, the model learns from its own generated responses via GRPO. For each query, $G=8$ responses are sampled in-group, scored by the reward model $\phi$, normalized to compute advantage estimates, and used to update the policy.

   → Mechanism: SFT establishes foundational capability → pipeline evaluation is distilled into a fast reward model → RL explores the unlabeled space.
   → Design Motivation: Minimizes the need for human annotation while maximizing learning efficiency.

4. **KL Divergence Constraint**: The GRPO loss includes a KL penalty term ($\beta = 0.2$) to prevent the model from forgetting its original conversational intelligence while acquiring paralinguistic capabilities. This term is applied jointly to both the audio and text token streams.

   → Mechanism: Balances newly acquired capabilities against pre-existing ones.
   → Design Motivation: Ablation experiments show that $\beta = 0$ leads to severe degradation on VoiceBench (catastrophic forgetting).

### Loss & Training

- **SFT**: Standard next-token prediction, optimizing jointly over audio stream $a_o$ and text stream $t_o$
- **Reward Model**: Cross-entropy loss, treating Likert scores as a single-character prediction task
- **GRPO**: Group relative policy optimization with KL penalty, using rewards from the distilled reward model

Training configuration:
- Base model: Kimi-Audio
- SFT/GRPO: 8× NVIDIA H100, FSDP
- SFT learning rate: 1e-5, global batch size: 64
- RL learning rate: 5e-4, query batch size: 32, group size: 8
- Reward model: single H100, LoRA fine-tuning, learning rate: 1e-6

## Key Experimental Results

### Main Results

**Comprehensive S2S Model Comparison** (ParaS2SBench score, 5-point scale)

| Model | Synthetic Avg | Real Avg | Overall Avg |
|-------|--------------|---------|------------|
| Whisper-GPT-TTS (Pipeline Baseline) | 3.022 | 3.487 | 3.176 |
| GPT-4o Voice mode | 3.284 | 3.639 | 3.403 |
| Qwen2.5 Omni | 3.248 | 3.612 | 3.369 |
| GLM-4-Voice | 3.033 | 3.037 | 3.034 |
| Kimi-Audio (Base) | 2.892 | 1.265 | 2.350 |
| **Kimi-Audio SFT** | **4.076** | **3.714** | **3.955** |
| **Kimi-Audio GRPO** | **4.441** | **4.161** | **4.382** |
| GPT-TTS (Topline) | 4.705 | 4.766 | 4.725 |

Key observations:
- All existing S2S models perform comparably to the pipeline baseline (~3.0–3.4), confirming their lack of paralinguistic awareness
- SFT yields a 68% relative improvement; GRPO further improves upon SFT by 11%
- The GRPO model approaches the performance of the Topline (ideal TTS system)

### Ablation Study

| Configuration | Key Metric | Notes |
|--------------|-----------|-------|
| RL w/ 10h SFT warm-up | Matches 50h SFT | RL is highly label-efficient |
| RL w/ 20h SFT warm-up | Matches 100h SFT | 1/5 of annotations suffices |
| KL $\beta = 0$ | VoiceBench drops significantly | No KL constraint leads to catastrophic forgetting |
| KL $\beta = 0.2$ | Best on both metrics | Optimal trade-off |
| Group size $G < 8$ | Significant performance drop | $G=2$ often yields identical scores for both samples |
| Group size $G \geq 8$ | No additional gain | $G=8$ provides sufficient learning signal |
| More resources to SFT vs. RM | SFT is more beneficial | 10h annotations suffice for a usable reward model |

### Key Findings

1. **High alignment with human ratings**: Pearson correlation consistently exceeds 0.7; model rankings nearly match human evaluations (only one swap observed).
2. **RL is far more label-efficient than SFT**: 10h SFT warm-up + RL matches 50h pure SFT; SFT requires more than 10× the data to reach parity with RL.
3. **Generalization from synthetic to real speech**: SFT and GRPO trained on synthetic data transfer effectively to real speech test sets (IEMOCAP, MELD).
4. **Cross-domain generalization**: Models RL-trained on IEMOCAP also improve on MELD, and vice versa.
5. **Human subjective evaluation**: The GRPO model outperforms SFT by 7.6% in subjective evaluation, even though general users are more tolerant of paralinguistic errors.

## Highlights & Insights

1. **Discovery and quantification of the "tone-deaf" problem**: This is the first systematic study to reveal severe paralinguistic deficiencies in current SOTA S2S models — all evaluated models perform comparably to a style-ignoring pipeline baseline.
2. **End-to-end speech-level evaluation**: ParaS2SBench is the first benchmark to directly assess both content and style appropriateness of S2S output speech, rather than evaluating only text responses.
3. **RL unlocks paralinguistic capability**: The paper demonstrates that a DeepSeek-R1-style RL paradigm is effective in the speech domain — with minimal SFT warm-up, models can acquire paralinguistic awareness through self-exploration.
4. **Exceptional data efficiency**: Just 10 hours of annotated data combined with RL surpasses all existing models, challenging the assumption that large amounts of high-quality annotations are required.
5. **Practical automated evaluation pipeline**: Complex speech quality assessment is decomposed into a composable pipeline of ASR + style recognition + LLM scoring, which is both efficient and consistent with human judgment.

## Limitations & Future Work

1. **Evaluation pipeline correlation does not reach 0.9**: Although exceeding 0.7 and statistically significant, there remains room for improvement, particularly in scenarios requiring fine-grained style distinctions.
2. **Reliance on multiple external models**: The evaluation pipeline depends on Whisper-v3, AudioReasoner, and ChatGPT; biases inherent in these models propagate into evaluation results.
3. **Fidelity of TTS synthetic data**: The instability of gpt-4o-mini-tts necessitates generating 10 candidates with manual selection; the naturalness of synthesized speech still lags behind real recordings.
4. **Validation on a single base model**: Although the framework claims general applicability to any LM-based S2S model, experiments are conducted on only one base model.
5. **Computational cost**: Stage 2 requires generating 320k speech responses and scoring them through the evaluation pipeline, incurring non-trivial computational and API costs.
6. **Directions for extension**: Supporting more paralinguistic dimensions (e.g., accent, speech rate, pausing), style consistency across multi-turn dialogue, and multi-speaker scenarios.

## Related Work & Insights

- **DeepSeek-R1**: The seminal work showing that RL can elicit reasoning without SFT; ParaS2S transplants this idea into the speech domain.
- **StyleTalk / ParalinGPT**: Early works on paralinguistic-aware dialogue, but limited to speech-to-text and entirely reliant on SFT.
- **GOAT-SLM**: The only S2S model emphasizing paralinguistics, using a multi-stage SFT pipeline. ParaS2S replaces the heavy annotation requirement with RL.
- **Align-SLM**: Uses DPO to align speech models, but focuses on long-range semantics rather than paralinguistics.
- Insight: The power of RL in modality extension is underestimated — even "soft skills" such as emotional awareness can be efficiently acquired through RL.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First application of an RL framework to paralinguistic-aware S2S; both problem formulation and solution are original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Exceptionally comprehensive: benchmark validation, model comparison, ablation, data efficiency, generalization, and human evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with detailed experiments, though some sections are redundant.
- Value: ⭐⭐⭐⭐⭐ — Fills an important gap in the S2S field; both the benchmark and the method offer long-term value.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[NeurIPS 2025\] Checklists Are Better Than Reward Models For Aligning Language Models](../../NeurIPS2025/reinforcement_learning/checklists_are_better_than_reward_models_for_aligning_langua.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[ICLR 2026\] Towards Strategic Persuasion with Language Models](towards_strategic_persuasion_with_language_models.md)
- [\[ICLR 2026\] Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models](co-rewarding_stable_self-supervised_rl_for_eliciting_reasoning_in_large_language.md)

<!-- RELATED:END -->
