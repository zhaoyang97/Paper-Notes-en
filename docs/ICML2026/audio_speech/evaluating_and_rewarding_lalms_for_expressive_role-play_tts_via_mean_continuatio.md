---
title: >-
  [Paper Note] Evaluating and Rewarding LALMs for Expressive Role-Play TTS via Mean Continuation Log-Probability
description: >-
  [ICML 2026][Audio & Speech][Role-Play TTS] This paper formulates the "continuation probability of ground-truth speech tokens by a pre-trained large audio language model" as an objective style consistency metric named MCL…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Role-Play TTS"
  - "LALM"
  - "Mean Continuation Log-Probability"
  - "GRPO"
  - "Style Consistency"
date: 2026-05-08
content_hash: 06cfdccc1d2ed14d
---

# Evaluating and Rewarding LALMs for Expressive Role-Play TTS via Mean Continuation Log-Probability

**Conference**: ICML 2026  
**arXiv**: [2601.22661](https://arxiv.org/abs/2601.22661)  
**Code**: https://github.com/y-ren16/MCLP  
**Area**: Audio/Speech  
**Keywords**: Role-Play TTS, LALM, Mean Continuation Log-Probability, GRPO, Style Consistency  

## TL;DR
This paper formulates the "continuation probability of ground-truth speech tokens by a pre-trained large audio language model" as an objective style consistency metric named MCLP. By employing a gated hybrid reward combining MCLP and CER through GRPO on the newly constructed WenetSpeech-RP-TTS dataset, the subjective MOS of role-play TTS is improved from 1.86 to 3.58.

## Background & Motivation

**Background**: LLM-style TTS (e.g., CosyVoice, VALL-E, Step-Audio) has achieved strong zero-shot voice cloning. Recent Instruct-TTS models allow style control via natural language descriptions, while Speech Role-Playing Agents (e.g., OmniCharacter, SpeechRole, VoxRole) further aim to portray specific characters across multi-turn dialogues.

**Limitations of Prior Work**: In "Role-Playing TTS (RP-TTS)" scenarios—where **style is controlled but timbre is not**—existing methods struggle. Instruct-TTS handles only single utterances and treats style as a static attribute, failing to maintain personas across multiple turns. Role-Playing Agents emphasize semantic alignment over acoustic style, often sacrificing expressiveness for coherence. Attempts to align style via RL are hindered by the **lack of objective style metrics**, often defaulting to emotion classifiers as proxy rewards, which only cover the emotional dimension.

**Key Challenge**: Style is a continuous, context-dependent, "high-dimensional concept" mixing prosody, emotion, and paralinguistic information. Existing evaluations and rewards attempt to approximate it using discrete labels (emotion categories, speaker IDs), which inevitably loses information. Furthermore, single rewards (only CER or only similarity) easily trigger reward hacking, leading to either highly expressive but unintelligible "gibberish" or extremely clear but monotone "plain speech."

**Goal**: To address both issues: (1) define an **interpretable, continuous, and human-aligned** style metric; (2) integrate it into an RL pipeline while maintaining content fidelity.

**Key Insight**: The authors hypothesize that LALMs pre-trained on massive speech data **implicitly learn a continuous latent space of speech styles**. Given a transcript and "candidate speech," if the candidate's style matches the ground truth, the pre-trained LALM should have a higher probability of continuing the ground-truth speech tokens when using the candidate speech as context. This translates "style consistency" into "continuation likelihood," which can be quantified numerically.

**Core Idea**: Directly use the "mean log-likelihood of ground-truth audio tokens by a pre-trained LALM" as the style metric MCLP. This functions both for offline evaluation and as a reward in GRPO, with CER used as a gate to prevent reward hacking.

## Method

### Overall Architecture

The pipeline consists of three stages: (a) **MCLP Continuation Model Pre-training**: Initialized with Step-Audio-2, performing "conversation-level" autoregressive training on 3 million hours of transcribed speech, with the loss calculated only on audio tokens to learn continuation capabilities. (b) **SFT Stage**: Based on Step-Audio-2-mini-Base, supervised fine-tuning is performed on the self-constructed WenetSpeech-RP-TTS to generate interleaved TA4 token sequences for turn $j$ based on scene description $\mathcal{S}$, character profile $\mathcal{P}$, and history $\mathcal{H}_{<j}$. (c) **GRPO Reinforcement Learning Stage**: Rollouts are performed only for the final turn, using a composite reward driven by MCLP style reward, CER content penalty, and gated aggregation.

The data loop is built on WenetSpeech: 17k videos were filtered by "YouTube + drama" tags, resulting in 8556 downloads. Demucs was used for accompaniment removal and pyannote for speaker diarization. DeepSeek-R1 inferred drama/episode titles and generated character profiles $\mathcal{P}$, while Qwen-VL-7B generated scene descriptions $\mathcal{S}$. Scenes were split by 5-second silences (30s maximum duration), resulting in 311k scenes (1435 hours), averaging 7.3 sentences and 2.33 speakers. The test set used strict video-level hold-out (200 videos, 900 scenes), stratified across 2–10 turns.

### Key Designs

1.  **MCLP Metric Design with "Dual Transcript + Reverse Continuation" Context**:
    *   **Function**: Quantifies style consistency between candidate audio $\mathbf{z}^{eval}$ and ground-truth audio $\mathbf{z}^{gt}$ using LALM likelihood.
    *   **Mechanism**: Constructs context as $\mathcal{H}=[\mathbf{w},\mathbf{z}^{eval},\mathbf{w}]$, where the same transcript $\mathbf{w}$ appears twice, sandwiching the candidate audio. The LALM then continues $\mathbf{z}^{gt}$ following $\mathcal{H}$. MCLP is defined as $\text{MCLP}=\frac{1}{|\mathbf{z}^{gt}_A|}\sum_{k\in \mathbf{z}^{gt}_A}\log P_\theta(z_k^{gt}\mid \mathcal{H},z_{<k}^{gt})$, averaged only over audio tokens.
    *   **Design Motivation**: Repeating $\mathbf{w}$ "pins" the textual content under teacher-forcing, ensuring likelihood variations **stem solely from style signals**. Step-Audio-2 was chosen because its semantic speech tokenizer preserves style over timbre details, biasing the metric toward "style similarity." Normalizing by the ground-truth length ensures fair comparison across multiple candidates. Compared to emotion classifiers, MCLP provides a **continuous, dense, and interpretable** scalar covering prosody, rhythm, and emotion.

2.  **Gated MCLP + CER Composite Reward for Reward Hacking Prevention**:
    *   **Function**: Simultaneously optimizes expressiveness (MCLP) and clarity (CER) during GRPO, ensuring the model learns to "speak clearly" before pursuing style.
    *   **Mechanism**: Style branch $R_{style}=\text{MCLP}(\mathbf{z}^{roll},\mathbf{z}^{gt})+C$ with offset $C=15$ to shift MCLP to a positive range. Content branch $R_{content}=\lambda\cdot\text{CER}(\hat{\mathbf{w}},\mathbf{w})$ with $\lambda=10$, using Step-Audio Token2Wav for rollout decoding and ASR for $\hat{\mathbf{w}}$. The final reward is $R(\mathbf{z})=R_{style}-R_{content}$ if $\text{CER}\le\tau=0.2$, otherwise 0.
    *   **Design Motivation**: Ablations showed that MCLP-only rewards caused CER to explode to 60%+ (generating repeated acoustic patterns), while CER-only rewards resulted in a low MOS of 2.33 (monotone). The gate acts as a **curriculum curve**, requiring a clarity threshold to be met before style points are granted, preventing "expressive gibberish" while allowing style optimization once clarity is sufficient.

3.  **GRPO-Based Last-Turn RL Alignment**:
    *   **Function**: Optimizes the policy only for the last turn of each dialogue post-SFT, using relative advantage normalization for stability.
    *   **Mechanism**: For each query $\mathbf{q}=(\mathcal{S},\mathcal{P},\mathcal{H})$, $G=8$ rollouts are sampled. Advantage is computed as $\hat{A}_i=(R_i-\text{mean})/\text{std}$ within the group. The objective is the clipped importance ratio $\rho_{i,t} \cdot \hat{A}_i$ plus a token-level KL constraint $\beta\mathbb{D}_{KL}$ ($\beta=0.001$). Training used 16,186 high-quality scenes (2–6 turns, last sentence >10 Chinese characters, non-Neutral style) for 1000 steps on 32 H800 GPUs.
    *   **Design Motivation**: RL is applied only to the last turn with history kept as ground truth to prevent context contamination by error accumulation. GRPO is more memory-efficient than PPO as it lacks a critic network. The filtering rules ensure samples have high potential for expressiveness improvement.

### Loss & Training

SFT: 1 epoch, batch size 64, learning rate $1\times 10^{-5}$ with cosine decay, max sequence length 16,384, AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay 0.1, grad clip 1.0). Objective: $\theta^*=\arg\min_\theta\sum -\log P_\theta(\mathbf{y}\mid \mathcal{S},\mathcal{P},\mathcal{H},\mathcal{I})$. RL: learning rate $1\times 10^{-6}$, global batch size 128, $G=8$ rollouts, temperature 1.0, max decode 1024.

## Key Experimental Results

### Main Results

| Model | Setting | CER↓ | CAM++↑ | Emo2Vec↑ | MCLP↑ | MOS↑ |
|-------|---------|------|--------|----------|-------|------|
| Ground Truth | — | — | — | — | — | 4.461 |
| GPT-Audio | w/ history | 11.97 | 0.636 | 0.875 | -4.849 | 1.915 |
| MiMo-Audio-7B | w/ history | 10.60 | 0.699 | 0.902 | -4.753 | 2.484 |
| Step-Audio-2-mini | w/ history | 3.28 | 0.629 | 0.864 | -4.829 | 1.856 |
| OV-InstructTTS | w/o history | 7.19 | 0.669 | 0.900 | -4.768 | 2.864 |
| **Ours** | w/ history | **1.13** | **0.724** | **0.917** | **-4.636** | **3.576** |
| **Ours** | w/o history | **1.63** | **0.704** | **0.910** | **-4.687** | **3.576** |

Ours reduced CER to nearly 1/10 of baselines (GPT-Audio w/o history 44.7% → Ours w/o history 1.63%). MOS is 0.71 points higher than the strongest Instruct-TTS (OV-InstructTTS 2.864) and 1.09 points higher than the strongest LALM (MiMo 2.484).

### Ablation Study

| Configuration | CER (w/ hist) | MCLP (w/ hist) | MOS |
|---------------|---------------|----------------|-----|
| Step-Audio-2-mini (baseline) | 3.28 | -4.829 | 1.856 |
| + SFT only | 3.33 | -4.725 | 3.178 |
| Full (SFT + RL with hybrid reward) | **1.13** | -4.636 | **3.576** |
| w/o CER Reward (MCLP-only) | 61.14 | **-4.590** | 1.145 |
| w/o MCLP Reward (CER-only) | **0.78** | -4.752 | 2.331 |

### Key Findings
- SFT alone improved MOS from 1.86 to 3.18 (+1.32), validating the dataset. RL provided an additional +0.40, proving MCLP reward delivers style gains beyond SFT.
- Removing the CER reward led to the **highest** MCLP (-4.59) but a catastrophic CER (61%) and MOS (1.15), confirming that reward hacking produces expressive gibberish and that the gated mechanism is essential.
- Using only CER reduced CER to 0.78% (lower than the full model) but yielded a MOS of only 2.33, demonstrating that peak clarity is not equivalent to peak user experience; MCLP captures quality nuances that humans care about but CER/ASR overlooks.
- Human pairwise experiments showed a win rate exceeding 0.8 when $\Delta\text{MCLP}>0.1$, indicating MCLP is a reliable preference predictor.

## Highlights & Insights
- **"Reverse Continuation + Dual Transcript" is a clever normalization technique**: While intuition suggests "using GT to predict eval," the authors reversed it. This ensures fixed GT length for fair normalization and pins the text via teacher-forcing so variations stem only from style. This paradigm of "leveraging LALM linguistic priors to quantify fuzzy concepts" is transferable to style in images or motion in video.
- **Gated Hybrid Reward is a general RL-for-TTS template**: The combination of hard gating, positive shifting, and content penalty is more robust against hacking than simple weighted sums. It can be applied to any "style + content" dual-target scenario like emotion-TTS or accent-TTS.
- **LLM-driven Data Refinement**: The automated pipeline (using LLMs for plot inference and VLMs for scene description) demonstrates how to upgrade traditional ASR corpora into structured datasets with character profiles and scene labels.

## Limitations & Future Work
- Main experiments focus on Chinese drama; while English results are provided in the appendix, systematic multilingual validation is needed to confirm if MCLP remains effective across languages.
- Evaluation is limited to TV dramas; domains like audiobooks (long-form narrative), game NPCs (short reactions), or virtual assistants (neutral commands) are not covered. Reward hyperparameters may require per-domain tuning.
- All descriptions are LLM/VLM generated, introducing noise and bias (e.g., R1 hallucinating character traits), which limits the RL upper bound.
- The MCLP model itself is a 7B LALM, making each rollout forward pass during RL computationally expensive compared to lightweight classifiers. Distilling a smaller MCLP estimator could be beneficial.

## Related Work & Insights
- **vs. CosyVoice3 / Higgs / OV-InstructTTS**: These rely on instruction prompts for single-sentence control. This work proves that incorporating multi-turn audio history and RL-based style alignment yields a >0.7 MOS improvement.
- **vs. Emotion Classifier Rewards (Wang 2025, Gao 2025b)**: Those methods use discrete labels; MCLP uses continuous likelihood, covering prosodic dimensions beyond emotion and providing dense reward signals.
- **vs. Speech Role-Playing Agents (SpeechRole, VoxRole, OmniCharacter)**: Those focus on end-to-end semantic alignment. This work focuses on acoustic style alignment in a content-specified setting, allowing the proposed method to serve as a high-fidelity TTS backend for agents.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Using LALM likelihood as a continuous style metric is a natural yet novel perspective in audio research; the reverse continuation design is ingenious.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comparison against 7 strong baselines, objective and subjective metrics, and human validation make for a robust evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear equations, cohesive motivation, and well-described reward mechanisms.
- **Value**: ⭐⭐⭐⭐ The MCLP metric and RL pipeline offer a ready-to-use engineering solution for Role-Play TTS, with the dataset serving as a valuable community contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Computational Narrative Understanding for Expressive Text-to-Speech](../../ACL2026/audio_speech/computational_narrative_understanding_for_expressive_text-to-speech.md)
- [\[ICML 2026\] MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety](multibreak_a_scalable_and_diverse_multi-turn_jailbreak_benchmark_for_evaluating_.md)
- [\[ACL 2026\] ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis](../../ACL2026/audio_speech/restyle-tts_relative_and_continuous_style_control_for_zero-shot_speech_synthesis.md)
- [\[ACL 2026\] DRInQ: Evaluating Conversational Implicature with Controlled Context Variation](../../ACL2026/audio_speech/drinq_evaluating_conversational_implicature_with_controlled_context_variation.md)
- [\[ACL 2026\] FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations](../../ACL2026/audio_speech/fc-tts_style_and_timbre_control_in_zero-shot_text-to-speech_with_disentangled_sp.md)

</div>

<!-- RELATED:END -->
