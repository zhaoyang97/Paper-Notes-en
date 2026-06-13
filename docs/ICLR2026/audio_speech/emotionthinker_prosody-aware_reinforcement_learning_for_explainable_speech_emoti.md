---
title: >-
  [Paper Note] EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning
description: >-
  [ICLR 2026][Audio & Speech][Speech Emotion Recognition] This work is the first to reformulate Speech Emotion Recognition (SER) as a deep reasoning problem…
tags:
  - "ICLR 2026"
  - "Audio & Speech"
  - "Speech Emotion Recognition"
  - "Explainable Reasoning"
  - "Reinforcement Learning"
  - "Prosody-Aware"
  - "Chain-of-Thought"
date: 2026-05-08
content_hash: d231d82a43b63e7b
---

# EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning

**Conference**: ICLR 2026 Oral  
**arXiv**: [2601.15668](https://arxiv.org/abs/2601.15668)  
**Code**: [Available](https://github.com/dingdongwang/EmotionThinker)  
**Area**: Audio & Speech
**Keywords**: Speech Emotion Recognition, Explainable Reasoning, Reinforcement Learning, Prosody-Aware, Chain-of-Thought

## TL;DR
This work is the first to reformulate Speech Emotion Recognition (SER) as a deep reasoning problem, leveraging a prosody-enhanced backbone model combined with GRPO-PTR (Progressive Trustworthy Reasoning reward) reinforcement learning to generate explainable emotion reasoning grounded in acoustic evidence.

## Background & Motivation
- Current SpeechLLMs still treat emotion recognition as a simple classification task, producing labels without explaining "why."
- Existing SFT-based descriptive methods remain at the level of acoustic feature description, lacking a **causal reasoning chain** from acoustic observations to emotional judgments.
- Three major challenges:
  1. Absence of high-quality reasoning datasets; existing emotion corpora lack fine-grained acoustic annotations.
  2. Weak prosody perception in SpeechLLMs (insufficient sensitivity to pitch, energy, speaking rate, and stress).
  3. Standard RL relies solely on rule-based rewards (outcome accuracy), which cannot supervise open-ended reasoning quality.

## Method

### Overall Architecture
Three-stage training pipeline:
1. **Data Construction**: Build the EmotionCoT-35K prosody-aware CoT reasoning dataset.
2. **Prosody-Enhanced SFT**: Train a prosody-aware backbone model, EmotionThinker-Base, on Qwen2.5-Omni-7B.
3. **GRPO-PTR Reinforcement Learning**: Progressively introduce trustworthy reasoning rewards to refine reasoning quality.

### Key Designs

**EmotionCoT-35K Dataset Construction**:
- 35K audio–reasoning pairs (~200 hours) covering 9 emotion categories (Neutral/Happy/Sad/Angry/Contempt/Confused/Whisper/Surprise/Fear).
- Sources: IEMOCAP, MELD, Expresso, MEAD, EARS.
- Automated annotation pipeline extracting:
    - Low-level features: speaking rate, pitch, energy (standard speech tools).
    - Stressed words: identified from transcripts via WhiStress.
    - Intonation contours: frame-level pitch-energy trajectories smoothed with Savitzky-Golay filtering, classified into coarse-grained styles (expressive/flat) and fine-grained patterns (rising/falling/rise-fall/fall-rise).
    - Speaker attributes: gender and age group via wav2vec2.0 classifier.
- All prosodic annotations are used as contextual prompts for GPT-4o to generate step-by-step reasoning trajectories.
- The first prosody-aware CoT dataset, covering dimensions far beyond existing speech description datasets.

**Prosody-Enhanced SFT (EmotionThinker-Base)**:
- ~500 hours of prosody-enhanced data across four task types:
  1. Word-level stress perception (Stress-17K dataset).
  2. Prosodic attribute classification (pitch/energy/speaking rate/intonation level).
  3. Comparative prosody enhancement (utterances with modified prosodic parameters concatenated to train the model to identify correct orderings).
  4. Cold-start reasoning on 5K EmotionCoT samples.
- Joint optimization of audio encoder, audio adapter, and LLM backbone.

**GRPO-PTR: Progressive Trustworthy Reasoning Reward**:

Three reward signals:
1. **Format Reward $R_f$**: Binary (0/1) signal indicating whether the output follows the think/answer XML format.
2. **Outcome Accuracy Reward $R_o$**: Binary (0/1) signal indicating whether the predicted label matches the ground truth.
3. **Reasoning Quality Reward $R_t$**: Scored by a trained reward model (Qwen2.5-Omni-3B fine-tuned on 101.4K samples) across four dimensions (1–5 scale):
    - Factual alignment
    - Interpretative quality
    - Caption completeness
    - Fluency and structural clarity

**Trustworthiness Weight $\tau$**:
- Sampled responses are divided into two groups (correct/incorrect outcomes); the mean reasoning reward of each group is computed.
- When the correct group's reasoning reward $\geq$ the incorrect group's, $\tau = 1$; otherwise $\tau = \exp(\text{difference})$, applying exponential decay suppression.
- Acts as a group-level alignment gate to prevent reward hacking where high reasoning scores co-occur with wrong answers.
- In essence, the reasoning reward signal is trusted only when reasoning quality and outcome correctness are consistent at the group level.

**Progressive Scheduling**:
- Early training uses only $R_o + R_f$ until emotion accuracy stabilizes at ~50%.
- $R_t$ is introduced afterward to avoid interference from multiple unstable reward signals during early convergence.

### Loss & Training
- Final reward: $R_i = 0.3 \cdot R_f + 1.0 \cdot R_o + 0.5 \cdot \tau \cdot R_t$
- KL divergence coefficient: 0.04; learning rate: 1e-6; $K=8$ candidates sampled per input.
- RL training for 3,000 steps based on Qwen2.5-Omni-7B.

## Key Experimental Results

### Main Results

| Model | IEMOCAP | MELD | RAVDESS | SAVEE | Avg Acc | Reasoning Quality Avg |
|---|---|---|---|---|---|---|
| Kimi-Audio | 57.72 | 59.13 | 61.07 | 55.21 | 58.83 | 2.72 |
| BLSP-Emo | 76.00 | 57.30 | 72.00 | 63.73 | 65.41 | 2.73 |
| Qwen2.5-Omni-7B | 45.70 | 54.64 | 64.77 | 52.49 | 50.83 | 2.87 |
| MiniCPM-O | 35.54 | 52.78 | 40.93 | 35.47 | 43.60 | 3.01 |
| **EmotionThinker** | **77.68** | **59.71** | **71.56** | **73.96** | **68.89** | **3.98** |

EmotionThinker achieves the highest emotion accuracy (68.89%) among 16 open-source models and substantially outperforms all baselines in reasoning quality (3.98 vs. second-best 3.04).

| Prosody Perception Test | Pitch | Speaking Rate | Energy | Intonation | Stress |
|---|---|---|---|---|---|
| Qwen2.5-Omni-7B | 25.71 | 29.94 | 27.67 | 25.83 | 30.24 |
| EmotionThinker-Base | **75.11** | **68.70** | **69.42** | **60.25** | **71.50** |

### Ablation Study

| Variant | SER Acc | Reasoning Quality |
|---|---|---|
| Qwen2.5-Omni-7B (Baseline 1) | 50.83 | 2.87 |
| EmotionThinker-Base (Baseline 2) | 52.63 | 3.41 |
| SFT (V1) | 53.91 | 3.78 |
| GRPO (V2) | 62.91 | 3.45 |
| GRPO-PTR w/o trained RM (V3) | 66.67 | 3.36 |
| GRPO-PTR w/o trustworthiness weight (V4) | 67.71 | 3.74 |
| GRPO-PTR w/o progressive scheduling (V5) | 62.80 | 3.76 |
| **GRPO-PTR full (V6)** | **68.89** | **3.98** |

### Key Findings
1. SFT improves reasoning quality but yields limited accuracy gains; GRPO substantially boosts accuracy but at the cost of reasoning quality; GRPO-PTR achieves both simultaneously.
2. An untrained reward model introduces noise (V3 vs. V6), confirming that training the reward model is critical.
3. Removing the trustworthiness weight (V4) has a minor impact on accuracy but degrades reasoning quality, indicating that $\tau$ primarily prevents logically flawed reasoning from being rewarded.
4. Disabling progressive scheduling (V5) causes a substantial accuracy drop to 62.80%, highlighting the stability challenges of multi-signal RL.
5. Varying $K$ from 4 to 16 yields limited performance differences; $K=8$ is selected as an efficiency–performance trade-off.

## Highlights & Insights
- **First work** to reformulate SER from a classification problem into an RL-driven deep reasoning task.
- Prosody-enhanced SFT is a critical prerequisite: without prosody perception capability, reasoning cannot be grounded in genuine acoustic cues.
- The trustworthiness weight $\tau$ in GRPO-PTR is an elegant design; the group-level alignment mechanism effectively prevents reward hacking.
- The four-dimensional reasoning quality evaluation framework is transferable to reasoning quality assessment in other modalities.
- Human evaluation and GPT-based automatic evaluation rankings are consistent, validating the reliability of the evaluation scheme.

## Limitations & Future Work
- The reward model is fine-tuned from only a 3B model, which may introduce evaluation bias.
- The nine emotion categories may lack sufficient granularity (e.g., sarcasm, mixed emotions).
- Validation is conducted exclusively on English datasets; cross-lingual generalization remains unknown.
- Reasoning generation increases inference latency, limiting applicability in real-time scenarios.

## Related Work & Insights
- Conceptually aligned with DeepSeek-R1 (RL-incentivized reasoning), but extended to the speech modality with task-specific PTR customized for emotion recognition.
- Advances beyond descriptive methods such as SECap and OSUM-EChat by establishing a causal chain from acoustic features to emotional inference.
- The prosody-enhanced SFT strategy (particularly the comparative enhancement tasks) is generalizable to other speech understanding tasks.

## Rating
- Novelty: 5/5 (First RL-driven explainable speech emotion reasoning framework; PTR strategy is highly original)
- Experimental Thoroughness: 4/5 (Four benchmarks, 16 baselines, human evaluation, and comprehensive ablation studies)
- Writing Quality: 4/5 (Modular and clear presentation; rigorous mathematical formulation)
- Value: 5/5 (Establishes a new paradigm for speech emotion reasoning; methodology is transferable)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization](avere_improving_audiovisual_emotion_reasoning_with_preference_optimization.md)
- [\[ACL 2026\] Privacy-preserving Prosody Representation Learning](../../ACL2026/audio_speech/privacy-preserving_prosody_representation_learning.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[ICLR 2026\] Stitch: Simultaneous Thinking and Talking with Chunked Reasoning for Spoken Language Models](stitch_simultaneous_thinking_and_talking_with_chunked_reasoning_for_spoken_langu.md)
- [\[ICLR 2026\] ParaS2S: Benchmarking and Aligning Spoken Language Models for Paralinguistic-Aware Speech-to-Speech Interaction](paras2s_benchmarking_and_aligning_spoken_language_models_for_paralinguistic-awar.md)

</div>

<!-- RELATED:END -->
