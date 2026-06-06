---
title: >-
  [Paper Note] UniSRM: A Unified Speech Reward Model for Fine-Grained Speech Evaluation
description: >-
  [ACL 2026][Audio & Speech][Speech Reward Model] This paper proposes UniSRM, a unified speech reward model. Through a two-stage training (SFT+GRPO) and Reasoning Consistency Reward (RCR) mechanism…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Speech Reward Model"
  - "Multi-dimensional Evaluation"
  - "Reasoning Consistency"
  - "TTS Evaluation"
date: 2026-05-08
content_hash: d1aaca05e01079e5
---

# UniSRM: A Unified Speech Reward Model for Fine-Grained Speech Evaluation

**Conference**: ACL 2026  
**arXiv**: [2605.23261](https://arxiv.org/abs/2605.23261)  
**Code**: https://github.com/lavendery/UniSRM  
**Area**: Speech Evaluation / Reward Models  
**Keywords**: Speech Reward Model, Multi-dimensional Evaluation, Reasoning Consistency, TTS Evaluation

## TL;DR
This paper proposes UniSRM, a unified speech reward model. Through a two-stage training (SFT+GRPO) and Reasoning Consistency Reward (RCR) mechanism, it supports multi-dimensional, interpretable speech evaluation ranging from utterance-level quality to dialogue-level coherence, significantly outperforming existing methods on multiple evaluation tasks.

## Background & Motivation

**Background**: Speech generation quality assessment long relied on Mean Opinion Scores (MOS), which are costly, subjective, and difficult to scale. Recent explorations have begun using Large Audio-Language Models (LALMs) as automatic scorers, such as WavReward, SageLM, and SpeechJudge.

**Limitations of Prior Work**: (1) Existing methods cover limited tasks—most only handle utterance-level quality or single-turn dialogue, ignoring multi-turn interactions and contextual consistency; (2) Incomplete evaluation dimensions—some methods miss key metrics like speaker similarity; (3) Uncontrollable reasoning processes—rule-based RL lacks sufficient supervision over reasoning steps, leading to inconsistencies between generated rationales and final decisions; (4) Lack of transparency in scoring—traditional metrics (WER, SIM, UTMOS) each capture only a single aspect.

**Key Challenge**: The contradiction between the diversity of speech evaluation tasks (from isolated utterances to dialogue contexts) and the single-adapter nature of existing reward models; the tension between the freedom of rationale generation and the accuracy of final scoring.

**Goal**: Construct a unified reward model capable of (1) supporting multiple speech evaluation tasks; (2) outputting interpretable multi-dimensional scores and reasoning processes; (3) ensuring consistency between reasoning and decision-making.

**Key Insight**: It is observed that existing LLM-based scorers perform poorly when integrating text or multi-turn dialogue context—this suggests improvements can be made through better training strategies and reasoning supervision. Meanwhile, a multi-dimensional decomposition of the evaluation process aligns with the intuitive logic of human scoring.

**Core Idea**: Replace simple end-to-end fine-tuning with a "staged training + reasoning consistency reward" paradigm, allowing the model to receive explicit supervision while generating dimension-level intermediate reasoning, thereby improving overall reliability.

## Method

### Overall Architecture

UniSRM adopts a two-stage training pipeline. In the first stage (SFT), a pre-trained speech-language model (Qwen2.5-Omni-7B-thinker) is instruction-tuned on a unified dataset, UniSRM-Data, to learn to output multi-dimensional scores and reasoning in a structured format. In the second stage (GRPO), Group Relative Policy Optimization is used with a Reasoning Consistency Reward to further align model predictions with human preferences.

The input consists of task-related multimodal context, and the output has two parts: (1) dimension-level reasoning within `<think>` tags; (2) the final decision (binary preference or MOS-like scores) within `<answer>` tags.

### Key Designs

1.  **Reasoning Consistency Reward (RCR)**:
    - **Function**: Explicitly supervises the model's dimension-level scoring behavior during reasoning, ensuring high alignment between intermediate reasoning and final decisions.
    - **Mechanism**: For paired tasks, the consistency reward is calculated as $R_{\text{rc}}(o) = \frac{1}{D}\sum_{i=1}^{D}\mathbf{1}[\text{sign}(a_i - b_i) = \text{sign}(a_i^{\star} - b_i^{\star})]$, meaning the relative order of two samples in each dimension should match the labels. For quality scoring tasks, a normalized multi-dimensional scoring error is used as the reward.
    - **Design Motivation**: Pure result accuracy rewards can lead to "shortcut" maneuvers where the final prediction is correct but dimension scores are contradictory, undermining interpretability. RCR enforces dimension-level consistency by directly constraining intermediate steps.

2.  **Multi-task Unified Framework & Structured Output**:
    - **Function**: Processes four complementary speech evaluation tasks in a single model: utterance-level paired preference, utterance-level quality scoring, scenario-aware style consistency (with text context), and multi-turn dialogue evaluation (with history).
    - **Mechanism**: All tasks are unified as conditional generation problems. System prompts enforce a two-part structure (reasoning + answer). While answer formats differ across tasks (binary decision vs. multi-dimensional score vector), the reasoning part always includes task-relevant dimension-level scores.
    - **Design Motivation**: A unified framework allows a single model to learn general evaluation capabilities, while structured output facilitates penalizing format violations during RL optimization.

3.  **Componentized Reward Function**:
    - **Function**: Combines format rewards, accuracy rewards, and reasoning consistency rewards, $R(x,o) = \lambda_{\text{fmt}}R_{\text{fmt}}(o) + \lambda_{\text{acc}}R_{\text{acc}}(o) + \lambda_{\text{rc}}R_{\text{rc}}(o)$, to comprehensively constrain model behavior.
    - **Mechanism**: Format rewards apply a $-1$ penalty for malformed outputs; accuracy rewards use $\mathbf{1}[y^{(g)} = y^{\star}]$ for paired tasks and normalized distance rewards for quality scoring; and RCR follows the design above.
    - **Design Motivation**: These three components constrain the model from different dimensions, requiring correct final answers, standardized generation processes, and consistent dimensional reasoning.

### Loss & Training

The SFT stage uses the standard auto-regressive maximum likelihood objective. The GRPO stage employs intra-group relative policy optimization with KL regularization. For each input $x$, $G$ responses are sampled from the current policy, and advantages are normalized using the group mean and standard deviation: $A^{(g)} = (R^{(g)} - \mu(x))/(\sigma(x) + \epsilon)$, followed by a clipped policy gradient objective.

## Key Experimental Results

### Main Results

| Model | Task 1 (Paired) | Task 2 (Quality) | Task 3-En (Scenario) | Task 3-Zh | Task 4 (Dialogue) |
|------|-------------|----------------|-----------------|---------|------------|
| WER / SIM / UTMOS / DNSMOS | 59.24–84.10 | 0.274–0.449 | 33.21–61.44 | 48.19–63.04 | 40.48–50.79 |
| GPT-4o-Audio | 61.04 | 0.060 | 64.02 | 64.82 | 71.96 |
| Gemini-2.5-Flash | 60.44 | 0.522 | 65.68 | 71.74 | 71.43 |
| **UniSRM (Ours)** | **65.06** | **0.551** | **85.61** | **91.30** | **88.89** |

UniSRM achieves the best performance across all tasks, particularly in tasks requiring text or dialogue context integration (Tasks 3 and 4), showing an improvement of over 20 percentage points compared to the strongest baseline.

### Ablation Study

| Configuration | Task 1 | Task 2 | Task 3-En | Task 3-Zh | Task 4 |
|------|------|------|--------|--------|------|
| SFT Only (w/o GRPO) | 60.24 | 39.20 | 67.16 | 70.95 | 74.60 |
| GRPO w/o RCR | 60.44 | 37.58 | 80.81 | 81.42 | 82.54 |
| **UniSRM Full** | **65.06** | **39.74** | **85.61** | **91.30** | **88.89** |

Key Findings: (1) GRPO provides improvements over pure SFT; (2) The addition of RCR generally brings further improvements, with a maximum gain of 8.88 percentage points (Task 4); (3) **Counter-intuitive phenomenon**: GRPO without RCR performed worse than SFT in some dimensions, indicating that pure accuracy rewards induce "shortcut" behavior. RCR effectively prevents this via dimension-level supervision.

### Cross-dataset Generalization

| Dataset | Metric | DNSMOS | Gemini-2.5-Pro | **UniSRM** |
|--------|------|--------|----------------|-----------|
| BVCC | PCC | 0.299 | 0.339 | **0.498** |
| SOMOS-Clean | PCC | 0.048 | 0.250 | **0.261** |
| SOMOS-Full | PCC | 0.053 | 0.222 | **0.235** |

UniSRM demonstrates strong cross-domain capability on human-annotated external datasets (BVCC, SOMOS), suggesting the model learns genuine evaluation abilities rather than overfitting to LLM-generated labels.

## Highlights & Insights

- **Ingenious RCR Design**: RCR does not simply punish errors; it enforces "logical self-consistency" at the dimension level—ensuring consistency in relative comparisons. This constraint shifts the optimization goal from being "correct at the last step" to being "reasonable throughout," significantly reducing the model's room to "cheat."
- **Synergy of Unified Data and Multi-task Learning**: By carefully designing UniSRM-Data, four seemingly different tasks are unified into a "multi-dimensional reasoning + structured answer" format, enabling a single model to learn cross-task general evaluation capabilities.
- **SFT and GRPO Collaboration**: SFT teaches the model to mimic the rationales and decisions of annotators. With GRPO, the model improves final accuracy while increasing the diversity and interpretability of reasoning through varied sampling.

## Limitations & Future Work

**Limitations acknowledged by authors**: (1) Current benchmarks have limited coverage of difficult scenarios such as heavy accents or overlapping speech; (2) High computational costs for training and inference limit scalability and low-latency deployment.

**Self-discovered limitations**: (1) Evaluation dimension definitions are relatively fixed and may not adapt to emerging application scenarios (e.g., multilingual mixing, specific character accents); (2) Reliance on LLM-generated labels in data sources—though cross-dataset generalization is proven, LLM systemic biases might be implicitly imported.

**Future Work**: (1) Explore distillation strategies to make UniSRM lightweight; (2) Introduce active learning to prioritize annotating high-uncertainty samples; (3) Research adaptive dimension selection.

## Related Work & Insights

- **vs WavReward / SageLM**: These methods focus on single-turn or utterance-level evaluation and often use rule-based RL. UniSRM covers richer task scenarios and explicitly constrains dimension-level consistency via RCR.
- **vs SpeechJudge**: While it also generates evaluation rationales, it targets the utterance level with limited dimensions. UniSRM extends this to the dialogue level.
- **vs QualiSpeech / AudioJudge**: These focus on low-level quality features, whereas UniSRM incorporates high-level, context-aware evaluations.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The RCR design is a creative improvement for multi-step reasoning evaluation. While multi-task unified frameworks have precedents, the comprehensiveness and execution quality of this work are at the industry forefront.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Includes 4 complementary tasks, 3 levels of ablation, fine-grained dimension analysis, and cross-dataset generalization, providing comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ The logic is clear and motivation is sufficient, though discussion on the trade-off between computational cost and practicality is slightly lacking.
- **Value**: ⭐⭐⭐⭐⭐ Provides a reusable paradigm and public dataset for reward modeling in speech generation, significantly contributing to the completion of the speech RLHF ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training](towards_fine-grained_and_multi-granular_contrastive_language-speech_pre-training.md)
- [\[ACL 2026\] SegTune: Structured and Fine-Grained Control for Song Generation](segtune_structured_and_fine-grained_control_for_song_generation.md)
- [\[ACL 2026\] UniVocal: Unified Speech-Singing Code-mixed Synthesis](univocal_unified_speech-singing_code-switching_synthesis.md)
- [\[ACL 2026\] UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions](unisonate_a_unified_model_for_speech_music_and_sound_effect_generation_with_text.md)
- [\[ACL 2026\] SpeechLLM-as-Judges: Towards General and Interpretable Speech Quality Evaluation](speechllm-as-judges_towards_general_and_interpretable_speech_quality_evaluation.md)

</div>

<!-- RELATED:END -->
