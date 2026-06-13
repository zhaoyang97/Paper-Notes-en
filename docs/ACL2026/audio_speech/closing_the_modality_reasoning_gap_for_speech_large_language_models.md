---
title: >-
  [Paper Note] Closing the Modality Reasoning Gap for Speech Large Language Models
description: >-
  [ACL 2026][Audio & Speech][Speech LLM] This paper proposes TARS (Trajectory Alignment for Reasoning in Speech), a reinforcement learning-based framework. By utilizing two dense reward signals—representation alignment and…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Speech LLM"
  - "Modality Reasoning Gap"
  - "Reinforcement Learning"
  - "Representation Alignment"
  - "Trajectory Alignment"
date: 2026-05-08
content_hash: 4a5006dc15e63791
---

# Closing the Modality Reasoning Gap for Speech Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2601.05543](https://arxiv.org/abs/2601.05543)  
**Code**: [https://github.com/AmphionTeam/TARS](https://github.com/AmphionTeam/TARS)  
**Area**: LLM Evaluation  
**Keywords**: Speech LLM, Modality Reasoning Gap, Reinforcement Learning, Representation Alignment, Trajectory Alignment

## TL;DR

This paper proposes TARS (Trajectory Alignment for Reasoning in Speech), a reinforcement learning-based framework. By utilizing two dense reward signals—representation alignment and behavior alignment—it aligns reasoning trajectories under speech conditions with those under text conditions. TARS achieves SOTA performance in 7B-scale models, with the Modality Recovery Rate (MRR) approaching or even exceeding 100%.

## Background & Motivation

**Background**: Speech Large Language Models (Speech LLMs) have made significant progress by adopting a three-stage architecture consisting of a speech encoder, an adapter, and a text LLM. This setup allows speech inputs to leverage the reasoning capabilities of the text LLM.

**Limitations of Prior Work**: Reasoning performance under speech input is significantly weaker than under text input, a phenomenon known as the "modality reasoning gap." (1) Input-side fusion methods (e.g., training adapters with a frozen LLM) only achieve superficial alignment; subtle representation differences are amplified as they propagate through Transformer layers. (2) Output-side supervision methods (e.g., knowledge distillation) enforce token-level matching offline. however, the distribution of speech conditions differs from text, making strict matching an unreachable goal and leading to exposure bias.

**Key Challenge**: While the underlying logical reasoning process should be modality-invariant, existing methods either only align input representations or enforce output alignment offline. Neither effectively guides the alignment of the reasoning trajectories themselves.

**Goal**: To design an online policy optimization framework that simultaneously aligns internal representations and external behaviors under speech and text conditions to eliminate the modality reasoning gap.

**Key Insight**: Utilizing the GRPO reinforcement learning framework, reasoning trajectories under text conditions are used as a moving reference. Asymmetric rewards are designed: speech completions are optimized for task accuracy, representation alignment, and behavior alignment simultaneously, while text completions are optimized only for accuracy.

**Core Idea**: Through online policy exploration and dense alignment rewards, the speech modality co-evolves with the continuously improving text reasoning capability, avoiding the exposure bias issues of offline supervision.

## Method

### Overall Architecture

TARS is based on GRPO and generates both speech-conditioned and text-conditioned completions for each prompt. The text branch is optimized using base rewards and serves as the alignment reference for the speech branch. The speech branch receives two additional dense rewards: representation alignment and behavior alignment. During training, the text branch continues to improve, providing an increasingly stronger alignment target for the speech branch.

### Key Designs

1.  **Representation Alignment Reward**:
    - **Function**: Reduces the drift of layer-wise hidden states between speech and text conditions.
    - **Mechanism**: For each Transformer layer $l$, the hidden states $\mathbf{H}^{(l)}$ of the reasoning tokens are mean-pooled into a fixed vector $\bar{\mathbf{h}}^{(l)}$. The average cosine similarity across $L$ layers between speech and text completions is calculated: $R_{\text{rep}} = \frac{1}{L}\sum_{l=1}^{L}\text{CosSim}(\bar{\mathbf{h}}_{\text{speech}}^{(l)}, \bar{\mathbf{h}}_{\text{text}}^{(l)})$.
    - **Design Motivation**: Representation drift is a major source of the modality reasoning gap. Layer-wise hidden state alignment provides coarse-grained but dense representation-level feedback.

2.  **Behavior Alignment Reward**:
    - **Function**: Ensures semantic consistency of the final outputs under speech and text conditions.
    - **Mechanism**: An external embedding model (e.g., Qwen3-Embedding-0.6B) is used to calculate the semantic cosine similarity between the speech completion $y_{\text{speech}}$ and the text reference $y_{\text{text}}^*$: $R_{\text{beh}} = \text{CosSim}(\mathcal{E}(y_{\text{speech}}), \mathcal{E}(y_{\text{text}}^*))$.
    - **Design Motivation**: Behavior alignment complements representation alignment, allowing the model to learn diverse effective reasoning trajectories as long as the final semantic behavior is consistent.

3.  **Modality-Specific Normalization**:
    - **Function**: Prevents speech completions from consistently receiving negative advantages due to systemically lower scores.
    - **Mechanism**: Advantages are calculated separately within the speech and text groups: $\hat{A}_{i,m} = r_{i,m} - \mu_m$, where $\mu_m$ is the mean reward within modality $m$.
    - **Design Motivation**: In naive full-group normalization, text completions naturally receive higher rewards, which would suppress the learning signal for speech completions.

### Loss & Training

A DAPO loss estimator (a variant of Dr. GRPO) is used. The total reward is $R_{\text{total}} = R_{\text{base}} + \alpha \cdot R_{\text{rep}} + \beta \cdot R_{\text{beh}}$ ($\alpha = \beta = 1.0$), where $R_{\text{base}} = R_{\text{acc}} + 0.5 \cdot R_{\text{fmt}}$. Training involves LoRA fine-tuning of all linear layers, while the audio encoder and projector are frozen.

## Key Experimental Results

### Main Results

**Accuracy of 7B Models on Reasoning Benchmarks (%)**

| Model | MMSU(A) | MMSU(T) | OBQA(A) | OBQA(T) | Avg(A) | MRR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-Omni | 61.51 | 67.94 | 81.09 | 84.40 | 71.30 | 91.76% |
| TARS(Qwen2.5-Omni) | **67.96** | 68.54 | **85.71** | 88.57 | **76.84** | **98.89%** |
| Phi-4-MM | 54.81 | 72.15 | 71.65 | 84.62 | 63.23 | 79.59% |
| TARS(Phi-4-MM) | **70.14** | 75.76 | **89.45** | 91.87 | **79.80** | **100.45%** |

### Ablation Study

| Training Strategy | MMSU(A) | OBQA(A) | Avg(A) | MRR |
| :--- | :--- | :--- | :--- | :--- |
| SFT | 60.83 | 81.54 | 71.19 | 89.36% |
| DPO | 59.99 | 79.78 | 69.89 | 89.39% |
| Standard GRPO | 63.73 | 82.86 | 73.30 | 94.04% |
| + Rep. Alignment | 65.91 | 84.40 | 75.16 | 96.43% |
| + Beh. Alignment | 66.20 | 84.84 | 75.52 | 95.57% |
| + Both (TARS) | **67.96** | **85.71** | **76.84** | **98.89%** |

### Key Findings

- TARS improves the MRR of Phi-4-MM to 100.45%, meaning speech reasoning performance exceeds that of text.
- Representation alignment and behavior alignment are complementary; used individually, they each provide approximately a 2% gain, with the combination performing best.
- Modality-specific normalization is crucial for training stability; naive normalization suppresses speech learning.
- TARS not only improves speech performance but also enhances text performance (Qwen2.5-Omni: 76.17→78.56%).
- End-to-end models outperform cascaded ASR+LLM systems, indicating that direct processing of speech signals avoids ASR errors.

## Highlights & Insights

- The asymmetric reward design is ingenious: the text branch improves continuously by optimizing only for task accuracy, providing an increasingly stronger alignment target for the speech branch, leading to co-evolution.
- Even when task accuracy for all speech completions is zero (in difficult reasoning scenarios), alignment rewards still provide effective gradient signals.
- The MRR > 100% result suggests that knowledge learned from speech processing can, in turn, enhance text reasoning.

## Limitations & Future Work

- Evaluation is limited to multiple-choice QA benchmarks; performance on free-form generation tasks remains to be verified.
- Representation alignment using mean pooling may lose positional information; more fine-grained alignment methods could be more effective.
- Reliance on synthetic speech for training requires further validation of robustness on real-world speech (with noise and accents).

## Related Work & Insights

- **vs AlignChat/DeSTA**: These methods freeze the LLM and only train the adapter, achieving only input-side alignment; TARS aligns the entire reasoning trajectory via RL.
- **vs Knowledge Distillation**: KD enforces token-level matching offline and suffers from exposure bias; TARS avoids this through online exploration.
- **vs SoundMind-RL**: Concurrent work also applies RL to speech reasoning but uses only sparse rule-based rewards, lacking dense alignment signals.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to apply an RL framework with trajectory alignment to eliminate the speech-text reasoning gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two base models, multiple baseline comparisons, and detailed ablations, though benchmarks are limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Problems are clearly defined, the methodology is intuitive, and formulas are concise.
- Value: ⭐⭐⭐⭐⭐ The MRR > 100% result is a landmark, opening a new direction for multimodal reasoning alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models](vapo_end-to-end_slide-enhanced_speech_recognition_with_omni-modal_large_language.md)
- [\[ACL 2026\] Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models](do_we_need_distinct_representations_for_every_speech_token_unveiling_and_exploit.md)
- [\[ACL 2026\] Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)
- [\[ACL 2026\] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)
- [\[NeurIPS 2025\] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](../../NeurIPS2025/audio_speech/thinksound_chain-of-thought_reasoning_in_multimodal_large_language_models_for_au.md)

</div>

<!-- RELATED:END -->
