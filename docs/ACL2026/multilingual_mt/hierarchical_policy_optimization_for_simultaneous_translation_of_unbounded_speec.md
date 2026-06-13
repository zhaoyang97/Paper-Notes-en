---
title: >-
  [Paper Note] Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech
description: >-
  [ACL 2026][Multilingual & Machine Translation][Simultaneous Translation] This paper proposes Hierarchical Policy Optimization (HPO), which performs post-training on LLM-based simultaneous translation models through hiera…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Simultaneous Translation"
  - "Reinforcement Learning"
  - "Hierarchical Reward"
  - "LLM Speech Translation"
  - "GRPO"
date: 2026-05-08
content_hash: e088494be710683f
---

# Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech

**Conference**: ACL 2026 Oral  
**arXiv**: [2604.21045](https://arxiv.org/abs/2604.21045)  
**Code**: [https://github.com/owaski/HPO](https://github.com/owaski/HPO)  
**Area**: Image Segmentation  
**Keywords**: Simultaneous Translation, Reinforcement Learning, Hierarchical Reward, LLM Speech Translation, GRPO

## TL;DR
This paper proposes Hierarchical Policy Optimization (HPO), which performs post-training on LLM-based simultaneous translation models through hierarchical reward design. By suppressing latency optimization when translation quality fails to meet a threshold, it achieves a +7 COMET improvement in translation quality at a 1.5-second latency.

## Background & Motivation

**Background**: Simultaneous Speech Translation (SST) requires generating translations while receiving partial speech input. Recently, LLM-based methods have become the mainstream solution for handling unbounded long speech (e.g., InfiniSST) by modeling SST as a multi-turn dialogue task and utilizing KV cache reuse to eliminate redundant computation.

**Limitations of Prior Work**: These methods rely heavily on synthetic read-write trajectory data for supervised fine-tuning (SFT). However, existing trajectory synthesis methods have significant flaws. Word alignment-based methods ignore the future context required for translation timing; LLM-based methods simulating interpreters exhibit unstable segmentation and fail to guarantee the generation of valid read-write trajectories. Consequently, the quality of SFT data is suboptimal, causing the model to learn incorrect behaviors.

**Key Challenge**: There is an inherent trade-off between translation quality and latency. When using reinforcement learning to jointly optimize both, latency rewards are easier to optimize (latency can be reduced simply by translating early, regardless of accuracy). This leads the model to over-optimize for latency at the expense of translation quality.

**Goal**: Design a post-training method to correct the erroneous behaviors of SFT models while stably balancing the optimization of translation quality and latency.

**Key Insight**: It is observed that the fundamental reason latency rewards dominate optimization is the scale difference and asymmetric optimization difficulty between the two rewards. By introducing a "quality gating" mechanism—where latency optimization is permitted only after translation quality reaches a certain standard—an effective hierarchical constraint can be established.

**Core Idea**: A hierarchical reward structure is used to constrain GRPO training. When translation quality does not exceed a threshold, the latency reward is set to its worst possible value, ensuring the model prioritizes accuracy before pursuing speed.

## Method

### Overall Architecture
The system consists of a streaming speech encoder and an LLM translator. Speech is divided into fixed-duration chunks; the encoder incrementally encodes each new chunk and reuses the previous KV cache. The LLM decodes the next segment of translation based on interleaved speech features and previously generated translations. Based on this SFT model, HPO samples multiple translation hypotheses, calculates hierarchical rewards, and optimizes using GRPO.

### Key Designs

1.  **Sentence-level Segmenting and Alignment (SEGALE)**:
    - **Function**: Segments long-text translation hypotheses into sentences and aligns them with the reference translation.
    - **Mechanism**: Sentence segmentation is performed using spaCy, followed by an embedding-based aligner and adaptive search to align hypothesis sentences with reference sentences, while identifying over-translation ($R_k = \phi$) and under-translation ($H_k = \phi$).
    - **Design Motivation**: Traditional mwersegmenter forces alignment by minimizing word error rate, assigning alignments even if the hypothesis is nonsensical gibberish. This leads to "reward hacking" when combined with non-robust neural metrics.

2.  **Hierarchical Reward**:
    - **Function**: Jointly evaluates translation quality and latency to prevent over-optimization of latency.
    - **Mechanism**: Quality scores $q^{j,k}$ (MetricX) and latency scores $l^{j,k}$ (LAAL) are calculated for each aligned sentence pair. If the quality score is below the threshold $q_{\text{thres}}$, the latency score is set to the worst value $l_{\max}$. These are averaged across all sentences within a hypothesis, group-normalized for each sample, and then weighted to obtain the final reward $r^j = \bar{q}^j - \lambda \cdot \bar{l}^j$.
    - **Design Motivation**: Direct weighted combinations of quality and latency rewards bias the model toward "translating early but poorly." Quality gating ensures low latency is only rewarded when the translation is sufficiently good, establishing an optimization priority of quality first, then speed.

3.  **Group Normalization Strategy**:
    - **Function**: Eliminates scale differences between quality and latency rewards.
    - **Mechanism**: For $n$ hypotheses sampled from the same prompt, group normalization (subtracting mean and dividing by standard deviation) is applied separately to quality and latency scores to bring both rewards onto the same scale.
    - **Design Motivation**: Quality metrics (e.g., MetricX range -25 to 0) and latency metrics (seconds) naturally have different scales; direct summation leads to unstable training.

### Loss & Training
The GRPO framework is adopted, sampling 16 translation trajectories for each speech segment, using clipped importance sampling and on-policy KL divergence regularization. MetricX serves as the quality reward model, with a quality threshold $q_{\text{thres}} = -5$, latency weight $\lambda = 0.5$, and KL penalty weight 0.01. Training was conducted for approximately 20 hours (500 steps) on three 8×H100 nodes, with one node dedicated to reward calculation.

## Key Experimental Results

### Main Results
In En→Zh/De/Ja directions on the ACL 60/60 dev set, HPO significantly outperforms the InfiniSST baseline across COMET, MetricX, and BLEURT metrics.

| Direction | Latency (s) | Gain (COMET) | Gain (MetricX) | Gain (BLEURT) |
|-----------|-------------|--------------|----------------|---------------|
| En→Zh     | ~1.5s       | +7           | +1.25          | +4            |
| En→De     | ~1.5s       | Significant  | Significant    | Significant   |
| En→Ja     | ~1.5s       | Significant  | Significant    | Significant   |

### Ablation Study

| Configuration | StreamLAAL | COMET | MetricX |
|---------------|------------|-------|---------|
| SFT (Baseline)| 1216       | 0.7348| -4.52   |
| Normalize     | 1555       | 0.7977| -3.41   |
| Normalize + Truncation (SeqPO) | 1805 | 0.8058 | -3.39 |
| Normalize + Hierarchical-Doc | 1544 | 0.8157 | -3.27 |
| Normalize + Hierarchical-Sent (HPO) | 1383 | 0.8234 | -3.21 |

### Key Findings
- Sentence-level hierarchical reward (Hierarchical-Sent) comprehensively outperforms document-level hierarchical reward and simple truncation methods, achieving better translation quality at lower latency.
- MetricX is the only quality reward function among six that demonstrates consistent performance across all automatic metrics and human evaluations.
- BLEU is a notable exception; HPO does not necessarily outperform the baseline on BLEU. Gemini evaluations also confirm that neural reward-based optimization can lead to reward hacking.
- Models using mwersegmenter exploit weaknesses in segmentation and metrics for reward hacking; gibberish hypotheses could still receive high scores.

## Highlights & Insights
- The hierarchical reward design is ingenious: rather than a simple weighted balance of two objectives, it establishes a "quality first" hard constraint, permitting speed optimization only when quality is sufficient. This logic can be extended to any multi-objective RL scenario involving a "primary objective + auxiliary objective."
- The study exposes the reward hacking vulnerabilities of the mwersegmenter + neural metric combination, where gibberish text could receive nearly perfect MetricX scores after segmentation—a significant finding for the translation evaluation field.
- In certain settings, HPO even exceeds the quality of offline translation models, suggesting that RL post-training not only corrects SFT errors but also discovers strategies superior to full-sequence offline translation.

## Limitations & Future Work
- The method was only validated on one architecture (InfiniSST), one data synthesis method, and three directions with English as the source language.
- MetricX remains imperfect as a reward model, sometimes favoring fluency over accuracy, which may lead to reward hacking.
- BLEU and Gemini evaluations indicate a risk of overfitting to neural rewards, necessitating more robust quality reward models.
- The effect of applying HPO to offline translation models (standard GRPO) has not been explored.

## Related Work & Insights
- **vs InfiniSST (SFT)**: While InfiniSST only uses synthetic trajectories for SFT, HPO uses RL post-training to correct behaviors, significantly outperforming SFT at all latency levels.
- **vs SeqPO-SiMT**: SeqPO uses truncation and normalization for latency rewards but is limited to text translation and does not support unbounded speech; HPO's hierarchical reward outperformed truncation in ablations.
- **vs traditional RL-SST**: Previous RL methods were based on encoder-decoder Transformer text translation; HPO is the first to extend RL to LLM-based unbounded speech simultaneous translation.

## Rating
- Novelty: ⭐⭐⭐⭐ The hierarchical reward idea is simple and effective, though the GRPO framework itself is not a new contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage includes three language directions, six reward functions, multi-dimensional ablations, human evaluation, and reward hacking analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is clear, method descriptions involve rigorous logic, and charts are well-designed.
- Value: ⭐⭐⭐⭐ Provides direct guidance for RL training in simultaneous translation; the hierarchical reward concept has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TLPO: Token-Level Policy Optimization for Mitigating Language Confusion in Large Language Models](tlpo_token-level_policy_optimization_for_mitigating_language_confusion_in_large_.md)
- [\[ACL 2026\] Efficient Training for Cross-lingual Speech Language Models](efficient_training_for_cross-lingual_speech_language_models.md)
- [\[ACL 2026\] Prosody as Supervision: Bridging the Non-Verbal–Verbal for Multilingual Speech Emotion Recognition](prosody_as_supervision_bridging_the_non-verbal--verbal_for_multilingual_speech_e.md)
- [\[ACL 2026\] NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs](niutranslmt_toward_inclusive_and_scalable_multilingual_machine_translation_with_.md)
- [\[ICLR 2026\] From Utterance to Vividity: Training Expressive Subtitle Translation LLM via Adaptive Local Preference Optimization](../../ICLR2026/multilingual_mt/from_utterance_to_vividity_training_expressive_subtitle_translation_llm_via_adap.md)

</div>

<!-- RELATED:END -->
