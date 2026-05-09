---
title: >-
  [Paper Note] When Eyes and Ears Disagree: Can MLLMs Discern Audio-Visual Confusion?
description: >-
  [AAAI 2026][Multimodal VLM][Multimodal Large Language Models] This paper identifies a critical phenomenon termed "audio-visual confusion" in MLLMs, wherein models are heavily dominated by visual information and fail to recognize missing audio when audio-visual inputs are asymmetric. The authors propose the AV-ConfuseBench benchmark and the RL-CoMM method — combining a stepwise reasoning reward that incorporates an external audio model as reference with answer-centered confidence optimization — achieving 10–30% accuracy improvements over baselines using only approximately 20% of the training data.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Multimodal Large Language Models
  - Audio-Visual Confusion
  - Hallucination
  - Reinforcement Learning
  - Collaborative Multi-Model
date: 2026-05-08
content_hash: 3382559cf83c557c
---

# When Eyes and Ears Disagree: Can MLLMs Discern Audio-Visual Confusion?

**Conference**: AAAI 2026  
**arXiv**: [2511.10059](https://arxiv.org/abs/2511.10059)  
**Code**: [https://github.com/rikeilong/AVConfusion](https://github.com/rikeilong/AVConfusion)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Large Language Models, Audio-Visual Confusion, Hallucination, Reinforcement Learning, Collaborative Multi-Model

## TL;DR

This paper identifies a critical phenomenon termed "audio-visual confusion" in MLLMs, wherein models are heavily dominated by visual information and fail to recognize missing audio when audio-visual inputs are asymmetric. The authors propose the AV-ConfuseBench benchmark and the RL-CoMM method — combining a stepwise reasoning reward that incorporates an external audio model as reference with answer-centered confidence optimization — achieving 10–30% accuracy improvements over baselines using only approximately 20% of the training data.

## Background & Motivation

**Audio-Visual Confusion — An Overlooked Deficiency in MLLMs**:

MLLMs (e.g., Qwen2.5-Omni, Gemini 2.5) have achieved remarkable progress in audio-visual understanding tasks. However, this paper identifies a critical problem:

> When the given audio-visual information is asymmetric, can an MLLM recognize that a visually present object has its corresponding audio missing?

Experiments reveal striking results:

**Audio-muted scenario**: After muting a specific instrument in a video and asking "Is there a sound of this instrument?", Qwen2.5-Omni-7B still answers "Yes" with 90.41% probability — **almost entirely misled by visual information**.

**Audio-modified scenario**: When background music is replaced with bird calls, MLLMs continue to describe visual content predominantly, remaining largely insensitive to the actual audio.

**Even Gemini 2.5 Pro with chain-of-thought enabled**: 38.36% of responses are still incorrectly guided by visual input.

**Open-source models perform worse**: Video-LLaMA2 achieves only 2.73% accuracy, answering "Yes" nearly 100% of the time.

**Root cause analysis**: MLLMs trained on synchronized audio-visual data develop a bound perception between vision and audio — "seeing an object implies hearing its sound." The model's reasoning is dominated by visual information; even when audio-level uncertainty is high (as observed through persistently elevated entropy scores in later layers), models still tend to rely on vision for judgment.

## Method

### Overall Architecture

RL-CoMM (Reinforcement Learning-based Collaborative Multi-MLLM) is built upon Qwen2.5-Omni-3B and consists of three training stages:

1. **Warm-up**: Supervised fine-tuning on a small set of high-quality Q&A pairs to establish a structured reasoning format.
2. **Step-RR** (Step-wise Reasoning Reward): GRPO optimization with stepwise reasoning rewards.
3. **Ans-CO** (Answer-centered Confidence Optimization): Optimization of answer-level confidence.

The core innovation is the introduction of an **external Large Audio Language Model (LALM)** as the reference model $\pi_{ref}$, with the **Omni-LLM** serving as the policy model $\pi_\theta$, compensating for the Omni-LLM's audio perception weakness through heterogeneous model collaboration.

### Key Designs

#### 1. AV-ConfuseBench Construction

The proposed mini-benchmark includes two settings:

**Audio-muted setting**:
- A specific instrument is muted in a multi-instrument performance scene
- Question format: "This is a video of audio corruption where some instrument sound is muted. Is there a/an {muted-object} sound?"
- Ground truth is uniformly "No"
- 39 videos yielding 73 Q&A pairs
- Metrics: accuracy + proportion of "Yes" responses

**Audio-modified setting**:
- Background audio is replaced with entirely unsynchronized sounds (wind, birdsong, rain, drilling, thunder)
- 20 videos × 5 sound types = 100 Q&A pairs
- Question: "Describe what you see and what you hear"
- Metrics: AI-assisted scoring (audio accuracy A-Acc, visual accuracy V-Acc, on a 0–5 scale)

#### 2. Step-wise Reasoning Reward (Step-RR)

The key idea of Step-RR is to leverage an external LALM to provide a "pure audio perspective" reference reasoning. A structured output format is defined:

- The policy model outputs three tagged segments: `<a-think>` (audio reasoning), `<v-think>` (visual reasoning), and `<answer>` (final answer)
- The reference model generates pure audio reasoning `<a-think>` conditioned on the ground truth

**Three reward signals**:

**Format reward** $r_{format}$: Whether the output conforms to the specified three-part format (0/1).

**Audio Reasoning Rationality reward (ARR)** $r_{arr}$: The semantic similarity between the policy model's audio reasoning $o_1^i$ and the reference model's audio reasoning $o_{ref}$ is computed using Qwen3 Embedding-0.6B:

$$r_{arr}^i = \begin{cases} 1, & \text{if } \mathcal{S}(o_1^i | o_{ref}) > \omega \text{ and } o_3^i = y \\ 0, & \text{otherwise} \end{cases}$$

where $\omega = 0.8$ is the similarity threshold. Design motivation: to ensure the policy model's audio reasoning is not contaminated by visual information and remains semantically consistent with the pure audio reference.

**Audio-Visual Correlation reward (AVC)** $r_{avc}$: Evaluates the correlation between audio and visual reasoning, incorporating a soft-matching mechanism:

$$r_{avc}^i = \begin{cases} 1 + \mathcal{I}(o_1^i | o_2^i), & \text{if } o_3^i = y \\ \mathcal{I}(o_1^i | o_2^i), & \text{if } o_3^i \neq \text{null}, \neq y \\ 0, & \text{otherwise} \end{cases}$$

Design motivation: to reward sound cross-modal audio-visual reasoning — providing a higher base score plus correlation score when the answer is correct, while still rewarding the correlation score when the answer is incorrect but a reasoning process exists.

**Group advantage computation**: $r^i = r_{format} + r_{arr} + r_{avc}$, normalized following standard GRPO:

$$A^i = \frac{r^i - \text{mean}(\{r^1, ..., r^G\})}{\text{std}(\{r^1, ..., r^G\})}$$

Note: The **KL penalty is removed** during training, as the KL divergence between the heterogeneous reference and policy models is semantically meaningless.

#### 3. Answer-centered Confidence Optimization (Ans-CO)

Addresses answer uncertainty arising from heterogeneous reasoning discrepancies:

$$\mathcal{L}_{OP} = \underbrace{-\frac{1}{T}\sum_{t=1}^{T} \log \pi_\theta(o_t | o_{<t}, x)}_{\text{NLL Loss}} + \lambda \cdot \underbrace{\frac{1}{|\mathcal{N}|}\sum_{t \in \mathcal{N}} H_t}_{\text{Entropy Minimization}}$$

where $\mathcal{N} = \{t | t > T_{prompt+think}\}$ restricts the entropy computation to answer tokens only, and $H_t$ denotes token-level entropy. $\lambda = 0.5$; when answer uncertainty $u > 0.75$, $\lambda = 0$ to prevent overconfidence from degrading generalization.

Design motivation: While Step-RR optimizes the reasoning process, audio and visual reasoning may generate conflicting signals. Ans-CO reduces the entropy of answer prediction to ensure determinism in the final response.

### Loss & Training

- **Base model**: Qwen2.5-Omni-3B
- **Reference model**: Large Audio Language Model (LALM, e.g., Qwen-Audio)
- **Warm-up**: SFT on 100 high-quality Q&A pairs using LLaMA-Factory
- **Semantic evaluation model**: Qwen3 Embedding-0.6B
- **Hardware**: 8 × NVIDIA A800 GPUs
- **Few-shot learning**: Step-RR and Ans-CO use only a small subset of training samples from Music-AVQA and AVQA

## Key Experimental Results

### Main Results

**Results on Music-AVQA and AVQA**:

| Method | Exist | Localis | Count | Comp | Temp | Avg | AVQA Avg |
|--------|-------|---------|-------|------|------|-----|----------|
| PSTP-Net (specialized) | 76.18 | 73.23 | 71.80 | 71.79 | 69.00 | 72.57 | 90.20 |
| CAD (specialized) | 83.42 | 73.97 | 76.37 | 74.88 | 76.16 | 76.96 | 92.20 |
| Qwen2.5-Omni-3B | 60.02 | 53.84 | 61.29 | 58.16 | 46.57 | 54.95 | 83.78 |
| + SFT | 73.67 | 74.09 | 75.43 | 68.47 | 60.44 | 70.41 | 90.41 |
| + GRPO | 77.69 | 71.10 | 67.33 | 64.23 | 70.14 | 70.05 | 85.31 |
| + **RL-CoMM** | **85.61** | **76.68** | **84.08** | **70.74** | **76.30** | **79.46** | **95.87** |
| Δ (vs baseline) | +25.59 | +22.84 | +22.79 | +12.58 | +29.73 | **+24.51** | +12.09 |

**AVHBench hallucination evaluation results**:

| Method | Audio-driven Visual Hallucination Acc | Video-driven Audio Hallucination Acc | Audio-Visual Matching Acc |
|--------|--------------------------------------|-------------------------------------|--------------------------|
| OneLLM | 53.7 | 44.3 | 60.1 |
| Qwen2.5-Omni-3B | 65.85 | 59.65 | 48.77 |
| + GRPO | 72.98 | 62.84 | 49.73 |
| + **RL-CoMM** | **78.96** | **65.63** | **51.85** |

### Ablation Study

**Training strategy comparison on AV-ConfuseBench**:

| Method | Audio-muted Acc ↑ | Yes Rate ↓ | Audio-modified A-Acc ↑ | V-Acc ↑ |
|--------|------------------|-----------|----------------------|---------|
| Qwen2.5-Omni-3B baseline | 8.22 | 91.78% | 1.14 | 4.10 |
| + SFT | 5.48 (worse!) | 94.52% | — | — |
| + GRPO | 15.07 | 84.93% | 1.84 | 4.47 |
| + **RL-CoMM** | **27.40** | **72.60%** | **2.36** | **4.54** |

**Component ablation of Step-RR and Ans-CO** (average accuracy on Music-AVQA):

| Configuration | Average Accuracy |
|---------------|-----------------|
| Qwen2.5-Omni-3B baseline | 54.95 |
| + Format + Accuracy reward | 70.05 |
| + Format + Step-wise Reasoning reward | 74.49 |
| + Format + Step-wise Reasoning + **Ans-CO** | **79.46** |

### Key Findings

1. **SFT is counterproductive**: On AV-ConfuseBench, SFT reduces accuracy from 8.22% to 5.48% — SFT reinforces the visual-audio binding and exacerbates confusion.
2. **RL-based reasoning training substantially outperforms SFT**: Forcing the model to "reflect and explore" during training mimics human reasoning to address challenging audio-visual tasks.
3. **RL-CoMM achieves substantial gains**: +24.51% over the baseline on Music-AVQA and +19.18% on AV-ConfuseBench.
4. **Step-RR outperforms simple accuracy reward**: 74.49% vs. 70.05%, demonstrating that stepwise reasoning rewards effectively correct visual bias.
5. **Challenges of heterogeneous reference models**: The ARR reward exhibits high variance during training with a lower-than-expected peak, indicating that the Omni-LLM still struggles to perform pure audio reasoning without visual interference.
6. **Complementary role of Ans-CO**: An additional 5 percentage point gain beyond reasoning rewards demonstrates that reasoning optimization and answer optimization are decoupled yet mutually complementary.

## Highlights & Insights

1. **The discovery of "audio-visual confusion" is highly valuable**: It reveals a fundamental cognitive deficiency in MLLMs — visual dominance leading to "blindness" in audio perception.
2. **Clever RL design through heterogeneous model collaboration**: Pure audio model reasoning is used as reference to correct the visual bias of Omni models.
3. **Counter-intuitive finding that SFT is harmful**: This reinforces the insight that "synchronized training ≠ independent perception."
4. **Justified removal of KL penalty**: KL divergence between heterogeneous models is semantically meaningless, motivating careful reconsideration of default components in RL frameworks.
5. **Complete experimental design (muted/modified settings)**: Separately evaluates the ability to detect missing audio and to balance conflicting audio-visual information.

## Limitations & Future Work

1. **Small base model scale**: Only Qwen2.5-Omni-3B is used; larger models may alleviate some issues.
2. **Limited scale of AV-ConfuseBench**: Only 73+100 samples, which may be insufficient for comprehensive evaluation.
3. **Suboptimal performance on audio-visual matching**: RL-CoMM achieves limited improvement on this task; the reward model for audio-visual composition requires further refinement.
4. **Restricted to music/instrument scenarios**: Broader audio-visual contexts (speech, ambient sounds, etc.) remain untested.
5. **Warm-up data construction**: The construction criteria and reproducibility of the 100 high-quality Q&A pairs require more detailed documentation.

## Related Work & Insights

- **Qwen2.5-Omni / Gemini 2.5**: State-of-the-art omni-modal large models.
- **AVHBench**: Audio-visual hallucination evaluation benchmark.
- **GRPO** (DeepSeek R1): Critic-free RL optimization framework.
- **Music-AVQA / AVQA**: Standard audio-visual question answering benchmarks.
- **Entropy-based metric**: A method for quantifying model prediction uncertainty.
- Insight: "Modality dominance" is a pervasive problem in multimodal models (vision typically dominates audio/text), necessitating explicit modality disentanglement mechanisms during training.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Both the discovery of audio-visual confusion and the heterogeneous collaborative RL framework exhibit strong originality.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four benchmarks (Music-AVQA / AVQA / AVHBench / AV-ConfuseBench) with thorough ablations, though AV-ConfuseBench is limited in scale.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clearly presented, method description is largely complete, and mathematical notation is accurate.
- Value: ⭐⭐⭐⭐⭐ — Reveals a fundamental deficiency in multimodal LLMs and opens a new direction for reliability research in multimodal reasoning.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval](../../ACL2026/multimodal_vlm/omni-embed-audio_leveraging_multimodal_llms_for_robust_audio-text_retrieval.md)
- [\[AAAI 2026\] AbductiveMLLM: Boosting Visual Abductive Reasoning Within MLLMs](abductivemllm_boosting_visual_abductive_reasoning_within_mll.md)
- [\[CVPR 2026\] When to Think and When to Look: Uncertainty-Guided Lookback](../../CVPR2026/multimodal_vlm/when_to_think_and_when_to_look_uncertainty-guided_lookback.md)
- [\[AAAI 2026\] Explore How to Inject Beneficial Noise in MLLMs](explore_how_to_inject_beneficial_noise_in_mllms.md)
- [\[NeurIPS 2025\] Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM](../../NeurIPS2025/multimodal_vlm/watch_and_listen_understanding_audio-visual-speech_moments_with_multimodal_llm.md)

<!-- RELATED:END -->
