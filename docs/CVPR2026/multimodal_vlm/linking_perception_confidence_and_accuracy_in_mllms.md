---
title: >-
  [Paper Note] Linking Perception, Confidence and Accuracy in MLLMs
description: >-
  [CVPR 2026][Multimodal VLM][Multimodal Large Language Models] This paper reveals a severe confidence miscalibration problem in MLLMs—accuracy drops sharply when visual inputs are degraded while confidence remains unchanged—and proposes CDRL (Confidence-Driven Reinforcement Learning with clean-noisy image pairs) for perception-sensitive training. The calibrated confidence is then leveraged for adaptive test-time scaling via CA-TTS, achieving an average improvement of 8.8% across four benchmarks.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Multimodal Large Language Models
  - Confidence Calibration
  - Reinforcement Learning
  - Test-Time Scaling
  - Visual Perception
date: 2026-05-08
content_hash: 2bccda9ba54e2cc9
---

# Linking Perception, Confidence and Accuracy in MLLMs

**Conference**: CVPR 2026
**arXiv**: [2603.12149](https://arxiv.org/abs/2603.12149)
**Code**: [https://github.com/anotherbricki/CA-TTS](https://github.com/anotherbricki/CA-TTS)
**Area**: Multimodal VLM
**Keywords**: Multimodal Large Language Models, Confidence Calibration, Reinforcement Learning, Test-Time Scaling, Visual Perception

## TL;DR
This paper reveals a severe confidence miscalibration problem in MLLMs—accuracy drops sharply when visual inputs are degraded while confidence remains unchanged—and proposes CDRL (Confidence-Driven Reinforcement Learning with clean-noisy image pairs) for perception-sensitive training. The calibrated confidence is then leveraged for adaptive test-time scaling via CA-TTS, achieving an average improvement of 8.8% across four benchmarks.

## Background & Motivation
Recent MLLM research has focused primarily on enhancing visual perception to improve accuracy, while a critical question has been overlooked: **does the model know when it does not know?**

The authors design a probing experiment in which noise is progressively added to key visual evidence while simultaneously monitoring model confidence and accuracy. The results reveal that confidence remains nearly unchanged while accuracy drops substantially, exposing a severe confidence miscalibration in MLLMs—models maintain high confidence even when visual perception degrades significantly.

Existing confidence calibration methods for LLMs operate at the token level, whereas visual perception in MLLMs is global in nature (spanning the entire response), creating a granularity mismatch. Furthermore, LLM calibration methods do not account for the influence of the visual component on calibration.

The core ideas are: (1) train with RL on clean-noisy image pairs, rewarding confidence discrepancy to enhance perceptual sensitivity while aligning accuracy with confidence for calibration; (2) the resulting calibrated confidence naturally serves as a routing signal for test-time scaling—a "free lunch," since calibration itself endows the model with TTS capability.

## Method

### Overall Architecture
The framework consists of two stages: (1) **CDRL training**—the model is trained with GRPO on clean-noisy image pairs to enhance perceptual sensitivity and calibrate confidence; (2) **CA-TTS inference**—the calibrated confidence signal is used to adaptively schedule three decoupled reasoning modules (Self-Consistency, Self-Reflection, and Self-Check), coordinated by an Expert Model serving as Planner, Voter, and Critic.

### Key Designs

1. **Confidence-Driven Reinforcement Learning (CDRL)**:

    - **Function**: Enhance the perceptual sensitivity of MLLMs (i.e., responsiveness to visual degradation) and calibrate confidence (high confidence when correct, low confidence when incorrect).
    - **Mechanism**: CLIP attention maps are used to identify key visual regions, which are then corrupted to generate image pairs $(i, i')$. Confidence is defined as the Negative Mean Log-Probability: $C = \frac{1}{T}\sum_{t=1}^T \text{Conf}_{\text{token}_t}$, $\text{Conf}_{\text{token}} = -\frac{1}{k}\sum_{i=1}^k \log p_{(i)}$. The confidence calibration reward is: $R_{\text{Conf},j} = \underbrace{\alpha \tanh(\beta \cdot \Delta C)}_{\text{Perception Term}} + \underbrace{(2 \cdot R_{\text{Output},j} - 1) \cdot C_j^{norm}}_{\text{Calibration Term}}$
    - **Design Motivation**: The Perception Term rewards confidence discrepancy between the clean and noisy images ($\Delta C = C_j - C_j'$), encouraging the model to be sensitive to visual degradation. The Calibration Term rewards high confidence when the answer is correct (+$C_j$) and penalizes high confidence when incorrect (−$C_j$), achieving accuracy-confidence alignment.

2. **Self-Consistency**:

    - **Function**: Sample multiple responses, then obtain a robust answer via confidence-weighted voting combined with external calibration from an Expert Model.
    - **Mechanism**: $V_{internal}[k] = \sum_{i=1}^n C_i \cdot \mathbb{I}(A_i = k)$ computes the internal confidence-weighted vote. The Expert Model (Voter) provides external confidence $C_{expert}$ for each candidate, and the final vote is $V_{final}[k] = V_{internal}^{norm}[k] + \tau_1 \cdot c_k$.
    - **Design Motivation**: Compared to plain majority voting, confidence-weighted voting gives greater weight to high-confidence correct answers, while the Expert Model provides independent external verification.

3. **Self-Reflection**:

    - **Function**: The Expert Model acts as a Critic to generate critiques of the question, guiding the base model to reconsider its response.
    - **Mechanism**: $Crit = M_{expert}^{Critic}(i, q, P_{critique})$, $(CoT_{reflect}, A_{reflect}) = M_{base}(i, q, Crit)$; the reflected answer is incorporated into the final vote with weight $\tau_2$.
    - **Design Motivation**: Low-confidence predictions can be corrected through externally guided reflection.

4. **Self-Check**:

    - **Function**: Perform self-checking at the visual level using Visual Contrastive Decoding (VCD) to contrast outputs from clean and noisy images.
    - **Mechanism**: $\log P_{VCD}(y|i,q) = (1+\alpha) \cdot \log P_\theta(y|i,q) - \alpha \cdot \log P_\theta(y|i',q)$; the contrastively decoded answer is incorporated into the final vote with weight $\tau_3$.
    - **Design Motivation**: Visual-level verification exploits the contrast between "spurious confidence" on noisy images and "true signal" from clean images to highlight reliable visual reasoning.

### Loss & Training
GRPO training is employed with total reward $r_j = R_{\text{Conf},j} + R_{\text{Output},j} + R_{\text{Format},j}$. The base model is Qwen2.5-VL-7B-Instruct, fine-tuned with full parameters on 8×H100 GPUs using 1,936 training samples. The Expert Model is Gemini-2.5-Pro.

## Key Experimental Results

### Main Results

| Method | Math-Vista | Math-Vision | MMStar | MMMU |
|--------|-----------|------------|--------|------|
| Pass@1 (base) | 64.7 | 23.0 | 60.2 | 48.8 |
| Majority Voting | 69.8 | 30.1 | 69.0 | 57.5 |
| VL-Rethinker | 74.1 | 30.7 | 63.4 | 55.6 |
| We-Think | 73.3 | 29.7 | 65.1 | 55.7 |
| **Ours (CDRL+CA-TTS)** | **79.5** | **42.4** | **71.3** | **66.3** |

### Ablation Study

| Configuration | Math-Vision ALL | Note |
|---------------|----------------|------|
| Training-Free (Pass@1) | 22.96 | Baseline |
| CDRL only | 26.38 | Better model state after calibration |
| CA-TTS only | 37.99 | TTS framework yields significant gains |
| CDRL + CA-TTS | **42.35** | Synergistic combination, best performance |

### Key Findings
- After CDRL training, the model's confidence drop under visual perturbation increases by 4–8× (e.g., Noised: −0.32 → −1.39), demonstrating that the model genuinely "knows when it does not know."
- The CA-TTS scaling slope $\beta_1 = 3.65$ is 2.2× that of Majority Voting (1.64) and 3.1× that of DeepConf (1.19), confirming that calibrated confidence enables more efficient TTS.
- Using Qwen2.5-VL-7B itself as the Expert Model still yields substantial gains over Majority Voting, demonstrating that the approach does not depend on an exceptionally strong Expert.
- On MMMU, the proposed method achieves 66.3% vs. VL-Rethinker's 55.6%, a gain of 10.7 percentage points.

## Highlights & Insights
- The probing experiment ("does the model know when it does not know?") intuitively and powerfully exposes a fundamental deficiency in MLLMs.
- The dual-term reward design in CDRL is elegant: the Perception Term uses image pairs to enhance sensitivity, while the Calibration Term aligns confidence with accuracy.
- "Calibrated confidence as a free lunch"—calibration training directly translates into TTS capability at inference time, with no additional cost.
- The three CA-TTS modules are fully decoupled and order-agnostic, each contributing only votes, yielding a flexible and robust architecture.

## Limitations & Future Work
- CA-TTS relies on an Expert Model (e.g., Gemini-2.5-Pro), introducing external API costs and latency.
- The training set consists of only 1,936 samples; scaling up may further improve calibration quality.
- The VCD in Self-Check requires additional inference over noisy images, increasing inference overhead.
- The voting weights for the three modules are fixed at $\tau_1 = \tau_2 = \tau_3 = 0.5$; adaptive weight assignment may yield better performance.

## Related Work & Insights
- DeepConf applies confidence for TTS but is limited to mathematical reasoning and lacks calibration training; this paper complements it by incorporating the training phase.
- VCD was originally proposed to mitigate hallucinations; this paper integrates it into the TTS framework as a visual self-check module.
- Compared to tree-search methods such as ToT, CA-TTS's decoupled multi-stage verification is more robust and avoids single-point failures.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First systematic study of visual perception–confidence calibration in MLLMs; the CDRL+CA-TTS framework is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four benchmarks, extensive ablations, scaling curve analysis, sensitivity experiments, and case studies are all well-executed.
- **Writing Quality**: ⭐⭐⭐⭐ The probing experiment provides a compelling introduction; the framework description is clear and well-organized.
- **Value**: ⭐⭐⭐⭐⭐ Identifies a fundamental problem in MLLMs and provides a systematic solution; the average 8.8% improvement is highly significant.

## Key Terminology
- **NMLP (Negative Mean Log-Probability)**: A sequence-level confidence measure; lower values indicate higher certainty.
- **Perceptual Bluntness**: The phenomenon in which a model is insensitive to degradation of visual inputs.
- **VCD (Visual Contrastive Decoding)**: Decoding that exploits the logit difference between clean and noisy images.
- **Free Lunch**: The TTS capability gain obtained at no additional cost as a byproduct of calibration training.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] CodePercept: Code-Grounded Visual STEM Perception for MLLMs](codepercept_code-grounded_visual_stem_perception_for_mllms.md)
- [\[CVPR 2026\] LFPC: Learning to Focus and Precise Cropping for MLLMs](lfpc_learning_to_focus_and_precise_cropping_for_mllms.md)
- [\[CVPR 2026\] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](cropvlm_learning_to_zoom_for_fine_grained_vision_language_perception.md)
- [\[NeurIPS 2025\] Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs](../../NeurIPS2025/multimodal_vlm/struct2d_a_perception-guided_framework_for_spatial_reasoning_in_mllms.md)
- [\[CVPR 2026\] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](mmrad_multimodal_anomaly_detection.md)

<!-- RELATED:END -->
