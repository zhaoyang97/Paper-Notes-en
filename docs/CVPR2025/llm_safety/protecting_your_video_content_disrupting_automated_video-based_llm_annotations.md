---
title: >-
  [Paper Note] Protecting Your Video Content: Disrupting Automated Video-Based LLM Annotations
description: >-
  [CVPR 2025][LLM Safety][Video Privacy Protection] This paper proposes two types of adversarial video watermarking methods—Ramblings (which induce video LLMs to generate incorrect descriptions) and Mutes (which induce video LLMs to generate extremely short or empty descriptions)—to protect personal videos from unauthorized automated annotation via imperceptible adversarial perturbations. It also demonstrates that these low-quality annotations degrade the performance of downstr…
tags:
  - "CVPR 2025"
  - "LLM Safety"
  - "Video Privacy Protection"
  - "Adversarial Watermarking"
  - "Video LLM"
  - "Adversarial Attack"
  - "Content Protection"
date: 2026-05-08
content_hash: b3a2903d18047a0a
---

# Protecting Your Video Content: Disrupting Automated Video-Based LLM Annotations

**Conference**: CVPR 2025  
**arXiv**: [2503.21824](https://arxiv.org/abs/2503.21824)  
**Code**: [https://github.com/ttthhl/Protecting_Your_Video_Content](https://github.com/ttthhl/Protecting_Your_Video_Content)  
**Area**: Video Generation  
**Keywords**: Video Privacy Protection, Adversarial Watermarking, Video LLM, Adversarial Attack, Content Protection

## TL;DR

This paper proposes two types of adversarial video watermarking methods—Ramblings (which induce video LLMs to generate incorrect descriptions) and Mutes (which induce video LLMs to generate extremely short or empty descriptions)—to protect personal videos from unauthorized automated annotation via imperceptible adversarial perturbations. It also demonstrates that these low-quality annotations degrade the performance of downstream text-to-video generation models.

## Background & Motivation

**Background**: Video LLMs (e.g., Video-ChatGPT, Video-LLaMA) have made significant progress in video understanding and automatic annotation, enabling the generation of high-quality dense descriptions for unlabeled videos. These annotated video-text pairs are subsequently used to fine-tune text-to-video generation models (e.g., AnimateDiff), forming an automated pipeline of "Video LLM Annotation $\to$ T2V Training".

**Limitations of Prior Work**: This automated pipeline brings serious privacy and security concerns. Massive amounts of personal videos on multimedia platforms may be automatically annotated without authorization using video LLMs, and the generated video-text pairs are used to train downstream models. Users' personal content is exploited without their knowledge. Currently, there are almost no protection mechanisms against this threat.

**Key Challenge**: The powerful understanding capability of video LLMs is a double-edged sword—it serves legitimate purposes but also facilitates unauthorized data mining. How can one render videos "invalid" for automatic LLM annotation without compromising the human viewing experience?

**Goal**: Design imperceptible adversarial perturbations (i.e., protective watermarks) added to video frames, such that video LLMs processing these videos either generate completely incorrect descriptions (Ramblings) or generate almost no description at all (Mutes), thereby protecting video content.

**Key Insight**: Reposition the adversarial attack paradigm (typically viewed as a security threat) as a privacy defense tool. Leverage PGD optimization to generate adversarial perturbations under $l_\infty$ constraints to attack the visual encoder or logit outputs of video LLMs.

**Core Idea**: Design two pairs of complementary perturbation strategies: feature-level/logit-level Ramblings to deviate generated content from the ground truth, and EOS token probability manipulation Mutes to force early termination or completely empty generation.

## Method

### Overall Architecture

Given an original video $\boldsymbol{x}$, the objective is to optimize a perturbation $\delta$ under an $l_\infty$-norm constraint ($\|\delta\|_\infty < \epsilon$, where $\epsilon = 16/255$), such that the adversarial video $\boldsymbol{x}' = \boldsymbol{x} + \delta$ produces abnormal outputs when processed by video LLMs. PGD iterative optimization is used, with 200 iterations for Ramblings and 500 iterations for Mutes.

The processing pipeline of video LLMs is: visual encoder $g(\cdot)$ extracts video features $\to$ LLM $h(\cdot)$ combines text prompts to generate hidden states $\to$ Softmax layer produces token probability distributions $\to$ autoregressive text generation. The four attack methods target different positions along this pipeline.

### Key Designs

1. **Rambling-F (Feature-level Misdirection)**:

    - **Function**: Shift video representations in the feature space, preventing the LLM from correctly associating video content.
    - **Mechanism**: Maximize the $l_2$ distance between the adversarial video and the original video at two levels: the video features $\mathcal{L}_{video}$ output by the visual encoder, and the LLM features $\mathcal{L}_{LLM}$ at the LLM hidden layer. The total loss is $\mathcal{L}_{RF} = \alpha \cdot \mathcal{L}_{video} + \beta \cdot \mathcal{L}_{LLM}$.
    - **Design Motivation**: Attacking features at both levels is more effective than attacking only one, as shifts in the visual encoder might be partially corrected by the LLM.

2. **Rambling-L (Logit-level Misdirection)**:

    - **Function**: Directly cause the generated content to deviate from the correct description at the output distribution level.
    - **Mechanism**: Maximize the autoregressive loss between the adversarial video and the original correct description $y$: $\mathcal{L}_{RL} = \mathcal{L}_{ar}(\mathcal{F}(\boldsymbol{x}', c_{in}), y)$. By increasing this loss, the token probability distribution of the model is pushed away from the ground-truth sequence.
    - **Design Motivation**: More direct than feature-level attacks, as it directly optimizes the degradation of output quality.

3. **Mute-S (Short Text Output)**:

    - **Function**: Induce the video LLM to terminate generation prematurely, producing short and useless text snippets.
    - **Mechanism**: Maximize the average probability of the EOS token across all positions: $\mathcal{L}_{MS} = \frac{1}{N} \sum_{i=1}^{N} f_i^{\text{EOS}}(\boldsymbol{x}', c_{in} \oplus y_{out})$. Note that $y_{out}$ is updated in each iteration, allowing the optimization to track the current generation state.
    - **Design Motivation**: Increasing the EOS probability at all positions is an untargeted attack, requiring no precise control over the termination position, making it easier to optimize.

4. **Mute-N (Null Output)**:

    - **Function**: Induce the video LLM to output EOS at the very first token, achieving completely empty generation.
    - **Mechanism**: Minimize the autoregressive loss to force the model to output EOS as the first token with high confidence: $\mathcal{L}_{MN} = -\mathcal{L}_{ar}(\mathcal{F}(\boldsymbol{x}', c_{in}), [\text{EOS}])$.
    - **Design Motivation**: This represents the most extreme protection—zero information leakage. As a targeted attack, the optimization goal is highly explicit.

### Loss & Training

The four methods share the PGD optimization framework, with a step size of $1/255$ and a perturbation bound of $\epsilon = 16/255$. The inference temperature is set to 0.2 for all models, with a maximum output of 512 tokens. Assuming white-box attacks (with access to model parameters), transfer attacks under black-box scenarios are also discussed in the appendix.

## Key Experimental Results

### Main Results

**Ramblings Performance (CLIP Score / BLEU, lower is better, indicating more effective protection):**

| Base Model | Method | OpenVid-1M CLIP↓ | MSR-VTT CLIP↓ | WebVid-10M CLIP↓ |
|---------|------|----------|----------|----------|
| Video-ChatGPT | Original | 0.762 | 0.674 | 0.609 |
| Video-ChatGPT | Rambling-F | 0.668 | 0.604 | 0.496 |
| Video-ChatGPT | Rambling-L | **0.627** | **0.603** | **0.483** |
| Video-LLaMA | Original | 0.788 | 0.624 | 0.609 |
| Video-LLaMA | Rambling-F | **0.583** | **0.493** | **0.429** |

**Mutes Performance (text length and EOS rate):**

| Base Model | Method | OpenVid-1M Length↓ | EOS Rate↑ | MSR-VTT Length↓ | EOS Rate↑ |
|---------|------|----------|----------|----------|----------|
| Video-LLaMA | Original | 203.5 | 0% | 208.6 | 0% |
| Video-LLaMA | Mute-S | 11.6 | 7% | 17.3 | 13% |
| Video-LLaMA | Mute-N | **0.0** | **100%** | **0.0** | **100%** |
| Video-ChatGPT | Original | 30.5 | 0% | 30.3 | 0% |
| Video-ChatGPT | Mute-N | **0.0** | **100%** | **0.0** | **100%** |

### Ablation Study

**Prompt Transferability (Video-LLaMA, OpenVid-1M):**

| Method | Attack Prompt | Test Prompt 1 CLIP↓ | Test Prompt 2 CLIP↓ | Test Prompt 3 CLIP↓ |
|------|-------------|----------|----------|----------|
| Rambling-F | "What is this video about?" | 0.583 | 0.600 | 0.607 |
| Rambling-L | Same as above | 0.609 | 0.613 | 0.625 |
| Mute-N | Same as above | Length 0 / EOS 100% | Length 64 / EOS 73% | Length 231 / EOS 19% |

### Key Findings

- **Random noise is completely ineffective**: Random perturbations have almost no impact on annotation quality (CLIP Score remains comparable to original videos), demonstrating that gradient-optimized adversarial perturbations are necessary.
- **Mute-N achieves striking performance**: It achieves a 100% EOS rate (completely silent output) across all models and datasets, serving as the strongest protection mechanism.
- **Mute-S compresses text length by nearly 20 times on Video-LLaMA** (from 203.5 to 11.6), showing significant efficacy.
- **Prompt transferability exists but is limited**: Ramblings transfer well across different prompts (with a stable decline in CLIP Score), but Mute-N is sensitive to different prompts (longer prompts drop the EOS rate from 100% to 19%).
- **Verification of downstream impact**: Fine-tuning AnimateDiff with annotations from protected videos leads to a significant decrease in video generation quality (VQAA, VQAT), proving the effectiveness of end-to-end protection.

## Highlights & Insights

- **Viewing adversarial attacks as a privacy protection tool** is a clever paradigm shift. While adversarial attacks are traditionally regarded as security threats, this paper utilizes them for defense.
- **The four methods cover different protection intensities**: From feature shift (mild) to complete silence (extreme), allowing users to choose based on their scenarios. The Rambling series is suitable for scenarios where "one wants attackers to receive incorrect data", whereas the Mute series is suitable for scenarios where "one wants to leak absolutely no information".
- **End-to-end evaluation** up to the downstream T2V model. It demonstrates not only that annotations are disrupted, but also that the disrupted annotations indeed impair downstream tasks.

## Limitations & Future Work

- **White-box assumption**: Primary experiments assume full access to the video LLM parameters, whereas in real scenarios, many models are deployed as black-box APIs.
- **Prompt sensitivity**: Mute-N exhibits limited transferability across different evaluation prompts; attackers might bypass protection by changing the phrasing of queries.
- **Model transferability**: The paper does not thoroughly evaluate whether perturbations optimized on one model can transfer to models with completely different architectures.
- **Perturbation detection**: Although $\epsilon = 16/255$ is imperceptible to the human eye, it could potentially be flagged by automated detectors.
- **Ethical duality**: This technology could also be exploited for malicious purposes, such as disrupting legitimate video analysis. Ethical usage guidelines need to be established.
- Future work can explore more robust black-box attacks and cross-model transfer attacks.

## Related Work & Insights

- **vs. Image Adversarial Attacks**: This work extends PGD attacks from the image domain to the video LLM scenario. The core difference lies in transitioning the attack target from the decision boundary of a classifier to the generation process of an autoregressive model.
- **vs. Traditional Watermarking**: Traditional digital watermarks (e.g., embedded in DWT or DCT domains) aim for copyright tracking rather than functional disruption. The "watermark" in this paper is functional—directly destroying the processing capability of models.
- **vs. Jailbreak Attacks**: Jailbreak attacks (e.g., visual adversarial examples) induce LLMs to generate harmful content, whereas the objective here is the opposite—preventing the model from generating useful content.

## Rating

- Novelty: ⭐⭐⭐⭐ Using adversarial attacks for video privacy protection is a novel application direction, and the design of the four methods is well-structured.
- Experimental Thoroughness: ⭐⭐⭐⭐ Highly comprehensive coverage using three models and three datasets, and downstream verification is convincing. However, it lacks cross-model transferability and robustness analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-defined problem formulation.
- Value: ⭐⭐⭐⭐ Addresses an important and timely issue, though its utility is constrained by the white-box assumption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Evaluation of Vision-LLMs in Surveillance Video](../../NeurIPS2025/llm_safety/evaluation_of_vision-llms_in_surveillance_video.md)
- [\[NeurIPS 2025\] VMDT: Decoding the Trustworthiness of Video Foundation Models](../../NeurIPS2025/llm_safety/vmdt_decoding_the_trustworthiness_of_video_foundation_models.md)
- [\[ICCV 2025\] Temporal Unlearnable Examples: Preventing Personal Video Data from Unauthorized Exploitation](../../ICCV2025/llm_safety/temporal_unlearnable_examples_preventing_personal_video_data_from_unauthorized_e.md)
- [\[ICLR 2026\] ExpGuard: LLM Content Moderation in Specialized Domains](../../ICLR2026/llm_safety/expguard_llm_content_moderation_in_specialized_domains.md)
- [\[CVPR 2025\] Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks](steering_away_from_harm_an_adaptive_approach_to_defending_vision_language_model_.md)

</div>

<!-- RELATED:END -->
