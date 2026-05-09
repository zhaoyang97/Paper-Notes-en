---
title: >-
  [Paper Note] Training-free Online Video Step Grounding
description: >-
  [NeurIPS 2025][Multimodal VLM][Video step grounding] This paper proposes BaGLM, a training-free online video step grounding method that integrates LLM-estimated step dependencies and LMM-estimated step progress into zero-shot LMM predictions via Bayesian filtering, outperforming existing trained offline methods on three datasets.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Video step grounding
  - Bayesian filtering
  - LMM zero-shot
  - online inference
  - training-free
date: 2026-05-08
content_hash: 7b5dffed42f894fa
---

# Training-free Online Video Step Grounding

**Conference**: NeurIPS 2025
**arXiv**: [2510.16989](https://arxiv.org/abs/2510.16989)
**Code**: [GitHub](https://lucazanella.github.io/baglm/)
**Area**: Multimodal VLM
**Keywords**: Video step grounding, Bayesian filtering, LMM zero-shot, online inference, training-free

## TL;DR

This paper proposes BaGLM, a training-free online video step grounding method that integrates LLM-estimated step dependencies and LMM-estimated step progress into zero-shot LMM predictions via Bayesian filtering, outperforming existing trained offline methods on three datasets.

## Background & Motivation

Video Step Grounding (VSG) aims to identify which steps are executed in a video, given a set of procedural step descriptions. This has significant applications in real-time AR/XR guidance (e.g., cooking assistance, furniture assembly).

Existing VSG methods face two core limitations:

**Training data dependency**: They require collected and annotated training data (e.g., narration text from HowTo100M), incurring high annotation costs and potentially biasing models toward specific video and task distributions.

**Offline processing requirement**: They assume access to the complete video, making them unsuitable for real-time video stream scenarios.

The central question explored in this paper is: **Can VSG be performed online without any training?** The authors first make a surprising finding — directly applying zero-shot LMMs to predict steps segment-by-segment **already surpasses trained offline SOTA methods** (InternVL2.5-8B exceeds MPTVA by 6.4% on CrossTask and NaSVA by 16.1% on Ego4D GoalStep). This motivates a natural improvement direction: can information from past frames be injected into LMM predictions while preserving the zero-shot advantage?

## Method

### Overall Architecture

BaGLM formulates VSG as a Bayesian filtering problem: the state is the step $a$ corresponding to the current segment, and the observation is the current video segment $\mathbf{S}_t$. The posterior belief $\text{bel}_t(a)$ is recursively estimated through a predict step (leveraging inter-step transition relations) and an update step (leveraging LMM observation predictions).

### Key Designs

1. **LMM as a Zero-Shot Observation Model**

   VSG is reformulated as a multiple-choice question: given the current video segment $\mathbf{S}_t$ and all step options (including "none"), the LMM is prompted to output probabilities for each option. After normalization, a step probability distribution $f_{\text{LMM}}^{\text{VSG}}: \mathcal{V} \times \mathcal{A} \to \Delta^{K+1}$ is obtained.

   InternVL2.5-8B is used, with only the current 2-second segment and the step list as input — no full-video access is required.

2. **PREDICT Step: LLM-Driven Step Transition Model**

   An LLM (LLaMA3-70B-Instruct) is used to estimate an inter-step dependency matrix $\mathbf{D} \in \mathbb{R}^{K \times K}$, where $\mathbf{D}_{i,j}$ denotes the probability that step $a_j$ is a prerequisite of step $a_i$. The transition matrix is initialized as $\mathbf{T} = \mathbf{D}^\top$.

   A key innovation is dynamically adjusting the transition matrix based on **step progress**. Two metrics are introduced:

   - **Readiness**: the degree to which prerequisites of step $a_i$ have been completed:
   $\mathbf{r}_t[i] = \frac{\sum_j \mathbf{D}_{i,j} \cdot \max_{\tau < t} \text{progress}_\tau[j]}{\sum_j \mathbf{D}_{i,j}}$

   - **Validity**: whether the successors of step $a_i$ have not yet been executed (preventing repeated attribution):
   $\mathbf{v}_t[i] = \frac{\sum_j \mathbf{D}_{j,i} \cdot (1 - \max_{\tau < t} \text{progress}_\tau[j])}{\sum_j \mathbf{D}_{j,i}}$

   The adjusted transition matrix is:
   $\tilde{\mathbf{T}}_t[i,j] = \frac{\mathbf{T}[i,j] \cdot \mathbf{r}_t[j] \cdot \mathbf{v}_t[j]}{\sum_k \mathbf{T}[i,k] \cdot \mathbf{r}_t[k] \cdot \mathbf{v}_t[k]}$

   Step progress is estimated by querying the LMM for the completion degree of each step on a 0–9 scale.

3. **UPDATE Step: Fusing LMM Predictions with the Bayesian Prior**

   The final belief is obtained as the product of the LMM observation likelihood and the predict-step prior:

   $\text{bel}_t(a_i) = \frac{1}{\mathcal{Z}} \cdot f_{\text{LMM}}(\mathbf{S}_t, \pi_{\text{VSG}})[i] \cdot \sum_{a_j \in \mathcal{A}} \tilde{\mathbf{T}}_t[j,i] \cdot \text{bel}_{t-1}(a_j)$

   where $\mathcal{Z}$ is a normalization factor. This formulation elegantly integrates instantaneous LMM predictions with historically accumulated beliefs and inter-step dependencies.

### Loss & Training

**No training is required.** All components rely on the zero-shot capabilities of pretrained models: InternVL2.5-8B for observation and progress estimation, and LLaMA3-70B for dependency matrix extraction. Videos are segmented into non-overlapping 2-second clips.

## Key Experimental Results

### Main Results

| Method | Setting | HT-Step R@1 | CrossTask Avg.R@1 | Ego4D R@1 |
|--------|---------|-------------|-------------------|-----------|
| VINA | Offline+Trained | 39.1 | 44.8 | - |
| NaSVA | Offline+Trained | 53.1 | 46.7 | 29.1 |
| MPTVA | Offline+Trained | - | 47.9 | - |
| VSLNet | Offline+Trained | - | - | 24.3 |
| NaSVA (online) | Online+Trained | 46.1 | - | 24.2 |
| **BaGLM** | **Online+Training-free** | **57.4** | **59.8** | **43.3** |

BaGLM surpasses NaSVA by **+4.3/+13.1/+14.2%** on HT-Step/CrossTask/Ego4D, respectively.

### Ablation Study: Transition Model Components

| Configuration | HT-Step | CrossTask | Ego4D |
|--------------|---------|-----------|-------|
| Static transition matrix only | 55.9 | 58.0 | 42.1 |
| + Readiness | 57.0 | 58.8 | 42.0 |
| + Validity | 56.4 | 58.8 | 43.1 |
| + Readiness + Validity | **57.4** | **59.8** | **43.3** |

### Oracle Experiments

| Configuration | HT-Step | CrossTask | Ego4D |
|--------------|---------|-----------|-------|
| Estimated dependencies + Estimated progress | 57.4 | 59.8 | 43.3 |
| Oracle dependencies + Oracle progress | **62.6** | **66.9** | **82.2** |

The oracle setting yields a 38.9% improvement on Ego4D, demonstrating that the Bayesian filtering framework itself is highly effective and that improvements in progress and dependency estimation would yield substantial further gains.

### Key Findings

- Zero-shot LMMs observing only the current segment already constitute a strong VSG baseline, surpassing specialized methods trained on HowTo100M.
- A 2-second segment duration is the optimal trade-off: shorter segments lack sufficient visual cues, while longer ones span multiple steps.
- BaGLM consistently improves over all LMMs on HT-Step and CrossTask, but gains are limited on Ego4D GoalStep (longer videos, coarser step descriptions).
- The choice of LLM (LLaMA-3.3-70B vs. GPT-4.1-mini) has minimal impact, indicating robustness to LLM selection.

## Highlights & Insights

- **The combination of Bayesian filtering and LMMs is highly elegant**, seamlessly integrating classical probabilistic inference with modern large model capabilities.
- Fully training-free, online inference, and surpassing trained offline methods — significant advantages for real-world deployment.
- The dynamic transition model based on step progress estimation and the dependency matrix is the key innovation, injecting task-specific knowledge into the Bayesian framework.
- Oracle experiments clearly identify directions for future improvement.

## Limitations & Future Work

- The method relies on the quality of LLM-estimated step dependencies, which may be inaccurate for ambiguous or highly domain-specific steps.
- Step progress estimation depends on the LMM's subjective judgment, limiting precision.
- Improvements are marginal on long videos such as Ego4D (average 28 minutes), indicating that long-range dependency modeling needs to be strengthened.
- Each segment requires two LMM calls (step prediction + progress estimation), and real-time performance is constrained by LMM inference speed.

## Related Work & Insights

- The evolution of the VSG field from weak supervision (Zhukov et al.) to LLM-assisted pseudo-labeling (NaSVA, MPTVA).
- Works such as VQAScore demonstrate that LMMs can replace CLIP for video-language alignment evaluation.
- The classical application of Bayesian filtering in tracking and localization is creatively adapted here for step grounding.
- Inspiration: this framework can be generalized to other sequential video understanding tasks (e.g., action anticipation, procedural anomaly detection).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The training-free online paradigm combining Bayesian filtering with LMMs is entirely original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, four LMMs, ablation studies, oracle analysis, and segment duration analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematical modeling is clear; the progression from preliminary findings to method design is logically coherent.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a new paradigm for training-free online inference in video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection](monitor_exploiting_large_language_models_with_instruction_for_online_video_anoma.md)
- [\[NeurIPS 2025\] See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model](seetrek_training-free_spatial_prompting_for_multimodal_large_language_model.md)
- [\[NeurIPS 2025\] iFinder: Structured Zero-Shot VLM Grounding for Dash-Cam Video Reasoning](ifinder_structured_zero-shot_vision-based_llm_grounding_for_dash-cam_video_reaso.md)
- [\[NeurIPS 2025\] Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?](can_multi-modal_llms_provide_live_step-by-step_task_guidance.md)
- [\[NeurIPS 2025\] Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning](can_llms_reason_over_non-text_modalities_in_a_training-free_manner_a_case_study_.md)

</div>

<!-- RELATED:END -->
