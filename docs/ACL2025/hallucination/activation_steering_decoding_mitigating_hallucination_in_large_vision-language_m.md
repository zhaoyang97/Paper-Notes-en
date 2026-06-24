---
title: >-
  [Paper Note] Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention
description: >-
  [ACL 2025][Hallucination Detection][Hallucination Mitigation] This paper proposes ASD (Activation Steering Decoding), a training-free, inference-time hallucination mitigation method. By identifying hallucination direction patterns within the intermediate hidden states of LVLMs, it leverages bidirectional steering and contrastive decoding to suppress hallucinated outputs while preserving the model's performance on general visual understanding tasks.
tags:
  - "ACL 2025"
  - "Hallucination Detection"
  - "Hallucination Mitigation"
  - "Activation Steering"
  - "Contrastive Decoding"
  - "Training-Free Inference Intervention"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: 6dfc0b91fbf80988
---

# Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention

**Conference**: ACL 2025  
**Code**: None  
**Area**: Hallucination Detection  
**Keywords**: Hallucination Mitigation, Activation Steering, Contrastive Decoding, Training-Free Inference Intervention, Vision-Language Models

## TL;DR

This paper proposes ASD (Activation Steering Decoding), a training-free, inference-time hallucination mitigation method. By identifying hallucination direction patterns within the intermediate hidden states of LVLMs, it leverages bidirectional steering and contrastive decoding to suppress hallucinated outputs while preserving the model's performance on general visual understanding tasks.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) have demonstrated powerful capabilities in multimodal understanding, but they frequently generate "hallucinations" — content that is inconsistent with the input images. This issue severely hinders the deployment of LVLMs in application scenarios requiring high factual accuracy, such as medical image analysis, autonomous driving description, and visual fact-checking.

**Limitations of Prior Work**: Existing hallucination mitigation methods mainly fall into two categories: (1) training-time methods, which require extra high-quality data and substantial computational resources for model fine-tuning (e.g., RLHF, DPO), incurring high costs and altering model weights; (2) inference-time contrastive decoding methods, such as VCD (Visual Contrastive Decoding), which suppress hallucinations by contrasting output distributions between a perturbed version and the original version. However, these methods usually intervene only at the output space (logits-level) without deeply utilizing internal model activation information.

**Key Challenge**: Hallucination is not merely an output-level problem; it is rooted in deviations within the model's internal activation states during inference. Correcting only during the output phase is akin to "treating the symptoms rather than the cause," making it difficult to fundamentally eliminate the source of hallucinations. Furthermore, directly intervening in internal activations faces a key challenge: how to accurately locate the representation pattern of hallucinations within the activation space?

**Goal**: (1) To investigate the representational characteristics of hallucinated content in the activation space of LVLM's intermediate layers; (2) To design a training-free intervention method that directly suppresses hallucination patterns within the activation space; (3) To ensure that the intervention does not compromise the model's general visual understanding capabilities.

**Key Insight**: By analyzing the intermediate hidden states of LVLMs during generation, the authors find that hallucinated content and factual content exhibit identifiable, directional difference patterns in the activation space. This implies that hallucinations can be suppressed by "steering" activation states in specific directions. This finding aligns with the technological concept of activation steering, which has recently been successfully applied in the LLM safety domain.

**Core Idea**: First use a small calibration set to identify the directional pattern of hallucinations in the activation space, then during inference, apply bidirectional hidden state intervention to perform positive steering (towards facts) and negative steering (away from hallucinations) simultaneously, ultimately generating more reliable outputs in conjunction with a contrastive decoding mechanism.

## Method

### Overall Architecture

The workflow of ASD consists of two stages: **offline stage**, where a small calibration dataset containing pairs of known factual and hallucinated samples is used to extract and analyze the differences in intermediate-layer activations when the model generates factual vs. hallucinated content, thereby learning a steering vector for hallucinations; **online inference stage**, where for each input image and question, bidirectional interventions are applied to the hidden states of selected layers during the forward pass — with positive steering towards the factual direction and negative steering away from the hallucination direction, followed by contrasting the intervened and original output distributions to generate the final token.

### Key Designs

1. **Hallucination Direction Pattern Identification (Calibration Stage)**:

    - **Function**: Extract the hallucination direction vector in the activation space from a small calibration set
    - **Mechanism**: Prepare a group of calibration samples, where each sample contains a factually correct answer and a hallucinated incorrect answer under the same image-question input. Feed these sample pairs into the model and extract hidden states at selected intermediate layers. By computing the difference direction (e.g., using PCA or mean difference) between the hidden states of factual answers and hallucinated answers, a hallucination direction vector $\mathbf{v}_h$ is obtained, indicating the shift direction "from fact to hallucination" in the activation space. The calibration set does not need to be large, as the directional patterns of hallucinations remain consistent across different samples.
    - **Design Motivation**: Rather than trying to understand the specific mechanism of each hallucination, it is more effective to statistically generalize the common features of hallucinations. A small amount of calibration data can capture this commonality, making the method extremely lightweight.

2. **Bidirectional Hidden State Intervention (Inference-time Steering)**:

    - **Function**: Perform directional modifications on intermediate-layer activations of the model during inference to suppress hallucinations
    - **Mechanism**: At selected layers during the model's forward pass, the hidden states are simultaneously intervened in two directions: (1) **Positive steering**: shifting hidden states along the opposite direction of the hallucination vector (i.e., the factual direction $-\mathbf{v}_h$) to encourage the model to generate factual content; (2) **Negative steering**: shifting hidden states along the hallucination vector $+\mathbf{v}_h$ to intentionally "induce" hallucinations. These two intervened versions yield a positive predictive distribution $P^+$ and a negative predictive distribution $P^-$, respectively.
    - **Design Motivation**: Bidirectional intervention is more effective than unidirectional steering — positive steering provides a signal of "where to go," while negative steering provides a signal of "where not to go." Combining the two allows for more precise correction of model behavior.

3. **Contrastive Decoding Mechanism**:

    - **Function**: Synthesize prediction results from positive and negative interventions to generate high-quality outputs
    - **Mechanism**: The probability of final output tokens is determined by calculating the contrastive difference between the positive prediction $P^+$ and negative prediction $P^-$. Specifically, tokens that have high probability under positive steering but low probability under negative steering (which are "confident facts") are enhanced. Meanwhile, tokens that have high probability under both directions (basic vocabulary generated regardless of steering direction, not requiring special handling) and those that have high probability under negative steering but low under positive steering (hallucination candidates) are suppressed. The contrastive decoding formula is similar to $\log P_{final} = (1+\alpha) \log P^+ - \alpha \log P^-$, where $\alpha$ controls the contrast strength.
    - **Design Motivation**: Pure positive steering may introduce new biases ("over-correction"); contrastive decoding achieves more fine-grained and robust hallucination suppression by referencing signals from both positive and negative directions simultaneously.

### Loss & Training

ASD requires absolutely no training — the calibration stage only needs a single forward pass over a small number of samples to extract the direction vectors, and interventions during the inference stage are performed on the fly. It does not modify model parameters and does not require gradient computation. The calibration set typically requires only tens to hundreds of samples. The entire method can be directly applied to any deployed LVLM.

## Key Experimental Results

### Main Results

| Method | CHAIR↓ | POPE Acc | MME-H | Training Requirement |
|------|--------|----------|-------|---------|
| LLaVA-1.5 Baseline | High | Medium | Medium | - |
| VCD (Contrastive Decoding) | Medium | Medium-High | Medium-High | Training-Free |
| OPERA | Medium | Medium-High | Medium-High | Training-Free |
| DoLa | Medium | Highest | Medium | Training-Free |
| ASD (Ours) | Lowest | Highest | Highest | Training-Free |

### Ablation Study

| Configuration | Hallucination Suppression Effect | General Performance | Description |
|------|-----------|---------|------|
| Full ASD Method | Best | Maintained/Improved | Bidirectional steering + Contrastive decoding |
| Positive Steering Only | Good | Slight decrease | Lacks negative signals |
| Negative Steering Only | Medium | Maintained | Knows what to avoid but not what to pursue |
| No Contrast (Directly using positive output) | Moderate-to-good | Prone to over-correction | Lacks contrastive regulation |
| Different Calibration Set Sizes | Improves with size | Unchanged | Saturates at 50-100 samples |
| Different Intervention Layers | Optimal at intermediate layers | - | Shallow intervention ineffectual, deep intervention too late |

### Key Findings

- ASD significantly reduces hallucination rates across multiple hallucination benchmarks, while leaving performance on general visual understanding benchmarks unaffected or even slightly improved.
- Bidirectional intervention is consistently superior to unidirectional steering — the complementarity of positive and negative signals is key to the method's success.
- The size of the calibration set has a minimal impact, indicating that directional hallucination patterns are highly consistent across different samples, representing a systematic bias at the model level.
- Intermediate layers are the most effective intervention points — layers that are too shallow have not yet formed semantic-level hallucination patterns, while layers that are too deep are already too close to output decisions to be corrected.
- The inference overhead of the method is minimal, only adding approximately a 2x forward pass cost (original + positive steering + negative steering), without requiring additional model replicas.

## Highlights & Insights

- **Identifiable Hallucinations in Activation Space**: Hallucinatory content is not random; it exhibits consistent directional patterns in the activation space. This finding provides a new perspective for understanding the inner mechanisms of LVLM hallucinations and suggests that the model's reliability can be "diagnosed" through deeper activation analysis.
- **Complementarity of Bidirectional Steering**: Positive steering + negative steering mimics a "carrot and stick" approach, which is more effective than using either alone. This concept of bidirectional intervention can be extended to other scenarios requiring directional generation control (e.g., toxicity control, style transfer).
- **Training-Free Plug-and-Play**: Does not modify model parameters or require extra training, directly applying to deployed models. For existing LVLMs in production environments, hallucination suppression capabilities can be added at zero cost.
- **Congruence with Representation Engineering in LLM Safety**: Activation steering has been used in LLM safety to control honesty, toxicity, etc. This work extends the concepts to multimodal hallucination control, building a technical bridge between the two domains.

## Limitations & Future Work

- The construction of the calibration set relies on pre-annotated factual/hallucinated sample pairs, which themselves may contain bias or incompleteness.
- The current approach assumes that hallucinations possess a single linear direction in the activation space, whereas different types of hallucinations (e.g., object, attribute, relation hallucinations) may correspond to different directions.
- It increases computation cost by approximately 2 times during inference (two extra forward passes). Although it requires no training, this can be a bottleneck in latency-sensitive applications.
- Multi-vector steering (using different direction vectors for different types of hallucinations) can be explored to achieve finer-grained hallucination control.
- Combining it with training-time methods (e.g., DPO) might yield stronger hallucination suppression effects.
- Interpretability analysis of the direction vectors (their corresponding semantic meanings) warrants in-depth research.

## Related Work & Insights

- **vs. VCD (Visual Contrastive Decoding)**: VCD constructs contrast by perturbing images at the input end (e.g., adding noise), whereas this work constructs contrast internally within the activation space. The perturbations in VCD might alter the model's overall behavior, while ASD's activation steering is more precise.
- **vs. OPERA**: OPERA is an attention-based method that detects hallucinated generation by analyzing attention patterns to identify over-focus on summary tokens. ASD starts from hidden states instead of attention weights, capturing different dimensions of hallucinated signals.
- **vs. DoLa (Decoding by Contrasting Layers)**: DoLa contrasts logit distributions of different layers to suppress hallucinated predictions. ASD similarly capitalizes on inter-layer information but via direction vector steering instead of directly contrasting logits, offering more flexible control.
- **vs. Representation Engineering (Zou et al.)**: RepE utilizes activation steering in the LLM safety domain to control attributes like honesty. ASD systematically applies a similar concept to multimodal hallucination control for the first time.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of bidirectional activation steering and contrastive decoding is novel in the field of LVLM hallucination mitigation, and activation space analysis provides a new perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluations across multiple hallucination and general benchmarks are comprehensive, and ablation studies are detailed.
- Writing Quality: ⭐⭐⭐⭐ The logical chain from observation to method to experiments is clear and compact.
- Value: ⭐⭐⭐⭐⭐ The training-free plug-and-play property gives it high practical application value. 22 citations indicate solid academic recognition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](../../ICLR2026/hallucination/dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)
- [\[ICCV 2025\] ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models](../../ICCV2025/hallucination/only_onelayer_intervention_sufficiently_mitigates_hallucinat.md)
- [\[ACL 2025\] ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs](icr_probe_tracking_hidden_state_dynamics_for_reliable_hallucination_detection_in.md)
- [\[ACL 2025\] Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models](retrieval_visual_contrastive_decoding_to_mitigate_object_hallucinations_in_large.md)
- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](../../CVPR2026/hallucination/residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)

</div>

<!-- RELATED:END -->
