---
title: >-
  [Paper Note] Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Theory of Mind] This paper proposes VisionToM, a lightweight vision-based intervention framework that probes and intervenes on attention heads sensitive to visual input and ToM reasoning within MLLMs. Without fine-tuning the backbone, VisionToM substantially enhances Theory of Mind reasoning in multimodal large language models, achieving significant performance gains on the EgoToM benchmark.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Theory of Mind
  - Multimodal Large Language Models
  - Attention Intervention
  - Visual Reasoning
  - Hallucination Mitigation
date: 2026-05-08
content_hash: e0f415126553f813
---

# Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.24484](https://arxiv.org/abs/2603.24484)
**Code**: None (Project page: [https://founce.github.io/VisionToM](https://founce.github.io/VisionToM))
**Area**: Multimodal VLM / Theory of Mind
**Keywords**: Theory of Mind, Multimodal Large Language Models, Attention Intervention, Visual Reasoning, Hallucination Mitigation

## TL;DR

This paper proposes VisionToM, a lightweight vision-based intervention framework that probes and intervenes on attention heads sensitive to visual input and ToM reasoning within MLLMs. Without fine-tuning the backbone, VisionToM substantially enhances Theory of Mind reasoning in multimodal large language models, achieving significant performance gains on the EgoToM benchmark.

## Background & Motivation

1. **Background**: Theory of Mind (ToM) refers to the capacity to infer mental states—desires, beliefs, and intentions—of oneself and others in order to predict behavior. As LLMs have advanced, their ToM capabilities have drawn increasing attention. However, existing ToM evaluations predominantly focus on textual input, leaving scenarios grounded in purely visual information underexplored.

2. **Limitations of Prior Work**: (1) Most MLLMs perform poorly on ToM tasks with vision-only input, particularly exhibiting large gaps relative to human baselines on Belief and Action reasoning; (2) existing approaches treat models as black boxes with little investigation into the internal behavior of attention in multiple-choice QA; (3) the influence of LLM hallucinations on ToM tasks has not been sufficiently studied from an interpretability perspective; (4) most multimodal ToM benchmarks rely on simulated environments and lack ecological validity in real-world settings.

3. **Key Challenge**: When handling ToM tasks, MLLMs over-rely on linguistic priors and neglect visual evidence. When visual information conflicts with linguistic priors, models tend to produce inaccurate inferences based on language patterns, leading to hallucinations. Existing interpretability-based enhancement methods are confined to the text modality.

4. **Goal**: How can one enhance an MLLM's visual attention and ToM reasoning capabilities—and reduce dependence on spurious linguistic priors—by intervening on internal representations, without fine-tuning the model?

5. **Key Insight**: Interpretability analysis reveals that MLLMs exhibit cross-task consistency in visual attention across multiple ToM tasks, while internal representations for ToM reasoning diverge across tasks but are consistent within each task. This observation provides a principled basis for targeted intervention.

6. **Core Idea**: Linear probes identify attention heads sensitive to visual input and ToM reasoning. Intervention vectors are computed pointing from incorrect to correct representations, and are injected at inference time to guide the model toward attending to visual evidence and producing correct reasoning.

## Method

### Overall Architecture

VisionToM proceeds in four stages: (1) **Internal Representation Extraction**—constructing positive/negative sample pairs and extracting representations of visual attention and ToM reasoning from MLLM attention heads; (2) **Probing**—training linear classifiers to identify which attention heads are most sensitive to visual input and ToM reasoning; (3) **ToM Reasoning Representation Disentanglement**—using clustering and encoder networks to align negative sample representations toward positive samples; (4) **Intervention**—injecting the computed intervention vectors into sensitive attention heads at inference time. The MLLM backbone remains frozen throughout.

### Key Designs

1. **Visual Attention Enhancement**:
    - **Function**: Reduce the MLLM's over-reliance on linguistic priors and strengthen attention to visual input.
    - **Mechanism**: PGD adversarial attacks ($\epsilon=16/255$, 300 steps) generate visually perturbed samples as negatives while keeping textual questions unchanged. Activation values from all attention heads are extracted from positive/negative pairs, and the mean offset vector is computed as: $\{\delta_{V,l}^h\} = \frac{1}{S}\sum_{i=1}^{S}(X_{V,i,l}^{pos,h} - X_{V,i,l}^{neg,h})$. This offset encodes the direction from "correct visual information" to "perturbed visual information"; applying it in reverse guides the model to attend more faithfully to genuine visual input.
    - **Design Motivation**: PGD attacks more effectively expose attention failure modes than random noise—after PGD attack, Goal accuracy drops from 61.5% to 29.1%, whereas random noise only reduces it to 47.0%, confirming that adversarial samples provide more accurate gradient directions.

2. **ToM Reasoning Guidance**:
    - **Function**: Strengthen the model's ToM reasoning ability and guide correct mental state inference.
    - **Mechanism**: Visual input is fixed; correct answers serve as positive samples and incorrect answers as negatives. Since the semantic diversity of negative samples produces non-uniform representation distributions, offset vectors cannot be computed directly. A clustering-based approach is adopted: negative samples for each sensitive attention head are clustered (with $k$ selected automatically in the range 2–15), and a dedicated encoder network $f_{h,c}$ is trained per cluster to learn correction vectors $\delta_{h,c,i} = f_{h,c}(x_{T,i}^{neg,h})$ that map negative to positive representations. At inference time, the nearest cluster center determines which encoder to apply.
    - **Design Motivation**: Different types of reasoning failures require interventions in different directions. Naïve averaging blurs distinctions among failure modes; the clustering plus dedicated encoder design achieves more fine-grained, targeted correction.

3. **Probing & Intervention**:
    - **Function**: Identify sensitive heads and precisely apply interventions at inference time.
    - **Mechanism**: An independent linear binary probe (logistic regression) is trained for each attention head; heads with high validation accuracy are designated "sensitive heads." Key finding: visually sensitive heads are distributed across layers and consistent across tasks, while ToM reasoning sensitive heads are concentrated in middle layers and diverge across tasks. Intervention selects the Top-$K$=64 sensitive heads and injects the sum of visual and ToM intervention vectors: $T_{l+1} = T_l + \sum_{h=1}^H(Attn_l^h(P_l^hT_l) + \alpha \times \Delta) \cdot W_l^o$, with intervention strength $\alpha=1.0$.
    - **Design Motivation**: Cross-task consistency of visual attention sensitive heads enables sharing the same head set; within-task consistency of ToM reasoning enables VisionToM to probe task-specific ToM embeddings. Reverse intervention ($-\alpha\Delta$) causes a sharp performance drop, validating the correctness of the intervention direction.

### Loss & Training

- **Probe training**: Standard cross-entropy loss for optimizing logistic regression parameters.
- **Encoder training**: $L_{total} = \sum_h \sum_{c=1}^{k_h^*} \frac{1}{|C_{h,c}|} \sum_{i \in C_{h,c}} \|(x_{T,i}^{neg,h} + \delta_{h,c,i}) - x_{T,i}^{pos,h}\|^2$
- Probes and encoders are trained on a 30% calibration split; inference is performed on the remaining 70% evaluation split.
- One-time calibration: probe training takes approximately 0.2 hours, encoder training approximately 1 hour; the MLLM backbone remains frozen.

## Key Experimental Results

### Main Results

| Model | Task | Baseline | +VisionToM | Gain |
|--------|------|------|----------|------|
| LLaVA-Next-Video-7B | Goal | 61.5% | 74.5% | +13.0% |
| LLaVA-Next-Video-7B | Belief | 38.9% | 45.3% | +6.4% |
| LLaVA-Next-Video-7B | Actions | 24.0% | 29.7% | +5.7% |
| Qwen2.5-VL-7B | Goal | 86.9% | 88.9% | +2.0% |
| Qwen2.5-VL-7B | Belief | 35.6% | 42.0% | +6.4% |
| Qwen2.5-VL-7B | Actions | 31.1% | 37.6% | +6.5% |
| Human Baseline | Goal/Belief/Actions | 88/72/78% | - | - |

### Ablation Study

| Configuration | Goal | Belief | Actions | Note |
|------|---------|------|------|------|
| LLaVA Baseline | 61.5% | 38.9% | 24.0% | No intervention |
| Visual only (w/o $\delta_T$) | 73.2% | 39.2% | 25.3% | Large gain on Goal; small on Belief/Actions |
| ToM only (w/o $\delta_V$) | 72.6% | 45.3% | 29.0% | Large gain on Belief/Actions |
| Random intervention (Rnd-$\Delta$) | 62.1% | 39.2% | 25.4% | Random direction nearly ineffective |
| Reverse intervention ($-\alpha\Delta$) | 50.6% | 20.6% | 10.1% | Sharp drop; validates direction correctness |
| Full (+$\alpha\Delta$) | 74.5% | 45.3% | 29.7% | Both interventions complementary |

### Key Findings

- Visual attention enhancement is most effective for the Goal task (+11.7%), as goal inference relies more heavily on visual cues.
- ToM reasoning intervention is critical for Belief and Actions tasks, which require deeper cognitive reasoning.
- The two intervention directions are orthogonally complementary; applying both simultaneously outperforms either alone.
- PGD attacks provide more precise intervention directions than random noise: after PGD-based intervention, Goal accuracy recovers from 29.1% to 74.5%, whereas random noise intervention recovers only from 47.0% to 70.4%.
- VisionToM is also effective on open-ended generation tasks: LLaVA-Next-Video's True∧Info metric improves from 8.5% to 27.2%.
- Qwen2.5-VL with VisionToM on the Goal task (88.9%) approaches the human baseline (88%).

## Highlights & Insights

- A significant finding from interpretability analysis: visual attention is consistent across tasks, whereas ToM reasoning representations cluster within tasks but diverge across them—an insight that provides theoretical grounding for precise intervention.
- The ingenious use of adversarial attacks as probing tools: PGD is not employed to attack the model but to uncover vulnerable directions in visual attention; reversing the perturbation direction serves as an enhancement mechanism.
- The clustering plus dedicated encoder strategy for ToM reasoning correction accounts for the diversity of reasoning failures, offering finer-grained correction than a simple mean direction.
- The entire method is lightweight with a frozen backbone—probes are linear classifiers and encoders are two-layer MLPs; once calibrated, intervention vectors are reusable.

## Limitations & Future Work

- Evaluation is currently limited to the EgoToM benchmark; generalization to other ToM benchmarks (e.g., MMToM-QA, GridToM) remains unknown.
- Intervention vectors are computed once on a calibration set and may require recalibration for out-of-distribution video scenarios.
- Although the automatic determination of cluster count (Silhouette + Elbow + CH Index) is reasonable, it may be unstable with small samples.
- Substantial gaps relative to human baselines remain on Belief and Actions tasks (45.3% vs. 72%; 29.7% vs. 78%), indicating that attention intervention is only one component of a broader solution.

## Related Work & Insights

- **vs. GridToM**: GridToM derives intervention directions from linear probe coefficient vectors in binary classification settings; VisionToM introduces clustering plus encoders to handle the heterogeneity of multi-class negative samples, enabling finer-grained intervention.
- **vs. ICT (CVPR'25)**: ICT uses random noise to guide visual attention; VisionToM uses PGD adversarial samples, providing more accurate directional estimates.
- **Insights**: The visual attention intervention paradigm is transferable to other tasks requiring enhanced VLM visual reasoning (e.g., visual commonsense reasoning, causal inference); the core pattern is "probe → find direction → intervene."

## Rating

- Novelty: ⭐⭐⭐⭐ Novel perspective combining interpretability probing and intervention for ToM enhancement.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two models, three tasks, comprehensive ablations, but only a single benchmark dataset.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear; visualizations (PCA, KDE) aid understanding.
- Value: ⭐⭐⭐⭐ Provides an interpretable approach to enhancing cognitive reasoning in MLLMs, though practical application scenarios are limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents](mindpower_enabling_theoryofmind_reasoning_in_vlmba.md)
- [\[CVPR 2026\] GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding](groundvts_visual_token_sampling_in_multimodal_large_language_models_for_video_te.md)
- [\[CVPR 2026\] StructXLIP: Enhancing Vision-Language Models with Multimodal Structural Cues](structxlip_enhancing_vision-language_models_with_multimodal_structural_cues.md)
- [\[CVPR 2026\] Predictive Regularization Against Visual Representation Degradation in Multimodal Large Language Models](predictive_regularization_against_visual_representation_degradation_in_multimoda.md)
- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
