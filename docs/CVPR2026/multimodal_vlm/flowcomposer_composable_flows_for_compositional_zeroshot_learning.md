---
title: >-
  [Paper Note] FlowComposer: Composable Flows for Compositional Zero-Shot Learning
description: >-
  [CVPR 2026][Multimodal VLM][Compositional Zero-Shot Learning] FlowComposer is the first work to introduce Flow Matching into Compositional Zero-Shot Learning (CZSL). It learns two primitive flows—an attribute flow and an object flow—to transport visual features into their corresponding text embedding spaces, and employs a learnable Composer to explicitly combine velocity fields into a compositional flow. A leakage-guided augmentation strategy further converts imperfect feature disentanglement into auxiliary supervision signals. As a plug-and-play module, FlowComposer consistently improves CZSL performance across three benchmarks.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Compositional Zero-Shot Learning
  - Flow Matching
  - CLIP
  - Velocity Field Composition
  - Leakage-Guided Augmentation
date: 2026-05-08
content_hash: 0b4e701533077668
---

# FlowComposer: Composable Flows for Compositional Zero-Shot Learning

**Conference**: CVPR 2026
**arXiv**: [2603.16641](https://arxiv.org/abs/2603.16641)
**Code**: [https://hkust-longgroup.github.io/FlowComposer/](https://hkust-longgroup.github.io/FlowComposer/)
**Area**: Multimodal VLM / Compositional Zero-Shot Learning
**Keywords**: Compositional Zero-Shot Learning, Flow Matching, CLIP, Velocity Field Composition, Leakage-Guided Augmentation

## TL;DR

FlowComposer is the first work to introduce Flow Matching into Compositional Zero-Shot Learning (CZSL). It learns two primitive flows—an attribute flow and an object flow—to transport visual features into their corresponding text embedding spaces, and employs a learnable Composer to explicitly combine velocity fields into a compositional flow. A leakage-guided augmentation strategy further converts imperfect feature disentanglement into auxiliary supervision signals. As a plug-and-play module, FlowComposer consistently improves CZSL performance across three benchmarks.

## Background & Motivation

1. **Background**: CZSL aims to recognize unseen attribute-object compositions by recombining seen primitive attributes and objects. Dominant approaches leverage vision-language models such as CLIP with parameter-efficient fine-tuning (PEFT) via prompt learning.
2. **Limitations of Prior Work**: Existing methods suffer from two fundamental deficiencies: (1) *implicit composition construction*—compositions are formed merely through token-level concatenation rather than explicit operations in the embedding space, causing the embeddings of unseen compositions to drift from image embeddings; and (2) *residual feature entanglement*—visual disentanglers cannot strictly separate attribute and object features, leading to cross-branch information leakage.
3. **Key Challenge**: These two deficiencies cause existing methods to overfit seen compositions, exhibiting a strong seen-bias where accuracy on seen compositions increases during training while accuracy on unseen compositions continuously declines.
4. **Goal**: To design a framework that performs explicit compositional operations in the embedding space, and simultaneously converts imperfect disentanglement into useful supervision.
5. **Key Insight**: The velocity fields of Flow Matching inherently support composition and decomposition—primitive flows can be learned and their velocity fields subsequently combined.
6. **Core Idea**: Two Flow Matching models are trained separately to learn attribute and object transport flows; a Composer network then combines their velocity fields to realize explicit composition in the embedding space.

## Method

### Overall Architecture

FlowComposer is built on top of existing CZSL baselines (e.g., CSP, Troika). Given an image, attribute, object, and compositional visual features along with text embeddings are obtained from the baseline encoders. FlowComposer operates within this shared feature space: it learns attribute and object flows that transport visual features to text embeddings, then applies the Composer to combine velocity fields. The resulting compositional flow score augments compositional recognition.

### Key Designs

1. **Attribute and Object Primitive Flow Models**:

   - **Function**: Learn velocity fields that transport visual features to their corresponding text embeddings.
   - **Mechanism**: For each branch $i \in \{a, o\}$, a linear interpolation path is constructed via Rectified Flow: $x^i_t = (1-t)x^i_0 + tx^i_1$. A velocity network $v_{\theta_i}$ is trained to regress the target velocity $x^i_1 - x^i_0$, supplemented by a cross-entropy loss to ensure that predicted endpoints are correctly classified. At inference, a single-step transport is performed: $\hat{x}^i_1 = x^i_0 + v_{\theta_i}(x^i_0, 0)$.
   - **Design Motivation**: The velocity fields of Flow Matching provide a continuous mapping from visual space to text space that naturally supports compositional operations.

2. **Composer**:

   - **Function**: Learn how to combine attribute and object velocity fields into a compositional velocity field.
   - **Mechanism**: The compositional velocity is approximated as $v^*_c = a^* v^*_a + b^* v^*_o$. Primitive velocities are first normalized to unit directions $\hat{\Delta}_a, \hat{\Delta}_o$, and the target combination coefficients $(a^*, b^*)$ are solved via least squares. The Composer network learns to predict these coefficients from the primitive velocities and is trained with an MSE loss.
   - **Design Motivation**: The relative contribution of attributes and objects varies across samples, necessitating adaptive combination beyond simple token concatenation.

3. **Leakage-Guided Augmentation**:

   - **Function**: Convert cross-branch information leakage arising from imperfect disentanglement into additional supervision.
   - **Mechanism**: In addition to standard intra-branch supervision (attribute visual features → attribute text embeddings), each primitive flow is additionally trained on leaked features—e.g., visual features extracted from the object branch are used to target attribute text embeddings, or compositional branch features are directed toward primitive text embeddings. This enriches the velocity supervision signals.
   - **Design Motivation**: Perfect disentanglement is unattainable in practice; rather than attempting to eliminate leakage, the method exploits it, converting a limitation into an advantage.

### Loss & Training

Total loss = baseline original loss + attribute flow loss (MSE + CE) + object flow loss (MSE + CE) + Composer loss (MSE) + leakage augmentation loss. FlowComposer is a model-agnostic plug-and-play module that can be appended to any CZSL pipeline.

## Key Experimental Results

### Main Results

| Dataset | Metric (HM↑) | Troika | +FlowComposer | Gain |
|---------|-------------|--------|--------------|------|
| MIT-States (CW) | HM | 39.2 | 40.2 | +1.0 |
| C-GQA (CW) | HM | 29.7 | 34.0 | +4.3 |
| UT-Zappos (CW) | HM | 55.4 | 58.6 | +3.2 |
| MIT-States (OW) | AUC | 12.5 | 15.9 | +3.4 |

Significant gains are also observed on the CSP baseline: C-GQA HM improves from 19.3 to 22.9 (+3.6).

### Ablation Study

| Configuration | HM (MIT-States) | AUC | Notes |
|---------------|-----------------|-----|-------|
| Troika baseline | 39.2 | 12.5 | Without FlowComposer |
| + Primitive flows (w/o Composer) | 39.7 | 13.8 | Transport flows only |
| + Composer | 40.0 | 15.0 | With velocity field composition |
| + Leakage augmentation (full) | 40.2 | 15.9 | Complete model |

### Key Findings

- FlowComposer consistently improves baseline performance across all three datasets and both settings (closed-world / open-world).
- The Composer module contributes the largest gain, particularly in the open-world setting (substantial AUC improvement), demonstrating that explicit composition is critical for generalization.
- Leakage-guided augmentation is most effective on C-GQA (+4.3 HM), likely because feature disentanglement is more challenging on that dataset.
- Training dynamics are more stable, with a more balanced seen/unseen accuracy trade-off and reduced seen-bias.

## Highlights & Insights

- **Compositionality of FM velocity fields**: This work is the first to identify that the velocity fields of Flow Matching are inherently well-suited to the compositional/decompositional nature of CZSL—an elegant conceptual correspondence.
- **Turning flaws into features**: Converting imperfect disentanglement (information leakage) into additional supervision is a clever and general strategy.
- **Plug-and-play design**: Operating entirely in the representation space without modifying any encoder, the module is transferable to any CZSL method.

## Limitations & Future Work

- The flow models introduce additional parameters and training overhead.
- Single-step inference is an approximation; multi-step inference may improve accuracy at the cost of efficiency.
- Validation is limited to the CLIP feature space; generalization to other VLMs requires further investigation.
- Future work may explore non-linear transport paths (e.g., ODE solvers) for more accurate transport.

## Related Work & Insights

- **vs. CSP/Troika**: These methods compose only at the token level; FlowComposer performs explicit composition in the embedding space.
- **vs. Diffusion classifiers**: Diffusion-based classifiers use generative models for classification but do not exploit the compositionality of velocity fields.
- **vs. FM for generation**: Conventional Flow Matching is applied to image generation; FlowComposer is the first to leverage its compositional properties for classification.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Applying FM to CZSL is a wholly new direction; the velocity field composition idea is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three datasets, two baselines, detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear; mathematical derivations are well-presented.
- **Value**: ⭐⭐⭐⭐ The plug-and-play design offers strong practical utility, though the target domain is relatively niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning](../../NeurIPS2025/multimodal_vlm/tomcat_test-time_comprehensive_knowledge_accumulation_for_compositional_zero-sho.md)
- [\[CVPR 2026\] AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](anomalyvfm_--_transforming_vision_foundation_models_into_zero-shot_anomaly_detec.md)
- [\[CVPR 2026\] Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](noiseaware_fewshot_learning_through_bidirectional.md)
- [\[CVPR 2026\] No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models](no_hard_negatives_required_concept_centric_learning_leads_to_compositionality_wi.md)
- [\[CVPR 2026\] No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection](no_need_for_real_anomaly_mllm_empowered_zero-shot_video_anomaly_detection.md)

</div>

<!-- RELATED:END -->
