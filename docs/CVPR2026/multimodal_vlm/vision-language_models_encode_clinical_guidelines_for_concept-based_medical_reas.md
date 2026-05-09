---
title: >-
  [Paper Note] Vision-Language Models Encode Clinical Guidelines for Concept-Based Medical Reasoning
description: >-
  [CVPR 2026][Multimodal VLM][Concept Bottleneck Models] This paper proposes MedCBR, a framework that integrates clinical diagnostic guidelines (e.g., BI-RADS) into the training and inference pipeline of concept bottleneck models. By leveraging LVLMs to generate guideline-consistent reports for enhanced concept supervision, combining multi-task CLIP training with a large reasoning model for structured clinical explanation generation, MedCBR achieves AUROCs of 94.2% and 84.0% on ultrasound and mammography cancer detection, respectively.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Concept Bottleneck Models
  - Medical Imaging
  - Explainable AI
  - Clinical Guidelines
  - CLIP
date: 2026-05-08
content_hash: 115c2d937a5c54e2
---

# Vision-Language Models Encode Clinical Guidelines for Concept-Based Medical Reasoning

**Conference**: CVPR 2026
**arXiv**: [2603.08921](https://arxiv.org/abs/2603.08921)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: Concept Bottleneck Models, Medical Imaging, Explainable AI, Clinical Guidelines, CLIP

## TL;DR
This paper proposes MedCBR, a framework that integrates clinical diagnostic guidelines (e.g., BI-RADS) into the training and inference pipeline of concept bottleneck models. By leveraging LVLMs to generate guideline-consistent reports for enhanced concept supervision, combining multi-task CLIP training with a large reasoning model for structured clinical explanation generation, MedCBR achieves AUROCs of 94.2% and 84.0% on ultrasound and mammography cancer detection, respectively.

## Background & Motivation
**Background**: Concept Bottleneck Models (CBMs) connect model predictions to human-interpretable concepts through an intermediate concept layer, representing a dominant paradigm in explainable AI and proving particularly valuable in medical imaging.

**Limitations of Prior Work**: Standard CBMs rely on discrete concept representations, neglecting broader clinical context such as diagnostic guidelines and expert heuristics, which leads to reduced reliability in complex cases. Specific issues include: (a) concept annotations are noisy and incomplete due to inter-observer variability; (b) CBMs fail to capture experience-driven reasoning, such as cases that appear benign but require holistic assessment within the context of clinical guidelines.

**Key Challenge**: CBMs require complete and noise-free concept annotations and assume that diagnostic reasoning is a deterministic function of concept presence—yet medical diagnosis depends on contextual information and structured reasoning embedded in clinical guidelines.

**Goal**: (a) Address concept annotation noise and incompleteness; (b) remedy the lack of clinical context in concept-to-diagnosis reasoning; (c) provide auditable explanations for model predictions.

**Key Insight**: Diagnostic reasoning is modeled as inference over multiple evidence sources rather than a direct function of concepts, with clinical guidelines introduced as a structured knowledge source.

**Core Idea**: Enrich concept representations via LVLM-generated guideline-consistent reports, combined with multi-task contrastive learning and a large reasoning model for interpretable diagnostic narrative generation.

## Method

### Overall Architecture
MedCBR comprises three stages: (1) **guideline-driven concept enrichment**—converting discrete concept labels into guideline-consistent textual reports using an LVLM; (2) **vision-language concept modeling**—training CLIP with multi-task objectives that jointly optimize cross-modal alignment, concept prediction, and diagnostic classification; (3) **concept-based clinical reasoning**—using a frozen large reasoning model (LRM) to integrate predicted concepts with guidelines to produce structured diagnostic explanations.

### Key Designs

1. **Guideline-Driven Concept Enrichment**:

    - **Function**: Transforms discrete concept vectors $c$ into continuous, guideline-conditioned textual representations $r$.
    - **Mechanism**: An LVLM receives the image $x$, the positive concept label set $c^+$, the label $y$, and clinical guidelines $\mathcal{G}$, and generates a structured report describing visual findings and summarizing their diagnostic implications according to $\mathcal{G}$.
    - **Design Motivation**: Discrete concept labels merely indicate which findings are present and cannot express inter-concept relationships or diagnostic significance. LVLM-generated enriched reports capture contextual and relational semantics among concepts, providing a more consistent supervision signal.

2. **Multi-Task Vision-Language Concept Model**:

    - **Function**: Jointly learns image-text alignment, concept prediction, and diagnostic classification.
    - **Mechanism**: Built on a CLIP backbone, the model is simultaneously optimized with three losses: a contrastive loss $\mathcal{L}_{CLIP}$ aligning images with LVLM-generated reports; a diagnostic loss $\mathcal{L}_y$ for cancer classification over visual embeddings; and a concept loss $\mathcal{L}_c$ predicting individual concepts via $N_c$ dedicated lightweight adapters. The total loss is $\mathcal{L} = \lambda\mathcal{L}_{CLIP} + \mu\mathcal{L}_y + \nu\mathcal{L}_c$.
    - **Design Motivation**: Multi-task training simultaneously enforces (i) cross-modal consistency, (ii) concept-level interpretability, and (iii) diagnostic discriminability, yielding representations that are both semantically rich and clinically grounded.

3. **Concept-Based Clinical Reasoning**:

    - **Function**: Converts model predictions into structured diagnostic narratives.
    - **Mechanism**: A frozen LRM receives a structured prompt $\pi = (\mathcal{Q}, \hat{y}, \hat{c}, \mathcal{G})$ comprising task instructions, predicted cancer probability, concept prediction confidences, and clinical guidelines, and generates a step-by-step diagnostic reasoning explanation.
    - **Design Motivation**: Since the LRM operates on structured inputs and explicit guidelines $\mathcal{G}$, its reasoning is anchored to verifiable clinical knowledge, reducing the risk of hallucination.

## Key Experimental Results

### Main Results — Cancer Detection

| Method | BUS-BRA (AUROC) | CBIS-DDSM (AUROC) | CUB-200 (Acc.) |
|---|---|---|---|
| CBM | 84.8 | 79.6 | 62.9 |
| CLIP ViT-L/14 | 93.5 | 82.4 | 85.7 |
| AdaCBM | 87.9 | 75.6 | 69.8 |
| Label-free CBM | 60.0 | 70.0 | 74.3 |
| **MedCBR** | **94.2** | **84.0** | **86.1** |

### Ablation Study — Component Contributions

| Configuration | BUS-BRA | CBIS-DDSM | CUB-200 |
|---|---|---|---|
| CLIP ViT | 93.5 | 82.4 | 85.7 |
| CLIP+CBL | 91.8 | 81.8 | 67.0 |
| CLIP+CBL+Guideline | 92.0 | 83.1 | 72.9 |
| CLIP+MTL | 93.6 | 83.2 | 82.3 |
| CLIP+MTL+Guideline (MedCBR) | **94.2** | **84.0** | **86.1** |

### Key Findings
- MedCBR consistently outperforms all CBM variants and vanilla CLIP across all three datasets, demonstrating the superiority of combining guideline-driven concept enrichment with multi-task learning.
- Introducing the concept bottleneck layer (CBL) alone degrades performance; however, incorporating guidelines recovers and surpasses the baseline, indicating that guideline information effectively compensates for the information loss induced by the bottleneck structure.
- Strong performance on CUB-200 bird classification (86.1%) validates the framework's generalizability beyond the medical domain.
- Concept-level detection performance is also consistently superior, with multi-modal supervision enabling the model to simultaneously capture visually grounded and modality-specific features.

## Highlights & Insights
- **Clinical Guidelines as a Structured Knowledge Source**: Unlike prior work that treats concepts or guidelines as auxiliary context, MedCBR integrates guidelines throughout the entire pipeline from training to inference, ensuring that concept-to-decision reasoning is constrained and validated.
- **LVLM-Driven Concept Enrichment**: The framework cleverly leverages LVLMs to transform noisy and incomplete discrete annotations into high-quality structured reports, addressing the practical challenge of concept annotation in medical data.
- **End-to-End Interpretable Pipeline**: The full chain from image → concepts → guidelines → diagnostic explanation is auditable at every step, satisfying the stringent transparency requirements of clinical practice.

## Limitations & Future Work
- The inference stage depends on an external frozen LRM, increasing deployment complexity and latency.
- Evaluation is limited to binary classification (benign/malignant); multi-class or finer-grained grading tasks have not been explored.
- Guidelines are provided as fixed text, with no exploration of dynamic retrieval or personalized guideline adaptation.
- The concept set relies on manual definition; extending the framework to new diseases requires domain experts to redefine the concept taxonomy.
- Radiologist evaluation covers only 20 cases, limiting statistical power.

## Related Work & Insights
- **vs. AdaCBM**: AdaCBM mitigates CLIP domain shift via learnable adapters but does not incorporate clinical knowledge; MedCBR provides stronger inductive bias through guideline-driven training.
- **vs. Label-free CBM**: Automatically generated concepts may omit clinically important features or introduce spurious correlations; MedCBR constrains concept discovery through guidelines.
- **vs. Agent-based methods (e.g., MAGDA, MedRAX)**: These approaches use guidelines or tools as reasoning aids but do not deeply integrate them into model training.

## Rating
- Novelty: ⭐⭐⭐⭐ — Deeply integrating clinical guidelines into CBM training and inference is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dataset validation with ablation and clinical evaluation, though the clinical evaluation sample size is limited.
- Writing Quality: ⭐⭐⭐⭐ — The framework is clearly presented, formulations are rigorous, and clinical relevance is well-motivated.
- Value: ⭐⭐⭐⭐ — Provides a practical guideline-integration paradigm for medical explainable AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Medic-AD: Towards Medical Vision-Language Model's Clinical Intelligence](medic-ad_towards_medical_vision-language_models_clinical_intelligence.md)
- [\[CVPR 2026\] Concept-wise Attention for Fine-grained Concept Bottleneck Models](coat_cbm_concept_wise_attention.md)
- [\[AAAI 2026\] Concept-RuleNet: Grounded Multi-Agent Neurosymbolic Reasoning in Vision Language Models](../../AAAI2026/multimodal_vlm/concept-rulenet_grounded_multi-agent_neurosymbolic_reasoning.md)
- [\[CVPR 2026\] What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](what_do_visual_tokens_really_encode_uncovering_sparsity_and_redundancy_in_multim.md)
- [\[CVPR 2026\] Towards Calibrating Prompt Tuning of Vision-Language Models](towards_calibrating_prompt_tuning_of_vision-language_models.md)

</div>

<!-- RELATED:END -->
