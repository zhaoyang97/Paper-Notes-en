---
title: >-
  [Paper Note] MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization
description: >-
  [ICML 2025][Multimodal VLM][Medical VLM] This paper proposes MMedPO, a clinical-aware multimodal medical preference optimization method. By constructing multimodal preference data through believable hallucination injection and localized lesion noise addition, and leveraging collaboration among multiple medical LLMs to evaluate clinical relevance as a weighting signal integrated into DPO training, it achieves average improvements of 14.2% and 51.7% on Med-VQA and report genera…
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "Medical VLM"
  - "Preference Optimization"
  - "Clinical Relevance"
  - "Multimodal Alignment"
  - "Hallucination Mitigation"
date: 2026-05-08
content_hash: 45c754b3c7cf1d3a
---

# MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization

**Conference**: ICML 2025  
**arXiv**: [2412.06141](https://arxiv.org/abs/2412.06141)  
**Code**: [https://github.com/aiming-lab/MMedPO](https://github.com/aiming-lab/MMedPO)  
**Area**: LLM Alignment/RLHF  
**Keywords**: Medical VLM, Preference Optimization, Clinical Relevance, Multimodal Alignment, Hallucination Mitigation

## TL;DR
This paper proposes MMedPO, a clinical-aware multimodal medical preference optimization method. By constructing multimodal preference data through believable hallucination injection and localized lesion noise addition, and leveraging collaboration among multiple medical LLMs to evaluate clinical relevance as a weighting signal integrated into DPO training, it achieves average improvements of 14.2% and 51.7% on Med-VQA and report generation tasks, respectively.

## Background & Motivation
**Background**: Medical Large Vision-Language Models (Med-LVLMs) are widely applied in clinical diagnosis and treatment planning, but they still face severe factuality issues—the models tend to prioritize textual knowledge learned during training over actual visual input, resulting in hallucinations.

**Limitations of Prior Work**: Existing preference optimization methods directly apply the preference data construction pipelines of general-domain LVLMs, ignoring the **clinical relevance** inherent to the medical field. Consequently, the dispreferred responses in preference data lack clinical significance and are easily distinguished by the models, which diminishes training effectiveness. For example, obvious factual errors like "gallstones in the right lung lobe" can be recognized without alignment training.

**Key Challenge**: The effectiveness of preference optimization depends heavily on the quality of preference data—dispreferred responses need to be sufficiently "believable" to provide effective learning signals; simultaneously, understanding lesion regions in the visual modality is critical for medical accuracy, yet existing methods rarely guide models to focus on localized lesions.

**Goal**: Design a multimodal preference optimization method that considers clinical relevance to make the alignment training of Med-LVLMs more effective.

**Key Insight**: Simultaneously improve from two dimensions: preference data construction (two dispreference strategies) and preference sample importance quantification (clinical relevance scoring).

**Core Idea**: Not all preference samples are equally important—samples with higher clinical relevance (i.e., those containing more "believable" dispreferred responses or more accurate lesion detection) should receive larger training weights.

## Method

### Overall Architecture
MMedPO consists of three steps:
1. **Multimodal Preference Data Construction**: Construct preference pairs through two strategies: hallucination injection (textual) and localized lesion noise addition (visual).
2. **Clinical Relevance Scoring**: Quantify the clinical importance of each sample using a collaborative multi-Med-LLM system and visual tool confidence.
3. **Clinical-Aware Preference Optimization**: Integrate the normalized clinical relevance score as a weight into the DPO loss function.

### Key Designs

1. **Dispreferred Response Generation via Hallucination Injection (Strategy 1, $\mathcal{D}_t$):**

    - **Function**: Utilize the target Med-LVLM and GPT-4o to generate hallucinated responses containing **believable medical errors**.
    - **Mechanism**:
      1. Sample from the target Med-LVLM multiple times to collect candidate responses that may contain hallucinations.
      2. Evaluate candidates using GPT-4o, selecting the response with the most prominent hallucination (most clearly conflicting with the ground truth).
      3. If no candidate meets the criteria, GPT-4o directly generates a hallucinated response based on the ground truth.
    - **Design Motivation**: Ensure that the dispreferred response contains targeted medical errors (misinterpretation of images, misleading descriptions, incorrect diagnoses) rather than random, meaningless content.
    - **Difference from Standard DPO**: Standard DPO might directly use substandard model outputs as dispreferred data, making clinical relevance uncontrollable.

2. **Dispreferred Construction via Localized Lesion Noise Addition (Strategy 2, $\mathcal{D}_v$):**

    - **Function**: Locate lesion regions using medical visual tools (e.g., MedKLIP) and add noise *only* to the lesion area to construct dispreferred samples.
    - **Mechanism**: Noise addition formula: $x_v^* = \sqrt{\bar{\xi}_k} \cdot (x_v \odot h) + \sqrt{1-\bar{\xi}_k} \cdot (\epsilon \odot h) + (x_v \odot (1-h))$
    - Where $h = \mathcal{T}(x_v)$ is the lesion heatmap predicted by the visual tool, and $\epsilon$ is random noise.
    - Preference Pair Construction: The original image + ground truth is preferred, while the noisy image + ground truth is dispreferred.
    - **Design Motivation**: Localized instead of global noise addition forces the model to learn to focus on lesion regions rather than relying solely on global information.

3. **Multi-Med-LLM Collaborative Clinical Relevance Scoring:**

    - **Function**: Evaluate the clinical relevance of dispreferred responses through consensus debate among multiple medical LLMs (e.g., Med42-7B, Med42-70B, BioMistral-7B).
    - **Workflow**:
      1. The first Med-LLM $\mathcal{G}_1$ evaluates the clinical relevance score $s_1$ of $y_l$.
      2. Subsequent Med-LLM $\mathcal{G}_i$ reviews the previous score $s_{i-1}$, adopting it if in agreement, or suggesting a new score otherwise.
      3. Iterate until a consensus is reached or the maximum number of rounds is met, then compute the average $\hat{s} = \frac{\sum s_i}{|S|}$.
    - **Design Motivation**: A single Med-LLM may be biased; collaborative evaluation across multiple models yields more reliable scores.
    - **Visual Tool Confidence**: For samples from the noise-addition strategy, the confidence $s_v$ of the visual tool in detecting the lesion is used as the clinical relevance.

4. **Clinical-Aware Weighted DPO Loss:**

    - Normalize each clinical relevance score: $s' = \frac{s-\mu}{\sigma}$, clipped to $[\alpha, \beta]$.
    - Weighted DPO loss:
    $$ \mathcal{L}_{mmedpo} = -\mathbb{E}_{(x,x^*,y_w,y_l,s') \sim \mathcal{D}_o}\left[s' \log \sigma\left(\alpha \log \frac{\pi_\theta(y_w|x)}{\pi_o(y_w|x)} - \alpha \log \frac{\pi_\theta(y_l|x^*)}{\pi_o(y_l|x^*)}\right)\right] $$
    - Samples with high clinical relevance receive greater weights, while the influence of low-quality samples is suppressed.

### Loss & Training
- Based on the weighted DPO loss (Eq. 3), utilizing LoRA for fine-tuning.
- Training hyperparameters: batch size 4, lr 1e-7, 3 epochs.
- Base model: LLaVA-Med-1.5 7B.
- Preference data $\mathcal{D}_o = \mathcal{D}_t \cup \mathcal{D}_v$, merging both strategies.

## Key Experimental Results

### Main Results (Comparison with baseline methods based on LLaVA-Med v1.5)

| Method | SLAKE Open | SLAKE Closed | VQA-RAD Open | VQA-RAD Closed | IU-Xray BLEU | IU-Xray ROUGE-L |
|---|---|---|---|---|---|---|
| LLaVA-Med v1.5 | 44.26 | 61.30 | 29.24 | 63.97 | 14.56 | 10.31 |
| + DPO | 49.30 | 62.02 | 29.76 | 64.70 | 16.08 | 12.95 |
| + POVID | 52.43 | 70.35 | 31.77 | 65.07 | 20.80 | 24.33 |
| + FiSAO | 52.69 | 70.46 | 32.70 | 64.11 | 21.06 | 25.72 |
| **+ MMedPO** | **53.99** | **73.08** | **36.36** | **66.54** | **23.49** | **29.52** |

### Ablation Study

| Configuration | SLAKE Avg. | VQA-RAD Avg. | IU-Xray Avg. | Notes |
|---|---|---|---|---|
| Strategy 1 w/o CRS | 55.65 | 47.23 | 10.95 | Hallucination injection, no weighting |
| Strategy 1 w/ CRS | 57.62 | 48.67 | 15.66 | Clinical score weighting is effective |
| Strategy 2 w/o CRS | 60.59 | 45.94 | 19.30 | Lesion noise, no weighting |
| Strategy 2 w/ CRS | 60.88 | 46.97 | 25.00 | Significant improvement in report generation |
| Single Med-LLM | 56.09 | 48.67 | 15.67 | Potential bias |
| Multi Med-LLM | 57.53 | 51.14 | 15.86 | Consensus is more reliable, +3.6% |
| Global Noise | 58.88 | 46.91 | 24.88 | Imprecise |
| **Localized Noise** | **59.88** | **46.98** | **25.00** | Focusing on lesions is more effective |

### Key Findings
- MMedPO outperforms the best baseline by an average of 14.2% on Med-VQA and 51.7% on report generation.
- The improvement on open-ended questions is larger than on closed-ended questions, indicating that MMedPO is more effective for free-form generation.
- The two preference strategies are complementary: hallucination injection is more effective for VQA, while lesion noise addition is more effective for report generation.
- The gain from clinical relevance scoring on the report generation task (+18.5%) is much larger than on VQA (+2.3%), showing that sample quality varies more in report generation.
- Attention map visualization shows that MMedPO significantly enhances focus on lesion regions.
- MMedPO is compatible with SFT pre-training and stronger backbone models (LLaVA-Med++), demonstrating good scalability.

## Highlights & Insights
- **Introducing clinical relevance to preference optimization** is a key innovation—not all preference pairs are equally effective, and weighting is highly justified.
- The two dispreference strategies cover text errors (hallucination injection) and visual errors (lesion noise addition), achieving true "multimodal" preference optimization.
- The multi-Med-LLM collaborative scoring mechanism effectively mitigates single-model bias.
- Attention map visualizations intuitively demonstrate how MMedPO enhances the model's focus on lesion regions.
- The design is highly modular, allowing individual components to be used independently.

## Limitations & Future Work
- Reliance on GPT-4o for hallucination generation and evaluation is costly and limited by API availability.
- The lesion detection accuracy of the visual tool (MedKLIP) directly impacts noise addition quality, which may not be precise enough in some domains.
- Validation was conducted only on chest X-ray datasets; generalization to other medical imaging modalities (CT, MRI, ultrasound) remains to be verified.
- The efficiency of multi-Med-LLM collaborative scoring could be a bottleneck, particularly for large-scale datasets.
- Future work could explore directly integrating the clinical relevance score into reward model training, rather than only using it as a loss weight.

## Related Work & Insights
- Compared to general VLM preference optimization methods such as POVID and FiSAO, it emphasizes the importance of medical-specific design.
- The concept of multi-agent collaborative evaluation for clinical relevance can be extended to other domains requiring expert assessment.
- The localized noise addition strategy, inspired by the diffusion model's noising process, represents a novel approach to constructing visual preference data.
- This provides a complete pipeline for Med-LVLM alignment, offering substantial practical value.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] HSCR: Hierarchical Self-Contrastive Rewarding for Aligning Medical Vision Language Models](../../ACL2025/multimodal_vlm/hscr_hierarchical_self-contrastive_rewarding_for_aligning_medical_vision_languag.md)
- [\[CVPR 2025\] Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization](../../CVPR2025/multimodal_vlm/debiasing_multimodal_large_language_models_via_noise-aware_preference_optimizati.md)
- [\[CVPR 2026\] Dynamics-Aware Preference Optimization for Vision-Language Models](../../CVPR2026/multimodal_vlm/dynamics-aware_preference_optimization_for_vision-language_models.md)
- [\[CVPR 2026\] Medic-AD: Towards Medical Vision-Language Model's Clinical Intelligence](../../CVPR2026/multimodal_vlm/medic-ad_towards_medical_vision-language_models_clinical_intelligence.md)
- [\[NeurIPS 2025\] Aligning by Misaligning: Boundary-aware Curriculum Learning for Multimodal Alignment](../../NeurIPS2025/multimodal_vlm/aligning_by_misaligning_boundaryaware_curriculum_learning_fo.md)

</div>

<!-- RELATED:END -->
