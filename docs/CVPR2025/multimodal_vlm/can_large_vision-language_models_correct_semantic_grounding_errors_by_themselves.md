---
title: >-
  [Paper Note] Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?
description: >-
  [CVPR 2025][Multimodal VLM][VLM] This work systematically investigates the self-correction capabilities of VLMs in semantic grounding tasks. It reveals that intrinsic self-correction (without external feedback) actually degrades performance (by -7 to -17 points). However, iterative correction guided by feedback from the same VLM acting as a binary verifier can improve performance by up to 8.4 percentage points, highlighting that feedback quality is the critical bottleneck for…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "VLM"
  - "semantic grounding"
  - "self-correction"
  - "visual prompting"
  - "iterative refinement"
date: 2026-05-08
content_hash: ddc9a1774fed757c
---

# Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?

**Conference**: CVPR 2025  
**arXiv**: [2404.06510](https://arxiv.org/abs/2404.06510)  
**Authors**: Yuan-Hong Liao, Rafid Mahmood, Sanja Fidler, David Acuna  
**Affiliations**: University of Toronto, Vector Institute, NVIDIA, University of Ottawa  
**Area**: Multimodal VLMs  
**Keywords**: VLM, semantic grounding, self-correction, visual prompting, iterative refinement

## TL;DR
This work systematically investigates the self-correction capabilities of VLMs in semantic grounding tasks. It reveals that intrinsic self-correction (without external feedback) actually degrades performance (by -7 to -17 points). However, iterative correction guided by feedback from the same VLM acting as a binary verifier can improve performance by up to 8.4 percentage points, highlighting that feedback quality is the critical bottleneck for self-correction.

## Background & Motivation
**Background**: Large Vision-Language Models (VLMs) have achieved significant progress in tasks like visual question answering and image captioning. However, they still make numerous mistakes in semantic grounding—the precise identification of semantic categories corresponding to specific image regions. Even state-of-the-art models like GPT-4V/4o only achieve around 40% grounding accuracy on ADE20k.

**Limitations of Prior Work**: (1) Traditional approaches rely heavily on fine-tuning with extensive in-domain annotated data, which is costly and generalization to new domains is difficult. (2) Recent self-correction research primarily focuses on textual LLMs (e.g., self-debugging in code generation), while self-correction of VLMs in visual tasks has not been systematically investigated. (3) Simply asking VLMs to "think again" (i.e., intrinsic self-correction) has been observed to degrade performance in multiple NLP tasks.

**Key Challenge**: Self-correction requires the model to accurately identify its own errors. However, if the model's understanding of the task is inherently flawed, how can it generate correct feedback for self-correction? This is not just a capability issue, but a fundamental cognitive contradiction—the model does not know what it does not know.

**Goal**: Two core research questions: (1) Can VLMs receive and understand oracle-level grounding feedback to improve their predictions? (2) Can VLMs achieve self-correction by providing high-quality binary feedback themselves?

**Key Insight**: This work systematically studies self-correction by decoupling it into two independent dimensions: "correction capability" and "feedback quality." By comparing the performance gap between oracle feedback and VLM-generated feedback, the exact bottleneck is pinpointed.

**Core Idea**: Self-correction of VLMs should not rely on intrinsic reflection. Instead, it should leverage the same model repurposed as a binary verifier to provide external feedback.

**Theoretical Motivation**: Prior work (Huang et al., 2024) proved that intrinsic self-correction of LLMs without external feedback is theoretically limited—the model has already utilized all available information during its initial prediction, and reasoning again does not introduce new evidence. This paper extends this theoretical insight from text to multimodal scenarios and experimentally verifies that this hypothesis also holds for visual grounding tasks.

## Method

### Overall Architecture
An iterative self-correction framework is proposed, consisting of three core components: an initial Predictor, a Verifier, and a Corrector, all powered by the same VLM instance. The pipeline is: the VLM first makes an initial semantic grounding prediction for an image region $\rightarrow$ the Verifier judges whether the prediction is correct $\rightarrow$ if incorrect, the Corrector makes a new prediction based on the feedback $\rightarrow$ repeat for multiple rounds. The key innovation lies in the systematic comparison of various feedback types (oracle binary, oracle class labels, VLM binary self-verification, and intrinsic self-correction).

### Key Designs

1. **Semantic Grounding Task Formulation**:

    - Function: Given an image $x$ and a region $r_i$ (demarcated by a bounding box or mask), the VLM outputs the semantic category label for this region.
    - Mechanism: Zero-shot CoT prompting is adopted where the model first describes the region's content before outputting the classification. Visual marks (such as red circles or arrows) or the Set-of-Mark (SoM) method are used to annotate the target region.
    - Design Motivation: Semantic grounding is one of the most fundamental visual understanding capabilities of VLMs. It suffers from a high error rate (>40%) and can be precisely quantified, making it an ideal testbed for studying self-correction.

2. **Systematic Comparison of Feedback Types**:

    - Function: Decouple the impact of feedback quality on self-correction performance.
    - Mechanism: Design four types of feedback—(a) Oracle Binary: directly informs the model whether the prediction is correct/incorrect; (b) Oracle Class Label: directly provides the ground-truth category name; (c) Intrinsic Self-Correction: asks the model to "think again" without providing any external information; (d) VLM Binary Verification: uses the same VLM as a verifier to judge whether its prediction is correct.
    - Design Motivation: By comparing the performance gap between oracle feedback and automatic feedback, the "feedback bottleneck" can be precisely quantified to guide future improvements.

3. **Binary Verifier Design**:

    - Function: Enable the VLM to judge whether its own (or another VLM's) prediction is correct.
    - Mechanism: Three visual enhancement strategies are designed: (a) Visual Marks: annotate the target region on the original image with a red circle and ask "Is this region X?"; (b) RoI Crop: crop and zoom in on the target region to reduce background distraction; (c) Combined: a combination of both. The query format is unified as binary question answering (Yes/No).
    - Design Motivation: Verification (judging correctness) is much simpler than generation (providing the correct answer)—the model may fail to accurately identify an object, but it can still judge that "this region does not look like a table." Reducing generative tasks to discriminative ones is a core strategy to improve self-correction capabilities.

4. **Iterative Correction Pipeline**:

    - Function: Multi-round correction to progressively approach the correct prediction.
    - Mechanism: In each iteration, regions judged incorrect by the verifier in the previous round are re-predicted, while predictions judged correct are kept unchanged. A maximum number of rounds (5 in experiments) is set to observe the convergence behavior.
    - Design Motivation: Single-round correction has limited coverage (some errors require multiple attempts to resolve), and the iterative mechanism allows the model to progressively cover more errors through stochasticity.

### Evaluation Metrics
The grounding performance is evaluated using semantic grounding accuracy and the F1 score. The F1 score considers both the precision and recall of the verifier. The authors find that the F1 score has a stronger correlation with the final improvement of iterative correction (Spearman $\rho=0.72$), making it a better predictor for the effectiveness of self-correction.

## Key Experimental Results

### Main Results: Correction Performance under Different Feedback Types

| Model | Baseline Accuracy | +Intrinsic Self-Correction | Change | +Oracle Binary (5 rounds) | Change | +VLM Verification (5 rounds) | Change |
|------|-----------|------------|------|-----------------|------|--------------|------|
| LLaVA-1.5-13B | 35.86% | 28.54% | -7.32 | 53.20% | +17.34| 40.29% | +4.43 |
| CogVLM-17B | 15.98% | — | — | 22.12% | +6.14 | 18.64% | +2.66 |
| ViP-LLaVA-13B | 35.90% | — | — | 42.46% | +6.56 | 38.14% | +2.24 |
| GPT-4V | 40.36% | 22.95% | -17.41 | 53.27% | +12.91 | 42.40% | +2.04 |
| GPT-4o | 33.81% | 26.49% | -7.32 | 57.78% | +23.97 | 41.18% | +7.37 |

### Verifier Method Comparison (F1 Score)

| Verification Method | LLaVA-1.5 | CogVLM | ViP-LLaVA | GPT-4V | GPT-4o |
|---------|-----------|--------|-----------|--------|--------|
| Intrinsic (No External Feedback) | 0.419 | — | — | 0.284 | 0.374 |
| Visual Marks | 0.567 | 0.452 | **0.649** | — | — |
| RoI Crop | **0.616** | **0.545** | 0.594 | — | — |
| Combined | 0.598 | 0.529 | 0.631 | — | — |
| Oracle Binary | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### Ablation Study: Number of Iteration Rounds vs. Accuracies

| Model | Round 0 (Baseline) | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |
|------|-----------|-------|-------|-------|-------|-------|
| LLaVA-1.5 (Oracle) | 35.86 | 45.31 | 49.72 | 51.43 | 52.58 | 53.20 |
| LLaVA-1.5 (VLM Verification) | 35.86 | 38.95 | 39.74 | 40.02 | 40.16 | 40.29 |
| GPT-4o (Oracle) | 33.81 | 48.22 | 53.81 | 57.78 | — | — |
| GPT-4o (VLM Verification) | 33.81 | 39.44 | 40.51 | 41.02 | 41.11 | 41.18 |

### Key Findings
- **Intrinsic self-correction is harmful**: All tested models show performance degradation under intrinsic self-correction without external feedback (LLaVA drops by 7.32 points, and GPT-4V drops by 17.41 points). This aligns with observations in the NLP domain: models tend to abandon correct answers when asked to "think again."
- **The oracle feedback ceiling is high**: Oracle binary feedback yields improvements of 17-24 percentage points, proving that VLMs have the inherent capability to utilize feedback and improve predictions. The bottleneck lies in the feedback quality rather than the correction capability.
- **VLM verifier is effective but capped**: Self-verification by VLM improves accuracy by 2-8 percentage points (corresponding to 25%-35% of the gains from oracle feedback), underscoring that verifier accuracy is the core bottleneck.
- **RoI Crop vs. Visual Marks varies across models**: LLaVA and CogVLM perform better with RoI Crop, whereas ViP-LLaVA benefits more from Visual Marks. This reflects that different VLMs' visual encoders process spatial information differently.
- **F1 is more predictive than Accuracy**: The Spearman correlation between the verifier's F1 score and the final iterative gain is 0.72, whereas the correlation with verifier Accuracy is only 0.48.
- **GPT-4o has the highest correction potential**: Although its baseline accuracy is lower than GPT-4V's, GPT-4o improves the most under Oracle feedback (+23.97), indicating that its internal representations contain richer corrective signals.
- **Diminishing returns from iteration**: Most of the improvements occur in the first two rounds, and subsequent rounds show diminishing marginal returns, approaching convergence after 5 rounds.

## Highlights & Insights
- **"Negative results" of intrinsic self-correction are highly valuable**: The systematic proof that directly prompting VLMs to "think again" is not only ineffective but harmful provides an important negative benchmark for the community. This calls for a re-examination of prompt engineering efforts aimed at achieving VLM self-correction.
- **Verification is simpler than generation**: Splitting self-correction into verification and regeneration is a highly clever design. Verification is a binary discrimination problem, whereas generation is an open-set classification problem, the former being inherently simpler. This insight can be migrated to self-improvement frameworks for other VLM tasks.
- **Training-free correction scheme**: The entire framework requires no extra training, fine-tuning, or new data. It achieves non-trivial performance gains solely through prompt design and inference strategies, offering high practical value in data-constrained applications.
- **Quantification of the feedback quality bottleneck**: Through systematic comparison between oracle and self-generated feedback, the gap between "ideal feedback" and "achievable feedback" is precisely quantified (with ~65%-75% of potential gains still untapped), pointing the way for future research.

## Limitations & Future Work
- Because the verifier and predictor employ the same VLM, their errors might be highly correlated, which limits the upper bound of self-correction. Using a different model as the verifier could introduce complementarity.
- The experiments are conducted only on two datasets (ADE20k and COCO, with 100 images each). Due to the small sample size, the findings' generalizability needs further validation.
- The self-correction capabilities in visual tasks beyond semantic grounding (e.g., VQA, image captioning) have not yet been explored.
- The computational cost of iterative correction scales linearly (requiring verification and re-prediction at each round), which might be prohibitive in real-world deployment.
- There is still significant room for improvement in the binary verifier's accuracy. How to design better visual prompts to assist VLMs in judging semantic correctness remains an open question.

## Related Work & Insights
- **vs Self-Refine (Madaan et al.)**: Self-Refine works well in code generation thanks to executable feedback (e.g., compiler errors), whereas semantic grounding lacks such structured feedback. The proposed binary verifier bridges this gap.
- **vs Intrinsic Self-Correction (Huang et al.)**: Prior work theoretically proved the limitations of self-correction in the absence of external signals. This paper experimentally demonstrates this theory in multimodal scenarios.
- **vs Set-of-Mark (Yang et al.)**: SoM uses visual marks on images to facilitate spatial understanding in VLMs. This paper extends this concept as a form of visual feedback inside the correction loop.
- **vs ViP-LLaVA**: An architecture specifically designed for visual prompting. It performs best under the Visual Marks setting, demonstrating the importance of aligning model architecture with prompting methods.

## Rating
- Novelty: ⭐⭐⭐⭐ First work to systematically investigate VLM self-correction with an elegant experimental design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across 5 models, multiple feedback types, and iterative analyses, though the dataset scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear research questions and rigorous experimental logic.
- Value: ⭐⭐⭐⭐ The negative results concerning intrinsic self-correction and the binary verification approach provide valuable references to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[CVPR 2025\] Calico: Part-Focused Semantic Co-Segmentation with Large Vision-Language Models](calico_part-focused_semantic_co-segmentation_with_large_vision-language_models.md)
- [\[ICCV 2025\] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves](../../ICCV2025/multimodal_vlm/ideator_jailbreaking_and_benchmarking_large_visionlanguage_m.md)
- [\[CVPR 2025\] Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves](skip_tuning_pre-trained_vision-language_models_are_effective_and_efficient_adapt.md)
- [\[CVPR 2025\] Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding](your_large_vision-language_model_only_needs_a_few_attention_heads_for_visual_gro.md)

</div>

<!-- RELATED:END -->
