---
title: >-
  [Paper Note] HEIE: MLLM-Based Hierarchical Explainable AIGC Image Implausibility Evaluator
description: >-
  [CVPR 2025][Multimodal VLM][AIGC image evaluation] This work proposes HEIE, a Hierarchical Explainable AIGC Image Implausibility Evaluator based on Multimodal Large Language Models (MLLMs). Through a CoT-driven trinity evaluator, it simultaneously outputs defect heatmaps, scores, and textual explanations. An Adaptive Hierarchical Implausibility Mapper is employed to achieve precise localization of both global and local defects, achieving state-of-the-art (SOTA) performance on…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "AIGC image evaluation"
  - "multimodal large language model"
  - "explainability"
  - "defect heatmap"
  - "Chain-of-Thought"
date: 2026-05-08
content_hash: 338721ccafb2d5bb
---

# HEIE: MLLM-Based Hierarchical Explainable AIGC Image Implausibility Evaluator

**Conference**: CVPR 2025  
**arXiv**: [2411.17261](https://arxiv.org/abs/2411.17261)  
**Code**: [https://yfthu.github.io/HEIE/](https://yfthu.github.io/HEIE/) (project page)  
**Area**: Multimodal VLM  
**Keywords**: AIGC image evaluation, multimodal large language model, explainability, defect heatmap, Chain-of-Thought

## TL;DR

This work proposes HEIE, a Hierarchical Explainable AIGC Image Implausibility Evaluator based on Multimodal Large Language Models (MLLMs). Through a CoT-driven trinity evaluator, it simultaneously outputs defect heatmaps, scores, and textual explanations. An Adaptive Hierarchical Implausibility Mapper is employed to achieve precise localization of both global and local defects, achieving state-of-the-art (SOTA) performance on the RichHF-18K and AbHuman datasets.

## Background & Motivation

**Background**: AIGC image generation technologies (e.g., Stable Diffusion, DALL·E 3) have advanced rapidly, yet generated images often suffer from quality defects like artifacts, unnatural textures, and structural errors. Existing evaluation methods predominantly output scalar scores, and only a few works (e.g., RichHF) have begun to predict defect heatmaps.

**Limitations of Prior Work**: (1) Task-specific small models (e.g., RAHF) lack explainability—they can localize defects but fail to explain "why there is an issue here," making it difficult for users to understand and improve; (2) Task-specific models lack common sense and logical reasoning abilities, and limited training data leads to poor generalization. Conversely, directly employing MLLMs (e.g., GPT-4o) also poses challenges: (1) difficulty in precisely localizing fine-grained defects (e.g., tiny areas like eye corners or fingers); (2) inability to output pixel-level heatmaps, as they are typically restricted to text outputs.

**Key Challenge**: Task-specific small models excel at pixel-level localization but lack understanding and explanation capabilities, whereas MLLMs excel at understanding and reasoning but lack fine-grained localization and pixel-level output capabilities. How to combine the strengths of both?

**Goal**: (1) Enable MLLMs to output pixel-level implausibility heatmaps; (2) achieve synergetic output of heatmaps, scores, and textual explanations; (3) precisely localize both large global defects and tiny local defects.

**Key Insight**: Special `[MAP]` and `[SCORE]` tokens are designed to "bridge" the high-level semantic understanding of MLLMs to pixel-level outputs. Additionally, CoT decomposes the complex evaluation task into a chain of subtasks from easy to hard, enabling the heatmaps, scores, and explanations to mutually reinforce each other.

**Core Idea**: By introducing special tokens and a hierarchical mapper into the MLLM, this work achieves CoT-driven Trinity explainable AIGC image implausibility evaluation consisting of heatmaps, scores, and explanations.

## Method

### Overall Architecture

An input AIGC image first undergoes feature extraction via a ViT. The extracted image features are then fed into the LLM (based on InternVL-8B) to proceed sequentially through a CoT pipeline: image description $\rightarrow$ problematic region identification $\rightarrow$ `[MAP]` token injection $\rightarrow$ problem analysis $\rightarrow$ `[SCORE]` token injection. The features of the `[MAP]` token and image features are processed by the Adaptive Hierarchical Implausibility Mapper to generate a heatmap. The features of the `[SCORE]` token, combined with the heatmap, are processed by the Verisimilitude Scorer to output a score, while the LLM simultaneously outputs a textual explanation.

### Key Designs

1. **Adaptive Hierarchical Implausibility Mapper**:

    - **Function**: Generate pixel-level defect heatmaps from MLLM while handling both global and local defects.
    - **Mechanism**: Three-layered design. **Base Mapper**: Define a special `[MAP]` token in the LLM, extract its last-layer hidden state feature $T$, and fuse it with ViT image features $F$ via two bidirectional cross-attention layers ($T\rightarrow F$ and $F\rightarrow T$) to generate a heatmap. **Hierarchical Mapping**: The image is adaptively segmented into $N$ patches based on resolution. The image encoder separately processes the thumbnail (global features $F_g$) and each patch (local features $F_i$). The LLM outputs $N$ local `[MAP]` tokens and 1 global `[MAP]` token to generate a local heatmap $H_l$ (built by splicing patch heatmaps) and a global heatmap $H_g$. **Adaptive Fusion**: The two heatmaps are modeled as Laplace distributions and adaptively fused using uncertainty estimates $p_{\text{uncertainty}} = e^{-\sigma}$ as weights.
    - **Design Motivation**: AIGC image defects can be either global (e.g., an extra leg) or local (e.g., deformed fingers), requiring detection at different granularities. The uncertainty-based fusion allows the model to decide whether to trust the global or local prediction.

2. **CoT-Driven Explainable Trinity Evaluator**:

    - **Function**: Co-generate heatmaps, scores, and textual explanations through chain-of-thought reasoning.
    - **Mechanism**: A five-step CoT pipeline is designed: (1) Image description: LLM describes key image elements; (2) Problematic region identification: localizes potential issues based on the description; (3) `[MAP]` token: injects defect information into the mapper based on the analysis; (4) Problem analysis: provides detailed textual explanations (type, cause, etc.) based on the localization; (5) `[SCORE]` token: injects overall scoring information based on comprehensive understanding. This easy-to-hard task decomposition fully leverages the step-by-step reasoning capability of LLMs, with each step's output providing context for the next.
    - **Design Motivation**: Having the LLM complete the complex evaluation task all at once yields poor performance. CoT decomposition correlates and mutually enhances the heatmap, analysis, and score: textual descriptions provide semantic context to guide heatmap generation, the visual saliency of the heatmap aids in score quantification, and the score in turn calibrates the focus of the heatmap.

3. **Verisimilitude Scorer**:

    - **Function**: Predict the overall realism score of the image.
    - **Mechanism**: Define a `[SCORE]` token in the LLM, extract its hidden state and regress the initial score $S_{\text{token}}$ through an FFN. Concurrently, extract a heatmap score $S_{\text{map}}$ from the predicted heatmap using a convolution network and an FFN. The final score $S = \text{Calib}(S_{\text{token}}, S_{\text{map}})$ fuses the two via a calibration function.
    - **Design Motivation**: LLMs are insensitive to direct numerical outputs (directly outputting numeric scores performs poorly). Regressing the hidden state of a special token encodes score information more accurately. The heatmap and score are strongly correlated, and fusing them improves accuracy.

### Loss & Training

The heatmap employs a focal loss to tackle the issue of imbalanced positive and negative samples. Both heatmaps of the hierarchical mapper are trained using the negative log-likelihood of a Laplace distribution:

$$\min_{H,\sigma} \left(\frac{\sqrt{2}}{\sigma}|H - H^{gt}| + \log(\sigma)\right)$$

which simultaneously learns the prediction and the uncertainty. Based on InternVL-8B, the model is fine-tuned using DeepSpeed, with a learning rate of $3 \times 10^{-4}$, a warmup ratio of 0.03, and a batch size of 16.

## Key Experimental Results

### Main Results

**RichHF-18K dataset heatmap + scoring:**

| Method | MSE (All) ↓ | KLD ↓ | CC ↑ | AUC-Judd ↑ | PLCC ↑ | SRCC ↑ |
|------|-------------|-------|------|-----------|--------|--------|
| CLIP encoder (fine-tuned) | 0.01437 | 2.462 | 0.251 | 0.747 | 0.390 | 0.378 |
| RAHF (augmented) | 0.00920 | 1.652 | 0.556 | 0.913 | 0.693 | 0.681 |
| **HEIE (Ours)** | **0.00825** | **1.634** | **0.574** | **0.915** | **0.697** | **0.683** |

**Textual Explanation Quality (Expl-AIGI-Eval):**

| Method | GPT-4o Eval ↑ | Human Eval ↑ |
|------|--------------|-------------|
| GPT-4o | 3.828 | 3.999 |
| Claude-3.5-Sonnet | 3.938 | 4.081 |
| **HEIE (Ours)** | **4.582** | **4.353** |

### Ablation Study

**Hierarchical Mapper Ablation (RichHF-18K):**

| Configuration | MSE ↓ | KLD ↓ | CC ↑ |
|------|-------|-------|------|
| Global token only | 0.01071 | 1.950 | 0.502 |
| Local token only | 0.00980 | 1.921 | 0.504 |
| Global + Local, fixed weights | 0.00954 | 1.874 | 0.511 |
| Global + Local, learnable weights | 0.00873 | 1.680 | 0.557 |
| Global + Local, uncertainty-adaptive | **0.00825** | **1.634** | **0.574** |

**CoT System Ablation:**

| Configuration | Heatmap MSE ↓ | Heatmap CC ↑ | Score PLCC ↑ |
|------|-----------|----------|-----------|
| w/o CoT Text | 0.00913 | 0.553 | 0.669 |
| w/ CoT Text | 0.00825 | 0.574 | 0.697 |
| w/ GT CoT Text | 0.00792 | 0.580 | 0.701 |

### Key Findings

- Uncertainty-adaptive fusion significantly outperforms fixed weights (CC: 0.574 vs 0.511), indicating that different images require different global/local weights.
- CoT textual reasoning in turn boosts the accuracy of both heatmaps and scores (MSE: 0.00825 vs 0.00913), validating that the three outputs indeed mutually reinforce each other.
- HEIE's textual explanation score outperforms GPT-4o and Claude-3.5-Sonnet (4.582 vs 3.828/3.938), indicating that task-specific CoT fine-tuning is more effective than general large models.
- In zero-shot cross-domain generalization experiments, HEIE significantly outperforms small-model baselines, verifying the advantage of MLLMs' common-sense knowledge.

## Highlights & Insights

- **Special tokens bridging text and pixels**: By utilizing special `[MAP]` and `[SCORE]` tokens, the LLM, which originally only outputs text, is elegantly enabled to "indirectly" output pixel-level heatmaps and regression scores. This design pattern can be extended to other tasks requiring MLLMs to output non-text modalities.
- **Easy-to-hard CoT task decomposition**: The five-step progressive reasoning chain naturally allows the output of each step to provide context for the next. In particular, the "localize before explaining" paradigm aligns with human cognitive processes. The key innovation lies in the mutual reinforcement of the three components rather than making independent predictions.
- **Expl-AIGI-Eval dataset construction pipeline**: A three-stage pipeline consisting of visual prompting, free-form LLM output, and ICL formatting is proposed to construct explanatory annotations, which can be reused for other tasks requiring fine-grained labeling.

## Limitations & Future Work

- Built upon InternVL-8B, the inference cost is relatively high, hindering real-time evaluation of a massive volume of generated images.
- Heatmap prediction relies on the image feature resolution of the ViT, which might still lack precision for extremely small defects (e.g., 1-2 pixel-level flaws).
- The annotation of the Expl-AIGI-Eval dataset relies on Claude-3.5 and GPT-4o, meaning the annotation quality is bounded by the capabilities of these models.
- Temporal consistency evaluation for AIGC video content is not yet explored.

## Related Work & Insights

- **vs RAHF**: RAHF is a task-specific small model that predicts heatmaps but cannot explain them. In contrast, HEIE utilizes the reasoning capabilities of MLLMs to achieve explainable evaluation and obtains better zero-shot generalization via the common sense of the MLLM.
- **vs Direct evaluation by GPT-4o**: GPT-4o can understand and explain but fails to output pixel-level heatmaps and is insensitive to subtle defects. HEIE addresses both problems using special tokens and a hierarchical mapper.
- **Potential combination with segmentation models like SAM**: HEIE's `[MAP]` token mechanism is similar to prompt-based segmentation in SAM, which points to future explorations of combining the two.

## Rating

- Novelty: ⭐⭐⭐⭐ First to utilize MLLMs for explainable AIGC image defect heatmap prediction; the CoT-driven trinity design is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Conducted on two datasets, with zero-shot generalization, detailed ablation, and human evaluation; extremely thorough.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-defined problem.
- Value: ⭐⭐⭐⭐ Holds direct guiding significance for both AIGC image quality evaluation and the improvement of generative models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MLLM-as-a-Judge for Image Safety without Human Labeling](mllm-as-a-judge_for_image_safety_without_human_labeling.md)
- [\[ICCV 2025\] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/aigi-holmes_towards_explainable_and_generalizable_ai-generated_image_detection_v.md)
- [\[CVPR 2025\] Efficient Motion-Aware Video MLLM](efficient_motion-aware_video_mllm.md)
- [\[ACL 2025\] VF-Eval: Evaluating Multimodal LLMs for Generating Feedback on AIGC Videos](../../ACL2025/multimodal_vlm/vf_eval_aigc_video_feedback.md)
- [\[ICCV 2025\] DocThinker: Explainable Multimodal Large Language Models with Rule-based Reinforcement Learning for Document Understanding](../../ICCV2025/multimodal_vlm/docthinker_explainable_multimodal_large_language_models_with.md)

</div>

<!-- RELATED:END -->
