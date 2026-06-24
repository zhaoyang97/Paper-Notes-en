---
title: >-
  [Paper Note] R-VLM: Region-Aware Vision Language Model for Precise GUI Grounding
description: >-
  [ACL 2025][Multimodal VLM][GUI grounding] Proposes R-VLM, which introduces region proposals and IoU-aware losses from traditional object detection into VLM-based GUI element grounding. Through two-stage zoom-in inference and an IoU-weighted cross-entropy loss, it achieves an average improvement of 13% in grounding accuracy on ScreenSpot and AgentStudio.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "GUI grounding"
  - "vision-language models"
  - "region-aware"
  - "IoU loss"
  - "two-stage zoom-in"
  - "GUI automation"
date: 2026-05-08
content_hash: 63da8cad216ad520
---

# R-VLM: Region-Aware Vision Language Model for Precise GUI Grounding

**Conference**: ACL 2025  
**arXiv**: [2507.05673](https://arxiv.org/abs/2507.05673)  
**Authors**: Joonhyung Park (KAIST), Peng Tang, Sagnik Das, Srikar Appalaraju, Kunwar Yashraj Singh, R. Manmatha, Shabnam Ghadar (AWS AI Labs)  
**Code**: Closed source  
**Area**: Multimodal VLM  
**Keywords**: GUI grounding, vision-language models, region-aware, IoU loss, two-stage zoom-in, GUI automation  

## TL;DR

Proposes R-VLM, which introduces region proposals and IoU-aware losses from traditional object detection into VLM-based GUI element grounding. Through two-stage zoom-in inference and an IoU-weighted cross-entropy loss, it achieves an average improvement of 13% in grounding accuracy on ScreenSpot and AgentStudio.

## Background & Motivation

### Background
GUI automation agents need to precisely locate interface elements (buttons, icons, text boxes, etc.) in screenshots, which is the foundation for all subsequent action execution. Existing vision-only GUI agents directly predict element coordinates from the full screenshot, facing two core challenges: (1) Screenshots contain a large number of irrelevant elements, with complex layouts and multiple scales; (2) VLMs treat coordinates as discrete tokens trained with standard cross-entropy, lacking learning signals that reflect localization quality.

### Limitations of Prior Work
- Although methods like **SeeClick** have made progress in GUI grounding pre-training, they directly predict coordinates from the full image, and small elements are particularly prone to localization failure.
- Existing methods train coordinate tokens using standard cross-entropy loss, failing to guide the model to focus on localization accuracy in the way IoU regression loss does in object detection.
- The authors analyzed the IoU distribution of SeeClick's predictions on ScreenSpot and found that the predicted IoU values were generally low (concentrated in the 0-0.3 range), and the accuracy for small elements was significantly lower than for large elements.

### Design Motivation
Adapts two strategies proven effective in classical object detection—**region proposal + crop-and-zoom** and **IoU-aware training objectives**—to the VLM GUI grounding scenario to compensate for the inherent limitations of VLMs in precise localization.

## Method

### Overall Architecture
Based on an existing VLM (Qwen-VL), R-VLM introduces two modules: (1) two-stage zoom-in grounding (inference stage) + instruction tuning on zoomed-in data (training stage); (2) IoU-aware weighted cross-entropy loss. The two are complementary—the IoU loss makes the first-stage predictions more accurate, and a more accurate region proposal, in turn, makes the second-stage zoom-in more effective.

### Key Designs 1: Two-Stage Zoom-in Grounding

**Inference Workflow**:
1. First Stage: Given the full screenshot and user instruction, the model predicts an initial bounding box, which serves as the region proposal.
2. Determine the zoom factor $k$ based on the size of the initial prediction—the smaller the element, the larger the zoom.
3. Crop and zoom in on the region centered around the initial prediction, and feed it into the model for the second-stage prediction.
4. Inverse-transform the coordinates predicted in the zoomed-in view back to the original image coordinates as the final output.

**Zoom-in Instruction Tuning Data Generation**:
- Instead of directly using the ground-truth (GT) box as the crop region, random perturbations are applied to the GT to generate noisy boxes with $\text{GIoU} > \sigma$ to simulate imprecise first-stage predictions.
- After cropping and zooming, specific instruction templates are attached (e.g., "Given the zoomed-in view centered on the initial prediction, predict a detailed bounding box for [INSTRUCTION]").
- Label coordinates are updated accordingly to relative coordinates within the zoomed-in view.

**Key Advantage**: This method can be applied **without training** (training-free) to any VLM, yielding significant improvements simply by executing two forward passes during inference.

### Key Designs 2: IoU-Aware Weighted Cross-Entropy Loss

**Core Idea**: Generate $M$ pseudo-GT boxes around the GT box, and assign different cross-entropy weights based on the $\text{GIoU}$ value of each pseudo-box relative to the true GT. The higher the $\text{GIoU}$, the closer the weight is to 1; the lower the $\text{GIoU}$, the smaller the weight. This guides the model to understand that "predictions with higher IoU are better than those with lower IoU."

**Loss Function**:
$$\mathcal{L}_{\text{IoU}}^{\text{CE}} = -\sum_{i=1}^{M} w_{\text{IoU}}^{(i)} \mathbf{b}^{(i)} \log \hat{\mathbf{b}}^{(i)} - \sum_{j=1}^{N} y_{\text{other}}^{(j)} \log \hat{y}_{\text{other}}^{(j)}$$

Where the weight is $w_{\text{IoU}}^{(i)} = 1 + \frac{1}{2}\log(\text{GIoU}(\mathbf{b}^{(i)}, \mathbf{b}^{(0)}))$, using a logarithmic scale to apply larger penalties to pseudo-boxes with low IoU.

**Efficient Implementation**:
- Concatenate the $M$ pseudo-boxes after the GT box to complete the loss computation for all pseudo-boxes in a **single forward pass**.
- Modify attention mask: Prevent pseudo-boxes from attending to each other.
- Modify RoPE positional encoding: Copy the positional encoding of the GT box to all pseudo-boxes to ensure the model outputs only a single coordinate during inference.

### Key Designs 3: Synergistic Effect
Ablation studies show a complementary effect between IoU loss and zoom-in grounding: IoU loss improves the precision of the first-stage prediction (bringing the region proposal closer to the target), which in turn makes the second-stage zoom-in more effective.

## Key Experimental Results

### Table 1: ScreenSpot GUI Grounding Accuracy

| Method | Mobile Text | Mobile Icon | Desktop Text | Desktop Icon | Web Text | Web Icon | Average |
|------|:-----------:|:-----------:|:------------:|:------------:|:--------:|:--------:|:----:|
| GPT-4V | 22.6 | 24.5 | 20.2 | 11.8 | 9.2 | 8.8 | 16.2 |
| CogAgent | 67.0 | 24.0 | 74.2 | 20.0 | 70.4 | 28.6 | 47.4 |
| SeeClick | 78.0 | 52.0 | 72.2 | 30.0 | 55.7 | 32.5 | 53.4 |
| **R-VLM** | **85.0** | **61.1** | **81.4** | **52.8** | **66.5** | **51.4** | **66.3 (+12.9)** |

The improvements on Desktop Icon and Web Icon are particularly significant (+22.8 and +18.9 respectively). In these scenarios, small icons are densely arranged, making the zoom-in strategy most effective.

### Table 2: AgentStudio GroundUI-1K Grounding Accuracy

| Method | Web | Desktop | Mobile | Average |
|------|:---:|:-------:|:------:|:----:|
| GPT-4o | 7.5 | 8.3 | 26.3 | 13.4 |
| Claude-3.5-Sonnet | 13.0 | 14.0 | 26.3 | 17.3 |
| Gemini-1.5-pro | 31.2 | 24.3 | 51.3 | 35.2 |
| SeeClick | 64.3 | 44.3 | 73.7 | 61.1 |
| **R-VLM** | **76.5** | **65.3** | **79.7** | **74.1 (+13.0)** |

Under the condition of using the same pre-training data and architecture, R-VLM substantially outperforms commercial large models and specialized agent models. The absolute improvement of 21% on Desktop is the most prominent.

### Table 3: Ablation Study (ScreenSpot)

| IoU Loss | Zoom-in Tuning | Zoom-in Inference | Average Accuracy |
|:-------:|:-------:|:-------:|:----------:|
| ✗ | ✗ | ✗ | 53.4 |
| ✗ | ✗ | ✓ | 61.7 (+8.3) |
| ✗ | ✓ | ✓ | 63.9 (+10.5) |
| ✓ | ✓ | ✓ | 66.4 (+12.9) |

Adding zoom-in only during inference yields an 8.3% improvement, and the combination of all three components achieves a total improvement of 12.9%.

### Table 4: GUI Navigation Tasks

On AITW, R-VLM achieves an average action matching score of 64.9% (+5.6%) and a click accuracy of 71.0% (+4.6%). On Mind2Web, the step success rate under the Cross-Website setting improves by 9.7% (16.4% -> 26.1%), showing that precise grounding directly translates into improved navigation success rates.

## Key Findings

- **Significant Right Shift in IoU Distribution**: The predicted IoU distribution of R-VLM shifts from being concentrated in the 0-0.3 range to being dominated by high IoU ranges, validating that the IoU loss indeed teaches the model to "pursue high IoUs."
- **Largest Gain on Small Elements**: Grouped analysis by element size reveals that while the baseline has extremely low accuracy on small elements, R-VLM substantially improves small element localization through the zoom-in mechanism.
- **Effective Even Training-Free**: Directly applying the two-stage zoom-in (without retraining) on the stronger UGround model (10M data, LLaVA-NeXT architecture) improves ScreenSpot by 1.8% and Multimodal-Mind2Web by 3.2% on average, proving the generalizability of the method.
- **Diminishing Marginal Returns of Zoom Steps**: Going from 2 steps to 4 steps only yields a 1.1% improvement, but doubles the inference latency (2.7s -> 5.6s/sample), making 2 steps the most cost-effective choice.

## Highlights & Insights

- **Elegant Transfer of Traditional Detection Concepts**: Adapts the region proposal + IoU regression approach from the Faster R-CNN era into the token space of VLMs, resolving the contradiction that "VLM coordinates are discrete tokens and cannot directly optimize IoU."
- **Efficient IoU Loss Implementation**: Compresses $M+1$ forward passes into a single pass through attention mask modifications and RoPE sharing, incurring almost no additional training overhead.
- **Plug-and-Play Inference Strategy**: Two-stage zoom-in requires no modifications to the training pipeline or model architecture; any VLM can use it directly.
- **Clever Data Augmentation Design**: Simulates actual first-stage prediction errors by controlling the offset of noisy boxes via a GIoU threshold, which is closer to real inference scenarios than directly cropping with GT.

## Limitations & Future Work

- **Ceiling of First-Stage Precision**: If the initial prediction deviates severely from the target (the zoomed-in region does not contain the target), the second stage cannot correct it. The authors suggest that generating multiple region proposals to improve recall is a potential direction for future work.
- **Only Validated on Qwen-VL**: The IoU loss and zoom-in tuning in the main experiments were only validated on Qwen-VL (9.6B) and have not been tested on larger or newer models (e.g., Qwen2-VL, InternVL).
- **Doubled Inference Latency**: Two-stage zoom-in requires two forward passes, increasing the inference time from 1.4s to 2.7s per sample.
- **GUI-Specific Method**: The zoom-in strategy relies on the prior that GUI elements are relatively small compared to the entire screenshot, and its performance has not been validated on natural image grounding.
- **Fixed Pre-training Data**: The model uses SeeClick's 1M pre-training data, leaving the impact of scaling up the data size unexplored.

## Related Work & Insights

- **SeeClick** (ACL 2024): The direct baseline for this paper, pre-trained on Qwen-VL for GUI grounding. R-VLM achieves substantial improvements using the exact same data and architecture.
- **CogAgent** (CVPR 2024): Introduces a specialized high-resolution path to process GUI screenshots, but its grounding accuracy remains low.
- **UGround** (ICLR 2025): Large-scale pre-training on LLaVA-NeXT with 10M data. R-VLM's training-free zoom-in strategy still achieves a 3%+ improvement on it.
- **Faster R-CNN** Series: The core inspiration for R-VLM; the two-stage detection concept still proves effective in new scenarios after twenty years.
- **ZoomEye** (Shen et al. 2024): Uses a similar zoom-in concept but with tree-structure search. R-VLM's approach is simpler—requiring only a single zoom-in.

**Insights**: The bottleneck of VLMs in precise localization tasks is not just a matter of model capability, but also one of training objectives and inference strategies. Paradigms well-proven in traditional CV (such as region-based detection) remain valuable in the VLM era; the key lies in how to adapt them to tokenized output spaces.

## Rating

- Novelty: ⭐⭐⭐⭐ — Adapting classical object detection methods to VLMs is clever, though the zoom-in idea itself has precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers both grounding and navigation tasks across 4 benchmarks, including training-free validation and detailed ablation analyses.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, with preliminary analysis making a solid case for the underlying motivation.
- Value: ⭐⭐⭐⭐ — The 13% grounding improvement has practical significance, and the plug-and-play nature lowers the barrier to entry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Aria-UI: Visual Grounding for GUI Instructions](aria-ui_visual_grounding_for_gui_instructions.md)
- [\[CVPR 2026\] DRS-GUI: Dynamic Region Search for Training-Free GUI Grounding](../../CVPR2026/multimodal_vlm/drs-gui_dynamic_region_search_for_training-free_gui_grounding.md)
- [\[ACL 2025\] Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference](activating_distributed_visual_region_within_llms_for_efficient_and_effective_vis.md)
- [\[ACL 2025\] RATE-Nav: Region-Aware Termination Enhancement for Zero-shot Object Navigation with Vision-Language Models](rate-nav_region-aware_termination_enhancement_for_zero-shot_object_navigation_wi.md)
- [\[ACL 2025\] Response Wide Shut: Surprising Observations in Basic Vision Language Model Capabilities](response_wide_shut_surprising_observations_in_basic_vision_language_model_capabi.md)

</div>

<!-- RELATED:END -->
