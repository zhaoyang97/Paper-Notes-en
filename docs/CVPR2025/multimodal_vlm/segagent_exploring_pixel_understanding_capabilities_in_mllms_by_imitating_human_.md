---
title: >-
  [Paper Note] SegAgent: Exploring Pixel Understanding Capabilities in MLLMs by Imitating Human Annotator Trajectories
description: >-
  [CVPR 2025][Multimodal VLM][Interactive segmentation] SegAgent models referring expression segmentation as an iterative operation process of a human annotator—the MLLM observes the current mask state and predicts the next click location, according to which the interactive segmentation model updates the mask, obtaining the final segmentation result after multiple steps of iteration. It significantly improves segmentation accuracy in complex scenarios through the StaR+ policy i…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Interactive segmentation"
  - "MLLM annotation agent"
  - "Multi-step MDP"
  - "Process reward model"
  - "Policy improvement"
date: 2026-05-08
content_hash: 9cf08001dbb8887a
---

# SegAgent: Exploring Pixel Understanding Capabilities in MLLMs by Imitating Human Annotator Trajectories

**Conference**: CVPR 2025  
**arXiv**: [2503.08625](https://arxiv.org/abs/2503.08625)  
**Code**: To be confirmed  
**Area**: Multimodal VLM  
**Keywords**: Interactive segmentation, MLLM annotation agent, Multi-step MDP, Process reward model, Policy improvement

## TL;DR
SegAgent models referring expression segmentation as an iterative operation process of a human annotator—the MLLM observes the current mask state and predicts the next click location, according to which the interactive segmentation model updates the mask, obtaining the final segmentation result after multiple steps of iteration. It significantly improves segmentation accuracy in complex scenarios through the StaR+ policy improvement and PRM with tree search.

## Background & Motivation

**Background**: Existing MLLM segmentation methods (such as LISA) typically require the MLLM to output implicit tokens to generate masks via a decoder in a single step. These methods are essentially single-step predictions and lack the capability of step-by-step refinement for complex shapes.

**Limitations of Prior Work**: (1) The single-step output paradigm demands extremely high fine-grained pixel understanding, whereas the visual encoders of MLLMs tend to lose spatial locality after abstractions like the Q-former. (2) The annotation quality and complexity of existing datasets like RefCOCO are limited (usually 1-2 clicks are enough to reach the target IoU), which fails to thoroughly evaluate the pixel-level understanding capability of MLLMs. (3) Human annotators perform segmentation through multiple rounds of interactive iterations, whereas existing methods completely ignore this progressive reasoning process.

**Key Challenge**: MLLMs excel at global semantic understanding and coarse-grained localization but struggle with pixel-level fine-grained segmentation. Conversely, interactive segmentation models (such as SAM) are proficient in fine-grained segmentation but do not comprehend natural language. How can the two be effectively combined?

**Goal**: How to enable the MLLM to act as a human annotator and cooperate with interactive segmentation tools through multi-step iteration to achieve high-quality referring segmentation?

**Key Insight**: Model the segmentation task as a Markov Decision Process (MDP)—the state is the current mask (overlaid on the original image), actions are predicted positive/negative click coordinates (pure text output), transitions are completed by the interactive segmentation model, and rewards are defined by the IoU with the GT (ground truth). Thus, the MLLM only needs to make decisions on "where to click," leaving the fine-grained segmentation to specialized models.

**Core Idea**: Enable the MLLM to mimic the click trajectories of human annotators, completing referring segmentation through multi-step interactive segmentation iterations, and utilizing RL-style policy improvement and a process reward model to enhance decision quality.

## Method

### Overall Architecture
The input consists of an image $I$ + a text prompt $P$ + an empty initial mask $M_0$. At each step, the current mask is semi-transparently overlaid onto the original image as the observation. The MLLM outputs the next click action (positive/negative prompt + normalized coordinates), based on which the interactive segmentation model (SAM/SimpleClick) updates the mask. The process iterates for $T$ steps ($T=7$ for RefCOCO, $T=11$ for HRES), taking the final mask as the output.

### Key Designs

1. **HLMAT: Human-Like Mask Annotation Task**:

    - **Function**: Automatically generate annotation trajectory data required for MLLM training.
    - **Mechanism**: Given a GT mask, a rule-based click simulator $F_{sim}$ is used to generate trajectories. At each step, the simulator identifies the largest error region (false positive/false negative) between the current mask and the GT, placing the next click at its center. Three filtering mechanisms guarantee trajectory quality: limiting the maximum length $T$, terminating when the IoU reaches $\tau_{stop}$, and discarding actions with an IoU gain smaller than $\tau_{diff}$.
    - **Design Motivation**: Manually collecting annotation trajectories is extremely costly, whereas a rule-based simulator can automatically generate large-scale training data from existing segmentation datasets. The MLLM can then learn to "observe mask state $\rightarrow$ decide where to click" through instruction tuning on these trajectories.

2. **StaR+ Policy Improvement**:

    - **Function**: Improve the policy by self-iteratively generating better training data.
    - **Mechanism**: Starting from the SFT baseline model $S_0$, the model generates new trajectories on the training set by itself, which are filtered by a reward function—any action where the step-wise IoU improvement falls below a threshold is replaced by a corrective action generated by $F_{sim}$. The corrected trajectories are then merged into the training set for re-finetuning. Distinct from the original STaR, filtering is performed based on step-wise reward changes rather than global trajectory correctness.
    - **Design Motivation**: SFT only learns the behavioral distribution of the simulator. Iterative development using the model's actual inference trajectories can cover broader distributions and correct failure modes. Experiments show that StaR+ achieves a $+15.12\%$ IoU improvement on ThinObject5K.

3. **Process Reward Model (PRM) + Greedy Tree Search**:

    - **Function**: Select the optimal step-wise action during inference via search.
    - **Mechanism**: The MLLM is trained to simultaneously predict the reward value at each step (output in textual format as "Current mIoU: 0.75"). During inference, $K$ candidate actions are generated through Multinomial Sampling at each step. Interactive segmentation is performed on each candidate and evaluated by the PRM to predict its reward. The action with the highest PRM score is selected. A simple greedy search is adopted instead of complex methods like MCTS.
    - **Design Motivation**: Segmentation is a task where process rewards (IoU) are naturally well-defined, allowing step-wise computation. The PRM enables the model to perform self-evaluation, and tree search expands the search space during inference. With $K=3$, the IoU on DIS5K increases from $81.17$ to $88.60$ ($+7.43$).

### Loss & Training
Standard instruction tuning loss (autoregressive cross-entropy) is used, with the image encoder frozen and the LLM + projector fine-tuned. Trained for $2$ epochs on $8 \times 80\text{GB}$ GPUs using DeepSpeed ZeRO-2. The base models support LLaVA-v1.5-7B and Qwen-VL-7B, while the interactive segmentation supports SAM and SimpleClick.

## Key Experimental Results

### Main Results

| Dataset | Method | IoU |
|--------|------|------|
| refCOCO val | LISA+SAM | 74.9 |
| refCOCO val | SAM4MLLM-Qwen | 77.1 |
| refCOCO val | **SegAgent-Qwen+SClick** | **79.69** |
| refCOCO+ val | LISA+SAM | 65.1 |
| refCOCO+ val | **SegAgent-Qwen+SClick** | **72.49** |
| refCOCOg val(U) | LISA+SAM | 67.9 |
| refCOCOg val(U) | **SegAgent-Qwen+SClick** | **75.11** |

### Ablation Study (HRES Dataset)

| Configuration | DIS5K IoU | ThinObject5K IoU |
|------|---------|---------|
| Baseline SFT | 71.45 | 71.45 |
| + StaR+ Policy Improvement | 78.81 (+7.36) | 86.57 (+15.12) |
| + PRM (K=1) | 81.17 (+2.36) | 75.54 (+4.09) |
| + Tree Search (K=3) | **88.60** (+7.43) | **86.13** (+10.59) |

### Key Findings
- **Multi-step iteration significantly outperforms single-step**: Simple datasets like RefCOCO only require 1-2 steps, whereas complex scenarios like DIS5K/ThinObject5K necessitate 7-11 steps, demonstrating the necessity of the iterative approach.
- **An interesting comparison between LLaVA and Qwen**: LLaVA+SAM is stronger in fine-grained mask refinement, whereas Qwen+SimpleClick performs better in coarse-grained localization. The Q-former architecture loses spatial locality during semantic abstraction.
- **Substantial improvement from StaR+**: Particularly on ThinObject5K with a $+15.12\%$ IoU improvement, revealing significant differences between the distribution of the model's self-generated trajectories and the simulator trajectories, highlighting the great value of self-improvement.
- **Complementary effects of PRM and tree search**: Using PRM alone ($K=1$) yields limited improvements, but when paired with tree search ($K=3$), the improvement is substantial, indicating that expanding the search space is critical for quality.

## Highlights & Insights
- **Reformulating segmentation as an MDP is a paradigm shift**: Instead of demanding the MLLM to output a mask in a single step, it acts as an "annotator" to make decisions, while fine-grained operations are delegated to professional tools. This human-AI collaborative paradigm can be migrated to other tasks requiring fine-grained annotation.
- **Natural suitability of PRM**: The IoU in segmentation tasks serves as a perfect process reward signal—it can be computed at each step without the need to train a separate reward model. This makes the application of RL-style methods to segmentation tasks exceptionally elegant.
- **Pure text coordinate output**: It requires no modifications to the MLLM architecture and no special tokens; actions are fully represented in text (e.g., "Positive point: (175,483)"), preserving the generality of the MLLM.

## Limitations & Future Work
- Only one round of StaR+ improvement is explored; more rounds could potentially bring further gains.
- Only greedy search is used instead of MCTS; more sophisticated search strategies might yield better results.
- It is unable to undo prior incorrect clicks, which may cause error propagation in complex scenarios.
- The advantage is not distinct enough on simple datasets like RefCOCO (where 1-2 steps suffice); the value of the method is fully realized only in complex scenarios.
- The issue where Qwen's Q-former architecture loses spatial locality is worth the VLM community's attention.

## Related Work & Insights
- **vs LISA**: LISA requires the MLLM to output implicit tokens and a decoder to generate masks in one step; SegAgent uses multi-step iteration with specialized tools, achieving a $+4.8\%$ IoU improvement on RefCOCO and $+7.4\%$ on RefCOCO+.
- **vs SAM4MLLM**: SAM4MLLM also employs interactive segmentation but only for a single step; SegAgent's multi-step MDP combined with RL improvement is significantly stronger.
- **Insights for MLLM architectures**: Although Q-former is strong in semantic understanding, it is inferior to direct projection in tasks requiring spatial localization, providing reference value for future VLM designs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Modeling segmentation as a human annotator's MDP is a highly innovative paradigm, and the introductions of StaR+ and PRM are both natural and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ It features comparative evaluations across multiple datasets and model combinations, clear ablation analysis, and introduces the high-quality HRES dataset.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, and the algorithm pseudocode is comprehensive.
- Value: ⭐⭐⭐⭐ The paradigm shift is inspiring, though practical deployment is constrained by the efficiency overhead of multi-step inference calls.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)
- [\[CVPR 2025\] PEACE: Empowering Geologic Map Holistic Understanding with MLLMs](peace_empowering_geologic_map_holistic_understanding_with_mllms.md)
- [\[ACL 2025\] OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference](../../ACL2025/multimodal_vlm/omnialign-v_towards_enhanced_alignment_of_mllms_with_human_preference.md)
- [\[ICLR 2026\] HumanPCR: Probing MLLM Capabilities in Diverse Human-Centric Scenes](../../ICLR2026/multimodal_vlm/humanpcr_probing_mllm_capabilities_in_diverse_human-centric_scenes.md)
- [\[CVPR 2025\] VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos](videoglamm_a_large_multimodal_model_for_pixel-level_visual_grounding_in_videos.md)

</div>

<!-- RELATED:END -->
