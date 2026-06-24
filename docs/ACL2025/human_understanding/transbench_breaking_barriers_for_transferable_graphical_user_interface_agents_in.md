---
title: >-
  [Paper Note] TransBench: Breaking Barriers for Transferable Graphical User Interface Agents in Dynamic Digital Environments
description: >-
  [ACL 2025][Human Understanding][GUI grounding] This paper proposes TransBench, the first benchmark to systematically evaluate the **transferability** (cross-version/cross-platform/cross-app) of GUI Agents. It covers 81 Chinese Apps, 1459 screenshots, and 22K+ annotated instructions. Experiments show that fine-tuning on older versions can effectively transfer to new versions and other platforms, with Android data exhibiting the strongest generalization in cross-platform migrat…
tags:
  - "ACL 2025"
  - "Human Understanding"
  - "GUI grounding"
  - "transferability"
  - "cross-version"
  - "cross-platform"
  - "cross-app"
  - "benchmark"
date: 2026-05-08
content_hash: 4692b85699e016af
---

# TransBench: Breaking Barriers for Transferable Graphical User Interface Agents in Dynamic Digital Environments

**Conference**: ACL 2025  
**arXiv**: [2505.17629](https://arxiv.org/abs/2505.17629)  
**Code**: TransBench (will be released)  
**Area**: Human-Computer Interaction / GUI Agent  
**Keywords**: GUI grounding, transferability, cross-version, cross-platform, cross-app, benchmark

## TL;DR
This paper proposes TransBench, the first benchmark to systematically evaluate the **transferability** (cross-version/cross-platform/cross-app) of GUI Agents. It covers 81 Chinese Apps, 1459 screenshots, and 22K+ annotated instructions. Experiments show that fine-tuning on older versions can effectively transfer to new versions and other platforms, with Android data exhibiting the strongest generalization in cross-platform migration.

## Background & Motivation

**Background**: GUI Agents autonomously operate digital interfaces through natural language instructions, with their core capability being grounding—mapping language intents to visual interface elements. Existing benchmarks like Mind2Web (Web) and GUI-Odyssey (Mobile) focus only on a single platform.

**Limitations of Prior Work**: In real-world environments, apps constantly update their versions (causing interface layout changes), users operate across multiple platforms (iOS/Android/Web), and tasks span across apps (e.g., shopping + watching review videos). Existing Agents behave fragilely in these dynamic scenarios, and there is a lack of benchmarks to systematically evaluate transferability.

**Key Challenge**: No existing dataset simultaneously covers version differences, platform differences, and app differences, making it impossible to evaluate and improve the generalization capability of GUI Agents.

**Goal**: (1) Formally define three levels of transferability for GUI Agents; (2) Construct a comprehensive transferability evaluation benchmark; (3) Verify the patterns of transferability through fine-tuning experiments.

**Key Insight**: Starting from practical usage scenarios, collect screenshots of the same App across different versions and platforms, and annotate unified grounding instructions.

**Core Idea**: Models fine-tuned on older versions/single-platform data can be effectively transferred to new versions and other platforms. Hybrid training with diverse data is the optimal strategy to improve comprehensive transferability.

## Method

### Overall Architecture
TransBench is a **benchmark dataset** and does not propose a new model. The core contributions lie in the data construction and systematic evaluation.

### Key Designs

1. **Three-level Transferability Definition**:

    - **Cross-Version Transfer**: Old Android → New Android of the same App (74.6% of screenshots have significant interface changes)
    - **Cross-Platform Transfer**: Android ↔ iOS ↔ Web (interface differences of the same App across different platforms)
    - **Cross-App Transfer**: Generalization between Apps with similar functions (e.g., JD → Pinduoduo) and Apps with different functions (e.g., Shopping → Finance)
    - Design Motivation: These three dimensions cover the main generalization challenges faced by GUI Agents in real-world usage.

2. **Three-step Data Collection Pipeline**:

    - **Screenshot Collection**: 81 Chinese Apps, covering 15 categories (shopping, video, social, travel, finance, etc.), collecting basic pages (homepage, messages, personal center) and domain-specific pages, totaling 1,459 screenshots (Android Old 393 + Android New 432 + iOS 429 + Web 205)
    - **Bounding Box Annotation**: GUI elements are first automatically detected using OmniParser (>65K boxes), and then manually verified and corrected by 4 annotators using an in-house GUILabeller tool, handling semantically equivalent elements (e.g., icon and text triggering the same action).
    - **Grounding Instruction Generation**: Using Qwen2VL to extract bounding box attributes and page summaries, then employing Qwen-plus to generate natural language instructions, and finally verified manually to achieve 95.5% accuracy, resulting in 22K+ high-quality instructions.

3. **Evaluation Metrics**:

    - **Accuracy (acc)**: Correct if the predicted point falls within the ground-truth (GT) bounding box.
    - **Average Distance (dis)**: Normalized Euclidean distance between the predicted point and the GT center (coordinates normalized to 0-100), serving as a complementary measure of precision.
    - Design Motivation: The distance metric can distinguish between "just inside the box" and "precisely targeted", compensating for the coarse granularity of the accuracy metric.

### Evaluated Models
Six models—CogAgent, SeeClick, Aria-UI, OS-Atlas, UGround, and Qwen2.5VL—covering different foundational bases and training strategies, were evaluated.

### Transferability Fine-Tuning Experiments
Using Aria-UI as the base model, multiple sets of data partitions were designed for fine-tuning:
- **Cross-Version**: Train on Old Android → Test on New Android
- **Cross-Platform**: Train on a single platform → Test on other platforms, and hybrid training
- **Cross-App**: Train on the first 7 categories of Apps → Test on other Apps of the same category and Apps of different categories

## Key Experimental Results

### Main Results (Standard Set, zero-shot)

| Model | Overall acc↑ | Android Old | Android New | iOS | Web |
|------|-------------|-------------|-------------|-----|-----|
| SeeClick | 39.90% | 46.86% | 46.42% | 43.57% | 15.37% |
| CogAgent | 72.16% | 76.04% | 75.70% | 68.61% | 66.69% |
| Aria-UI | 77.51% | 80.97% | 81.38% | 77.61% | 66.86% |
| OS-Atlas | 81.37% | 84.52% | 84.60% | 79.64% | 74.76% |
| UGround | 84.18% | 86.94% | 87.71% | 82.43% | 77.62% |
| **Qwen2.5VL** | **86.43%** | 88.87% | **90.29%** | **84.72%** | **79.79%** |

### Ablation Study (Aria-UI Fine-Tuning Transferability)

| Fine-Tuning Data | Android New | iOS | Web |
|---------|-------------|-----|-----|
| None (Base) | 81.38% | 77.61% | 66.86% |
| Android Old (5K) | **88.36%** | 82.57% | 73.61% |
| iOS (5K) | 87.06% | 82.03% | 73.66% |
| Web (4K) | 84.87% | 80.62% | 66.49% |
| **General Mix (5K)** | 88.15% | **83.15%** | **76.54%** |

### Key Findings
- **Cross-version transfer is effective**: After fine-tuning on the old Android version, the accuracy on the new version increased from 81.38% to 88.36% (+6.98%), and even for newly added UI elements in the new version, accuracy improved from 80.15% to 87.50%. This demonstrates that the grounding knowledge from older versions is transferable.
- **Android data has the strongest generalization**: Fine-tuning on Android data improved iOS performance by +4.96%, exceeding the +4.42% improvement from fine-tuning on iOS's own data; however, Android/iOS data is difficult to transfer to the Web platform.
- **Hybrid data is optimal**: General Mix achieved the best or near-best performance across all platforms, notably improving the Web from 66.86% to 76.54% (+9.68%).
- **Performance ranking: Android > iOS > Web**: All models consistently performed best on Android and worst on Web, reflecting platform heterogeneity.
- **Cross-app transfer is less constrained**: The performance gain gap between cross-app transfer in the same category versus different categories is minor, indicating that version/platform differences are more critical than app differences.

## Highlights & Insights
- **First transferability benchmark**: Systematically defines and evaluates the three-tiered transferability of GUI Agents, filling an important gap. The data construction pipeline (auto-annotation -> manual verification -> LLM-generated instructions) is reusable.
- **Introduction of the distance metric**: Adding a normalized distance metric alongside accuracy reveals that UGround's normalized coordinate output is more robust than Qwen2.5VL's absolute coordinates—offering practical insights for GUI Agent design.
- **The "old data is not obsolete" discovery**: Fine-tuning on older versions not only improves performance on the old version but also boosts performance on new versions and other platforms, highlighting that the value of historical data is often underestimated.

## Limitations & Future Work
- Only covers Chinese Apps; transferability for English/multilingual Apps is not evaluated.
- Grounding only evaluates single-step clicks, without addressing multi-step interactions or complex operations like scrolling/dragging.
- The fine-tuning experiments use only one model (Aria-UI); transferability patterns may differ across different base models.
- The amount of data for the Web platform is relatively small (205 screenshots vs. 825 for Android), which might affect the reliability of conclusions related to the Web.
- Does not evaluate end-to-end task completion rate; only evaluates grounding accuracy.

## Related Work & Insights
- **vs Mind2Web**: Only evaluates Web environments, lacking cross-version/cross-platform dimensions. TransBench is more comprehensive.
- **vs GUI-Odyssey**: Supports cross-app but does not support cross-version/cross-platform.
- **vs ScreenSpot**: Focuses on grounding but does not focus on transferability.
- **vs VisualAgentBench/WebHybrid**: Involves cross-platform but lacks cross-version evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ First to systematically define the three dimensions of GUI Agent transferability; the problem definition is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model and multi-dimensional evaluation, but fine-tuning was performed on only one model.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, systematic experimental design, and rich charts/tables.
- Value: ⭐⭐⭐⭐ Direct reference value for practical GUI Agent deployment; the benchmark can drive community research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] D3-Human: Dynamic Disentangled Digital Human from Monocular Video](../../CVPR2025/human_understanding/d3-human_dynamic_disentangled_digital_human_from_monocular_video.md)
- [\[CVPR 2026\] Breaking Spurious Correlations: Uncertainty-Driven Causal Transformers for AU Detection](../../CVPR2026/human_understanding/breaking_spurious_correlations_uncertainty-driven_causal_transformers_for_au_det.md)
- [\[CVPR 2025\] X-Dyna: Expressive Dynamic Human Image Animation](../../CVPR2025/human_understanding/x-dyna_expressive_dynamic_human_image_animation.md)
- [\[CVPR 2025\] MotionReFit: Dynamic Motion Blending for Versatile Motion Editing](../../CVPR2025/human_understanding/motionrefit_motion_editing.md)
- [\[ICLR 2026\] BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behaviour Analysis](../../ICLR2026/human_understanding/bah_dataset_for_ambivalencehesitancy_recognition_in_videos_for_digital_behaviour.md)

</div>

<!-- RELATED:END -->
