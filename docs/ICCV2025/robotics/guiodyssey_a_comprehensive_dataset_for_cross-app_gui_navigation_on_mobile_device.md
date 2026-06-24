---
title: >-
  [Paper Note] GUIOdyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices
description: >-
  [ICCV 2025][Robotics][GUI navigation] This paper presents GUIOdyssey, the first comprehensive dataset for cross-app GUI navigation on mobile devices (8,334 episodes, 212 apps, 1,357 app combinations), along with OdysseyAgent—a multimodal navigation agent equipped with a history resampling module that significantly improves cross-app task performance while balancing accuracy and inference efficiency.
tags:
  - "ICCV 2025"
  - "Robotics"
  - "GUI navigation"
  - "cross-app"
  - "mobile agent"
  - "multimodal large language model"
  - "history information modeling"
date: 2026-05-08
content_hash: 916c37b022441dd0
---

# GUIOdyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices

**Conference**: ICCV 2025  
**arXiv**: [2406.08451](https://arxiv.org/abs/2406.08451)  
**Code**: [GitHub](https://github.com/OpenGVLab/GUI-Odyssey)  
**Area**: Robotics  
**Keywords**: GUI navigation, cross-app, mobile agent, multimodal large language model, history information modeling

## TL;DR

This paper presents GUIOdyssey, the first comprehensive dataset for cross-app GUI navigation on mobile devices (8,334 episodes, 212 apps, 1,357 app combinations), along with OdysseyAgent—a multimodal navigation agent equipped with a history resampling module that significantly improves cross-app task performance while balancing accuracy and inference efficiency.

## Background & Motivation

GUI navigation agents on smartphones can autonomously execute user instructions, offering substantial value for visually impaired users and general productivity. Recent advances in large foundation models have made autonomous GUI navigation increasingly feasible. However, existing GUI navigation datasets and methods share a fundamental limitation:

**Nearly all datasets are confined to single-app navigation.** In practice, users frequently need to complete tasks spanning multiple apps—e.g., searching for information in a browser and recording it in a notes app, finding a song in a music app and sharing it to social media, or coordinating calendar and messaging apps to schedule events. Such cross-app tasks present three unique challenges:

**Longer action sequences**: Cross-app tasks require an average of 15.3 steps (far exceeding the 5–7 steps typical of single-app tasks), with compounding error propagation risk—a single mistake may invalidate all subsequent steps.

**More complex working memory management**: Critical UI elements and contextual information span multiple apps, requiring the agent to retain key information across app switches.

**Broader functional knowledge**: The agent must understand interaction patterns across different apps (file sharing, email composition, message sending, etc.) and establish workflows among them.

Prior evaluations have shown that **current models perform substantially worse on cross-app tasks than on single-app tasks**. Yet no dedicated cross-app training dataset exists to bridge this gap.

## Method

### Overall Architecture

This work comprises two components: (1) **GUIOdyssey dataset**—a cross-app navigation dataset constructed via human annotation augmented with GPT-4/4o; and (2) **OdysseyAgent**—a multimodal navigation agent fine-tuned on Qwen-VL, equipped with a history resampling module for efficient processing of long-horizon history.

### Key Designs

1. **Dataset Construction Pipeline**: A four-stage pipeline ensures data quality and diversity:

   **(a) Cross-app task proposal**: Six task categories—general utilities, information management, online shopping, media & entertainment, social sharing, and multi-app composite. A total of 91 high-level instruction templates were jointly designed by human participants and GPT-4.

   **(b) Flexible instruction instantiation**: Diversity is ensured through three strategies—replacing items in templates (e.g., "yoga" → "meditation"), selecting different apps to accomplish the same task (e.g., Spotify vs. Google Podcasts), and GPT-4-based paraphrasing to vary expression.

   **(c) Human annotation**: Trained annotators complete instructions step-by-step on Android emulators, recording screenshots and actions at each step. Six device types are covered (Pixel Pro, Tablet, Fold, etc.). The action set comprises 9 types: CLICK, SCROLL, LONG PRESS, TYPE, COMPLETE, IMPOSSIBLE, HOME, BACK, and RECENT.

   **(d) Fine-grained augmentation annotation**: GPT-4o generates three-level semantic annotations per step—screen description (current page content), contextual information (summary of preceding steps), and decision rationale (why the next action is taken). Low-level instructions are also generated as atomic decompositions of high-level instructions. Quality checks cover screenshot integrity, action accuracy, and instruction consistency.

2. **History Resampling Module in OdysseyAgent**: A core challenge in cross-app navigation is handling large quantities of historical screenshots and action sequences—the agent must recall results from prior apps to make current decisions, yet naively concatenating all historical screenshot tokens severely degrades inference speed.

   OdysseyAgent augments Qwen-VL with a **history resampler**—a single-layer cross-attention module:
    - Query: learnable embeddings
    - Key/Value: historical screenshot tokens
    - Output: compressed history tokens concatenated with current screenshot tokens, user instruction, and preceding actions, then fed into the LLM to predict the next action

   The training objective is standard next-token prediction:
   $\mathcal{L} = \sum_{i=1}^{N} P_\theta(A_i^t | X^{\{t, t-1, \cdots, t-\delta\}}, I_{user}, A_{<i}^t)$

   where $\delta$ is the historical image window size, and $\theta$ includes the trainable parameters of the VL adapter, history resampler, and LLM.

3. **Multi-dimensional Evaluation Design**: The dataset is split into four settings to comprehensively assess generalization:

    - **Train-Random & Test-Random** (in-domain)
    - **Train-App & Test-App** (unseen apps)
    - **Train-Task & Test-Task** (unseen task types)
    - **Train-Device & Test-Device** (unseen device types)

### Loss & Training

- Standard cross-entropy loss for next-action prediction
- Fine-tuned from Qwen-VL-Chat, retaining the visual encoder, LLM, and VL adapter
- Evaluation metric AMS (Action Matching Score): action type matching + CLICK/LONG PRESS within 14% screen distance + SCROLL direction matching + TYPE evaluated with ANLS
- Success Rate (SR): all steps must be correct to count as success; longer episodes are inherently harder

## Key Experimental Results

### Main Results

**Test-Random (In-domain) Comparison**

| Method | High-level AMS | Low-level AMS |
|--------|---------------|--------------|
| GPT-4o (zero-shot) | 13.19 | 42.71 |
| Claude3.5-Sonnet (zero-shot) | 15.80 | 34.18 |
| Claude3.5 + OmniParser | 32.88 | 63.91 |
| InternVL2-Pro + OmniParser | 14.69 | 54.31 |
| Qwen-VL (fine-tuned) | 74.67 | 86.32 |
| OdysseyAgent (fine-tuned) | 75.79 | 86.88 |
| **OdysseyAgent* (+ semantic annotation)** | **78.24** | **88.15** |

Fine-tuned methods substantially outperform zero-shot baselines; OdysseyAgent with semantic annotation achieves the best results.

### Ablation Study

**Effect of History Information Type (High-level Instruction AMS)**

| Config | Action | Screenshot | Context | Test-Random | Overall | SR |
|--------|--------|------------|---------|-------------|---------|-----|
| (1) | × | × | × | 66.13 | 55.60 | 1.49 |
| (2) | ✓ | × | × | 74.67 | 63.44 | 5.18 |
| (3) | × | ✓ | × | 71.22 | 60.30 | 4.20 |
| (4) | × | × | ✓ | 75.25 | 64.77 | 5.06 |
| (5) | ✓ | ✓ | × | 75.79 | 63.60 | 4.76 |
| **(6)** | **✓** | **✓** | **✓** | **77.06** | **66.84** | **6.32** |

### Key Findings

- **Contextual information** (textual summaries of historical steps) used alone outperforms the combination of actions + screenshots (Config 4 vs. 5)—abstract summaries generalize better than raw observations.
- Using all three history modalities achieves the best results, improving Overall AMS from 55.60 to 66.84 (+20.2%).
- Out-of-domain performance drops substantially: high-level AMS falls from 78.24 to 62.90 (−19.6%), while low-level AMS drops only 7.8%—indicating that complex reasoning and planning remain inadequate.
- CogAgent and SphAgent, despite performing well on other GUI tasks, perform poorly in cross-app settings (<16% AMS), confirming the large domain gap between single-app and cross-app navigation.
- OmniParser's GUI grounding capability substantially boosts closed-source model performance (Claude3.5: 15.80 → 32.88).

## Highlights & Insights

- **The first cross-app GUI dataset** fills an important gap—demonstrating that cross-app tasks require dedicated training data and cannot simply be addressed by single-app capabilities.
- The finding that **contextual information outperforms raw screenshots + actions** is insightful—abstract summarization of history is more beneficial than raw memory replay.
- The three-level semantic annotation design (screen description + context + decision rationale) mirrors human cognitive processes.
- The history resampler is a lightweight yet effective design—a single cross-attention layer achieves a strong balance between performance and efficiency.
- The dataset covers emerging device types such as foldables and tablets, demonstrating good forward-looking coverage.

## Limitations & Future Work

- Success Rate remains low (peak 11.61% in-domain); error accumulation in 15+ step sequences is severe.
- Evaluation is offline only (AMS); no online interactive evaluation on real devices is conducted, potentially overestimating actual capability.
- Existing methods perform worst on unseen task types (AMS 58.83), highlighting the need for stronger high-level reasoning and planning.
- Coordinate-based navigation is inherently brittle—reliance on precise screen coordinates makes the approach sensitive to resolution or layout changes.
- The Android emulator environment may differ physically from real device usage (touch response, animation latency, etc.).
- Integration with accessibility tree auxiliary information remains unexplored.

## Related Work & Insights

- **AITW** is the largest single-app GUI dataset (715K episodes) but lacks cross-app capability.
- **AndroidControl** provides high/low-level instructions but is limited to single-app settings.
- **OmniParser**'s GUI grounding capability serves as an inspiration—structured understanding can substantially improve zero-shot performance.
- Implication: Future GUI agents will require stronger **working memory and cross-app reasoning**, potentially necessitating retrieval-augmented mechanisms or scratchpad strategies to manage cross-app context.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The cross-app navigation dataset is a first-of-its-kind contribution, though OdysseyAgent's architectural innovations are limited.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four in-domain and out-of-domain settings, diverse baselines, and detailed history information ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Rich statistics, intuitive visualizations, and clear logical exposition.
- **Value**: ⭐⭐⭐⭐⭐ The dataset is of high community value and reveals the core challenges of cross-app GUI navigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Hoi! - A Multimodal Dataset for Force-Grounded, Cross-View Articulated Manipulation](../../CVPR2026/robotics/hoi_-_a_multimodal_dataset_for_force-grounded_cross-view_articulated_manipulatio.md)
- [\[CVPR 2025\] ShowUI: One Vision-Language-Action Model for GUI Visual Agent](../../CVPR2025/robotics/showui_one_vision-language-action_model_for_gui_visual_agent.md)
- [\[CVPR 2025\] GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities](../../CVPR2025/robotics/gigahands_a_massive_annotated_dataset_of_bimanual_hand_activities.md)
- [\[CVPR 2025\] SortScrews: A Dataset and Baseline for Real-time Screw Classification](../../CVPR2025/robotics/sortscrews_a_dataset_and_baseline_for_real-time_screw_classification.md)
- [\[CVPR 2025\] MoManipVLA: Transferring Vision-Language-Action Models for General Mobile Manipulation](../../CVPR2025/robotics/momanipvla_transferring_vision-language-action_models_for_general_mobile_manipul.md)

</div>

<!-- RELATED:END -->
