---
title: >-
  [Paper Note] Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents
description: >-
  [CVPR 2025][Multimodal VLM][Mobile Agent] The MONDAY framework automatically generates mobile navigation datasets from YouTube tutorial videos. Through an OCR-based scene transition detection and a 3-step action recognition pipeline with GPT-4o, it constructs 313K annotated frames covering both iOS and Android platforms at 1/17th of the cost of manual annotation ($0.34 vs $5.76 per video). After pre-training, the agent achieves a performance gain of 18.11% on the unseen Windo…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Mobile Agent"
  - "Automated Dataset Generation"
  - "YouTube Videos"
  - "Cross-Platform Navigation"
  - "OCR Scene Detection"
date: 2026-05-08
content_hash: c6192a997445e4ea
---

# Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents

**Conference**: CVPR 2025  
**arXiv**: [2505.12632](https://arxiv.org/abs/2505.12632)  
**Code**: To be confirmed  
**Area**: Robotics  
**Keywords**: Mobile Agent, Automated Dataset Generation, YouTube Videos, Cross-Platform Navigation, OCR Scene Detection

## TL;DR
The MONDAY framework automatically generates mobile navigation datasets from YouTube tutorial videos. Through an OCR-based scene transition detection and a 3-step action recognition pipeline with GPT-4o, it constructs 313K annotated frames covering both iOS and Android platforms at 1/17th of the cost of manual annotation ($0.34 vs $5.76 per video). After pre-training, the agent achieves a performance gain of 18.11% on the unseen Windows Mobile platform.

## Background & Motivation

**Background**: Training data for mobile GUI agents primarily relies on manual recording and annotation (e.g., AitW, AMEX), which is high in cost, small in scale, and covers only a single platform. In contrast, massive amounts of mobile tutorial videos ("how to change wallpaper on Android") exist on YouTube, but there is a lack of an automated pipeline from videos to structured action datasets.

**Limitations of Prior Work**: (1) Manual annotation is costly ($5.76 per video) and cannot be scaled up; (2) Existing datasets only cover a single platform (either iOS or Android), leading to poor agent generalization; (3) Scene transition detection in YouTube UI screenshots is difficult, as dark mode switching causes pixel-based methods to fail; (4) Precise localization of UI elements during action recognition is challenging, especially for small buttons in complex interfaces.

**Key Challenge**: YouTube video data is abundant but unstructured (lacking action annotations), and traditional annotation methods cannot scale. The challenge lies in automatically extracting precise action sequences from videos.

**Goal**: To design a fully automated pipeline to generate high-quality mobile navigation datasets from YouTube tutorial videos, and to validate the value of this data for cross-platform generalization of agents.

**Key Insight**: Utilizing the property that OCR text is more stable than visual pixels to detect scene transitions, employing multi-step reasoning with GPT-4o + Set-of-Mark to identify precise actions, and leveraging narration transcripts to assist in disambiguation.

**Core Idea**: Automatically generating large-scale cross-platform mobile navigation datasets from YouTube videos via OCR-driven scene transition detection and GPT-4o multi-step action recognition.

## Method

### Overall Architecture
A three-stage pipeline: (1) Video collection and filtering (from 129K down to 20K videos); (2) OCR scene transition detection (segmenting interface-changing frames); (3) 3-step action recognition (Scene Summary -> Multi-frame Contextual Action Recognition -> Zoom-in Refinement Localization). The final output consists of frame-level action annotations.

### Key Designs

1. **OCR Scene Transition Detection**:

    - **Function**: Detect transition points of user interfaces in sequence of mobile screenshots to segment videos into individual action steps.
    - **Mechanism**: Extracts text from the phone screen area at 4 FPS using PaddleOCR (where screen area is detected by GroundingDINO at 2 FPS and interpolated linearly), tracks text elements at the same locations, and calculates the Levenshtein distance. A scene transition is flagged when more than 20% of the text changes. The F1 score reaches 95.04%, which is 12.77% higher than SceneDetect (82.27%).
    - **Design Motivation**: Methods based on YUV color differences are highly sensitive to global appearance changes such as dark mode switching (F1 of only 70.86%), whereas OCR text content remains stable during these changes, making it exceptionally suitable for UI scenarios.

2. **3-Step Action Recognition (Scene Summary → Action ID → Refinement)**:

    - **Function**: Precisely recognize the user action and corresponding UI element coordinates in each frame.
    - **Mechanism**: Step 1 - Scene Summary: GPT-4o describes the interface layout from the unmarked original frame; Step 2 - Action Recognition: The current frame + scene summaries of current and adjacent 2 frames + Set-of-Mark (numbered UI elements) + video narration transcript are fed to GPT-4o to identify candidate actions; Step 3 - Refined Localization: A zoomed-in view is generated around candidate UI elements, and GPT-4o + SoM are used again for precise localization. The final coordinate is defined as the center point of the UI element's bounding box.
    - **Design Motivation**: Single-step action recognition accuracy is only 70.63%, which rises to 80.90% after introducing temporal context (+8.80%) and refined localization (+1.47%). Narration transcript assists GPT-4o in disambiguating visually similar UI elements (+2.70%).

3. **Large-Scale Video Filtering Pipeline**:

    - **Function**: Filter high-quality mobile tutorial videos from 129K YouTube videos.
    - **Mechanism**: Multi-stage filtering: GroundingDINO detects the phone screen (filtering out Android Watch/MacOS) -> MediaPipe detects hand occlusions (filtering out hand-held video recordings) -> GPT-4o samples frames to confirm the OS type. GPT-3.5 identifies task names from CommonCrawl posts to use for video searching.
    - **Design Motivation**: The quality of YouTube videos is highly inconsistent, making multi-stage filtering essential to ensure overall data quality. Ultimately, 20K videos were retained.

### Loss & Training
LoRA is used for agent pre-training and fine-tuning. The input consists of the current screenshot + task name + past 4 actions, and the output is the next action prediction. The checkpoint with the lowest validation loss is selected. The evaluation metrics are exact action matching + interaction area validation for touch/long press.

## Key Experimental Results

### Main Results

| Test Set | Model | Without MONDAY | With MONDAY | Gain |
|--------|------|------|----------|------|
| AitW (Avg. of 5 categories) | SeeClick | 66.98% | 68.47% | +1.49% |
| AitW (Avg. of 5 categories) | Llama-3.2-11B | 58.96% | 67.38% | +8.42% |
| AMEX | Llama-3.2-11B | 43.74% | 55.96% | +12.22% |
| **Windows Mobile (Unseen)** | **SeeClick** | **38.54%** | **51.71%** | **+13.17%** |
| **Windows Mobile (Unseen)** | **Llama-3.2-11B** | **26.83%** | **50.24%** | **+23.41%** |
| MONDAY self | SeeClick | 40.66% | 63.39% | +22.73% |

### Ablation Study

| Method | Overall Action Accuracy | Touch Accuracy |
|------|---------|---------|
| 3-step multi-frame (Complete) | **80.90%** | **91.84%** |
| 2-step (Without refinement) | 79.43% | 89.97% |
| 1-step (Direct recognition) | 70.63% | 74.67% |
| Without narration transcripts | 78.20% | 87.64% |
| Single-frame (Without temporal context) | 77.22% | 89.30% |

### Key Findings
- **Remarkable cross-platform generalization**: An average gain of 18.11% is achieved on Windows Mobile (a completely unseen platform), indicating that the diversity of iOS + Android dual-platform data enables the agent to acquire platform-agnostic UI comprehension.
- **OCR-based scene detection far outperforms visual methods**: F1 of 95.04% vs SceneDetect's 82.27%, proving that text in UI serves as the most stable signal.
- **Nearly perfect UI element detection**: Hit Ratio of 99.87% vs OmniParser's 91.83%, benefiting from mobile-specific heuristic filtering.
- **Extremely cost-efficient**: $0.34 per video vs manual cost of $5.76 per video, achieving a 17-fold reduction in costs.
- **Llama-3.2 benefits more than SeeClick**: This is likely because Llama's initial UI comprehension is relatively weak, thus benefiting more from the diverse data of MONDAY.

## Highlights & Insights
- **OCR-driven scene transition detection** is a highly practical innovation. Text content is more stable than pixel information in UI environments and can be extended to any UI video analysis task.
- **The design philosophy of the 3-step action recognition** (first scanning globally -> reasoning temporally -> zoom-in refinement) mimics the cognitive process of humans watching tutorial videos, making the contribution of each step clear and quantifiable.
- **YouTube videos represent a treasure trove for agent training**: 20K videos yield 313K annotated frames at an extremely low cost, naturally covering highly diverse apps and operational scenarios, which offers a qualitative advantage over manually constructed datasets.

## Limitations & Future Work
- Reliance on GPT-4o for action recognition, where API costs and rate limits may affect larger-scale data generation.
- Multi-stage filtering discards a large volume of videos (from 129K down to 20K), which might filter out valuable data.
- The 20% text change threshold is set empirically, and its applicability to other languages/scripts remains unvalidated.
- Action distributions in tutorial videos lean heavily towards simple actions (Touch 79.83%), with very few samples of complex gestures (Multi-touch, Zoom).
- The refinement step requires generating zoomed-in views, increasing computational overhead.

## Related Work & Insights
- **vs AitW / AMEX**: These are manually annotated, single-platform datasets, whereas MONDAY is an automatically generated, dual-platform dataset. Pre-training on MONDAY improves performance on both of these datasets.
- **vs OmniParser**: OmniParser achieves a UI element detection Hit Ratio of 91.83%, whereas MONDAY achieves 99.87% thanks to mobile-specific heuristic rules.
- Provides a low-cost scaling paradigm for the GUI agent community, eliminating the absolute dependence on manual annotation.

## Rating
- Novelty: ⭐⭐⭐⭐ The OCR scene transition detection and the automated YouTube-to-dataset pipeline are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Independent evaluation of individual dataset construction components + down-stream verification across multiple agents and platforms + comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ The pipeline is clearly described, and statistical information is complete.
- Value: ⭐⭐⭐⭐ High practical value; the dataset is directly beneficial to the mobile agent community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents](v-stylist_video_stylization_via_collaboration_and_reflection_of_mllm_agents.md)
- [\[CVPR 2025\] CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation](comm_a_coherent_interleaved_image-text_dataset_for_multimodal_understanding_and_.md)
- [\[CVPR 2025\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[CVPR 2025\] From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons](from_multimodal_llms_to_generalist_embodied_agents_methods_and_lessons.md)
- [\[CVPR 2025\] StarVector: Generating Scalable Vector Graphics Code from Images and Text](starvector_generating_scalable_vector_graphics_code_from_images_and_text.md)

</div>

<!-- RELATED:END -->
