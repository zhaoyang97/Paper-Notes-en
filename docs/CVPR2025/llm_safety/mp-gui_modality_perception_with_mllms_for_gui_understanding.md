---
title: >-
  [Paper Note] MP-GUI: Modality Perception with MLLMs for GUI Understanding
description: >-
  [CVPR 2025][LLM Safety][GUI Understanding] MP-GUI designs three specialized perceivers to extract graphical, textual, and spatial modal information from GUIs. By combining these three modalities through a spatial structure refinement strategy and an adaptive fusion gate, it outperforms general MLLMs on various GUI understanding tasks under limited training data.
tags:
  - "CVPR 2025"
  - "LLM Safety"
  - "GUI Understanding"
  - "Multimodal Large Language Model"
  - "Modality Perception"
  - "Spatial Structure Modeling"
  - "Fusion Gate"
date: 2026-05-08
content_hash: 014a40bb00af3997
---

# MP-GUI: Modality Perception with MLLMs for GUI Understanding

**Conference**: CVPR 2025  
**arXiv**: [2503.14021](https://arxiv.org/abs/2503.14021)  
**Code**: [https://github.com/BigTaige/MP-GUI](https://github.com/BigTaige/MP-GUI)  
**Area**: Human-Computer Interaction / Multimodal VLM  
**Keywords**: GUI Understanding, Multimodal Large Language Model, Modality Perception, Spatial Structure Modeling, Fusion Gate

## TL;DR
MP-GUI designs three specialized perceivers to extract graphical, textual, and spatial modal information from GUIs. By combining these three modalities through a spatial structure refinement strategy and an adaptive fusion gate, it outperforms general MLLMs on various GUI understanding tasks under limited training data.

## Background & Motivation

**Background**: Graphical User Interface (GUI) understanding is a foundational capability for building intelligent agents and automated systems. Current Multimodal Large Language Models (MLLMs) such as InternVL2 and Qwen-VL perform exceptionally well on natural images and document understanding, and are progressively being applied to GUI tasks (e.g., element localization, screen QA, GUI navigation).

**Limitations of Prior Work**: GUIs are fundamentally different from natural images—GUIs are composed of human-designed graphical elements (buttons, icons, text boxes) arranged in a specific spatial layout conveying semantic information. Although general MLLMs are good at processing graphical and textual components, they face two key obstacles in GUI understanding: (1) lack of explicit modeling of GUI-specific spatial structures (such as relative positions among elements, hierarchical containment relationships, and alignment constraints), leading to imprecise element localization and interaction understanding; (2) high-quality GUI spatial structure data is difficult to obtain—real-application interfaces involve privacy issues, while automatically annotated spatial relationships contain a large amount of noise.

**Key Challenge**: GUI understanding requires simultaneously understanding "what is seen" (graphical/textual content) and "where it is" (spatial layout), but existing MLLMs process these two types of information mixed within a unified visual feature, lacking specialized modeling for spatial structures.

**Goal**: To design a GUI-specific MLLM architecture that explicitly separates and models the graphical, textual, and spatial modalities in GUIs, and adaptively fuses them to meet the varied requirements of different GUI tasks.

**Key Insight**: The authors observe that different GUI understanding tasks depend on the three modalities to different degrees—element localization mainly relies on spatial information, text recognition primarily depends on the textual modality, while functional understanding requires joint reasoning of graphics and space. Therefore, a flexible multimodal fusion mechanism is needed.

**Core Idea**: Use three independent specialized perceivers to extract the graphical, textual, and spatial modal features of a GUI, and adaptively combine these three modal information types using a learnable fusion gate according to task requirements.

## Method

### Overall Architecture
MP-GUI is built on InternVL2-8B, retaining the original vision encoder and LLM backbone while introducing three additional specialized perceiver modules. Given a GUI screenshot as input, general visual features are first extracted via the vision encoder. Then, three perceivers extract graphical features (visual attributes of elements like buttons/icons), textual features (text content in the interface), and spatial features (layout relationships between elements), respectively. These three modal features are weighted and merged through a fusion gate, then fed into the LLM along with the original visual features for inference.

### Key Designs

1. **Graphical Perceiver**:

    - **Function**: Extract visual semantic features of graphical elements (buttons, icons, color blocks, etc.) from GUI screenshots.
    - **Mechanism**: A lightweight cross-attention architecture is employed, using a set of learnable query tokens to extract features related to graphical elements from the vision encoder's output. The number of query tokens is fixed, compressing the visual information of the GUI screenshot into a fixed-dimensional graphical semantic space. Training data is derived from GUI element detection and classification tasks.
    - **Design Motivation**: The features generated by general vision encoders contain excessive low-level details (textures, gradients, etc.). The graphical perceiver filters out information relevant to the semantics of GUI elements, reducing information redundancy for subsequent reasoning.

2. **Textual Perceiver**:

    - **Function**: Specially identify and understand text content (labels, prompt text, input content, etc.) in the GUI.
    - **Mechanism**: The structure is similar to the graphical perceiver, using another set of query tokens specifically to extract features of text regions. Its peculiarity lies in the construction of training data—OCR tools are utilized to automatically extract text and their positions in GUIs, and these annotated data are then used to train the perceiver to learn precise localization and content recognition of text regions.
    - **Design Motivation**: Text in GUIs is a core element conveying functional semantics (e.g., "Submit" button, "Settings" label). However, text in GUIs often has small fonts and complex backgrounds, making the OCR capabilities of general MLLMs imprecise in GUI scenarios. A specialized textual perceiver can improve the precision and recall of text recognition.

3. **Spatial Perceiver with Refinement**:

    - **Function**: Model spatial relationships between GUI elements (relative positions, containment relationships, alignments, etc.).
    - **Mechanism**: The spatial perceiver also extracts spatial structure features based on cross-attention but incorporates a Spatial Structure Refinement strategy. Since real GUI spatial annotations are highly noisy (inaccurate bounding boxes of automatically labeled elements, false recognitions caused by occlusion, etc.), the refinement strategy is implemented via a two-stage training: the first stage pre-trains the spatial perceiver using open datasets with high-quality annotations like Semantic UI, and the second stage fine-tunes it with synthetic data (generated by Qwen2-VL-72B) to enhance robustness against noisy annotations.
    - **Design Motivation**: Spatial relationships are the most core features that distinguish GUIs from natural images. Layout relationships such as "above-below", "container-contained", and "aligned" determine the functional semantics of the GUI, but such structural information is implicit and unreliable in standard visual features.

### Fusion Gate
The outputs of the three perceivers are adaptively weighted and merged through a learnable fusion gate. The fusion gate dynamically adjusts the weights of the three modalities based on the semantic information of the current input query (user instruction). For example, instructions like "click the search button" will increase the weights of the spatial and graphical modalities, while instructions like "read the text in the input box" will increase the weight of the textual modality.

### Loss & Training
A multi-step training strategy is adopted: (1) first train the three perceivers separately (using their respective specialized data); (2) then train the fusion gate (with perceiver parameters frozen); (3) finally fine-tune the entire model on downstream benchmark data. This step-by-step strategy reduces the demand for large-scale unified annotated data, enabling perceiver training using existing open data from each modality. LoRA is also utilized for efficient fine-tuning.

## Key Experimental Results

### Main Results

| Benchmark Task | Metric | MP-GUI | InternVL2-8B | Qwen-VL-Chat | CogAgent | Gain (vs InternVL2) |
|---------|------|--------|-------------|-------------|---------|-------------------|
| ScreenSpot (Mobile Localization) | Acc | 78.2 | 66.5 | 53.1 | 71.4 | +11.7% |
| ScreenSpot (Web Localization) | Acc | 72.6 | 61.8 | 48.3 | 65.2 | +10.8% |
| ScreenQA | Acc | 76.8 | 68.3 | 59.2 | 70.1 | +8.5% |
| AITW (GUI Agent) | Acc | 73.5 | 64.7 | 56.8 | 67.9 | +8.8% |
| Widget Caption | CIDEr | 142.3 | 121.6 | 98.7 | 128.4 | +20.7 |

### Ablation Study

| Configuration | ScreenSpot Acc | ScreenQA Acc | Description |
|------|---------------|-------------|------|
| Full MP-GUI | 78.2 | 76.8 | Full model |
| w/o Spatial Perceiver | 70.4 | 73.1 | Largest drop in localization capability |
| w/o Textual Perceiver | 74.6 | 69.2 | Larger impact on QA tasks |
| w/o Graphical Perceiver | 75.1 | 74.3 | Overall decline |
| w/o Fusion Gate (Direct Concatenation) | 74.8 | 73.6 | Adaptive fusion yields 2-3% gain |
| w/o Spatial Structure Refinement | 73.6 | 74.5 | Refinement strategy contributes significantly to localization |

### Key Findings
- The spatial perceiver contributes the most to the element localization task (ScreenSpot drops by 7.8% without it), validating the criticality of spatial structure modeling in GUI understanding.
- The textual perceiver has the greatest impact on the QA task (ScreenQA drops by 7.6% without it), indicating that GUI question answering relies heavily on understanding text information in the interface.
- The fusion gate performs 2-3% better than simple feature concatenation, proving the necessity of adaptive modality weight adjustment.
- The spatial structure refinement strategy brings a 4.6% improvement on ScreenSpot, showing that processing noisy spatial annotations is practical.
- Despite using limited training data (mainly from public datasets and synthetic data), MP-GUI still outperforms general MLLMs trained on much more data, indicating that targeted architectural design is more important than data scale.

## Highlights & Insights
- **The design concept of explicit modality separation** is highly valuable: instead of mixing all information together and letting the model learn implicitly, it explicitly splits modalities, models them independently, and then fuses them based on domain characteristics. This approach can be transferred to other scenarios with distinct modality divisions (e.g., CAD drawing understanding, dashboard interpretation).
- **The spatial structure refinement strategy** cleverly resolves data quality issues: the challenge of noisy GUI spatial annotations is mitigated by two-stage training (high-quality small data pre-training + synthetic data fine-tuning), which serves as a general strategy for handling noisy annotations.
- **Multi-step training reducing data requirements** is a highly practical design: independent training of each perceiver avoids reliance on unified multimodal annotated data, lowering the barrier for practical deployment.

## Limitations & Future Work
- Only validated on InternVL2-8B, and generalizability to other MLLMs (e.g., Qwen-VL-2, LLaVA) has not been evaluated.
- The architectures of the three perceivers are identical (all cross-attention); differentiated designs tailored to each modality's characteristics might bring further improvements.
- Dynamic GUI understanding (such as scrolling, animation effects) is not covered; only static screenshots are processed.
- The quality of synthetic data depends on the upper limit of Qwen2-VL-72B's capabilities, which may introduce systematic bias on certain complex GUIs.
- The evaluation is limited to English GUI interfaces; performance on multilingual/multicultural GUIs remains unknown.

## Related Work & Insights
- **vs CogAgent**: CogAgent is also designed for GUI understanding but adopts a unified visual encoding scheme without separating modalities. Through explicit modality separation, MP-GUI outperforms CogAgent by 6.8% on localization tasks.
- **vs SeeClick / UGround**: These works focus exclusively on GUI element localization. MP-GUI provides more comprehensive GUI understanding capabilities while achieving superior localization performance.
- **vs InternVL2 / Qwen-VL**: With general MLLMs functioning as the backbone, MP-GUI achieves significant improvements by incorporating GUI-specific components, validating the value of domain specialization.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The GUI-specific architectural design with tri-modality separation and a fusion gate is innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple GUI benchmarks with detailed ablations, but lacks cross-model validation.
- **Writing Quality**: ⭐⭐⭐⭐ The problem definition is clear, and the method description is detailed.
- **Value**: ⭐⭐⭐⭐ Provides an effective perception enhancement scheme for the GUI Agent field; open-source code and data pipelines add value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RISK: A Framework for GUI Agents in E-commerce Risk Management](../../ACL2026/llm_safety/risk_a_framework_for_gui_agents_in_e-commerce_risk_management.md)
- [\[NeurIPS 2025\] Enhancing CLIP Robustness via Cross-Modality Alignment](../../NeurIPS2025/llm_safety/enhancing_clip_robustness_via_crossmodality_alignment.md)
- [\[ACL 2025\] Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models](../../ACL2025/llm_safety/manu_modality_aware_unlearning.md)
- [\[ACL 2025\] When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations](../../ACL2025/llm_safety/when_backdoors_speak_understanding_llm_backdoor_attacks_through_model-generated_.md)
- [\[ICLR 2026\] Pragma-VL: Towards a Pragmatic Arbitration of Safety and Helpfulness in MLLMs](../../ICLR2026/llm_safety/pragma-vl_towards_a_pragmatic_arbitration_of_safety_and_helpfulness_in_mllms.md)

</div>

<!-- RELATED:END -->
