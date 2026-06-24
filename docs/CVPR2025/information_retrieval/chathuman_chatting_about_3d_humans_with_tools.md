---
title: >-
  [Paper Note] ChatHuman: Chatting about 3D Humans with Tools
description: >-
  [CVPR 2025][Information Retrieval & RAG][3D Human] ChatHuman is proposed, an LLM-based language-driven system that manages new tools by automatically selecting and integrating specialized 3D human analysis tools (3D pose estimation, shape recovery, contact detection, human-object interaction analysis, emotion recognition, etc.), utilizing academic papers as tool manuals along with RAG (Retrieval-Augmented Generation) to create in-context examples. It outperforms existing LLM…
tags:
  - "CVPR 2025"
  - "Information Retrieval & RAG"
  - "3D Human"
  - "Tool-use LLM"
  - "RAG"
  - "26 tools"
  - "DAG calling"
  - "LLaVA"
date: 2026-05-08
content_hash: a60417a23b02b775
---

# ChatHuman: Chatting about 3D Humans with Tools

**Conference**: CVPR 2025  
**arXiv**: [2405.04533](https://arxiv.org/abs/2405.04533)  
**Code**: To be confirmed  
**Area**: Information Retrieval  
**Keywords**: 3D Human, Tool-use LLM, RAG, 26 tools, DAG calling, LLaVA

## TL;DR
ChatHuman is proposed, an LLM-based language-driven system that manages new tools by automatically selecting and integrating specialized 3D human analysis tools (3D pose estimation, shape recovery, contact detection, human-object interaction analysis, emotion recognition, etc.), utilizing academic papers as tool manuals along with RAG (Retrieval-Augmented Generation) to create in-context examples. It outperforms existing LLM models in tool selection accuracy and overall performance on 3D human-related tasks.

## Background & Motivation
**Background**: Numerous methods have been proposed to detect, estimate, and analyze various human attributes in images (3D pose, shape, contact, human-object interaction, emotion, etc.). However, these methods suffer from high barriers to entry, requiring expert knowledge for selection, configuration, and interpretation.

**Limitations of Prior Work**: Highly fragmented tools—each method only addresses a specific problem (e.g., HMR for 3D pose, DECO for contact detection). Users must know which method to select, how to install and run it, and how to interpret complex 3D outputs (such as SMPL parameters and joint angle matrices).

**Key Challenge**: Adapting LLMs to 3D human tasks faces the challenges of domain-specific knowledge requirements and the interpretation of complex 3D outputs.

**Goal**: To build a conversational system capable of automatically selecting, applying, and interpreting a wide range of 3D human analysis tools.

**Key Insight**: Leveraging academic papers to teach LLMs how to use tools, and employing RAG models to create in-context examples for new tools.

**Core Idea**: Academic papers as tool manuals + RAG-generated usage examples + 3D-to-text output conversion to construct a unified conversational system for 3D human tasks.

## Method

### Overall Architecture
User input of natural language query + image/3D data $\to$ Paper-based RAG retrieves relevant tool usage examples $\to$ LLaVA-1.5-7B generates a tool invocation graph (Node/Chain/DAG modes) $\to$ Graph executor runs the toolchain $\to$ Tool-conditioned Transformation converts 3D outputs into VLM-compatible formats (SMPL $\to$ rendered RGB / contact labels $\to$ body part text / shape parameters $\to$ height and weight values) $\to$ LLM synthesizes multi-tool results to generate the final response. It integrates 26 tools: 9 perception, 10 generation, and 7 reasoning tools.

### Key Designs

1. **Paper-Driven Tool Learning**:

    - **Function**: Utilizing academic papers of tools to teach the LLM how to run and apply them.
    - **Mechanism**: Papers contain comprehensive information such as the input/output formats of the method, applicable scenarios, computational resource requirements, and limitations. The paper abstract, methodology description, and experimental setup are encoded into structured tool descriptions understandable by the LLM.
    - **Difference from Traditional API Descriptions**: API documentation often contains only function signatures and brief descriptions, whereas papers also include capabilities, failure modes, and comparisons with other methods, which are crucial for correct tool selection.
    - **Design Motivation**: Traditional tool descriptions are often superficial, whereas academic papers offer the most authoritative and comprehensive user guides.

2. **RAG-Enhanced In-Context Learning**:

    - **Function**: Using a retrieval-augmented generation model to construct usage examples for new tools.
    - **Mechanism**: When a new 3D human analysis tool is introduced, the most functionally similar tool in the existing tool usage database is retrieved, and its usage examples are adapted to serve as in-context examples for the new tool.
    - **Design Motivation**: New tools can be integrated without retraining by simply retrieving usage examples of similar tools, facilitating plug-and-play expansion.

3. **3D Output-to-Text Conversion**:

    - **Function**: Converting professional 3D outputs (e.g., SMPL parameters, joint coordinates, contact maps, mesh data) into user-interpretable natural language descriptions.
    - **Mechanism**: Designing domain-specific conversion templates to map joint angles to pose descriptions (e.g., "left hand raised to shoulder height") and contact maps to interaction descriptions (e.g., "right hand is touching the tabletop").
    - **Design Motivation**: Bridging the gap between specialized 3D outputs and general user understanding, enabling non-experts to access 3D human analysis results.

### Loss & Training
- Data: 90K tool execution instructions (generated by GPT-4 from COCO captions + 50 user queries per tool) + 88K tool feedback data (17K pose discrimination + 44K pose generation + 27K shape & contact).
- LLaVA-1.5-7B + LoRA (rank=128, alpha=256), AdamW lr=2e-4, 2 epochs, 8×A100-80G + DeepSpeed.
- Loss function: $L = CE(\hat{Y}_{tool}, Y_{tool}) + CE(\hat{Y}_t, Y_t)$

## Key Experimental Results

### Tool Selection Accuracy

| Method | SRt | SRact | SRargs | SR | IoU |
|------|:---:|:---:|:---:|:---:|:---:|
| Visual ChatGPT-3.5 | 0.498 | 0.319 | 0.237 | 0.251 | 0.791 |
| GPT4Tools | 0.609 | 0.547 | 0.525 | 0.520 | 0.566 |
| Visual ChatGPT-4 | 0.892 | 0.802 | 0.715 | 0.753 | 0.797 |
| **ChatHuman** | **1.000** | **0.974** | **0.950** | **0.970** | **0.975** |

### 3DPW Pose Estimation (MPJPE mm↓)

| Method | MPJPE | PA-MPJPE |
|------|:---:|:---:|
| SPIN | 102.9 | 62.9 |
| HMR 2.0 | 91.0 | 58.4 |
| ChatPose | 163.6 | 81.9 |
| **ChatHuman** | **91.3** | **58.7** |

Reasoning-based Pose Estimation (RPE, requiring reasoning to locate before estimation): ChatHuman 147.2mm vs HMR 225.2mm (MPJPE Gain: 34.6%)

### Ablation Study

| Configuration | SRt (Seen) | SRargs (Seen) | SRt (Unseen) | SRargs (Unseen) |
|------|:---:|:---:|:---:|:---:|
| Baseline (No RAG) | 0.998 | 0.93 | 0.82 | 0.84 |
| + RAG | 1.000 | 0.93 | 0.89 | 0.89 |
| + RAG + Paper | **1.000** | **0.95** | **0.99** | **0.94** |

### Key Findings
- Paper-based RAG improves the SRt for unseen tools from 0.82 to 0.99 (+20%); the abstract and introduction of the papers are the most effective, whereas methodology and experiments tend to introduce noise.
- Tool composition generalization: Trained on compositions of $\le 3$ tools; achieves 94.3% SRact for 4 tools, and 92.9% for 5 tools (exhibiting excellent OOD generalization).
- Multiple-choice discrimination reduces the mesh error on MixPose from 124–126mm to 119.6mm.
- HOI detection F1 score: LLaVA direct reasoning 0.39 $\to$ ChatHuman tool prediction 0.63 (+61.5%).
- Body shape estimation: ChatHuman average error 13.0cm vs LLaVA 22.9cm (-43%).

## Highlights & Insights
- The concept of **"papers as manuals"** is elegant and practical. Academic papers themselves represent the most detailed tool documentation, encompassing capacity boundaries and failure modes, which are much more comprehensive than manually written API documentation.
- The concept of **"translating" complex 3D outputs into natural language** is highly transferable to other specialized domains (e.g., generating diagnostic reports for medical image analysis or interpreting engineering simulation results) to build domain-specific AI assistants.
- The **plug-and-play nature of RAG** makes the system highly scalable. Once a new 3D human analysis method is published, it can be integrated simply by indexing its paper.

## Limitations & Future Work
- Under ambiguous queries, the system may select incorrect tools and lacks autonomous error correction (requiring user clarification and retries).
- Information loss occurs during the 3D output-to-text conversion, rendering it insufficient for tasks requiring precise numerical accuracy (e.g., body height error of 6.7cm).
- System performance is bounded by the best available academic tools; upgraded versions of tools require manual replacement (though retraining is not needed).
- The training data is generated via GPT-4, potentially introducing distribution biases.

## Related Work & Insights
- **vs Visual ChatGPT**: A general image tool framework, whereas ChatHuman is deeply customized for the 3D human domain, offering 26 domain-specific tools + 3D-to-text conversion, increasing SRact from 80.2% to 97.4%.
- **vs ChatPose**: It only focuses on a single pose estimation task, while ChatHuman covers 26 tools across three primary categories: perception, generation, and reasoning.
- **vs ToolLLM/Gorilla**: General tool-use frameworks, whereas ChatHuman's paper-driven RAG represents a domain-specific innovation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The paper-driven tool learning + RAG integration concept is remarkably novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Thorough validation across multiple tasks, with comprehensive evaluations on tool selection accuracy and dialogue quality.
- **Writing Quality**: ⭐⭐⭐⭐ Clear and easy to follow, with intuitive system architecture diagrams.
- **Value**: ⭐⭐⭐⭐ Highly valuable reference for integrating LLM tools within specialized domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Tools Are Under-Documented: Simple Document Expansion Boosts Tool Retrieval](../../ICLR2026/information_retrieval/tools_are_under-documented_simple_document_expansion_boosts_tool_retrieval.md)
- [\[CVPR 2025\] EZSR: Event-based Zero-Shot Recognition](ezsr_event-based_zero-shot_recognition.md)
- [\[CVPR 2025\] Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation](preserving_clusters_in_prompt_learning_for_unsupervised_domain_adaptation.md)
- [\[CVPR 2025\] COBRA: COmBinatorial Retrieval Augmentation for Few-Shot Adaptation](cobra_combinatorial_retrieval_augmentation_for_few-shot_adaptation.md)
- [\[CVPR 2025\] Advancing Myopia To Holism: Fully Contrastive Language-Image Pre-training](advancing_myopia_to_holism_fully_contrastive_language-image_pre-training.md)

</div>

<!-- RELATED:END -->
