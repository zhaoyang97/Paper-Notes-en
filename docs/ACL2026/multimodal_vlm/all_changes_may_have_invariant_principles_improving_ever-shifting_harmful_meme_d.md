---
title: >-
  [Paper Note] All Changes May Have Invariant Principles: Improving Ever-Shifting Harmful Meme Detection via Design Concept Reproduction
description: >-
  [ACL 2026][Multimodal VLM][Harmful meme detection] Proposes the RepMD method, which constructs a Design Concept Graph (DCG)—borrowing the concept of attack trees to describe the steps and logic used by malicious users to…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Harmful meme detection"
  - "design concept graph"
  - "attack tree"
  - "MLLM reasoning guidance"
  - "type drift"
date: 2026-05-08
content_hash: ea2b65c1b01ce3c4
---

# All Changes May Have Invariant Principles: Improving Ever-Shifting Harmful Meme Detection via Design Concept Reproduction

**Conference**: ACL 2026  
**arXiv**: [2601.04567](https://arxiv.org/abs/2601.04567)  
**Code**: [GitHub](https://github.com/jzySaber1996/RepMD)  
**Area**: Multimodal Safety / Meme Detection  
**Keywords**: Harmful meme detection, design concept graph, attack tree, MLLM reasoning guidance, type drift

## TL;DR

Proposes the RepMD method, which constructs a Design Concept Graph (DCG)—borrowing the concept of attack trees to describe the steps and logic used by malicious users to design harmful memes—to guide MLLMs in detecting evolving harmful memes, achieving 81.1% accuracy on GOAT-Bench.

## Background & Motivation

**Background**: Harmful memes on the internet are constantly evolving, characterized by type drift (new forms, new targets) and temporal evolution (closely related to current events), making detection extremely difficult.

**Limitations of Prior Work**: (1) Existing detection methods only learn combinations of harmful elements, lacking understanding of implicit expressions—such as implying racism by highlighting a person's accessories; (2) Emerging internet slang (e.g., GOAT, Stan) increases detection difficulty; (3) MLLMs, despite multi-modal understanding capabilities, struggle with these implicit harmful messages.

**Key Challenge**: While visual elements and expressions of harmful memes change, the design logic of malicious users may contain "invariant principles." How can these invariant principles be extracted from historical memes to guide the detection of new ones?

**Goal**: To define an interpretable structure to describe harmful meme design concepts and utilize it to guide MLLMs in detection.

**Key Insight**: Borrowing the "attack tree" concept from the security domain to model meme design intent as a structured graph containing methods, targets, and logic gates.

**Core Idea**: Different types of harmful memes, while seemingly distinct, may share the same design concepts (e.g., "specializing facts to specific groups to achieve an attack"), which can transfer across types.

## Method

### Overall Architecture

RepMD consists of three steps: (1) Constructing a Fail Reason Tree—analyzing historical memes where MLLM detection failed to summarize causes; (2) Deriving a Design Concept Graph (DCG) from failure reasons—describing design steps a malicious user might take; (3) Retrieving similar design steps from the DCG for target memes to form step-by-step guidance for MLLM detection.

### Key Designs

1.  **Fail Reason Tree Construction**:
    - **Function**: Systematically analyzes which memes MLLMs fail on and why.
    - **Mechanism**: Use 5 MLLMs to vote on detections for historical memes, treating those with $\ge 3$ failures as hard examples. Qwen3VL-235B is used to analyze failure reasons and classify them into 7 major categories (culture, politics, etc.), forming a hierarchical tree structure. It also includes prompt iteration optimization.
    - **Design Motivation**: Focuses only on memes that MLLMs truly cannot detect, ensuring design concepts target the most challenging cases.

2.  **Design Concept Graph (DCG)**:
    - **Function**: Describes the design logic of malicious users in a structured way.
    - **Mechanism**: References attack trees to define a three-level structure—Reproduction Method (malicious design steps), Logic Gate (AND/OR/NOT logic), and Reproduction Goal (design objective, e.g., "group specialization"). Each node is labeled as harmful or not. Derived from fail reason nodes.
    - **Design Motivation**: Attack trees successfully model attacker logic chains in cybersecurity; they are equally applicable to modeling the thought patterns of meme designers.

3.  **SVD Graph Pruning and Retrieval Guidance**:
    - **Function**: Streamlines the DCG and retrieves relevant design steps for target memes.
    - **Mechanism**: Uses SVD dimensionality reduction to prune redundant nodes in the DCG, preserving core design patterns. For a target meme, the most relevant design steps are retrieved via similarity, forming step-by-step guidance prompts to let the MLLM reason along the design logic.
    - **Design Motivation**: Directly using all DCG information would introduce noise; SVD pruning has proven effective in GNNs.

### Loss & Training

RepMD is a training-free method, completely based on the in-context learning capabilities of MLLMs. DCG construction and retrieval are both performed during inference.

## Key Experimental Results

### Main Results

| Method | GOAT-Bench Accuracy | OOD Generalization | Temporal Generalization |
| :--- | :--- | :--- | :--- |
| Baseline MLLM | Low | Significant drop | Decrease |
| **Ours** (RepMD) | **81.1%** | Only 2.1% drop | 0.3% increase |

### Ablation Study

| Config | Key Metric | Note |
| :--- | :--- | :--- |
| w/o DCG | Accuracy drops significantly | Design concepts are the core contribution |
| w/o SVD Pruning | Performance drops | Pruning removes noise to improve precision |
| Human Eval | 15-30s / meme | DCG effectively assists human identification |

### Key Findings
- RepMD loses only 2.1% accuracy in OOD generalization (new types) and even improves by 0.3% in temporal generalization (future quarters).
- Human evaluation confirms high interpretability of the DCG—assessors can use the DCG to judge meme harmfulness within 15-30 seconds.
- Different types of harmful memes indeed share design concepts, validating the "invariant principle" hypothesis.

## Highlights & Insights
- Borrowing the attack tree concept from the security domain to model meme design intent is a creative cross-domain transfer.
- The "invariant principle" hypothesis is experimentally validated—demonstrating strong generalization across both types and time.
- The method requires no training, fully utilizing MLLM reasoning capabilities and DCG guidance.

## Limitations & Future Work
- The current DCG needs to be constructed from failure cases, which may not be rich enough during a cold start.
- Tested only on English memes; memes from different cultures/languages may have different design patterns.
- Parameter selection for SVD pruning may require adjustment for different domains.
- Future expansion could include video memes and multilingual memes.

## Related Work & Insights
- **vs Traditional harmful content detection**: Not just detecting "if it is harmful," but also explaining "why it is harmful" and "how it was designed."
- **vs Attack Tree**: Creatively transfers security analysis methods to social media content analysis.
- **vs LLM-based detection**: Provides structured design concept guidance, which is more stable than pure prompting.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The cross-domain innovation from attack trees to DCGs is very unique.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes both type and temporal generalization experiments plus human evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear formal definitions and well-supported motivation.
- **Value**: ⭐⭐⭐⭐ Provides a-new-paradigm insight for harmful content detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Yes FLoReNce, I Will Do Better Next Time! Agentic Feedback Reasoning for Humorous Meme Detection](../../AAAI2026/multimodal_vlm/yes_florence_i_will_do_better_next_time_agentic_feedback_reasoning_for_humorous_.md)
- [\[AAAI 2026\] CAMU: Context Augmentation for Meme Understanding](../../AAAI2026/multimodal_vlm/trace_textual_relevance_augmentation_and_contextual_encoding_for_multimodal_hate.md)
- [\[ACL 2026\] Dynamic Emotion and Personality Profiling for Multimodal Deception Detection](dynamic_emotion_and_personality_profiling_for_multimodal_deception_detection.md)
- [\[CVPR 2026\] Concept-wise Attention for Fine-grained Concept Bottleneck Models](../../CVPR2026/multimodal_vlm/coat_cbm_concept_wise_attention.md)
- [\[ACL 2026\] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)

</div>

<!-- RELATED:END -->
