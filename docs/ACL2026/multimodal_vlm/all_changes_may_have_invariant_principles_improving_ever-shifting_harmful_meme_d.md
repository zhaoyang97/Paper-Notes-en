---
title: >-
  [Paper Note] All Changes May Have Invariant Principles: Improving Ever-Shifting Harmful Meme Detection via Design Concept Reproduction
description: >-
  [ACL 2026][Multimodal VLM][harmful meme detection] This paper proposes RepMD, a method that constructs a Design Concept Graph (DCG)—inspired by attack trees to model the steps and logic behind malicious meme creation—to guide MLLMs in detecting ever-shifting harmful memes, achieving 81.1% accuracy on GOAT-Bench.
tags:
  - ACL 2026
  - Multimodal VLM
  - harmful meme detection
  - design concept graph
  - attack tree
  - MLLM reasoning guidance
  - category drift
date: 2026-05-08
content_hash: b516929e18e86e5d
---

# All Changes May Have Invariant Principles: Improving Ever-Shifting Harmful Meme Detection via Design Concept Reproduction

**Conference**: ACL 2026
**arXiv**: [2601.04567](https://arxiv.org/abs/2601.04567)
**Code**: [GitHub](https://github.com/jzySaber1996/RepMD)
**Area**: Multimodal Safety / Meme Detection
**Keywords**: harmful meme detection, design concept graph, attack tree, MLLM reasoning guidance, category drift

## TL;DR

This paper proposes RepMD, a method that constructs a Design Concept Graph (DCG)—inspired by attack trees to model the steps and logic behind malicious meme creation—to guide MLLMs in detecting ever-shifting harmful memes, achieving 81.1% accuracy on GOAT-Bench.

## Background & Motivation

**Background**: Harmful memes on the Internet continuously evolve, exhibiting two key characteristics: category drift (new formats, new attack targets) and temporal evolution (tight coupling with current events), making detection extremely challenging.

**Limitations of Prior Work**: (1) Existing detection methods learn only the combination of harmful elements, lacking understanding of implicit expressions—e.g., implying racial discrimination by emphasizing a person's accessories; (2) Emerging internet slang (e.g., GOAT, Stan) further complicates detection; (3) MLLMs, despite their multimodal understanding capabilities, remain equally ineffective against such implicit harmful content.

**Key Challenge**: The visual elements and expressions of harmful memes constantly change, yet the underlying design logic of malicious creators may follow invariant principles. The core question is how to extract these invariant principles from historical memes to guide the detection of new ones.

**Goal**: To define an interpretable structure that captures the design concepts of harmful memes and leverage it to guide MLLMs in detection.

**Key Insight**: Drawing on the concept of attack trees from the security domain, the design intent of a meme is modeled as a structured graph comprising methods, goals, and logic gates.

**Core Idea**: Although harmful memes of different types appear superficially distinct, they may share the same design concepts (e.g., "specializing a general fact to a specific group to execute an attack"), and these concepts can transfer across types.

## Method

### Overall Architecture

RepMD proceeds in three steps: (1) constructing a Fail Reason Tree by analyzing historical memes where MLLMs fail and categorizing the reasons for failure; (2) deriving a Design Concept Graph (DCG) from the failure reasons to describe the design steps a malicious creator might take; and (3) retrieving relevant design steps from the DCG for a target meme to form step-by-step guidance that assists the MLLM in detection.

### Key Designs

1. **Fail Reason Tree Construction**:

    - Function: Systematically analyzes which memes MLLMs fail on and why.
    - Mechanism: Historical memes are evaluated by five MLLMs via majority voting; those failing on ≥3 models are treated as hard cases. Qwen3VL-235B is then used to analyze failure reasons and classify them into seven categories (cultural, political, etc.), forming a hierarchical tree structure. An iterative prompt optimization step is also included.
    - Design Motivation: By focusing exclusively on memes that MLLMs genuinely cannot detect, the resulting design concepts remain concentrated on the most challenging cases.

2. **Design Concept Graph (DCG)**:

    - Function: Describes the design logic of malicious creators in a structured manner.
    - Mechanism: Referencing attack trees, a three-level structure is defined—Reproduction Method (design steps taken by the malicious creator), Logic Gate (AND/OR/NOT combinational logic), and Reproduction Goal (the design objective, e.g., "crowd specialization"). Each node is labeled as harmful or not. The DCG is derived from the failure reason nodes.
    - Design Motivation: Attack trees have successfully modeled attacker logic chains in cybersecurity, and the same paradigm is equally applicable to modeling the reasoning of meme designers.

3. **SVD-based Graph Pruning and Retrieval-Guided Inference**:

    - Function: Compresses the DCG and retrieves relevant design steps for a target meme.
    - Mechanism: SVD dimensionality reduction is applied to prune redundant nodes from the DCG, retaining core design patterns. For a target meme, the most relevant design steps are retrieved from the DCG via similarity search, forming step-by-step guidance prompts that direct the MLLM to reason along the design logic.
    - Design Motivation: Using the full DCG directly introduces noise; SVD-based pruning has been shown effective in GNN settings.

### Loss & Training

RepMD is a training-free method that relies entirely on the in-context learning capabilities of MLLMs. Both DCG construction and retrieval are performed at inference time.

## Key Experimental Results

### Main Results

| Method | GOAT-Bench Accuracy | Out-of-Domain Generalization | Temporal Generalization |
|--------|--------------------|-----------------------------|------------------------|
| Baseline MLLM | Low | Large drop | Drop |
| RepMD | **81.1%** | Only −2.1% | +0.3% |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| w/o DCG | Significant accuracy drop | Design concepts are the core contribution |
| w/o SVD pruning | Performance degradation | Pruning removes noise and improves precision |
| Human evaluation | 15–30 sec/meme | DCG effectively assists human identification |

### Key Findings
- RepMD loses only 2.1% accuracy on out-of-domain generalization (new meme categories) and even gains 0.3% on temporal generalization (memes from future quarters).
- Human evaluation confirms the high interpretability of DCG—annotators can judge whether a meme is harmful within 15–30 seconds using the DCG.
- Harmful memes of different types do share design concepts, validating the "invariant principles" hypothesis.

## Highlights & Insights
- Borrowing the attack tree concept from the security domain to model meme design intent represents a creative cross-domain transfer.
- The "invariant principles" hypothesis is empirically validated—strong generalization across both meme categories and time periods is demonstrated.
- The method requires no training and fully leverages MLLM reasoning capabilities guided by the DCG.

## Limitations & Future Work
- The current DCG must be constructed from failure cases, which may be insufficient in cold-start scenarios.
- Evaluation is limited to English memes; memes from different cultures and languages may exhibit distinct design patterns.
- The hyperparameter selection for SVD pruning may require domain-specific tuning.
- Future work could extend the approach to video memes and multilingual memes.

## Related Work & Insights
- **vs. Traditional harmful content detection**: RepMD not only detects *whether* content is harmful, but also explains *why* it is harmful and *how* it was designed.
- **vs. Attack trees**: Security analysis methodology is creatively transferred to social media content analysis.
- **vs. LLM-based detection**: Providing structured design concept guidance yields more stable performance than pure prompting.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The cross-domain innovation from attack trees to design concept graphs is highly distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Category and temporal generalization experiments combined with human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Formal definitions are clear and motivation is well articulated.
- Value: ⭐⭐⭐⭐ Offers a new paradigm with broad implications for harmful content detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Yes FLoReNce, I Will Do Better Next Time! Agentic Feedback Reasoning for Humorous Meme Detection](../../AAAI2026/multimodal_vlm/yes_florence_i_will_do_better_next_time_agentic_feedback_reasoning_for_humorous_.md)
- [\[AAAI 2026\] CAMU: Context Augmentation for Meme Understanding](../../AAAI2026/multimodal_vlm/trace_textual_relevance_augmentation_and_contextual_encoding_for_multimodal_hate.md)
- [\[CVPR 2026\] Concept-wise Attention for Fine-grained Concept Bottleneck Models](../../CVPR2026/multimodal_vlm/coat_cbm_concept_wise_attention.md)
- [\[ACL 2026\] Dynamic Emotion and Personality Profiling for Multimodal Deception Detection](dynamic_emotion_and_personality_profiling_for_multimodal_deception_detection.md)
- [\[CVPR 2026\] MUPO: All Roads Lead to Rome - Incentivizing Divergent Thinking in Vision-Language Models](../../CVPR2026/multimodal_vlm/mupo_all_roads_lead_to_rome_incentivizing_divergent_thinking_in_vlms.md)

</div>

<!-- RELATED:END -->
