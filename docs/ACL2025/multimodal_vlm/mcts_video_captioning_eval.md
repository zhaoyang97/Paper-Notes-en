---
title: >-
  [Paper Note] Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search
description: >-
  [ACL 2025][Multimodal VLM][Video Captioning] Proposes the AutoCaption framework, which uses Monte Carlo Tree Search (MCTS) to automatically and iteratively generate fine-grained video captioning keypoints (averaging 122 per video). It builds the MCTS-VCB benchmark to evaluate the video captioning capabilities of 20+ MLLMs, and demonstrates that the generated data can be used for fine-tuning to significantly improve model performance.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Video Captioning"
  - "MCTS"
  - "Evaluation Benchmark"
  - "Keypoint Generation"
  - "Video Understanding"
date: 2026-05-08
content_hash: edb36d175f1228b9
---

# Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search

**Conference**: ACL 2025  
**arXiv**: [2506.11155](https://arxiv.org/abs/2506.11155)  
**Code**: [https://github.com/tjunlp-lab/MCTS-VCB](https://github.com/tjunlp-lab/MCTS-VCB)  
**Area**: Multimodal VLM  
**Keywords**: Video Captioning, MCTS, Evaluation Benchmark, Keypoint Generation, Video Understanding  

## TL;DR
Proposes the AutoCaption framework, which uses Monte Carlo Tree Search (MCTS) to automatically and iteratively generate fine-grained video captioning keypoints (averaging 122 per video). It builds the MCTS-VCB benchmark to evaluate the video captioning capabilities of 20+ MLLMs, and demonstrates that the generated data can be used for fine-tuning to significantly improve model performance.

## Background & Motivation
**Background**: Video captioning is an important task for evaluating the video understanding capability of MLLMs. Existing methods perform evaluation by creating keypoints (descriptive sentences) and comparing them with model-generated captions.

**Limitations of Prior Work**: (a) Keypoints in existing benchmarks are insufficient or homogeneous—for example, DREAM-1K averages only 6.3 keypoints per video, which easily misses details; (b) keypoints are mostly action-oriented, ignoring dimensions such as appearance, environment, and object attributes; (c) the cost of manual annotation is extremely high, making it difficult to scale.

**Key Challenge**: To comprehensively evaluate the video understanding capability of MLLMs, it is necessary to cover all dimensions and details of the video content. However, manually creating such fine-grained keypoints is both expensive and incomplete, causing existing evaluations to be easily overestimated or inaccurate.

**Goal** (a) How to automatically generate a large number of diverse, fine-grained video description keypoints? (b) How to build an evaluation benchmark that comprehensively covers various dimensions of video content? (c) Can the generated data be used to improve model performance?

**Key Insight**: Apply the iterative search capability of MCTS to video captioning—by defining 6 descriptive actions (overall, detail, temporal, spatial, background, camera movement) and continuously expanding new nodes in the search tree to discover deeper details in the video.

**Core Idea**: Use MCTS to iteratively search the description space of video content, automatically generating an average of 122 verified keypoints to build a more comprehensive fine-grained video captioning benchmark than manual annotation.

## Method

### Overall Architecture
Input: Video $v$. AutoCaption constructs a search tree $T$ via MCTS, where the root node is the video, each edge represents a descriptive action (one of six), and child nodes represent the status (new description) after executing the action. Through four iterative steps (Selection → Expansion → Evaluation → Backpropagation), it continuously discovers new video details. Finally, post-processing (de-duplication, verification) is performed on the descriptions of all nodes to obtain a set of verified keypoints.

### Key Designs

1. **Design of 6 Descriptive Actions**:

    - **Function**: Define action types that cover each dimension of video content.
    - **Mechanism**: A1 overall description (executed only once after the root node, initialized by GPT-4o and Gemini), A2 detailed description (sampling probability is double that of other actions, employing a two-stage process: first find new details → then extract undescribed attributes to describe further), A3 temporal perspective description, A4 spatial perspective description, A5 background description, and A6 camera movement description. Each node randomly expands 2 actions.
    - **Design Motivation**: Video content is multidimensional; action descriptions alone cannot provide comprehensive coverage. The 6 types of actions ensure that information is mined from multiple angles such as appearance, time, space, environment, and camera.

2. **MCTS Node Evaluation and Selection**:

    - **Function**: Balance correctness and diversity to select the most valuable nodes for expansion.
    - **Mechanism**: Node state value $Q(s,a) = \alpha^{1-MC(s)} \cdot \beta^{SM(s)}$, where $MC(s)$ is the Monte Carlo value (the ratio of keypoints that pass verification) and $SM(s)$ is the similarity to other nodes on the path. Nodes with high $MC$ (correct) and low $SM$ (novel) have higher value. Selection uses the PUCT algorithm: $s_i = \arg\max_{s \in L(T)}[Q(s,a) + c\frac{\sqrt{N_{parent}(s)}}{1+N(s)}]$.
    - **Design Motivation**: The core of MCTS lies in the explore-exploit balance. The MC value ensures that expanded nodes generate correct descriptions, while the SM value avoids repeating existing information.

3. **Keypoint Verification Process**:

    - **Function**: Automatically verify whether the generated keypoints accurately describe the video content.
    - **Mechanism**: Three-step verification—(i) extract key information to be verified from descriptions; (ii) generate verification questions (Yes/No) for each piece of information; (iii) use two different MLLMs (GPT-4o and Qwen2-VL-72B) to watch the video and answer the verification questions; keypoints are only retained if both models confirm they pass. This guarantees the accuracy of the keypoints.
    - **Design Motivation**: Descriptions generated by MLLMs may contain hallucinations. Using dual-model cross-verification filters out incorrect information and ensures the quality of the benchmark.

### Loss & Training
AutoCaption itself does not involve training. However, the paper demonstrates the effect of fine-tuning InternVL2.5-8B with approximately 10K samples generated by AutoCaption: a 25.0% performance gain on MCTS-VCB and a 16.3% gain on DREAM-1K.

## Key Experimental Results

### Main Results
F1 score comparison of 20+ MLLMs on MCTS-VCB (5 dimensions + overall):

| Model | Appearance | Action | Environment | Object | Camera | Overall F1 |
|------|------|------|------|------|------|--------|
| Gemini-1.5-Pro | - | - | - | - | - | **71.2** |
| GPT-4o | - | - | - | - | - | 70.6 |
| LLaVA-OV-72B | 56.9 | 68.3 | 70.9 | 55.7 | 57.7 | 64.1 |
| InternVL2.5-78B | 48.2 | 53.8 | 60.3 | 44.4 | 40.6 | 52.4 |
| InternVL2.5-8B | 46.8 | 51.0 | 59.4 | 42.7 | 40.1 | 50.8 |

### Ablation Study
Effect of fine-tuning InternVL2.5-8B with AutoCaption data:

| Configuration | MCTS-VCB F1 | DREAM-1K F1 |
|------|-------------|-------------|
| InternVL2.5-8B (Original) | 50.8 | Baseline |
| + AutoCaption 10K Fine-tuning | **63.5 (+25.0%)** | **+16.3%** |

### Key Findings
- **Gemini-1.5-Pro is the strongest but achieves only 71.2 F1**: This indicates that the evaluation difficulty of MCTS-VCB is high; even the strongest closed-source model fails to cover about 30% of the keypoints.
- **Significant gap between open-source and closed-source**: The best open-source model LLaVA-OV-72B (64.1) lags behind Gemini-1.5-Pro (71.2) by 7.1 pp.
- **Object attributes and camera movement are weak areas**: All models perform worst in the Object Description and Camera Movement dimensions, indicating that existing MLLMs have insufficient understanding of fine-grained object attributes and camera language.
- **Scale does not always equal quality**: In the InternVL2.5 series, the improvement from 8B to 78B is only 1.6 pp (50.8 → 52.4), while LLaVA-OV-7B directly reaches 62.8, which shows that training data and strategies are more important than model scale.
- **The fine-tuning effect of AutoCaption data is surprising**: It yields a 25.0% improvement with only 10K samples and a 16.3% improvement when transferred to DREAM-1K, proving that AutoCaption is not only a good evaluation tool but also an excellent data generation tool.
- **An average of 122 keypoints vs 6.3 in DREAM-1K**: Roughly 20 times the keypoint density makes the evaluation more comprehensive, preventing models from "getting lucky" on a few covered points.

## Highlights & Insights
- **Innovative Application of MCTS for Content Discovery**: MCTS is usually used for decision-making and reasoning, but here it is used to "search the description space of video content"—allowing AI to systematically discover every describable detail in videos. This idea can be transferred to scenes like detailed image captioning and document information extraction.
- **Quality Control through Dual-Model Cross-Verification**: Using two different MLLMs to verify the accuracy of keypoints is more reliable than single-model verification, effectively controlling hallucinations.
- **Dual Value of 'Evaluation as Data'**: The evaluation keypoints generated by AutoCaption are themselves high-quality training data; a single framework solves both evaluation and data scarcity issues.

## Limitations & Future Work
- **Dependency on powerful MLLMs for initialization**: The A1 overall description action requires GPT-4o and Gemini-1.5-Pro, and verification also requires GPT-4o, leading to high API cost requirements.
- **Limited ability to process long videos**: Most videos in MCTS-VCB are relatively short, and the applicability to long videos (> 5 minutes) remains unverified.
- **Heuristic design of the 6 actions**: Action types are predefined, which may fail to cover all types of video content (such as emotions, narrative structures, etc.).
- **Future directions**: Leave the action design to be automatically discovered by MCTS; extend the framework to multilingual video captioning.

## Related Work & Insights
- **vs DREAM-1K**: DREAM-1K uses manually annotated keypoints, averaging only 6.3 per video and biasing towards event descriptions; MCTS-VCB automatically generates 122 per video and covers 5 dimensions, providing a more comprehensive evaluation.
- **vs MSR-VTT/MSCV**: Traditional benchmarks only provide single-sentence descriptions, making it completely impossible to perform fine-grained evaluation.
- **vs MCTS in Reasoning**: MCTS is used in mathematical reasoning (such as AlphaProof) to search for problem-solving paths. This work uses it to search for "description paths," which is an interesting cross-domain transfer.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying MCTS to video content discovery is a novel application.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 20+ model evaluations + fine-tuning verification + cross-benchmark transfer.
- Writing Quality: ⭐⭐⭐⭐ Clear framework description, sufficiently explained motivation and workflows.
- Value: ⭐⭐⭐⭐ Serves as both an evaluation tool and a data generation tool, offering double the value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art](godbench_a_benchmark_for_multimodal_large_language_models_in_video_comment_art.md)
- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[ACL 2025\] Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models](unsolvable_problem_detection.md)
- [\[ACL 2025\] Evaluating Multimodal Language Models as Visual Assistants for Visually Impaired Users](evaluating_multimodal_language_models_as_visual_assistants_for_visually_impaired.md)
- [\[ACL 2025\] VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service](vlminferslow_evaluating_the_efficiency_robustness_of.md)

</div>

<!-- RELATED:END -->
