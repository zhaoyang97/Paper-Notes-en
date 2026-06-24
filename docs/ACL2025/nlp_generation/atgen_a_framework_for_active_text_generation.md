---
title: >-
  [Paper Note] ATGen: A Framework for Active Text Generation
description: >-
  [ACL 2025][Text Generation][active learning] The authors propose ATGen, the first systematic active learning (AL) framework for NLG. It integrates state-of-the-art (SOTA) AL strategies, human/LLM annotation interfaces, parameter-efficient fine-tuning (PEFT), and vLLM inference optimization. Evaluation on four NLG tasks (including TriviaQA and GSM8K) demonstrates that active learning can reduce annotation costs by 2 to 4 times.
tags:
  - "ACL 2025"
  - "Text Generation"
  - "active learning"
  - "NLG"
  - "annotation efficiency"
  - "LLM annotation"
  - "framework"
date: 2026-05-08
content_hash: cd9e4924dc6e0d1b
---

# ATGen: A Framework for Active Text Generation

**Conference**: ACL 2025  
**arXiv**: [2506.23342](https://arxiv.org/abs/2506.23342)  
**Code**: [GitHub](https://github.com/Aktsvigun/atgen)  
**Area**: Text Generation  
**Keywords**: active learning, NLG, annotation efficiency, LLM annotation, framework  

## TL;DR

The authors propose ATGen, the first systematic active learning (AL) framework for NLG. It integrates state-of-the-art (SOTA) AL strategies, human/LLM annotation interfaces, parameter-efficient fine-tuning (PEFT), and vLLM inference optimization. Evaluation on four NLG tasks (including TriviaQA and GSM8K) demonstrates that active learning can reduce annotation costs by 2 to 4 times.

## Background & Motivation

- **Background**: With the rapid development of NLG tasks (such as summarization, QA, and reasoning), domain-specific tasks still require high-quality annotated data. While LLM-based annotation can partially replace human annotators, it remains highly expensive.
- **Limitations of Prior Work**: (1) Existing AL frameworks primarily support classification and sequence labeling tasks, lacking support for NLG. (2) There is no unified platform for evaluating AL strategies in NLG. (3) AL for modern LLMs requires support for PEFT and highly efficient inference.
- **Key Challenge**: High annotation costs of NLG tasks vs. the lack of systematic tools to reduce annotation requirements.
- **Goal**: Construct a unified active learning framework for NLG to reduce annotation costs.
- **Key Insight**: Build a full-stack framework integrating strategy selection, annotation, training, and evaluation.
- **Core Idea**: Systematically apply AL to NLG to lower the costs of both human and LLM API annotation.

## Method

### Overall Architecture

ATGen provides: (1) a collection of AL strategies (HUDS, HADAS, Facility Location, etc.); (2) a web GUI for human annotation; (3) automatic LLM annotation (supporting OpenAI, Anthropic, or local models); (4) parameter-efficient training using LoRA/QLoRA; (5) inference acceleration with vLLM/SGLang; and (6) benchmarking scripts.

### Key Designs

**Design 1: Integration of NLG-specific AL Strategies**
- **Function**: Implement all SOTA NLG AL strategies under a unified interface.
- **Mechanism**: Strategies include HUDS (Uncertainty + Metric Learning), HADAS (Hallucination-Aware), Facility Location (Submodular Function), BLEUVar, IDDS, among others.
- **Design Motivation**: Traditional AL strategies for classification tasks (such as least confidence) perform poorly in NLG, requiring dedicated evaluations.

**Design 2: Support for Dual-Mode Annotation**
- **Function**: Simultaneously support both human and LLM annotation modes.
- **Mechanism**: For the human mode, the experimental design (ED) strategy is recommended to select and annotate instances in a single batch. For the LLM mode, OpenAI batch APIs are supported (reducing costs by 50%).
- **Design Motivation**: Human annotation is highly susceptible to the latency of active learning iterations. The ED strategy eliminates waiting times associated with model retraining and querying.

**Design 3: Integration of Efficient Training and Inference**
- **Function**: Support LoRA/QLoRA/DoRA along with vLLM/SGLang/Unsloth.
- **Mechanism**: The AL loop requires multiple iterations of fine-tuning and inference. Integrating PEFT and high-performance inference frameworks makes active learning of large language models practical.
- **Design Motivation**: Active learning for LLMs is computationally infeasible without efficient training and inference.

### Loss & Training

Each AL strategy computes query scores using its own formulation. Standard causal LM loss with PEFT is utilized during training. Models are evaluated using EM, F1, ROUGE-2, and AlignScore.

## Key Experimental Results

### Main Results

**TriviaQA (Human Annotation Simulation, Qwen3-1.7B)**

| Strategy | EM at 4% Data | EM at 12% Data |
|------|-----------|------------|
| Random | ~30 | ~42 |
| HUDS | ~42 | ~48 |
| HADAS | ~40 | ~46 |
| Facility Location | ~38 | ~45 |

### Ablation Study

| Dimension | Finding |
|------|------|
| Human vs. LLM Annotation | LLM annotation overall degrades performance by a few percentage points on GSM8K |
| ED vs. AL | ED is superior in scenarios sensitive to annotation latency |
| Different Acquisition Models | Qwen3-1.7B performs well |

### Key Findings

1. Three strategies—HUDS, HADAS, and Facility Location—consistently and significantly outperform random sampling across multiple tasks.
2. Active learning is equally effective in LLM-assisted annotation scenarios, reducing API call costs by 2 to 4 times.
3. Error accumulation is still observed when using DeepSeek-R1 for mathematical reasoning annotation, indicating that domain-expert human annotators remain necessary for specialized tasks.

## Highlights & Insights

1. This is the first comprehensive active learning framework designed specifically for NLG, successfully filling a critical tool gap.
2. The dual-mode (human + LLM) annotation design aligns well with current trends in AI-assisted annotation.
3. The framework is open-source under a community-friendly MIT license.

## Limitations & Future Work

1. Data distribution bias introduced by active learning was not investigated.
2. The computational overhead of active learning for large-scale LLMs remains substantial.
3. Evaluation was focused primarily on English tasks, leaving multilingual scenarios unexplored.

## Related Work & Insights

- Existing toolkits like ALToolbox only support classification and information extraction, whereas ATGen extends support to NLG.
- Insight: Active learning remains highly valuable in the LLM era—focusing on reducing API costs rather than purely minimizing human effort.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ★★★☆☆ |
| Value | ★★★★★ |
| Experimental Thoroughness | ★★★★☆ |
| Writing Quality | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Better Open-Ended Text Generation: A Multicriteria Evaluation Framework](towards_better_open-ended_text_generation_a_multicriteria_evaluation_framework.md)
- [\[ACL 2025\] PerSphere: A Comprehensive Framework for Multi-Faceted Perspective Retrieval and Summarization](persphere_a_comprehensive_framework_for_multi-faceted_perspective_retrieval_and_.md)
- [\[ACL 2025\] Personalized Text Generation with Contrastive Activation Steering](personalized_text_generation_with_contrastive_activation_steering.md)
- [\[ACL 2025\] Writing Like the Best: Exemplar-Based Expository Text Generation](writing_like_best_exemplar.md)
- [\[ACL 2025\] Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems](dehumanizing_machines_anthropomorphic.md)

</div>

<!-- RELATED:END -->
