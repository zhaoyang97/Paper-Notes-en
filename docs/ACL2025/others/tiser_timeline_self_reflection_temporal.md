---
title: >-
  [Paper Note] Learning to Reason Over Time: Timeline Self-Reflection for Temporal Reasoning
description: >-
  [ACL 2025][Temporal reasoning] The TISER framework is proposed to achieve test-time scaling for LLM temporal reasoning through a four-stage pipeline of "reasoning → timeline construction → self-reflection → answer generation." When combined with fine-tuning on synthetic reasoning trajectory data, this framework enables 7B open-source models to outperform GPT-4 on multiple temporal reasoning benchmarks and achieve SOTA results on tasks such as TGQA.
tags:
  - "ACL 2025"
  - "Temporal reasoning"
  - "self-reflection"
  - "timeline construction"
  - "test-time scaling"
  - "synthetic data"
date: 2026-05-08
content_hash: ccc179f8c0969edf
---

# Learning to Reason Over Time: Timeline Self-Reflection for Temporal Reasoning

**Conference**: ACL 2025  
**arXiv**: [2504.05258](https://arxiv.org/abs/2504.05258)  
**Code**: [https://github.com/amazon-science/TISER](https://github.com/amazon-science/TISER)  
**Area**: Others  
**Keywords**: Temporal reasoning, self-reflection, timeline construction, test-time scaling, synthetic data  

## TL;DR
The TISER framework is proposed to achieve test-time scaling for LLM temporal reasoning through a four-stage pipeline of "reasoning → timeline construction → self-reflection → answer generation." When combined with fine-tuning on synthetic reasoning trajectory data, this framework enables 7B open-source models to outperform GPT-4 on multiple temporal reasoning benchmarks and achieve SOTA results on tasks such as TGQA.

## Background & Motivation
**Background**: LLMs exhibit excellent performance across many tasks, but temporal reasoning (understanding event order, duration, and temporal interval relationships) remains a weak spot. Benchmarks like TRAM and TimeBench indicate that even state-of-the-art models frequently fail on complex temporal queries.

**Limitations of Prior Work**: Existing approaches rely on prompt engineering (CoT), specialized pre-training (Temp-T5), or mathematical reasoning modules; however, they all lack explicit temporal structure representations. Models do not explicitly organize and cross-reference temporal information during the reasoning process.

**Key Challenge**: Temporal reasoning requires models to concurrently excel at two tasks: (a) extracting and sequencing temporal events from text, and (b) making logical inferences based on temporal order. Pure CoT reasoning lacks structured temporal representations, making it error-prone under complex temporal dependencies.

**Goal**: How to enable LLMs to explicitly construct timelines during inference and leverage self-reflection to detect and correct inconsistencies in temporal reasoning?

**Key Insight**: Inspired by test-time scaling, the reasoning trajectory is extended to capture complex temporal dependencies. Instead of simply being longer, it is systematically structured into three phases: reasoning → timeline → reflection.

**Core Idea**: Allowing LLMs to explicitly construct an event timeline during reasoning to serve as a "scaffolding", and then self-reflect and correct reasoning outcomes against the timeline, thereby significantly improving temporal reasoning accuracy.

## Method

### Overall Architecture
Four-stage reasoning pipeline (iterative):
1. **Stage I - Reasoning**: Generate an initial CoT reasoning trajectory $r$ based on the question and temporal context.
2. **Stage II - Timeline Construction**: Extract temporal events from the reasoning trajectory and context, organizing them into an ordered timeline $t$.
3. **Stage III - Reflection**: Compare the reasoning trajectory $r$ with the timeline $t$, detect inconsistencies/omissions/errors, and generate an improved reasoning trajectory $r'$.
4. **Stage IV - Answer Generation**: Generate the final answer based on the refined reasoning and the timeline.

### Key Designs

1. **Explicit Timeline Construction (Stage II)**:

    - **Function**: Extract all relevant temporal events from the reasoning trajectory and original text, arranging them in chronological order.
    - **Mechanism**: Aggregate scattered temporal information in the text into an ordered structure, mimicking how humans draw timelines to solve complex temporal problems.
    - **Design Motivation**: The timeline serves as an "external memory", allowing the model to intuitively cross-reference the chronological order of events rather than relying on implicit parametric memory.

2. **Iterative Self-Reflection (Stage III)**:

    - **Function**: Compare initial reasoning against the timeline, detect inconsistencies (e.g., incorrect event sorting, missing key time points), and generate corrected reasoning.
    - **Mechanism**: Form a feedback loop of reasoning → timeline → cross-referencing → correction, which can iterate repeatedly until consistency is reached.
    - **Design Motivation**: The core of test-time scaling—improving accuracy by extending the reasoning process.

3. **Synthetic Reasoning Trajectory Dataset**:

    - **Function**: Starting from existing temporal reasoning datasets, use GPT-4 or DeepSeek to generate intermediate reasoning trajectories in the TISER format.
    - **Mechanism**: For each sample $(q, a, c)$, generate a complete trajectory containing reasoning $r$, timeline $t$, and reflection $f$. Retain only samples where the final answer $a'$ matches the gold answer $a$.
    - **Design Motivation**: Ensure the correctness of the synthetic reasoning process by retaining only trajectories leading to the correct answer.

4. **Structured Output Templates**:

    - Use XML tags to separate outputs of different stages: `<reasoning>`, `<timeline>`, `<reflection>`, `<answer>`.
    - LoRA fine-tuning enables the model to learn to output in this format.

### Loss & Training
- Base Models: Mistral-7B, Qwen2.5-7B
- Fine-tuning Method: LoRA SFT
- Training Data: Synthetic reasoning trajectory versions of TGQA + TempReason + TimeQA
- Data Generator: GPT-4 or DeepSeek V2.5

## Key Experimental Results

### Main Results (Exact Match / F1)

| Model | Inference Method | TGQA | TempReason L2 | TempReason L3 | TimeQA Easy | TimeQA Hard | Average |
|------|---------|------|---------------|---------------|-------------|-------------|------|
| GPT-4 | Standard | 72.5/82.5 | 78.6/86.2 | 81.9/88.3 | 83.6/93.7 | 76.0/85.3 | 78.5/87.2 |
| GPT-4 | TISER | 82.8/93.4 | 79.8/87.2 | 84.7/91.3 | 84.4/90.5 | 77.2/86.4 | 81.8/89.8 |
| Qwen2.5-7B | Standard | 46.1/48.9 | 51.0/53.6 | 40.1/42.7 | 70.9/73.5 | 53.2/55.8 | 52.3/55.0 |
| **Mistral-7B + TISER-FT (GPT-4)** | **TISER** | **80.5/87.4** | **82.5/84.3** | **87.1/88.5** | **97.5/98.5** | **95.9/96.4** | **88.7/91.0** |
| **Qwen2.5-7B + TISER-FT (GPT-4)** | **TISER** | **84.5/94.2** | **85.5/87.5** | - | - | - | - |

### Ablation Study

| Configuration | Average EM | Description |
|------|---------|------|
| Full TISER | 85.6 | Reasoning + Timeline + Reflection |
| w/o Reflection | Decrease | Without iterative correction |
| w/o Timeline | Decrease | Without explicit temporal structure |
| Standard CoT | 55.7 | Baseline standard fine-tuning |

### Key Findings
- **7B Model Outperforms GPT-4**: TISER-fine-tuned Mistral-7B achieves 88.7 EM, significantly outperforming GPT-4's 78.5 EM (+10.2).
- **Timeline is Core**: Explicitly constructing timelines provides a huge boost compared to pure CoT because it externalizes implicit temporal information into verifiable structures.
- **Self-Reflection is Effective but Relies on Timeline Support**: Self-reflection without a timeline as an "anchor" shows limited efficacy; reflection requires a structured reference.
- **Improvement Even in Standard Inference**: TISER-fine-tuned models perform better than standard fine-tuned models even when not using the TISER inference pipeline (i.e., using standard inference).
- **Robust OOD Generalization**: Performance is maintained or even improved on unseen benchmarks such as MultiHopRAG and Test-of-Time.

## Highlights & Insights
- **The "Drawing a Timeline" Concept** is highly natural and intuitive—exactly how humans resolve complex temporal problem-solving. Translating this cognitive strategy into an LLM reasoning pipeline is highly ingenious.
- **Synthetic Data Quality Control** is well-designed. Retaining only reasoning trajectories that lead to correct answers ensures the accuracy of the training signal.
- **Small Models Greatly Outperforming Large Models** carries substantial practical significance. A 7B model equipped with TISER beats GPT-4 by over 10 points, demonstrating that structured reasoning strategies matter more than model scale.

## Limitations & Future Work
- The model is currently evaluated only on temporal reasoning tasks. The underlying concept of TISER (explicitly building domain structures $\to$ self-reflection) could potentially be transferred to other structured reasoning tasks, such as spatial reasoning.
- Dependence on the quality of training data generated by GPT-4.
- The configuration of iterative reflection loops and stopping criteria requires further investigation.

## Related Work & Insights
- **vs TG-LLM (Xiong et al. 2024)**: TG-LLM also focuses on temporal reasoning but uses CoT. Ours introduces explicit timelines and self-reflection, resulting in a substantial performance gain.
- **vs s1 (Muennighoff et al. 2025)**: s1 utilizes budget forcing for general test-time scaling, whereas TISER is a specialized test-time scaling method tailored for temporal reasoning.
- **vs Self-Refine (Madaan et al. 2023)**: Self-Refine performs general self-reflection, whereas TISER introduces timelines as a structured reference for reflection.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combined design of timeline construction and self-reflection is both natural and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Benchmarked on multiple datasets (5+), multiple models, OOD evaluations, complete ablation studies, and impressive results outperforming GPT-4 using only a 7B model.
- **Writing Quality**: ⭐⭐⭐⭐ Clear flow with standard algorithm pseudocode.
- **Value**: ⭐⭐⭐⭐⭐ Provides a transferable paradigm for enhanced structured reasoning, with open-sourced code and data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Learning to Reason from Feedback at Test-Time](learning_to_reason_from_feedback_at_test-time.md)
- [\[ACL 2025\] Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate](improve_rule_retrieval_and_reasoning_with_self-induction_and_relevance_reestimat.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] Inducing Lexicons of In-Group Language with Socio-Temporal Context](inducing_lexicons_of_in-group_language_with_socio-temporal_context.md)
- [\[ACL 2025\] Contextual Experience Replay for Self-Improvement of Language Agents](contextual_experience_replay_for_self-improvement_of_language_agents.md)

</div>

<!-- RELATED:END -->
