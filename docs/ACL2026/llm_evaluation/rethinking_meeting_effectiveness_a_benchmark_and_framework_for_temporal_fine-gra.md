---
title: >-
  [Paper Note] Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation
description: >-
  [ACL 2026][LLM Evaluation][Meeting effectiveness evaluation] This paper redefines meeting effectiveness evaluation by proposing an objective "goal achievement / time cost" standard and a temporal fine-grained evaluation…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Meeting effectiveness evaluation"
  - "temporal fine-grained evaluation"
  - "LLM-as-Judge"
  - "topic segmentation"
  - "multi-party dialogue"
date: 2026-05-08
content_hash: 896ad96b01cd5e8f
---

# Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation

**Conference**: ACL 2026  
**arXiv**: [2604.17260](https://arxiv.org/abs/2604.17260)  
**Code**: [GitHub](https://github.com)  
**Area**: LLM Evaluation  
**Keywords**: Meeting effectiveness evaluation, temporal fine-grained evaluation, LLM-as-Judge, topic segmentation, multi-party dialogue

## TL;DR

This paper redefines meeting effectiveness evaluation by proposing an objective "goal achievement / time cost" standard and a temporal fine-grained evaluation paradigm. It constructs the AMI-ME dataset containing 2,459 annotated snippets from 130 meetings and develops an LLM-based automatic evaluation framework, achieving a Spearman correlation of 0.64.

## Background & Motivation

**Background**: Meetings are cornerstones of organizational collaboration, but their effectiveness evaluation has long relied on post-hoc surveys, producing a single coarse-grained score for the entire session. This approach is costly, difficult to scale, and lacks reproducibility.

**Limitations of Prior Work**: (1) A single score fails to capture the dynamic nature of meetings—one meeting may have alternating phases of high and low efficiency; (2) existing evaluation standards vary and are often based on subjective perceptions, lacking universality; (3) meeting data is scarce and involves privacy concerns, hindering large-scale quantitative analysis.

**Key Challenge**: There is a need for an evaluation method that is both objective and universal while capturing the temporal dynamics of meetings and overcoming the scalability bottleneck of manual annotation.

**Goal**: (1) Define objective and universal meeting effectiveness evaluation standards; (2) propose a temporal fine-grained evaluation method; (3) build a meta-evaluation dataset; (4) develop an LLM automatic evaluation framework.

**Key Insight**: Segmenting meetings into continuous topic snippets and evaluating effectiveness independently for each snippet increases annotation reliability and significantly expands the data volume (from 130 data points to 2,459).

**Core Idea**: Define meeting effectiveness as "goal achievement / time cost" and perform automatic evaluation using LLM-as-a-Judge on fine-grained topic snippets.

## Method

### Overall Architecture

The automatic evaluation framework consists of three steps: (1) **Meeting Goal Classification**—multi-label classification to identify meeting goals from 19 predefined goals; (2) **Topic Segmentation**—segmenting meeting transcripts into continuous fine-grained topic snippets; (3) **Snippet Effectiveness Evaluation**—using an LLM to score each snippet, assessing its contribution to overall goals and the efficiency of time utilization.

### Key Designs

1. **Objective Evaluation Standard Definition**:

    - Function: Provide universal evaluation standards independent of participants' subjective perceptions.
    - Mechanism: Effectiveness = Goal Achievement / Time Cost. "Goals" are defined as goals synthesized from content after the meeting (rather than pre-planned) to adapt to the natural evolution of topics. Evaluation is based on these emergent goals rather than the original agenda.
    - Design Motivation: Subjective standards (e.g., participant satisfaction) cannot be assessed independently from meeting content, whereas objective standards (goal achievement) can.

2. **Temporal Fine-grained Evaluation Method**:

    - Function: Evaluate effectiveness at the level of atomic topic units to capture meeting dynamics.
    - Mechanism: Using a reference-guided segmentation method—taking AMI original annotations as reference, LLMs (Gemini-2.5-Pro) generate continuous and finer-grained segments. Each snippet is the smallest topic unit that cannot be further divided. Total meeting effectiveness = time-weighted average of snippet scores.
    - Design Motivation: (1) Short snippets are easier to annotate accurately; (2) expansion from 130 meetings to 2,459 data points; (3) provides detailed analysis of meeting dynamics.

3. **Segmentation Alignment Evaluation**:

    - Function: Solve the issue of fair evaluation when model segmentation is inconsistent with ground truth segmentation.
    - Mechanism: For each ground truth snippet, calculate the time-weighted average score of all overlapping predicted snippets: $\hat{e}_{t_i}^{t_{i+1}} = \sum_j \hat{e}_j \cdot \Delta_{i,j} / \sum_j \Delta_{i,j}$.
    - Design Motivation: The number and boundaries of model segments and ground truth segments are typically different, making direct one-to-one comparison unfeasible.

### Loss & Training

This is a pure evaluation framework involving no training. Annotations were completed by 6 trained professional annotators (split into two groups) using a 5-point scale, reaching an ICC of 0.82-0.88 ("Good" level reliability).

## Key Experimental Results

### Main Results

| Model | Spearman (All Meetings) | Kendall (All Meetings) |
|------|-------------------|------------------|
| Qwen3-32B (Non-reasoning) | **0.6445** | **0.4803** |
| GPT-4o | 0.6341 | 0.4756 |
| Llama3.3-70B | 0.6072 | 0.4854 |
| DeepSeek-R1-70B | 0.6132 | 0.4663 |
| Gemini-2.5-Flash | 0.5624 | 0.4122 |

### Ablation Study

| Setting | Spearman | Explanation |
|------|----------|------|
| GT Segmentation + GT Goal | 0.6445 | Upper bound |
| Predicted Segmentation | 0.2256 | Segmentation errors drastically reduce performance |
| End-to-end (Speech to Eval) | 0.2180 | Minimal impact from ASR errors (close to predicted segmentation) |
| Theoretical Upper Bound | 0.6417 | Inherent penalty of segmentation inconsistency |

### Key Findings

- Topic segmentation quality is the bottleneck—Spearman drops from 0.64 to 0.23 when moving from ground truth to predicted segmentation.
- ASR errors have very little impact on effectiveness evaluation (0.23 vs 0.22), indicating LLM robustness to noisy text.
- Gemini-2.5-Flash performs the worst due to a tendency to overuse the lowest scores.
- Reasoning models (DeepSeek-R1) do not necessarily outperform non-reasoning models (Qwen3 non-reasoning mode performs best).

## Highlights & Insights

- The definition "Effectiveness = Goal Achievement / Time" is concise and universal—transforming the vague "Is the meeting good?" into a quantifiable ratio.
- Temporal fine-grained evaluation yields three benefits: more accurate annotation, more data, and dynamic analysis.
- The segmentation alignment mechanism elegantly solves boundary inconsistency issues in evaluation.

## Limitations & Future Work

- Current topic segmentation quality severely limits end-to-end performance (0.64 → 0.23).
- Validated only on the AMI corpus with limited meeting types.
- The limit of at most 3 goals per meeting may not apply to complex meetings.
- Multimodal signals (video, screen sharing, etc.) impact on effectiveness was not considered.

## Related Work & Insights

- **vs Traditional Meeting Evaluation**: Traditional methods produce single scores and depend on questionnaires; this work achieves automated temporal fine-grained evaluation for the first time.
- **vs G-Eval**: The evaluation part borrows G-Eval's CoT and form-filling paradigms but targets a completely different task.
- **vs Multi-party Dialogue Research**: Provides an evaluation foundation for proactive intervention by multi-party dialogue agents.

## Rating

- Novelty: ⭐⭐⭐⭐ Evaluation paradigm is novel, but the core method (LLM-as-Judge) is standard practice.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ End-to-end evaluation, across meeting types, and analysis of segmentation impact are very comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, precise problem definition, and transparent dataset construction process.
- Value: ⭐⭐⭐⭐ Provides a new paradigm and benchmark for meeting analysis, but practical application requires solving the segmentation bottleneck.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology](k-metbench_a_multi-dimensional_benchmark_for_fine-grained_evaluation_of_expert_r.md)
- [\[ACL 2026\] IF-Critic: Towards a Fine-Grained LLM Critic for Instruction-Following Evaluation](if-critic_towards_a_fine-grained_llm_critic_for_instruction-following_evaluation.md)
- [\[ICML 2026\] On Effectiveness and Efficiency of Agentic Tool-calling and RL Training](../../ICML2026/llm_evaluation/on_effectiveness_and_efficiency_of_agentic_tool-calling_and_rl_training.md)
- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)
- [\[ICCV 2025\] OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning](../../ICCV2025/llm_evaluation/omnidiff_a_comprehensive_benchmark_for_fine-grained_image_difference_captioning.md)

</div>

<!-- RELATED:END -->
