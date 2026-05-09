---
title: >-
  [Paper Note] Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation
description: >-
  [ACL 2026][LLM Evaluation][meeting effectiveness evaluation] This paper redefines meeting effectiveness evaluation by proposing an objective criterion of "goal achievement / time cost" and a temporal fine-grained evaluation paradigm. It constructs the AMI-ME dataset comprising 2,459 annotated segments from 130 meetings, and develops an LLM-based automatic evaluation framework achieving a Spearman correlation of 0.64.
tags:
  - ACL 2026
  - LLM Evaluation
  - meeting effectiveness evaluation
  - temporal fine-grained evaluation
  - LLM-as-Judge
  - topic segmentation
  - multi-party dialogue
date: 2026-05-08
content_hash: 1c400aaee379c505
---

# Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation

**Conference**: ACL 2026
**arXiv**: [2604.17260](https://arxiv.org/abs/2604.17260)
**Code**: [GitHub](https://github.com)
**Area**: LLM Evaluation
**Keywords**: meeting effectiveness evaluation, temporal fine-grained evaluation, LLM-as-Judge, topic segmentation, multi-party dialogue

## TL;DR

This paper redefines meeting effectiveness evaluation by proposing an objective criterion of "goal achievement / time cost" and a temporal fine-grained evaluation paradigm. It constructs the AMI-ME dataset comprising 2,459 annotated segments from 130 meetings, and develops an LLM-based automatic evaluation framework achieving a Spearman correlation of 0.64.

## Background & Motivation

**Background**: Meetings are a cornerstone of organizational collaboration, yet their effectiveness has long been assessed through post-meeting questionnaires that yield a single coarse-grained score for the entire session. Such approaches are costly, difficult to scale, and lack reproducibility.

**Limitations of Prior Work**: (1) A single score fails to capture the dynamic nature of meetings—a meeting may alternate between efficient and inefficient phases; (2) existing evaluation criteria are heterogeneous and often grounded in subjective perception, lacking generalizability; (3) meeting data are scarce and privacy-sensitive, impeding large-scale quantitative analysis.

**Key Challenge**: There is a need for an evaluation method that is both objective and universal while capturing the temporal dynamics of meetings, and that overcomes the scalability bottleneck of manual annotation.

**Goal**: (1) Define objective and universal criteria for meeting effectiveness evaluation; (2) propose a temporal fine-grained evaluation methodology; (3) construct a meta-evaluation dataset; (4) develop an LLM-based automatic evaluation framework.

**Key Insight**: Meetings are segmented into consecutive topical segments, each evaluated independently for effectiveness. This improves annotation reliability and substantially increases the volume of data points (from 130 to 2,459).

**Core Idea**: Meeting effectiveness is defined as "goal achievement / time cost," evaluated automatically at the granularity of fine-grained topical segments using LLM-as-a-Judge.

## Method

### Overall Architecture

The automatic evaluation framework proceeds in three steps: (1) **Meeting goal classification**—multi-label classification identifying meeting goals from 19 predefined categories; (2) **Topic segmentation**—splitting meeting transcripts into consecutive fine-grained topical segments; (3) **Segment effectiveness scoring**—using an LLM to score each segment on its contribution to overall goals and time utilization efficiency.

### Key Designs

1. **Definition of Objective Evaluation Criteria**:

    - Function: Provide universal evaluation criteria independent of participants' subjective perceptions.
    - Mechanism: Effectiveness = goal achievement / time cost. "Goals" are defined as those synthesized from meeting content after the meeting concludes (rather than pre-planned agendas), accommodating the natural evolution of topics during a meeting. Evaluation is based on these emergent goals rather than the original agenda.
    - Design Motivation: Subjective criteria (e.g., participant satisfaction) cannot be assessed independently from meeting content, whereas objective criteria (goal achievement) can.

2. **Temporal Fine-grained Evaluation Methodology**:

    - Function: Evaluate effectiveness at the level of minimal topical units to capture meeting dynamics.
    - Mechanism: A reference-guided segmentation approach is adopted—using original AMI annotations as reference, an LLM (Gemini-2.5-Pro) produces continuous and finer-grained segmentations. Each segment constitutes the smallest indivisible topical unit. Overall meeting effectiveness is computed as the duration-weighted average of segment scores.
    - Design Motivation: (1) Shorter segments are easier to annotate accurately; (2) data points expand from 130 meetings to 2,459 segments; (3) detailed analysis of meeting dynamics becomes feasible.

3. **Segmentation Alignment Evaluation**:

    - Function: Enable fair evaluation when model-produced segmentations differ from ground-truth segmentations.
    - Mechanism: For each ground-truth segment, the duration-weighted average score of all overlapping predicted segments is computed: $\hat{e}_{t_i}^{t_{i+1}} = \sum_j \hat{e}_j \cdot \Delta_{i,j} / \sum_j \Delta_{i,j}$
    - Design Motivation: Model segmentations and ground-truth segmentations typically differ in both count and boundaries, making direct one-to-one comparison infeasible.

### Loss & Training

This is a purely evaluative framework and involves no model training. Annotations were produced by six trained professional annotators divided into two groups, using a 5-point scale, with ICC values of 0.82–0.88 (indicating "good" reliability).

## Key Experimental Results

### Main Results

| Model | Spearman (All Meetings) | Kendall (All Meetings) |
|-------|------------------------|----------------------|
| Qwen3-32B (non-thinking) | **0.6445** | **0.4803** |
| GPT-4o | 0.6341 | 0.4756 |
| Llama3.3-70B | 0.6072 | 0.4854 |
| DeepSeek-R1-70B | 0.6132 | 0.4663 |
| Gemini-2.5-Flash | 0.5624 | 0.4122 |

### Ablation Study

| Setting | Spearman | Note |
|---------|----------|------|
| Ground-truth segmentation + ground-truth goals | 0.6445 | Upper bound |
| Predicted segmentation | 0.2256 | Segmentation error substantially degrades performance |
| End-to-end (speech → evaluation) | 0.2180 | ASR errors have minimal impact (close to predicted segmentation) |
| Theoretical upper bound | 0.6417 | Inherent penalty from segmentation inconsistency |

### Key Findings

- Topic segmentation quality is the primary bottleneck—Spearman drops from 0.64 to 0.23 when moving from ground-truth to predicted segmentations.
- ASR errors have negligible impact on effectiveness evaluation (0.23 vs. 0.22), indicating LLM robustness to noisy transcripts.
- Gemini-2.5-Flash performs worst due to its tendency to overuse the lowest score.
- Reasoning models (DeepSeek-R1) do not necessarily outperform non-reasoning models (Qwen3 in non-thinking mode achieves the best results).

## Highlights & Insights

- The definition of "effectiveness = goal achievement / time" is concise and universal, transforming the vague notion of "meeting quality" into a quantifiable ratio.
- Temporal fine-grained evaluation achieves three objectives simultaneously: more accurate annotation, more data, and dynamic analysis.
- The segmentation alignment mechanism elegantly resolves boundary inconsistency issues in evaluation.

## Limitations & Future Work

- Current topic segmentation quality severely limits end-to-end performance (0.64 → 0.23).
- Validation is conducted solely on the AMI corpus, covering a limited range of meeting types.
- The constraint of at most three goals per meeting may not generalize to complex meetings.
- Multimodal signals (video, shared screens, etc.) and their effects on effectiveness are not considered.

## Related Work & Insights

- **vs. Traditional Meeting Evaluation**: Traditional methods produce a single score relying on questionnaires; this paper is the first to achieve automated temporal fine-grained evaluation.
- **vs. G-Eval**: The evaluation component draws on G-Eval's chain-of-thought and form-filling paradigm, but targets an entirely different task.
- **vs. Multi-party Dialogue Research**: This work provides an evaluation foundation for proactive intervention by multi-party dialogue agents.

## Rating

- Novelty: ⭐⭐⭐⭐ The evaluation paradigm is novel, though the core method (LLM-as-Judge) is standard practice.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ End-to-end evaluation, cross-meeting-type analysis, and segmentation impact analysis are highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Logic is clear, problem definitions are precise, and the dataset construction process is transparent.
- Value: ⭐⭐⭐⭐ Introduces a new paradigm and benchmark for meeting analysis, though practical deployment requires addressing the segmentation bottleneck.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning](../../ICCV2025/llm_evaluation/omnidiff_a_comprehensive_benchmark_for_fine-grained_image_difference_captioning.md)
- [\[ICLR 2026\] Enabling Fine-Grained Operating Points for Black-Box LLMs](../../ICLR2026/llm_evaluation/enabling_fine-grained_operating_points_for_black-box_llms.md)
- [\[ACL 2026\] AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage](autoreproduce_automatic_ai_experiment_reproduction_with_paper_lineage.md)
- [\[ICLR 2026\] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home Agents](../../ICLR2026/llm_evaluation/simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_agents.md)
- [\[AAAI 2026\] NeSTR: A Neuro-Symbolic Abductive Framework for Temporal Reasoning in Large Language Models](../../AAAI2026/llm_evaluation/nestr_a_neuro-symbolic_abductive_framework_for_temporal_reasoning_in_large_langu.md)

</div>

<!-- RELATED:END -->
