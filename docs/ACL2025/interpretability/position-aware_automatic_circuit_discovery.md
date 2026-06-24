---
title: >-
  [Paper Note] Position-aware Automatic Circuit Discovery
description: >-
  [ACL 2025][Interpretability][Circuit Discovery] Proposes Position-aware Edge Attribution Patching (PEAP) and a dataset Schema mechanism to address the cancellation effect and overestimation of importance in automatic circuit discovery caused by ignoring position information, enabling smaller and more faithful circuit discovery.
tags:
  - "ACL 2025"
  - "Interpretability"
  - "Circuit Discovery"
  - "Position-awareness"
  - "Mechanistic Interpretability"
  - "Attribution Patching"
  - "Schema"
date: 2026-05-08
content_hash: e024e0645035ee01
---

# Position-aware Automatic Circuit Discovery

**Conference**: ACL 2025  
**arXiv**: [2502.04577](https://arxiv.org/abs/2502.04577)  
**Code**: [https://github.com/technion-cs-nlp/PEAP](https://github.com/technion-cs-nlp/PEAP)  
**Area**: Interpretability  
**Keywords**: Circuit Discovery, Position-awareness, Mechanistic Interpretability, Attribution Patching, Schema

## TL;DR

Proposes Position-aware Edge Attribution Patching (PEAP) and a dataset Schema mechanism to address the cancellation effect and overestimation of importance in automatic circuit discovery caused by ignoring position information, enabling smaller and more faithful circuit discovery.

## Background & Motivation

Circuit analysis is a core method for understanding the internal mechanisms of language models, revealing how a model works by finding the minimal computational subgraph that executes a specific task. However, existing **automatic** circuit discovery methods (such as EAP) suffer from a key blind spot: **ignoring position information**.

Specifically, position-independent methods suffer from two major issues:

**Cancellation Effect (Low Recall)**: The importance score of a component at different positions may have opposite signs. When summing across positions, positive and negative cancellation causes actually important edges to be missed. Experiments confirm that in the IOI task, the difference in top-1% edge ranking is as high as 17.1%.

**Importance Overestimation (Low Precision)**: Without considering positions, methods tend to select edges that have small impacts across multiple positions, while ignoring those that have large impacts in a small number of positions. Similarly, the difference in the top-1% reaches 17.5%.

Manual circuit discovery (e.g., IOI circuit, Greater-Than circuit) can distinguish positions but is not scalable and is prone to introducing human bias. The goal of this work is: **to maintain position sensitivity while automating the process**.

## Method

### Overall Architecture

The method consists of two core contributions:
1. **PEAP** (Position-aware Edge Attribution Patching): extending edge attribution patching to cross-position edges.
2. **Schema**: defining semantic labels to handle the position alignment problem of variable-length inputs.

### Key Designs

1. **Position-aware Edge Attribution Patching (PEAP)**:

    - The original EAP only handles edges at the same position. PEAP extends this to **cross-position attention edges**.
    - For an attention head $h^i_t$ at position $t$, it connects to nodes at other positions through three types of edges: value, key, and query.
    - By patching $v_{t'}$, $k_{t'}$, and $q_t$ separately, the attribution score for each type of edge is calculated.
    - Using a first-order linear approximation: $M(x|e=e_{x'}) - M(x) \approx (z^*_{h^i_t} - z_{h^i_t})^\top \nabla_{z_{h^i_t}} M(x)$
    - Key: Evaluating the edge importance of each position separately, rather than aggregating across positions.

2. **Dataset Schema**:

    - **Problem**: Input lengths of real-world datasets vary, making direct position alignment across samples impossible.
    - **Proposed Solution**: Define semantic spans (e.g., "Subject", "Year"), mapping samples of different lengths to a unified abstract computation graph.
    - Mapping function $f_\mathcal{S}^x$: Maps edges of the abstract graph to a set of edges in the specific sample's computation graph.
    - Schema-level attribution score: Summing the scores of all concrete edges mapped to the same abstract edge, and then averaging across samples.

3. **Automated Schema Generation Pipeline**:

    - Using LLMs (Claude 3.5 Sonnet) to automatically generate Schemas: sampling multiple subsets to generate them separately, and then unifying them into a final version.
    - **Saliency Enhancement**: Using input×gradient to calculate the importance of each token position to the target metric, generating a saliency mask.
    - Providing the mask to the LLM so that it considers the model's actual computation patterns when designing the Schema.
    - Schema application is also automated by the LLM, with a validity rate of $\geq 90\%$ considered successful.

### Loss & Training

This is an inference-time analysis method and does not involve model training. The key hyperparameter is the choice of circuit size—constructed by incrementally adding the most important edges using a greedy algorithm.

## Key Experimental Results

### Main Results

On the Greater-Than task (GPT2-small), the circuits discovered by PEAP are several orders of magnitude smaller than non-positional circuits at the same level of faithfulness.

**Hard Faithfulness Comparison (Multi-task, Multi-model)**:

| Task | Model | PEAP+Schema | Non-positional | Gain |
|------|------|------------|---------|------|
| Greater-Than | GPT2-small | Smaller circuit achieves equivalent faithfulness | Larger circuit required for faithfulness | Significant |
| IOI | GPT2-small | LLM + Mask ≈ Manual Schema | Large gap for non-positional | Significant |
| IOI | Llama-3-8B | LLM Schema slightly outperforms manual | - | Significant |
| Winobias | Llama-3-8B | Mask enhancement consistently improves | - | Significant |

### Ablation Study

**Comparison of Schema Generation Methods**:

| Method | Characteristics | Faithfulness |
|------|------|--------|
| Manual Schema | Gold standard | High |
| LLM + Mask | Automated + Saliency-guided | ≈ Manual |
| LLM Only | Purely automated | Slightly lower than LLM+Mask |
| No Schema (EAP) | Non-positional | Lowest |

**Quantifying Cancellation and Overestimation Effects (IOI, GPT2-small)**:

| K% | Cancellation Difference | Control Difference | Overestimation Difference | Control Difference |
|----|---------|---------|---------|---------|
| 1% | 17.1% | 3.9% | 17.5% | 3.6% |
| 5% | 13.4% | 2.4% | 14.6% | 2.1% |
| 10% | 12.1% | 2.3% | 12.4% | 2.2% |

### Key Findings

- Position-aware circuits achieve a better faithfulness-circuit size trade-off across all tasks and models.
- LLM-generated Schema + saliency mask can match or even outperform manual designs.
- On Llama-3-8B, the LLM-generated Schema even outperforms the manual Schema designed for GPT2-small, indicating that Schemas should be customized for specific models.
- The cancellation effect can occur **across positions within a single sample**, not just across samples.

## Highlights & Insights

- **Clear Theoretical Intuition**: The formalization of the cancellation and overestimation issues is intuitive. Simple diagrams (Figure 2) allow readers to quickly grasp the pitfalls of ignoring position.
- **Fully Automated Pipeline**: From Schema generation and application to circuit discovery, the entire process can be fully automated, drastically reducing the manual labor costs of mechanistic interpretability research.
- **Model-aware Schema Design**: Allowing LLMs to "see" the model's computation pattern through input×gradient saliency scores is an elegant AI-assisted interpretability method.
- **Discovery of an Counter-intuitive Phenomenon**: The manual Schema meticulously designed for GPT2-small underperforms when transferred to Llama-3-8B compared to the automatically generated schema by the LLM.

## Limitations & Future Work

- Schema requires spans in all samples to appear in the same order, which limits its applicability to more free-form text formats.
- There is a lack of a priori principles for what constitutes a "good" Schema; currently, it can only be evaluated a posteriori through downstream faithfulness.
- LLM automatic application of Schemas also suffers from failure rates, requiring filtering of invalid applications.
- Experiments were only conducted on GPT2-small and Llama-3-8B; scalability on larger models remains to be verified.
- This work only used Claude 3.5 Sonnet; other LLMs (Llama-3-70B, GPT-4o) failed to meet the standards for Schema application.

## Related Work & Insights

- Directly improves the methodology of EAP (Syed et al., 2023), introducing the position dimension to automatic circuit discovery.
- Complements manual circuit discovery work (such as the IOI circuit in Wang et al., 2023; the Greater-Than circuit in Hanna et al., 2024)—PEAP achieves automated position awareness.
- The concept of Schema is similar to the role labeling in the IOI dataset (IO, S1, etc.) but generalizes it into a systematic method.
- Saliency-guided Schema generation is an interesting example of using LLM-as-agent for automated interpretability.

## Rating

- Novelty: ⭐⭐⭐⭐ Position-aware circuit discovery is a natural yet important improvement; the Schema + automated pipeline adds practical value.
- Experimental Thoroughness: ⭐⭐⭐⭐ A comprehensive comparison across three tasks, two models, and multiple Schema generation methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear articulation of research motivation, well-designed diagrams, and structurally progressive method explanations.
- Value: ⭐⭐⭐⭐ Provides more precise tools and automated pipelines for mechanistic interpretability research, with the potential to become a standard practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](../../ICLR2026/interpretability/formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)
- [\[ICCV 2025\] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](../../ICCV2025/interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [\[ICLR 2026\] Setting Up for Failure: Automatic Discovery of the Neural Mechanisms of Cognitive Errors](../../ICLR2026/interpretability/setting_up_for_failure_automatic_discovery_of_the_neural_mechanisms_of_cognitive.md)
- [\[ACL 2025\] A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)
- [\[ICML 2026\] All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs](../../ICML2026/interpretability/all_circuits_lead_to_rome_rethinking_functional_anisotropy_in_circuit_and_sheaf_.md)

</div>

<!-- RELATED:END -->
