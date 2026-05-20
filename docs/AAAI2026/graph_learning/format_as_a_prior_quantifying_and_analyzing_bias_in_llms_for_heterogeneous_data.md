---
title: >-
  [Paper Note] Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data
description: >-
  [AAAI 2026][Graph Learning][Format Bias] This paper presents the first systematic investigation of format bias in LLMs when processing heterogeneous-format data (text / table / infobox / knowledge graph). Through a three…
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "Format Bias"
  - "LLM"
  - "Heterogeneous Data"
  - "Knowledge Graph"
  - "Attention Mechanism"
  - "Information Richness"
  - "Structural Quality"
date: 2026-05-08
content_hash: f4501a50979361fd
---

# Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data

**Conference**: AAAI 2026
**arXiv**: [2508.15793](https://arxiv.org/abs/2508.15793)  
**Code**: [github.com/NLPGM/Format-as-a-prior](https://github.com/NLPGM/Format-as-a-prior)  
**Area**: Graph Learning / LLMs with Heterogeneous Data
**Keywords**: Format Bias, LLM, Heterogeneous Data, Knowledge Graph, Attention Mechanism, Information Richness, Structural Quality

## TL;DR

This paper presents the first systematic investigation of format bias in LLMs when processing heterogeneous-format data (text / table / infobox / knowledge graph). Through a three-stage experimental framework, it reveals the existence of such bias, its data-level driving factors, and its internal causes at the attention mechanism level, and validates the effectiveness of attention rebalancing as an intervention.

## Background & Motivation

In real-world applications, LLMs must process external knowledge from multiple formats — unstructured text, semi-structured infoboxes, structured tables, and knowledge graphs (KGs). **These heterogeneous formats are complementary to each other**, and whether they can be fairly integrated is critical for knowledge-intensive applications.

However, when faced with heterogeneous-format inputs, LLMs may exhibit systematic preferences for certain formats, leading to:
- Neglect of critical information in non-preferred formats
- Increased reasoning errors and downstream task risks
- For example, over-relying on textual descriptions in clinical decision-making while ignoring anomalous indicators in tables

Prior work has explored multimodal bias, entity popularity bias, and temporal bias in LLMs, but **format itself as a source of bias has never been systematically studied**.

Three core questions:
1. Does format bias exist systematically?
2. What data-level factors drive the bias?
3. What internal mechanism in LLMs produces the bias?

## Method

### Overall Architecture

A three-stage progressive experimental study:
- **Stage 1**: Existence and directionality of bias (10 LLMs × 6 format pairs = 60 experimental groups)
- **Stage 2**: Data-level factor analysis (information richness / structural quality / format type)
- **Stage 3**: Attention mechanism analysis + lightweight intervention experiments

### Key Design 1: Conflict Scenario Construction and Evaluation Protocol

4,000 factual claims are sampled from ConflictBank, each paired with 3 counterfactuals, yielding 12,000 conflicting pairs. Evidence from both sides is randomly converted into different formats (text / table / infobox / KG) using GPT-4o-mini.

Two confounding factors are controlled:
- **Internal knowledge filtering**: Each claim is tested 16 times in a zero-shot setting; only samples the model cannot answer at all are retained.
- **Evidence order randomization**: Eliminates input-order bias.

Two core metrics:
- **DCR (Dual Coverage Rate)** = Both / (Pref-A + Pref-B + Both) — measures whether bias exists
- **FPR (Format Preference Rate)** = Pref-A / (Pref-A + Pref-B) — measures the direction of bias

### Key Design 2: Three-Factor Analysis at the Data Level

**Factor 1: Information Richness**
- Homogeneous setting: Comparing variants with 4 / 8 / 12 entries within the same format; LLMs consistently prefer variants with more entries.
- Heterogeneous setting: Structured data vs. text; LLMs' preference for structured formats increases as the number of entries grows.
- Conclusion: LLMs equate "more information" with "higher evidential value."

**Factor 2: Structural Quality**
- Controlled corruption is introduced to format-defining symbols (brackets, colons, delimiters) at corruption probabilities of 0.45 / 0.9.
- Homogeneous setting: LLMs consistently prefer intact formats, and this preference saturates after moderate corruption.
- Heterogeneous setting: LLMs' preference for structured formats drops sharply as corruption increases.
- Conclusion: Structural integrity acts as a "credibility signal."

**Factor 3: Format Type**
- Content is fixed (same number of entries); only the presentation format varies.
- Resulting preference hierarchy: Table > KG > Infobox (relative to Text).

| Metric | Infobox vs Text | Table vs Text | KG vs Text |
|--------|-----------------|---------------|------------|
| FPR | 0.235 | 0.398 | 0.336 |

### Key Design 3: Attention Mechanism Analysis and Intervention

Three models are analyzed: Qwen3-8B, Mistral-7B, and LLaMA-3.1-8B.

**Attention vs. Bias Existence**: The attention gap is negatively correlated with DCR (Spearman $\rho$ = -0.31 / -0.37 / -0.54); the more imbalanced the attention, the harder it is for the model to recognize information from both sides simultaneously.

**Attention vs. Bias Direction**: In 82.35% of one-sided responses, the model prefers the side receiving *less* attention — higher attention does not imply selection as the answer.

**Attention Rebalancing Intervention**: The total attention allocated to each evidence segment is normalized:

$$a'_j = \frac{\bar{m}}{m_k + \varepsilon} \cdot a_j, \quad j \in I_k$$

### Loss & Training

This paper is an empirical analysis study and does not involve the design of training loss functions. The intervention method directly modifies the attention distribution at inference time.

## Experiments

### Main Results: Existence of Format Bias (FPR Heatmap)

| Format Pair (A vs B) | Preferred Direction | FPR Range | Statistically Significant |
|----------------------|---------------------|-----------|--------------------------|
| Text vs Table | Prefers Text | 0.55–0.75 | Majority * |
| Text vs Infobox | Prefers Text | 0.65–0.85 | All * |
| Text vs KG | Prefers Text | 0.45–0.65 | Partial * |
| KG vs Table | Prefers KG | 0.55–0.80 | Majority * |
| KG vs Infobox | Prefers KG | 0.65–0.85 | All * |
| Table vs Infobox | Prefers Table | 0.50–0.65 | Partial * |

Preference hierarchy: **Text ≈ KG > Table > Infobox**, consistent across 10 LLMs and 6 model families.

### Ablation Study

| Condition | DCR Change | FPR Change |
|-----------|------------|------------|
| Heterogeneous input (baseline) | 3–24% | Significantly biased |
| Homogeneous input (text–text) | **28–53%** ↑↑ | Approaches 0.5 |
| Attention rebalancing (heterogeneous) | **Significant improvement** | **No significant change** |

### Key Findings

1. **Format bias exists systematically**: DCR is as low as 3–24% (heterogeneous) vs. 28–53% (homogeneous); format heterogeneity alone can independently impair multi-source information integration.
2. **Bias does not improve with model scale**: Within the Qwen3 series, DCR shows no clear improvement from 8B to 32B.
3. **All three factors significantly influence bias**: Higher information richness leads to greater preference, greater structural integrity leads to greater trust, and format type exhibits an inherent preference hierarchy.
4. **Bias existence is amenable to intervention; bias direction is not**: Attention rebalancing effectively improves DCR but does not alter FPR, suggesting that directional preferences may be rooted in pretraining.
5. **Downstream task impact**: Transitioning from homogeneous to heterogeneous inputs reduces accuracy on HotpotQA by 9% and on MuSiQue by 12%; attention rebalancing recovers 6.5% and 9.5%, respectively.

## Highlights & Insights

- **First systematic study of format bias** — a novel and important contribution filling a significant gap in LLM bias research.
- Large-scale and rigorous experiments: 10 LLMs × 6 model families × 6 format pairs + three progressive stages.
- **Strict confounding factor control**: Internal knowledge filtering (16 zero-shot tests) + order randomization.
- A complete research loop from "existence → factors → mechanism → intervention."
- Three actionable mitigation directions proposed: data preprocessing, inference-time intervention, and format-balanced training.

## Limitations & Future Work

- Format conversion relies on GPT-4o-mini, and conversion quality may introduce additional bias.
- Only 4 formats are considered (text / table / infobox / KG); formats such as JSON, XML, and code are not covered.
- The attention intervention is validated only on ~8B-scale models; its effectiveness on larger models (70B+) is unknown.
- Conflict scenario construction relies on factual claims from ConflictBank; generalization to open-domain reasoning requires further validation.
- The root cause of directional bias (pretraining data distribution? tokenizer?) is not subjected to in-depth causal analysis.

## Related Work & Insights

- **Heterogeneous reasoning benchmarks**: COMPMIX (Christmann et al. 2024) requires cross-format reasoning; CompMix-IR provides a unified retrieval framework.
- **LLM knowledge conflicts**: ConflictBank (Su et al. 2024) evaluates how LLMs handle conflicting information; Jin et al. (2024) study the tension between parametric knowledge and contextual evidence.
- **Multimodal bias**: Zhu et al. (2024) investigate cross-modal knowledge conflicts in vision-language models; Zhang et al. (2025) evaluate and steer multimodal preferences.
- **Bias mitigation**: Conflict-aware decoding (Yuan et al. 2024), attention pruning (Jin et al. 2024b), neuron reweighting (Shi et al. 2024).

## Rating

⭐⭐⭐⭐ (4/5)

The problem formulation is novel and important, the experimental design is systematic and comprehensive, and confounding factors are rigorously controlled. The three-stage progressive analysis is logically coherent, forming a complete loop from phenomena to factors to mechanisms to interventions. Points are deducted primarily for the relative simplicity of the attention intervention and the lack of in-depth analysis of the root causes of directional bias. For the graph learning community, this paper reveals that structured knowledge such as KGs may be systematically undervalued in LLM interactions, carrying important practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Self-Correction Distillation for Structured Data Question Answering](self-correction_distillation_for_structured_data_question_answering.md)
- [\[AAAI 2026\] Spiking Heterogeneous Graph Attention Networks](spiking_heterogeneous_graph_attention_networks.md)
- [\[AAAI 2026\] S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)
- [\[AAAI 2026\] EchoLess: Label-Based Pre-Computation for Memory-Efficient Heterogeneous Graph Learning](echoless_label-based_pre-computation_for_memory-efficient_heterogeneous_graph_le.md)
- [\[AAAI 2026\] MUG: Meta-path-aware Universal Heterogeneous Graph Pre-Training](mug_meta-path-aware_universal_heterogeneous_graph_pre-training.md)

</div>

<!-- RELATED:END -->
