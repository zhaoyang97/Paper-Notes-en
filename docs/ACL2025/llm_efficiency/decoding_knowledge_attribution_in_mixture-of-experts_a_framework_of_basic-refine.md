---
title: >-
  [Paper Note] Decoding Knowledge Attribution in Mixture-of-Experts: A Framework of Basic-Refinement Collaboration and Efficiency Analysis
description: >-
  [ACL2025][LLM Efficiency][Mixture-of-Experts] Proposes a cross-layer knowledge attribution algorithm to systematically analyze the "basic-refinement" collaboration framework of shared experts and routed experts in MoE models, revealing that MoEs achieve 31% higher layer-wise efficiency compared to dense models, and validating the decisive impact of architectural depth on robustness through a semantic-driven routing mechanism (attention head-expert correlation $r=0.68$) and ex…
tags:
  - "ACL2025"
  - "LLM Efficiency"
  - "Mixture-of-Experts"
  - "Knowledge Attribution"
  - "Interpretability"
  - "Expert Collaboration"
  - "Sparse Routing"
date: 2026-05-08
content_hash: e52f2b507db9ae6f
---

# Decoding Knowledge Attribution in Mixture-of-Experts: A Framework of Basic-Refinement Collaboration and Efficiency Analysis

**Conference**: ACL2025  
**arXiv**: [2505.24593](https://arxiv.org/abs/2505.24593)  
**Authors**: Junzhuo Li, Bo Wang, Xiuze Zhou, Peijie Jiang, Jia Liu, Xuming Hu (HKUST(GZ), Ant Group)
**Area**: Others  
**Keywords**: Mixture-of-Experts, Knowledge Attribution, Interpretability, Expert Collaboration, Sparse Routing

## TL;DR

Proposes a cross-layer knowledge attribution algorithm to systematically analyze the "basic-refinement" collaboration framework of shared experts and routed experts in MoE models, revealing that MoEs achieve 31% higher layer-wise efficiency compared to dense models, and validating the decisive impact of architectural depth on robustness through a semantic-driven routing mechanism (attention head-expert correlation $r=0.68$) and expert blocking experiments.

## Background & Motivation

Mixture-of-Experts (MoE) reduces computational overhead by sparsely activating a subset of experts, but its interpretability—especially how experts collaborate to process and refine knowledge in heterogeneous designs (such as shared expert modules)—remains an unsolved puzzle.

**Limitations of Prior Work:**
- Existing knowledge attribution methods (e.g., Knowledge Neurons, Transformer FFN analysis) are designed for dense models and cannot capture the dynamic routing-expert interactions in MoEs.
- There is a lack of comparative studies on heterogeneous MoE architectures, and empirical validation is missing for functional hypotheses of shared experts (whether they act as "general feature extractors" or "redundant backups").
- The cross-layer expert collaboration mechanism is opaque, hindering the systematic optimization of MoE models.

**Core Problem:** How do experts in MoE models collaboratively process and refine knowledge? What roles do shared experts and routed experts play, respectively? How does architectural depth affect robustness?

## Method

### Cross-Layer Knowledge Attribution Algorithm

Extends neuron-level attribution methods of dense models to MoEs, enabling simultaneous analysis of macro-architectural behaviors and micro-expert contributions.

**Importance Score of MoE Expert Neurons:**

For neuron $\mathbf{v}^l_{\mathcal{E}_j}$ of expert $\mathcal{E}_j$ in the $l$-th layer, the importance score is defined as the gain in the log-probability of target token prediction after adding this neuron. Here, $g^l_{i,j}$ represents the gating probability, and $\mathbf{u}^l$ is the intermediate representation after the attention output. This score measures the impact of each expert neuron on the final prediction, representing the gain introduced by this module during inference.

### Experimental Design

**MoE Models:** Qwen 1.5-MoE (24 layers, 64 experts + 4 shared), OLMoE (16 layers, 64 experts), Mixtral-8x7B (32 layers, 8 experts)
**Dense Baseline Models:** Qwen 1.5-7B (32 layers), Llama-7B (32 layers), Mistral-7B (32 layers)
**Evaluation Metrics:** HIT@10, MRR (Knowledge Prediction Task), Layer-wise Efficiency (FFN Gain / Number of Layers)

### "Basic-Refinement" Collaboration Framework

The division of labor between shared experts and routed experts is validated through ablation studies:

1. **Shared Experts = General Basic Processors**: Responsible for cross-domain basic tasks such as entity recognition and syntactic parsing.
2. **Routed Experts = Domain Refiners**: After shared experts provide basic representations, routed experts perform domain-specific attribute associations (e.g., mapping "Canada" to "Ottawa").
3. **Semantic-Driven Routing**: A strong temporal correlation exists between attention heads and expert selection (r=0.68, p<0.001), where attention heads actively guide expert selection.

### Three-Stage Processing Pattern (Taking Qwen 1.5-MoE as an Example)

- **Early Stage (Layers 1-13)**: Experts are initialized in parallel to extract basic features, contributing 6.1% of the total gain.
- **Middle Stage (Layers 14-19)**: Dynamic routing activates expert selection (top-4 gating), contributing 43.5% of the total gain.
- **Late Stage (Layers 20-24)**: Shared and routed experts collaborate on refinement, contributing 50.4% of the total gain.

### Causal Validation

Three intervention experiments confirm that attention heads causally drive expert selection:
- Inhibiting key attention heads $\rightarrow$ top-4 expert gating probability decreases by 54%, MRR decreases by 29%
- Forcing the activation of correct experts $\rightarrow$ recalls 85% of the original MRR
- Integrated gradient path analysis $\rightarrow$ 28% of expert activation attribution can be directly traced back to specific attention heads

## Key Experimental Results

### Table 1: Efficiency Comparison between Dense and MoE Models

| Model | HIT@10 | MRR | FFN Gain | Attn Gain | Peak Gain Position | Layer-wise Efficiency |
|------|--------|-----|----------|-----------|---------------|---------|
| Llama-7B | 0.90 | 0.70 | 5.16 | 4.03 | 77.6% | 0.161 |
| Qwen 1.5-7B | 0.79 | 0.62 | 6.49 | 4.14 | 90.2% | 0.203 |
| Mistral-7B | 0.88 | 0.71 | 6.74 | 2.96 | 83.2% | 0.211 |
| **Qwen 1.5-MoE** | 0.85 | 0.63 | **7.36** | 3.18 | 84.8% | **0.307** |
| **OLMoE** | 0.83 | 0.64 | 4.98 | 4.40 | 84.6% | **0.311** |
| **Mixtral-8x7B** | 0.90 | 0.73 | 6.79 | 3.03 | 83.3% | 0.212 |

**Findings:** Layer-wise efficiency of MoE models is significantly superior to dense models. Qwen 1.5-MoE achieves an efficiency of 0.307 with 24 layers, which is **51%** higher than the 32-layer Qwen 1.5-7B (0.203). OLMoE reaches the highest efficiency of 0.311 with only 16 layers. FFN Gain dominates in MoE models, indicating that expert networks are the core of knowledge processing.

### Table 5: Expert Blocking Experiment—Robustness Comparison (MRR)

| Task | Model | Original | Block Top1 | Block Top5 | Block Top10 |
|------|------|------|---------|---------|----------|
| name_birthplace | Qwen 1.5-MoE | 0.85 | 0.84 | 0.83 | 0.81 |
| name_birthplace | OLMoE | 0.82 | 0.80 | 0.80 | 0.60 |
| country_capital | Qwen 1.5-MoE | 0.71 | 0.68 | 0.68 | **0.40** |
| country_capital | OLMoE | 1.00 | 1.00 | 0.76 | 0.76 |
| country_language | Qwen 1.5-MoE | 0.94 | 0.92 | 0.92 | 0.92 |
| country_language | OLMoE | 0.96 | 0.92 | 0.87 | **0.68** |
| fruit_inside_color | Qwen 1.5-MoE | 0.74 | 0.68 | 0.63 | 0.60 |
| fruit_inside_color | OLMoE | 0.76 | 0.60 | 0.52 | **0.41** |
| object_superclass | Qwen 1.5-MoE | 0.83 | 0.82 | 0.80 | 0.79 |
| object_superclass | OLMoE | 0.80 | 0.75 | 0.70 | **0.66** |

**Findings:** The deep Qwen 1.5-MoE exhibits strong robustness on general tasks (blocking Top10 on name_birthplace only drops by 4.7%), but drops by 43% on the core sensitive task (country_capital). The shallow OLMoE degrades severely on most tasks: fruit_inside_color drops by 46%, and country_language drops by 29%. This validates the critical role of architectural depth and shared experts in redundant design.

### Table 2: Ablation Study of Qwen 1.5-MoE

| Configuration | HIT@10 | MRR |
|------|--------|-----|
| Top-1 Routed Expert Only | 0 | 0 |
| Top-2 Routed Expert Only | 0 | 0 |
| Shared Expert Only | 0.03 | 0.01 |
| Shared + Top-1 | 0.82 | 0.59 |
| Shared + Top-2 | 0.83 | 0.63 |
| Shared + Top-4 (Default) | 0.85 | 0.63 |

**Findings:** Activating any type of expert in isolation leads to severe degradation, proving that complex knowledge tasks must rely on the collaboration between shared and routed experts. An "effective expert threshold" exists—core knowledge is primarily captured by a small number of experts, and adding more experts yields diminishing returns.

## Highlights & Insights

- **First Systematic Framework for MoE Interpretability**: Proposes a cross-layer attribution algorithm, filling the gap in interpretability for sparse MoE architectures by enabling concurrent analysis of macro-architectural behaviors and micro-expert contributions.
- **"Basic-Refinement" Collaboration Paradigm**: Rigorously validates the division of labor—where shared experts serve as general processors and routed experts serve as domain refiners—through ablation studies, and confirms that attention heads drive expert selection via causal interventions.
- **Task Sensitivity Insights**: Distinguishes between "core sensitive tasks" (e.g., geographic reasoning, requiring concentrated expertise) and "distribution-tolerant tasks" (e.g., object attributes, utilizing broad participation), providing quantitative guidance for MoE design.
- **Actionable Design Principles**: Deep MoEs should deploy shared experts in early layers to guarantee redundancy and allocate routed experts in late layers for refinement. Shallow MoEs need to balance expert versatility and routing adaptability.

## Limitations & Future Work

- The analysis is limited to static-routing MoEs at the 7B parameter scale; whether larger-scale (100B+) or fully dynamic-routing MoEs exhibit the same specialization patterns remains unverified.
- Retrieval-augmented MoEs (e.g., Monet) are not covered, as the confounding factors introduced by external knowledge coupling lie beyond the scope of this analysis.
- The knowledge attribution method relies on changes in log-probabilities, and its explanatory power for complex scenarios like multi-step reasoning or compositional reasoning remains to be explored.
- The experimental dataset primarily consists of factual knowledge prediction; generalization to tasks requiring higher-level abstract reasoning (e.g., mathematical reasoning, code generation) has not been fully evaluated.

## Related Work & Insights

- **MoE Architecture**: From the classic framework of Jacobs et al. (1991) to Switch Transformer (Fedus et al., 2021), DeepSeekMoE (Dai et al., 2024), and Qwen-MoE (Team, 2024), MoEs balance efficiency and performance through sparse routing, yet interpretability research lags behind.
- **Knowledge Attribution**: Transformer FFN layers acting as key-value memories (Geva et al., 2021), knowledge neurons (Dai et al., 2022), knowledge circuits (Yao et al., 2024), and neuron-level attribution (Yu & Ananiadou, 2024) $\rightarrow$ these are all tailored for dense models and cannot handle dynamic routing and expert collaboration in MoEs.
- **Our Positioning**: For the first time, systematically extends knowledge attribution from dense models to heterogeneous MoE architectures, filling the gap in interpretability for sparse-routing models.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first systematic MoE knowledge attribution framework; the "basic-refinement" collaboration paradigm and semantic-driven routing mechanism are meaningful findings.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comparison across 3 MoE + 3 dense models, with ablation studies + causal interventions + expert blocking, providing thorough multi-angle validation.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and well-layered analysis, though heavily relying on mathematical notations; some experimental details require referencing the appendix.
- Value: ⭐⭐⭐⭐ — Provides actionable principles (depth-redundancy-task sensitivity) for MoE architecture design, offering valuable insights for understanding and optimizing large-scale MoEs like DeepSeek-V3.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Mixture of Lookup Experts](../../ICML2025/llm_efficiency/mixture_of_lookup_experts.md)
- [\[ACL 2025\] DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts](dive_moe_reconstruction.md)
- [\[ICML 2026\] Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](../../ICML2026/llm_efficiency/beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)
- [\[NeurIPS 2025\] On the Expressive Power of Mixture-of-Experts for Structured Complex Tasks](../../NeurIPS2025/llm_efficiency/on_the_expressive_power_of_mixture-of-experts_for_structured_complex_tasks.md)
- [\[ICLR 2026\] DirMoE: Dirichlet-Routed Mixture of Experts](../../ICLR2026/llm_efficiency/dirmoe_dirichlet-routed_mixture_of_experts.md)

</div>

<!-- RELATED:END -->
