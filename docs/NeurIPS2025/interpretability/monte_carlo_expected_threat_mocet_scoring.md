---
title: >-
  [Paper Note] Monte Carlo Expected Threat (MOCET) Scoring
description: >-
  [NeurIPS 2025][Interpretability][AI safety] This paper proposes the MOCET (Monte Carlo Expected Threat) scoring framework, which decomposes LLM-generated bioweapon synthesis protocols into sequential Bernoulli trials…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "AI safety"
  - "biosecurity"
  - "LLM risk assessment"
  - "Monte Carlo simulation"
  - "threat scoring"
  - "k-NN"
  - "ASL"
date: 2026-05-08
content_hash: 59d71116c9199355
---

# Monte Carlo Expected Threat (MOCET) Scoring

**Conference**: NeurIPS 2025
**arXiv**: [2511.16823](https://arxiv.org/abs/2511.16823)  
**Code**: available upon request  
**Area**: Interpretability
**Keywords**: AI safety, biosecurity, LLM risk assessment, Monte Carlo simulation, threat scoring, k-NN, ASL

## TL;DR

This paper proposes the MOCET (Monte Carlo Expected Threat) scoring framework, which decomposes LLM-generated bioweapon synthesis protocols into sequential Bernoulli trials, combines k-NN semantic embedding-based success probability estimation with Monte Carlo simulation, and produces interpretable, automatable threat quantification metrics for measuring the real-world risk of LLMs in the biosecurity domain.

## Background & Motivation

As LLM capabilities advance rapidly, their potential for misuse in the biosecurity domain has attracted increasing concern:

**Erosion of knowledge barriers**: Raw materials for synthesizing biochemical agents such as Ricin and Sarin are relatively accessible; historically, the primary barrier preventing malicious actors has been the difficulty of obtaining technical knowledge. LLMs may significantly lower this barrier.

**Inadequacy of existing evaluations**: Benchmarks such as LAB-Bench, BioLP-bench, and WMDP can assess domain knowledge but lack metrics that connect model capabilities to real-world risk.

**Shifting regulatory environment**: Recent federal deregulation of AI oversight in the United States, combined with the widespread proliferation of open-source models, creates an urgent need for quantifiable risk measurement tools.

**Scalability requirements**: Metrics must be both automatable and open-ended to keep pace with the rapid iteration of LLMs.

The paper's threat model focuses on **non-state actors** leveraging LLMs for bioweapon development, with particular emphasis on the "Build" phase — the critical bottleneck between research knowledge and actual production.

## Method

### Overall Architecture

The MOCET framework decomposes LLM-generated protocols into a sequence of steps, treating each as a Bernoulli trial, and computes expected threat via Monte Carlo simulation. The overall pipeline is: LLM generates protocol → step decomposition → k-NN probability estimation → Monte Carlo simulation → MOCET / Cumulative MOCET score.

### Key Designs

**Step-level probability modeling**: For an $n$-step protocol, the success indicator variable for each step is $X_i \sim \text{Bernoulli}(p_i)$, and the overall success probability is:

$$E[Y] = \prod_{i=1}^{n} E[X_i] = \prod_{j=1}^{m} p_j^{n_j}$$

where steps are grouped into $m$ categories, each with $n_j$ steps and success rate $p_j$.

**MOCET score** (expected threat per incident): Computed over $N$ Monte Carlo trials, weighted by a harm function $W$ calibrated from historical casualty data:

$$\text{MOCET} = \frac{1}{N} \sum_{i=1}^{N} W(Y_i) E[Y_i]$$

**Cumulative MOCET score** (annualized population-level expected threat):

$$\text{Cumulative MOCET} = \text{Rate of Occurrence} \times \text{MOCET}$$

The rate of occurrence is approximated using FBI mass murder data (30 incidents in 2017).

**k-NN probability estimation**: The core challenge is accurately estimating the per-step success probability $p_i$. The framework uses all-mpnet-base-v2 to generate semantic embeddings $\vec{v}_i \in \mathbb{R}^d$ for step descriptions, and retrieves the $k$ most semantically similar steps from a historical dataset via k-nearest neighbors:

$$p_i \approx \frac{1}{k} \sum_{j \in \mathcal{N}_i} X_j$$

where $\mathcal{N}_i$ denotes the $k$ historical steps most semantically similar to step $i$.

**Error analysis**: A Taylor expansion demonstrates that when per-step probability deviation satisfies $\|\alpha\| / p \sim 10\%$, the resulting error in $E[Y]$ and the MOCET score is only approximately $\sim 1\%$, establishing the robustness of the framework.

### Loss & Training

MOCET itself involves no training loss. The k-NN model employs pretrained Sentence-Transformers embeddings without additional fine-tuning. During validation on MMLU, GPQA, and WMDP, k-NN prediction accuracy is shown to significantly discriminate between correct and incorrect answers ($p \ll 0.01$, $k = 10, 20, 40$), confirming the reliability of the probability estimation approach.

## Key Experimental Results

### Main Results

**Historical bioweapon incident statistics** (used to calibrate the harm function $W$):

| Agent | Major Incidents since 1975 | Total Deaths | Total Injuries | Avg. Casualties/Incident |
|-------|--------------------------|-------------|---------------|--------------------------|
| Anthrax | 6 | 81+ | 217+ | 49.6+ |
| Ricin | 20+ | 6 | 5 | 0.55 |
| Sarin Gas | 5 | 1875+ | 9700+ | 2315 |

**Case study results** (Dolphin-2.9-Llama3-8B, a jailbroken open-source model):

| Agent | $E[Y]$ (Model) | $E[Y]$ (Human) | MOCET | Cumulative MOCET |
|-------|--------------|--------------|-------|-----------------|
| Sarin | 0.82% | 0.5% | 18.94 | 568.17 |
| Anthrax | 1.18% | 16.5% | 0.58 | 17.50 |

### Ablation Study

**Standard benchmarks vs. safety evaluation comparison**:

| Benchmark | Llama-3-8B-Instruct | Dolphin-2.9-Llama3-8B |
|-----------|--------------------|-----------------------|
| MMLU | 63.77% | 57.15% |
| WMDP-Bio | 71.01% | 65.99% |
| WMDP-Chem | 47.06% | 46.32% |
| GPQA | 29.46% | 27.46% |

The Dolphin model exhibits a slight performance decline on standard benchmarks, superficially suggesting improved safety. However, MOCET analysis reveals that the jailbroken model carries a non-zero real-world threat risk — a finding that standard benchmarks fail to capture for catastrophic risks.

**k-NN validation**: $k = 10, 20, 40$ all yield statistically significant results; k-NN prediction accuracy for correct answers is significantly higher than for incorrect answers ($p \ll 0.01$), validating the reliability of the probability estimation method.

### Key Findings

1. Open-source jailbroken LLMs can produce bioweapon synthesis guidance with non-zero MOCET scores, demonstrating that LLMs do lower the knowledge barrier for malicious actors.
2. MOCET provides metrics analogous to public safety statistics: per-incident MOCET is comparable to shooting incidents at 18.86 casualties/event; Cumulative MOCET is comparable to motor vehicle fatalities at 44,534/year.
3. Discrepancies exist between model estimates and human expert assessments (the model is conservative for Anthrax and slightly optimistic for Sarin), indicating that automated evaluation should complement rather than replace expert review.
4. Standard benchmarks (MMLU, WMDP) do not reflect the true safety risk of models.

## Highlights & Insights

1. **Strong interpretability**: MOCET scores correspond directly to expected casualties, making them accessible to policymakers and non-technical stakeholders.
2. **Dual scalability**: The framework is both automatable and open-ended, extending naturally to new threat categories without dependence on fixed benchmarks.
3. **Alignment with policy frameworks**: Compatible with established frameworks including the OpenAI Preparedness Framework, Anthropic RSP, and NIST AI RMF.
4. **Cross-domain methodology**: The combination of k-NN and Monte Carlo is generalizable and can be extended to other safety domains.

## Limitations & Future Work

1. **Assumption constraints**: The framework assumes actors cannot fact-check, nor employ best-of-n or multi-turn prompting; in practice, adversaries may be considerably more strategic.
2. **Data dependency**: The accuracy of the harm function and step-level probabilities depends on historical data, with limited magnitude estimation.
3. **Single-model evaluation**: Validation is conducted on only one open-source model; closed-source models such as GPT-4 and Claude are not evaluated.
4. **Biosecurity focus**: The framework is currently restricted to biosecurity and has not been extended to chemical, radiological, cyber, or other threat domains.
5. **Correctness ≠ risk**: The framework assumes informational correctness equates to risk, without accounting for partially correct but still dangerous instructions.

## Related Work & Insights

- **Anthropic RSP / OpenAI Preparedness Framework**: MOCET provides a quantifiable risk metric that complements these existing frameworks.
- **WMDP**: Assesses domain knowledge but not real-world risk; MOCET fills this gap.
- **LLM-as-Judge**: MOCET extends the LLM evaluation paradigm by shifting the focus from performance to safety risk quantification.
- **Insights**: Analogous probabilistic cascade modeling combined with Monte Carlo methods could be applied to assess other threat categories, including AI-assisted cyberattacks and chemical weapon synthesis.

## Rating

- ⭐⭐⭐⭐ **Novelty**: Combining Monte Carlo simulation with k-NN probability estimation for threat quantification is novel within the AI safety literature.
- ⭐⭐⭐ **Experimental Thoroughness**: The case study covers only one model and two biological agents; experimental scale is insufficient.
- ⭐⭐⭐⭐ **Value**: Provides an interpretable and scalable quantitative tool for AI safety evaluation with significant policy implications.
- ⭐⭐⭐ **Methodological Depth**: The probabilistic modeling is relatively straightforward; the accuracy of k-NN estimation requires more rigorous validation.

**Overall**: ⭐⭐⭐⭐ (3.5/5) — The paper presents a valuable framework for risk quantification in AI safety, with interpretability and policy alignment as its key strengths. However, the limited experimental scale and strong assumptions necessitate larger-scale validation before its effectiveness can be fully established.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SSA: Improving Performance With a Better Scoring Function](../../ACL2026/interpretability/ssa_improving_performance_with_a_better_scoring_function.md)
- [\[NeurIPS 2025\] Why Is Attention Sparse in Particle Transformer?](why_is_attention_sparse_in_particle_transformer.md)
- [\[NeurIPS 2025\] Cognitive Mirrors: Exploring the Diverse Functional Roles of Attention Heads in LLM Reasoning](cognitive_mirrors_exploring_the_diverse_functional_roles_of_attention_heads_in_l.md)
- [\[NeurIPS 2025\] OrdShap: Feature Position Importance for Sequential Black-Box Models](ordshap_feature_position_importance_for_sequential_black-box_models.md)
- [\[NeurIPS 2025\] Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders](transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)

</div>

<!-- RELATED:END -->
