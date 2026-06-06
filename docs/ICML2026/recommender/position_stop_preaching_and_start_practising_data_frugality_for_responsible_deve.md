---
title: >-
  [Paper Note] Position: Stop Preaching and Start Practising Data Frugality for Responsible Development of AI
description: >-
  [ICML 2026][Recommender Systems][data frugality] This position paper points out a "value-action gap" in the ML community regarding "data frugality." While researchers acknowledge that coresets save energy…
tags:
  - "ICML 2026"
  - "Recommender Systems"
  - "data frugality"
  - "coreset"
  - "carbon emissions"
  - "responsible AI"
  - "subset selection"
date: 2026-05-08
content_hash: 059bfc2a3a473ba9
---

# Position: Stop Preaching and Start Practising Data Frugality for Responsible Development of AI

**Conference**: ICML 2026  
**arXiv**: [2602.19789](https://arxiv.org/abs/2602.19789)  
**Code**: https://github.com/saintslab/data-frugality  
**Area**: AI Safety / Sustainable AI / Data-efficient  
**Keywords**: data frugality, coreset, carbon emissions, responsible AI, subset selection

## TL;DR
This position paper points out a "value-action gap" in the ML community regarding "data frugality." While researchers acknowledge that coresets save energy, few actually report energy consumption or carbon emissions. Using ImageNet-1K as a case study, the authors calculate a conservative lower bound of ~5.82 GWh / 2589 tCO2e for downstream training and storage, calling for data frugality to transition from a slogan to a measurable, actionable, and rewardable engineering practice.

## Background & Motivation

**Background**: Current mainstream AI training follows the scaling laws of Kaplan and Hoffmann, equating "larger datasets" with "better models," while benchmarks and leaderboards reward experiments on massive corpora. Simultaneously, a parallel movement advocates for "scaling down" through data subset methods like coreset selection and dataset condensation.

**Limitations of Prior Work**: The authors conducted a systematic survey of 10 representative coreset papers. They found that 8/10 cited "computational efficiency" as a motivation and 3/10 mentioned "energy efficiency," but **only 1/10 actually reported energy savings**, while 6/10 reported time savings. Publicity for data frugality far outpaces practice.

**Key Challenge**: Carbon emissions do not scale linearly with compute or data size—a 25% data pruning achieves 24%–40% time savings and 24%–33% energy savings across different architectures. Without measuring actual energy and carbon, "energy saving" claims remain rhetorical and useless for responsible AI decision-making. Furthermore, carbon estimates depend heavily on grid carbon intensity, which varies by over 60x globally (e.g., Turkmenistan vs. Lesotho).

**Goal**: (i) Quantify the downstream training and storage carbon costs for "national-level" datasets like ImageNet-1K to provide a citable lower bound; (ii) Empirically demonstrate that coresets save power without compromising accuracy; (iii) Translate these needs into concrete recommendations across three layers: People, Platforms, and Policies.

**Key Insight**: By combining the SAINTS Lab's Carbontracker measurements, OpenReview metadata, and Hugging Face download statistics, the authors construct a "conservative lower bound." They aim to show that even under the most minimal assumptions, the "opportunity cost" of ignoring data frugality is staggering.

**Core Idea**: Transition data frugality from an activist issue of a "value-action gap" into a reporting and peer-review norm aligned with the ML community—if a method claims energy efficiency, it **must** measure energy.

## Method

As a position paper, the "method" consists of three supporting arguments: carbon cost accounting, empirical coreset evidence, and an actionable framework.

### Overall Architecture

The paper is organized through a lifecycle perspective: the data lifecycle (collection, cleaning, storage, distribution) and the model lifecycle (training, selection, deployment) run in parallel, with coreset/subset selection impacting multiple stages. Based on this:

1.  **Quantifying the Baseline**: Using OpenReview metadata (ICLR 2017–2022) to estimate the number of papers training ImageNet-1K from scratch, then extrapolating to the full ML literature using dimensions.ai keyword indexing and projecting to 2023–2025.
2.  **Empirical Gains**: Measuring energy and time for full vs. 25% pruned training on A100 GPUs using ResNet-34 / ResNet-50 / Swin-T with Carbontracker. Referencing SOTA subset selection curves (Dyn-Unc, InfoMax) to show no accuracy loss.
3.  **Fairness Side Effects**: Comparing random, reweighted, and balanced sampling on Coloured-MNIST (99% majority bias) to prove coresets can simultaneously assist in debiasing.
4.  **Action Framework**: Consolidating evidence into appeals for People, Platforms, and Policies.

### Key Designs

1.  **Reproducible Lower Bound Estimation of ImageNet-1K Downstream Carbon Costs**:
    - **Function**: Converts "environmental cost of datasets" from an abstract slogan into a citable figure—5.82 GWh energy and 2589 tCO2e (global average).
    - **Mechanism**: Training cost = estimated training count $N \approx 46{,}179 \pm 1{,}154$ × ResNet-50 single epoch energy $\approx 0.394$ kWh × 300 epochs, resulting in $5.46 \pm 0.14$ GWh. Storage cost = $N \times 130$ GB × 60 kWh/TB/yr $\approx 360 \pm 9$ MWh. $N$ is derived from the proportion of "train from scratch" papers in OpenReview, extrapolated via indexing.
    - **Design Motivation**: The authors deliberately used public metadata to avoid dependence on proprietary vendor data. They acknowledge this as a "lower bound"—Hugging Face Hub counts 214 ImageNet-1K derivatives with over 2.5 million downloads (55x the paper count), suggesting the real cost is much higher.

2.  **Empirical Dual-Axis Core-set Gains: Accuracy Curves × Energy Tables**:
    - **Function**: Answers "how much is saved vs. how much is lost" using accuracy-pruning ratio curves and energy tables.
    - **Mechanism**: Accuracy is demonstrated using SOTA curves from Dyn-Unc (Swin-T) and InfoMax (ResNet-34), showing 25%–35% of ImageNet-1K can be pruned without Top-1 drop. Energy is measured for ResNet-34/50 and Swin-T over 10 epochs (3 runs on A100), recording both CPU and GPU usage via Carbontracker. Results show 25% pruning yields savings of 32% time / 29% energy for ResNet-34, 40% / 33% for ResNet-50, and 24% / 24% for Swin-T.
    - **Design Motivation**: It highlights that 25% data $\neq$ 25% energy, debunking the common assumption of data size as a perfect proxy for energy. It also honestly includes the one-time cost of coreset construction, noting it amortizes within 3–4 training runs.

3.  **The Framework: Actionable Proposals for People, Platforms, and Policies**:
    - **Function**: Translates ethical appeals into specific actions for conference calls, leaderboard rules, and funding terms.
    - **Mechanism**: (People) Use "Data-Pareto" plots (Accuracy vs. Data Volume) instead of single accuracy points; measure what you motivate. (Platforms) Mandate compute reporting (e.g., CVPR 2026) and introduce data-efficient challenges like BabyLM. (Policies) Standardize reporting, promote shared data infrastructure (e.g., Berzelius) to reduce local redundancy, and propose "data sunset laws" requiring approval for large-scale data use.
    - **Design Motivation**: The authors argue that the value-action gap persists because "thinking is easier than doing"; frugality must move from "personal virtue" to an "institutional default."

### Loss & Training
This is a position paper; no main model training loss is proposed. The measurement stack uses Carbontracker and CodeCarbon for training, and a storage intensity coefficient of 60 kWh/TB/yr (Selvan, 2025).

## Key Experimental Results

### Main Results: ImageNet-1K Downstream Environmental Cost Estimates

| Dimension | Estimated Value | Equivalent Carbon (445 g/kWh) | Equiv. Annual Per Capita Carbon Footprint |
|-----------|-----------------|-------------------------------|------------------------------------------|
| Training (46.2k × 300 ep × 0.394 kWh) | 5.46 ± 0.14 GWh | 2429 ± 61 tCO2e | ~514 ± 13 people |
| Storage (46.2k copies × 130 GB × 60 kWh/TB/yr) | 360 ± 9 MWh | 160 ± 4 tCO2e | ~34 ± 1 people |
| Total | 5.82 ± 0.15 GWh | 2589 ± 65 tCO2e | ~548 people |
| Same Energy @ Turkmenistan (1310 g/kWh) | — | 7624 ± 191 tCO2e | — |
| Same Energy @ Lesotho (21 g/kWh) | — | 122 ± 3 tCO2e | — |

### Ablation Study: Impact of 25% Data Pruning on Training Time/Energy (A100, Single GPU)

| Model | Params | min/epoch (full → 25%) | Time Savings | kWh/epoch (full → 25%) | Energy Savings |
|-------|--------|------------------------|--------------|------------------------|----------------|
| ResNet-34 | 21.8M | 35.2 → 23.8 | 32% | 0.2798 → 0.1989 | 29% |
| ResNet-50 | 25.6M | 40.7 → 24.3 | 40% | 0.3940 → 0.2645 | 33% |
| Swin-T | 28.3M | 58.7 → 44.6 | 24% | 0.7002 → 0.5300 | 24% |

### Key Findings
- **"25% Data $\neq$ 25% Energy"**: Energy reduction ratios vary from 24% to 33% across architectures, proving that data size is not a perfect proxy for energy.
- **Dyn-Unc and InfoMax can prune 25% / 35% of ImageNet-1K respectively without accuracy loss**; this represents a potential saving of 621–854 tCO2e.
- **Coreset construction costs amortize after 3–4 training runs**, even for "expensive" methods like Dyn-Unc.
- **Coloured-MNIST experiments** show that balanced/reweighted sampling significantly improves conflicted accuracy under 99% bias; data frugality can inherently assist in debiasing.
- **Iceberg Effect**: Downloads of ImageNet derivatives on Hugging Face are 55x higher than paper-based counts, indicating that the true environmental cost is far higher than the lower bound provided.

## Highlights & Insights
- **"Quantify lower bounds before calling for action"**: Unlike many sustainability pieces, the authors use public metadata to establish a solid lower bound, minimizing room for counter-argument.
- **The rule "measure what you motivate"** is a concise action that can be directly implemented in conference review forms.
- **Data-Pareto reporting paradigm**: Plotting accuracy against data volume is transferable to other fields (e.g., model compression).
- **Shared dataset infrastructure**: A neglected perspective—pointing out that every lab storing its own ImageNet copy constitutes dozens of MWh of wasteful redundancy.

## Limitations & Future Work
- The estimate only covers ImageNet-1K and training/storage costs, **excluding embodied costs** of collection, cleaning, and transmission.
- Empirical evidence is limited to **image classification**; generative models are more sensitive to long-tail data, and coreset migration remains an open problem.
- **Rebound effect**: Increased efficiency might lead to "running more experiments" rather than net energy reduction.
- **Policy feasibility**: High-level suggestions like carbon caps or data sunset laws require international regulation and are harder to implement than reporting norms.

## Related Work & Insights
- **vs. Kandpal & Raffel (2025)**: They argue the "human labor value" of training data is undervalued; this paper argues the environmental "price" is undervalued.
- **vs. Wang et al. (2025) / Goel et al. (2025)**: While they call for LLM "scaling down," this paper brings the argument to specific subset selection methods and empirical reporting.
- **vs. Strubell et al. (2019) & Luccioni et al. (2023)**: These focused on single-model training costs; this paper scales the perspective to **dataset-level** aggregate downstream costs.

## Rating
- Novelty: ⭐⭐⭐⭐ (Combination of lower-bound accounting + Data-Pareto paradigm + 3-layer framework).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Conservative, reproducible numbers; multi-architecture evidence).
- Writing Quality: ⭐⭐⭐⭐⭐ (Clean structure, restrained argumentation, proactive regarding limitations).
- Value: ⭐⭐⭐⭐ (Provides a template for integrating data frugality into conference and funding standards).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race](position_neglecting_the_sustainability_of_ai_is_fuelling_a_global_ai_arms_race.md)
- [\[NeurIPS 2025\] Position: Towards Bidirectional Human-AI Alignment](../../NeurIPS2025/recommender/position_towards_bidirectional_human-ai_alignment.md)
- [\[AAAI 2026\] MultiTab: A Scalable Foundation for Multitask Learning on Tabular Data](../../AAAI2026/recommender/multitab_a_scalable_foundation_for_multitask_learning_on_tabular_data.md)
- [\[ACL 2026\] Quality Over Clicks: Intrinsic Quality-Driven Iterative RL for Cold-Start E-Commerce Query Suggestion](../../ACL2026/recommender/quality_over_clicks_intrinsic_quality-driven_iterative_reinforcement_learning_fo.md)
- [\[AAAI 2026\] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](../../AAAI2026/recommender/semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)

</div>

<!-- RELATED:END -->
