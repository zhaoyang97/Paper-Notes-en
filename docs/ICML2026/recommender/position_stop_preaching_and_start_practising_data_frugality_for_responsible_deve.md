---
title: >-
  [Paper Note] Position: Stop Preaching and Start Practising Data Frugality for Responsible Development of AI
description: >-
  [ICML 2026][Recommender Systems][coreset] This position paper points out that the ML community has long been "preaching without practicing" regarding "data frugality"—while verbally acknowledging that coresets save energy, almost no one actually reports energy consumption or carbon emissions. Using ImageNet-1K as a case study, the authors calculate a conservat
tags:
  - ICML 2026
  - Recommender Systems
  - coreset
date: 2026-05-08
content_hash: 39b366933f86733a
---
# Position: Stop Preaching and Start Practising Data Frugality for Responsible Development of AI

**Conference**: ICML 2026  
**arXiv**: [2602.19789](https://arxiv.org/abs/2602.19789)  
**Code**: https://github.com/saintslab/data-frugality  
**Area**: AI Safety / Sustainable AI / Data Efficiency  
**Keywords**: Data frugality, coreset, carbon emissions, responsible AI, subset selection

## TL;DR
This position paper points out that the ML community has long been "preaching without practicing" regarding "data frugality"—while verbally acknowledging that coresets save energy, almost no one actually reports energy consumption or carbon emissions. Using ImageNet-1K as a case study, the authors calculate a conservative lower bound of approximately 5.82 GWh / 2589 tCO2e for downstream training and storage, calling for data frugality to evolve from a slogan into a measurable, actionable, and rewardable engineering practice.

## Background & Motivation

**Background**: Current mainstream AI training follows the scaling laws of Kaplan and Hoffmann, equating "larger datasets" with "better models," while benchmarks and leaderboards reward experiments run on larger corpora. Simultaneously, a parallel research stream advocates for "scaling down"—using data subset methods such as coreset selection and dataset condensation.

**Limitations of Prior Work**: The authors conducted a systematic survey of 10 representative coreset papers and found that while 8/10 listed "computational efficiency" as a motivation and 3/10 mentioned "energy efficiency," **only 1/10 actually reported energy savings**, and 6/10 reported time savings. In other words, the "promotion" of data frugality far outpaces its "practice."

**Key Challenge**: Carbon emissions cannot be linearly derived from computational volume or data scale—a 25% data reduction yields 24%–40% time gains and 24%–33% energy gains across different architectures, showing a non-proportional relationship. This implies that without actual measurement of energy and carbon, "energy saving" claims remain mere rhetoric and cannot support real decision-making for responsible AI. Furthermore, carbon estimation is highly dependent on grid carbon intensity (with a global variance exceeding 60x); the same training run can differ in emissions by dozens of times between Turkmenistan and Lesotho.

**Goal**: (i) Quantify the downstream training and storage carbon costs of a "foundational" dataset like ImageNet-1K to provide a citable lower bound; (ii) Empirically demonstrate that coresets save power without compromising accuracy; (iii) Translate these demands into specific recommendations across three layers: People, Platforms, and Policies.

**Key Insight**: The authors piece together available evidence from their own lab's (SAINTS Lab) Carbontracker measurement stack, OpenReview ICLR metadata scraping, and Hugging Face download statistics to construct a "conservative lower bound." Their aim is to show that even under the most minimal assumptions, the "opportunity cost" of failing to practice data frugality is staggering.

**Core Idea**: Translate data frugality from an activist issue of the "value-action gap" into reporting and review norms that the ML community can directly align with—if a method claims energy efficiency as a selling point, it **must** measure energy consumption.

## Method

As a position paper, the "method" consists of three mutually reinforcing arguments: carbon cost accounting, empirical coreset evidence, and an actionable framework.

### Overall Architecture

The paper is organized through a lifecycle perspective: the data lifecycle (collection, cleaning, storage, distribution) and the model lifecycle (training, selection, deployment) coexist, with coreset/subset selection impacting multiple stages of both. On this basis:

1.  **Quantitative Baseline**: ICLR 2017–2022 OpenReview metadata is used to estimate how many papers "train models from scratch on ImageNet-1K." This ratio is scaled to the entire ML literature using dimensions.ai keyword indexing and linearly extrapolated to 2023–2025.
2.  **Empirical Gains**: ResNet-34 / ResNet-50 / Swin-T are trained on A100 GPUs, using Carbontracker to measure actual energy/time for full vs. 25% pruned datasets; SOTA subset selection curves from Dyn-Unc (He et al., 2024) and InfoMax (Tan et al., 2025) are cited to show zero accuracy loss.
3.  **Fairness Side Effects**: Random, reweighted, and balanced sampling are compared on Coloured-MNIST (99% majority color) to prove that coresets can also facilitate debiasing.
4.  **Action Framework**: The above evidence is synthesized into calls for action targeting People, Platforms, and Policies.

### Key Designs

**1. Reproducible lower bound estimation of ImageNet-1K downstream carbon costs: Turning environmental costs from slogans into citable figures.**

The authors calculate the "environmental cost of the dataset itself" as a concrete lower bound: 5.82 GWh of energy and 2589 tCO2e of carbon emissions (based on global average carbon intensity). Training cost = estimated training runs $N\approx 46{,}179\pm 1{,}154$ × ResNet-50 single epoch energy $\approx 0.394$ kWh × 300 epochs $= 5.46\pm 0.14$ GWh; Storage cost = $N\times 130$ GB × 60 kWh/TB/yr $\approx 360\pm 9$ MWh. Here, $N$ is derived from the proportion of "train from scratch on ImageNet" papers in ICLR OpenReview, extrapolated and scaled via dimensions.ai. A 2.5% error rate is derived from LLM-based annotation validation. The authors deliberately use only public metadata to avoid dependence on proprietary corporate data, and they state that this is a lower bound—Hugging Face Hub's 214 ImageNet-1K derived datasets have over 2.5 million downloads, 55x higher than the paper count. This conservative stance makes the conclusion harder to refute.

**2. Dual-axis measurement of Coreset gains: Accuracy curves vs. Energy tables.**

On the accuracy side, SOTA curves from Dyn-Unc on Swin-T and InfoMax on ResNet-34 show that 25%–35% of ImageNet-1K can be pruned without Top-1 accuracy loss. On the energy side, ResNet-34/50 and Swin-T were run for 10 epochs × 3 trials (A100/DGX configuration), with Carbontracker recording both CPU and GPU energy. Results show that 25% pruning yields a 32% time + 29% energy saving for ResNet-34, 40% + 33% for ResNet-50, and 24% + 24% for Swin-T. The authors use these figures to debunk the common assumption that "data size is a proxy for energy"—25% data $\neq$ 25% energy. They also honestly calculate the one-time cost of coreset construction as "amortizable over 3–4 training runs" (using the expensive Dyn-Unc as an example), avoiding unfair comparisons.

**3. Coreset for debiasing: Achieving fairness through subset selection targets.**

The authors add an often-overlooked benefit: if "group balance" is used as the sampling target instead of "coverage" during coreset construction, subset selection becomes a debiasing tool. Using Coloured-MNIST, where digits are biased with a 99% majority color, they show that models otherwise default to learning color. Comparing three samplings—random (baseline), reweighted (weighting loss by inverse group frequency), and balanced (equalizing group samples in the coreset)—the latter two significantly increase "conflicted accuracy" (accuracy when color is not a cue). This shows that when initial data collection is biased, coresets can algorithmically remedy the bias at the selection layer.

**4. Actionable recommendations for People, Platforms, and Policies: Translating ethics into specific rules.**

To prevent the paper from being just another manifesto, requirements are divided into three layers. For **People**, they suggest replacing single-point accuracy with "Data-Pareto" (Accuracy vs. Data Volume) and the rule "measure what you motivate." For **Platforms**, they recommend mandatory compute reporting (inspired by CVPR 2026) and data-efficient leaderboards (like BabyLM). For **Policies**, they advocate for standardized carbon reporting, shared data centers (like Sweden's Berzelius) to reduce redundant local copies, and "data sunset laws" where large-scale data use requires approval and expiration dates, similar to biomedical data. The authors argue the value-action gap is caused by institutional defaults, and frugality must be upgraded from a "personal virtue" to an "institutional default."

### Loss & Training

This is a position paper; no primary model training loss is proposed. The energy measurement stack = Carbontracker + CodeCarbon for training; storage uses an energy intensity coefficient of 60 kWh/TB/yr (Selvan, 2025).

## Key Experimental Results

### Main Results: Estimated Downstream Environmental Cost of ImageNet-1K

| Dimension | Estimated Value | Equivalent Carbon (445 g/kWh) | Equivalent Annual Footprint/Person |
| :--- | :--- | :--- | :--- |
| Training (46.2k runs × 300 ep × 0.394 kWh) | 5.46 ± 0.14 GWh | 2429 ± 61 tCO2e | ~514 ± 13 people |
| Storage (46.2k copies × 130 GB × 60 kWh/TB/yr) | 360 ± 9 MWh | 160 ± 4 tCO2e | ~34 ± 1 people |
| Total | 5.82 ± 0.15 GWh | 2589 ± 65 tCO2e | ~548 people |
| Same Energy @ Turkmenistan (1310 g/kWh) | — | 7624 ± 191 tCO2e | — |
| Same Energy @ Lesotho (21 g/kWh) | — | 122 ± 3 tCO2e | — |

### Ablation Study: Impact of 25% Data Pruning on Training Time / Energy (A100, Single Card)

| Model | Parameters | min/epoch (full → 25%) | Time Saving | kWh/epoch (full → 25%) | Energy Saving |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ResNet-34 | 21.8M | 35.2 → 23.8 | 32% | 0.2798 → 0.1989 | 29% |
| ResNet-50 | 25.6M | 40.7 → 24.3 | 40% | 0.3940 → 0.2645 | 33% |
| Swin-T | 28.3M | 58.7 → 44.6 | 24% | 0.7002 → 0.5300 | 24% |

### Key Findings
- **"25% Data $\neq$ 25% Energy"**: Energy-data reduction ratios vary from 24% to 33% across architectures; data scale cannot be a proxy for energy.
- **Dyn-Unc and InfoMax can prune 25% / 35% of ImageNet-1K without accuracy loss**, representing a potential saving of 621–854 tCO2e.
- **Coreset construction costs are amortized after 3–4 runs**, making even "expensive" methods like Dyn-Unc worthwhile.
- **Coloured-MNIST experiments** show that balanced/reweighted sampling significantly raises conflicted accuracy under 99% bias, allowing data frugality to effectively "subsidize" debiasing.
- **The Iceberg Effect**: Downloads of ImageNet-1K derivatives on Hugging Face are 55x higher than paper-based counts, suggesting real downstream costs are much higher than the calculated lower bound.

## Highlights & Insights
- **"Lower bound before call to action"**: Unlike most sustainability position papers, the authors use only public metadata to calculate a conservative lower bound, minimizing room for rebuttal.
- **"Measure what you motivate"**: This single-sentence principle is a rare example of translating values into a tool that reviewers can actually use in a review form.
- **Data-Pareto Paradigm**: The Accuracy vs. Data Volume chart is transferable to other fields (e.g., Accuracy vs. FLOPs for compression).
- **Shared dataset infrastructure**: This is an underrated policy perspective—pointing out that redundant storage of ImageNet across individual labs is a waste on the scale of dozens of MWh.

## Limitations & Future Work
- The estimate only covers ImageNet-1K, training, and storage; it **excludes embodied costs** from data collection, cleaning, and network transfer.
- Empirical results are limited to **image classification**—generative models are more sensitive to long-tail patterns, and transferring data frugality to them remains an open question.
- **Rebound effect**: Efficiency gains are often consumed by running more experiments rather than net reduction; the paper acknowledges this but lacks a quantitative solution.
- **Varying feasibility of policy recommendations**: Higher-level suggestions like carbon caps and data sunset laws rely on international regulation and are harder to drive in the short term compared to People-layer reporting norms.

## Related Work & Insights
- **vs. Kandpal & Raffel (2025)**: They argue the "human labor value" of data is underestimated; this paper argues the environmental cost of treating data as a "free input" is underestimated.
- **vs. Wang et al. (2025) / Goel et al. (2025)**: They call for LLM "scaling down"; this paper provides specific subset methods and empirical reporting norms.
- **vs. McCoy et al. (2025)**: They propose capability-per-resource as a metric; this paper provides the Data-Pareto paradigm as a concrete implementation.
- **vs. Strubell et al. (2019) and Luccioni et al. (2023)**: Earlier works calculated energy for single models; this paper scales up to the **dataset-level** aggregate downstream cost.

## Rating
- Novelty: ⭐⭐⭐⭐ Not a new algorithm, but the combination of "lower bound calculation," "Data-Pareto reporting," and the "People/Platforms/Policies" framework is rare.
- Experimental Thoroughness: ⭐⭐⭐⭐ Figures are conservative and reproducible; uses Carbontracker across 3 architectures with SOTA curves and debiasing experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure (preaching vs. practicing), restrained argumentation, and proactive acknowledgement of limitations.
- Value: ⭐⭐⭐⭐ Data frugality will eventually be integrated into conference and funding mandates; this paper provides a reusable template and checklist.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race](position_neglecting_the_sustainability_of_ai_is_fuelling_a_global_ai_arms_race.md)
- [\[ICML 2025\] Position: The Right to AI](../../ICML2025/recommender/the_right_to_ai.md)
- [\[NeurIPS 2025\] Position: Towards Bidirectional Human-AI Alignment](../../NeurIPS2025/recommender/position_towards_bidirectional_human-ai_alignment.md)
- [\[ACL 2026\] Quality Over Clicks: Intrinsic Quality-Driven Iterative RL for Cold-Start E-Commerce Query Suggestion](../../ACL2026/recommender/quality_over_clicks_intrinsic_quality-driven_iterative_reinforcement_learning_fo.md)
- [\[AAAI 2026\] MultiTab: A Scalable Foundation for Multitask Learning on Tabular Data](../../AAAI2026/recommender/multitab_a_scalable_foundation_for_multitask_learning_on_tabular_data.md)

</div>

<!-- RELATED:END -->
