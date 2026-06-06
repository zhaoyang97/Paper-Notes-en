---
title: >-
  [Paper Note] Towards Resource-Efficient LLMs: End-to-End Energy Accounting of Distillation Pipelines
description: >-
  [ICML 2026][Model Compression][Distillation Energy Consumption] The authors developed a multi-stage GPU energy collection framework based on NVML…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Distillation Energy Consumption"
  - "End-to-End Accounting"
  - "Teacher-Side Cost"
  - "Pareto Frontier"
  - "Teacher Reuse"
date: 2026-05-08
content_hash: 08b35544726016e5
---

# Towards Resource-Efficient LLMs: End-to-End Energy Accounting of Distillation Pipelines

**Conference**: ICML 2026  
**arXiv**: [2605.13981](https://arxiv.org/abs/2605.13981)  
**Code**: https://github.com/StellarLuminosity/Energy (Available)  
**Area**: LLM Efficiency / Green AI / Knowledge Distillation  
**Keywords**: Distillation Energy Consumption, End-to-End Accounting, Teacher-Side Cost, Pareto Frontier, Teacher Reuse

## TL;DR
The authors developed a multi-stage GPU energy collection framework based on NVML, decomposing the distillation pipeline into "teacher-side + student-side + evaluation" for granular accounting. They discovered that for one-off runs, teacher logit caching or synthetic data generation represents the primary cost, causing KD and synthetic SFT on 1B–13B OLMo-2 students to consume approximately $2.4\times$ more energy than direct SFT. A closed-form break-even formula is provided to demonstrate that distillation is only "energy-saving" when teacher outputs are reused $N^*$ times or more.

## Background & Motivation

**Background**: The explosion in LLM deployment has surged GPU and electricity demands. The "Green AI" trend advocates for evaluating energy consumption alongside accuracy. In this context, knowledge distillation (KD) is widely regarded as a "cheaper and greener" production line for small models, with papers typically reporting student-side FLOPs, training duration, or inference energy as evidence of efficiency.

**Limitations of Prior Work**: Existing reports almost exclusively calculate student-side costs, treating the teacher's generation of logits, synthetic data, and hyperparameter sweeps as "sunk costs." When the cost of a 32B teacher generating billions of tokens for a 1B student is included, the claim that "distillation is more energy-efficient" becomes questionable—yet the community lacks a unified, reproducible, and stage-decomposed energy protocol to verify or debunk this.

**Key Challenge**: Teacher-side costs are nearly fixed high overheads that can only be diluted when spread across multiple students or hyperparameter sweeps, whereas student training is a variable cost that scales linearly with size. The relative magnitude of these two determines where the entire pipeline falls on the energy-quality Pareto frontier, a factor rarely quantified in previous work.

**Goal**: (a) Determine when KD/synthetic SFT achieves a better energy-quality trade-off than a strong SFT baseline under a fixed budget; (b) Quantify teacher-side costs relative to student training; (c) Identify when distillation is truly "energy-efficient" across dimensions like student scale, sequence length, teacher reuse, and quality targets.

**Key Insight**: The distillation pipeline is formalized into five non-overlapping stages: "prerun, data preprocessing, teacher forward, student training, and evaluation." Each segment is measured using numerical integration of GPU power time-series sampled via NVML every 0.5s, with CPU consumption estimated via CodeCarbon. Data is normalized to Joules/token to plot Pareto frontiers.

**Core Idea**: Distillation is not inherently "green"; energy efficiency is entirely a **workflow** issue. By measuring the pipeline in stages, a closed-form break-even reuse formula can be derived to provide actionable design guidelines for when to employ distillation.

## Method

### Overall Architecture
The complete pipeline is divided into three comparison regimes: baseline SFT, logit-based KD, and synthetic SFT. All are executed on a dedicated H100 80 GB node, using the OLMo-2 tokenizer, Adafactor optimizer, bf16 precision, sequence length of 1024, effective batch size of 4, a cosine LR with a 100-step warmup, and early stopping with a tolerance of $\epsilon = 2\times 10^{-3}$. The teacher is consistently OLMo-2-SFT (32B), while students cover 1B / 7B / 13B scales. Tasks utilize TULU-3 instructions, OpenR1-Math, and Open-R1 Codeforces datasets. The pipeline is structured such that each stage is timestamped with token counts. Energy is calculated as $E_{\text{GPU}} \approx \int_{t_s}^{t_e} P_{\text{GPU}}(t)\,dt$, and CO₂e is derived via $E_{\text{total}} \cdot \text{PUE} \cdot g_{\text{region}}$. Three Pareto frontiers are output: energy-quality, stage breakdown, and reuse amortization curves.

### Key Designs

1.  **Multi-Stage End-to-End Energy Accounting Protocol**:
    - **Function**: Decomposes total distillation energy into $E_{\text{prerun}} + E_{\text{teacher}} + E_{\text{student}} + E_{\text{eval}}$, where $E_{\text{teacher}}$ is further split into logit caching $E_{\text{logit}}$ or synthetic generation $E_{\text{gen}}$.
    - **Mechanism**: NVML samples GPU power every 0.5s as ground truth; CodeCarbon estimates CPU power in process-tracking mode. Units are unified to $1\,\text{kWh}=3.6\times 10^{6}\,\text{J}$. For comparability across scales/pipelines, stage energy is divided by tokens: $\text{J/token}=E^{(\text{stage})}_{\text{total}}/N_{\text{tokens}}$.
    - **Design Motivation**: Previous Green AI works relied on proxy metrics like GPU-hours or FLOPs, which are incomparable across pipelines. Decomposing by stage identifies whether the bottleneck lies with the teacher or student.

2.  **Energy-Quality Pareto Frontier and Unified Quality Score**:
    - **Function**: Compresses scores from five benchmarks into a single scalar for cross-student comparison and plots energy vs. quality to identify dominated pipeline-scale combinations.
    - **Mechanism**: The quality score is defined as the equal-weighted retention rate relative to the 32B teacher: $Q_i = \frac{1}{B}\sum_{b=1}^{B}\frac{s_{i,b}}{s_{\text{teacher},b}}$, where $B=5$ (AlpacaEval 2, IFEval, MT-Bench-101, GSM8K, MMLU). The $x$-axis represents total pipeline kWh, and the $y$-axis represents $Q$.
    - **Design Motivation**: Benchmark tables alone fail to show which (pipeline, scale) combinations are strictly dominated. Explicit Pareto plots reveal "obviously sub-optimal" configurations that waste electricity.

3.  **Teacher Amortization and Closed-Form Break-Even Threshold**:
    - **Function**: Quantifies when distillation surpasses baseline SFT in end-to-end energy efficiency when teacher products (cached logits/synthetic datasets) are reused across $N$ students or seeds.
    - **Mechanism**: The average energy per student is $E_{\text{teacher}}/N + E_{\text{student}}^{\text{distill}}$. The break-even point is $N^* = \dfrac{E_{\text{teacher}}}{E_{\text{student}}^{\text{baseline}}-E_{\text{student}}^{\text{distill}}}$. For inference, $T^* = \dfrac{E_{\text{extra-train,kWh}}\cdot 3{,}600{,}000}{j_{\text{ref}}-j_{\text{student}}}$ determines the number of serving tokens needed to recover training overhead.
    - **Design Motivation**: Efficiency is a workflow variable determined by reuse frequency. This simple ratio provides a "reuse-before-regenerate" guideline for new hardware or model families.

### Loss & Training
The KD objective is a standard Hinton-style mixture: $\mathcal{L}_{\text{KD}}(\theta_s) = \alpha\,\mathrm{CE}(y_{\mathrm{hard}}, p_s) + (1-\alpha)\,T^2\,\mathrm{KL}(p_t^{(T)} \,\|\, p_s^{(T)})$ with default $\alpha=0.5, T=1$. Sensitivity sweeps cover $T \in \{1, 2, 4\}$ and $\alpha \in \{0.3, 0.5, 0.8\}$. Synthetic SFT uses pure autoregressive $\mathcal{L}_{\text{SFT}}(\theta_s; x, y) = -\sum_{t=1}^s \log p_{\theta_s}(y_t \mid x, y_{<t})$ with teacher nucleus sampling. All regimes share the same batch size, scheduler, and early stopping rules to ensure energy differences are attributable solely to pipeline structure.

## Key Experimental Results

### Main Results
Teacher 32B → Students 1B/7B/13B. Values are averaged across three datasets with 2–3 repetitions.

| Pipeline | Scale | $E$ (kWh) | J/token | $Q$ | Notes |
|---|---|---|---|---|---|
| Baseline SFT | 1B | 7.00 | 0.84 | 0.69 | Lowest energy |
| Baseline SFT | 7B | 19.50 | 2.34 | 0.90 | Pareto-dominant for 7B/13B |
| Baseline SFT | 13B | 34.60 | 4.15 | 0.99 | Highest quality |
| KD | 1B | 16.90 | 2.03 | 0.70 | $\sim 2.4\times$ energy of 1B SFT |
| KD | 13B | 42.50 | 5.10 | 0.82 | Dominated by baseline 13B |
| Synthetic SFT | 13B | 40.70 | 4.88 | 0.85 | Teacher generation is the main cost |

### Ablation Study
Decomposed energy distribution (kWh) by stage:

| Pipeline | Student Scale | Data Preproc. | Teacher-side | Student Train | Eval |
|---|---|---|---|---|---|
| Baseline SFT | 1B / 13B | 0.37 / 0.37 | – | 6.30 / 33.15 | 0.33 / 1.08 |
| KD | 1B / 13B | 0.37 / 0.37 | 11.00 (Logits) | 5.20 / 30.05 | 0.33 / 1.08 |
| Synthetic SFT | 1B / 13B | 0.37 / 0.37 | 10.60 (Gen) | 5.35 / 28.65 | 0.33 / 1.08 |

Reuse thresholds: $N^*$ for KD is $\sim 10$ for 1B and $4-6$ for larger models; synthetic SFT is $\sim 11$ for 1B and $2-3$ for 13B.

### Key Findings
- Teacher-side costs push the distillation curve to the right; in one-off runs, KD/synthetic SFT are strictly Pareto-dominated by baseline SFT at 7B/13B scales.
- Smaller students find it harder to amortize the teacher; larger students break even faster. Thus, "reuse-before-regenerate" is most critical for small-scale students.
- Distillation student training kWh is consistently lower than identically sized baselines due to convergence speed (soft labels lead to earlier stopping), not because "distillation runs more efficiently on GPUs."
- In KD hyperparameters, $T$ is a second-order factor while $\alpha$ dominates the energy-quality trade-off. Some $(T, \alpha)$ combinations are Pareto-dominated, yielding negative returns for the extra electricity spent.
- In synthetic SFT, `max_new_tokens` is the primary driver of non-linear energy growth. Beyond medium lengths, marginal returns diminish; reducing prompt count/length should be prioritized over expanding generation.

## Highlights & Insights
- Transforms the "is distillation energy-efficient" marketing narrative into actionable engineering math: use the break-even formula $N^* = E_{\text{teacher}}/(E^{\text{baseline}}_{\text{student}} - E^{\text{distill}}_{\text{student}})$.
- True "green" AI involves treating teacher outputs as versioned, shared infrastructure—a direct workflow recommendation for enterprise R&D.
- The combination of NVML time-series integration and CodeCarbon provides an open-source harness for "energy auditing" of other post-training methods like quantization, pruning, or LoRA.

## Limitations & Future Work
- Experiments were limited to H100 single-node and the OLMo-2 model family; J/token may shift significantly on TPU or A100.
- The teacher was fixed at 32B; smaller teachers might drastically lower $N^*$.
- Tasks only covered instructions, math, and code; safety alignment, multilingual, and long-context scenarios remain unexplored.
- The $T^*$ inference amortization formula currently assumes same-scale substitution; it requires further refinement for scenarios where small distilled models replace significantly larger counterparts.

## Related Work & Insights
- **vs Schwartz et al. (Green AI)**: While they called for energy as an evaluation metric, Ours applies this to the specific sub-field of distillation with a reproducible protocol.
- **vs Rafat et al. (2023) / Yuan et al. (2024)**: Previous works treated teachers as sunk costs. Ours explicitly measures teacher forward passes as a first-class cost, leading to the conclusion that distillation is more energy-intensive for small student scenarios.
- **vs CodeCarbon**: Building on top of general estimators, Ours adds raw NVML sampling and explicit stage boundaries to reduce estimation error and enable pipeline-level Pareto analysis.

## Rating
- Novelty: ⭐⭐⭐⭐ Not a new algorithm, but the first to establish a quantifiable and reproducible break-even framework for teacher-side costs in distillation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Robust investment of 2000 GPU-hours across 3 pipelines, 3 scales, and 3 datasets.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with corresponding formulas, tables, and Pareto plots; practical recommendations are immediately applicable.
- Value: ⭐⭐⭐⭐⭐ Debunks the "distillation is greener" narrative and provides an open-source harness for continuous energy auditing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] End-to-End Compression for Tabular Foundation Models](end-to-end_compression_for_tabular_foundation_models.md)
- [\[CVPR 2026\] A Paradigm Shift: Fully End-to-End Training for Temporal Sentence Grounding in Videos](../../CVPR2026/model_compression/a_paradigm_shift_fully_end-to-end_training_for_temporal_sentence_grounding_in_vi.md)
- [\[ICML 2026\] Memory-Efficient Partitioned DNN Inference on Resource-Constrained Android Crowds](memory-efficient_partitioned_dnn_inference_on_resource-constrained_android_crowd.md)
- [\[ICML 2026\] Multi-Adapter Representation Interventions via Energy Calibration](multi-adapter_representation_interventions_via_energy_calibration.md)
- [\[ICML 2026\] Energy-Structured Low-Rank Adaptation for Continual Learning](energy-structured_low-rank_adaptation_for_continual_learning.md)

</div>

<!-- RELATED:END -->
