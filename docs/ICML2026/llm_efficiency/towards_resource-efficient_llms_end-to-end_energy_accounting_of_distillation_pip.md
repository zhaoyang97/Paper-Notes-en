---
title: >-
  [Paper Note] Towards Resource-Efficient LLMs: End-to-End Energy Accounting of Distillation Pipelines
description: >-
  [ICML 2026][LLM Efficiency][Distillation Energy Consumption] The authors built a staged GPU energy measurement framework based on NVML, decomposing the distillation pipeline into "teacher side + student side + evaluation…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "Distillation Energy Consumption"
  - "End-to-End Accounting"
  - "Teacher-Side Cost"
  - "Pareto Frontier"
  - "Teacher Reuse"
date: 2026-05-08
content_hash: cff8291f0ce8d867
---

# Towards Resource-Efficient LLMs: End-to-End Energy Accounting of Distillation Pipelines

**Conference**: ICML 2026  
**arXiv**: [2605.13981](https://arxiv.org/abs/2605.13981)  
**Code**: https://github.com/StellarLuminosity/Energy (available)  
**Area**: LLM Efficiency / Green AI / Knowledge Distillation  
**Keywords**: Distillation Energy Consumption, End-to-End Accounting, Teacher-Side Cost, Pareto Frontier, Teacher Reuse

## TL;DR
The authors built a staged GPU energy measurement framework based on NVML, decomposing the distillation pipeline into "teacher side + student side + evaluation" for segment-wise accounting. They found that one-off teacher logit caching/synthetic data generation dominates energy use, causing KD and synthetic SFT to consume about $2.4\times$ more energy than direct SFT for 1B–13B OLMo-2 students. They provide a closed-form break-even formula, showing distillation is only truly "energy-saving" when teacher outputs are reused more than $N^*$ times.

## Background & Motivation

**Background**: The surge in LLM deployment has driven up GPU and electricity demand, and the "Green AI" movement calls for energy consumption to be evaluated alongside accuracy. In this context, knowledge distillation is widely regarded as a "cheaper, greener" pipeline for producing small models, with papers typically reporting student-side FLOPs/training time/inference energy as evidence of greenness.

**Limitations of Prior Work**: Existing reports almost exclusively account for the student side, treating teacher logit generation, synthetic data creation, and hyperparameter sweeps as "sunk costs." Once the cost of a 32B teacher generating billions of tokens for a 1B student is included, the claim that "distillation saves energy" becomes untenable. However, the community lacks a unified, reproducible, stage-wise energy protocol to expose or support this issue.

**Key Challenge**: The teacher side is a nearly fixed, large expense that can only be diluted by sharing across multiple students or hyperparameter sweeps, while the student side is a linearly scaling variable cost. Their relative magnitudes determine where the entire pipeline falls on the energy-quality plane, a point rarely quantified in prior work.

**Goal**: (a) Identify when KD/synthetic SFT achieves a better energy-quality trade-off than a strong SFT baseline under a fixed budget; (b) Quantify the teacher-side cost relative to student training and when it dominates; (c) Determine, across student scale, sequence length, teacher reuse, and quality targets, when distillation is truly "energy-saving."

**Key Insight**: Formalize the distillation pipeline into five non-overlapping stages: "prerun / data preprocessing / teacher forward / student training / evaluation." Each stage is numerically integrated using NVML 0.5 s-sampled GPU power time series; CPU-side energy is estimated with CodeCarbon. All results are normalized to Joule/token, and Pareto frontiers are plotted.

**Core Idea**: Distillation is not inherently "green"; whether it saves energy is a **workflow** issue. By segmenting and measuring the pipeline, a closed-form break-even reuse formula can be derived, providing actionable design guidelines for "when to distill."

## Method

### Overall Architecture
The full pipeline is split into three comparative regimes: baseline SFT, logit-based KD, and synthetic SFT. All run on the same dedicated H100 80 GB node, using a fixed OLMo-2 tokenizer, Adafactor optimizer, bf16, sequence length 1024, effective batch 4, cosine LR + 100-step warmup, and early stopping with tolerance $\epsilon = 2\times 10^{-3}$. The teacher is always 32B OLMo-2-SFT; students are 1B/7B/13B. Tasks use TULU-3 instructions, OpenR1-Math, and Open-R1 Codeforces supervised datasets. The pipeline is structured so each stage is timestamped and token-counted; GPU power series are integrated as $E_{\text{GPU}} \approx \int_{t_s}^{t_e} P_{\text{GPU}}(t)\,dt$ to yield stage energy, which are summed for end-to-end kWh. CO₂e is estimated as $E_{\text{total}} \cdot \text{PUE} \cdot g_{\text{region}}$. Three Pareto frontiers are output: energy-quality, stage breakdown, and reuse amortization curves.

### Key Designs

1. **Stage-wise End-to-End Energy Accounting Protocol**:

    - **Function**: Hard-partitions total distillation energy into $E_{\text{prerun}} + E_{\text{teacher}} + E_{\text{student}} + E_{\text{eval}}$, with $E_{\text{teacher}}$ further split into logit caching $E_{\text{logit}}$ or synthetic generation $E_{\text{gen}}$. Each segment has explicit start/end boundaries, token counts, and GPU/CPU power sampling.
    - **Mechanism**: NVML samples GPU power at 0.5 s intervals as ground truth; CodeCarbon in process-tracking mode estimates CPU. All units are unified as $1\,\text{kWh}=3.6\times 10^{6}\,\text{J}$. To enable cross-scale/pipeline comparison, stage energy is divided by processed tokens to yield $\text{J/token}=E^{(\text{stage})}_{\text{total}}/N_{\text{tokens}}$. CO₂e, being deployment-dependent, is explicitly marked as a derived metric; main analysis focuses on measured energy.
    - **Design Motivation**: Previous Green AI work often used GPU-hours or FLOPs as proxies, which are not directly comparable across pipelines. Stage-wise breakdown immediately pinpoints whether the bottleneck is teacher or student, guiding which lever to adjust.

2. **Energy-Quality Pareto Frontier and Unified Quality Score**:

    - **Function**: Compresses five benchmark scores into a single cross-student comparable scalar, then plots energy vs. quality Pareto scatterplots to identify dominated pipeline-scale combinations.
    - **Mechanism**: Quality score is defined as the relative retention rate to the 32B teacher: $Q_i = \frac{1}{B}\sum_{b=1}^{B}\frac{s_{i,b}}{s_{\text{teacher},b}}$, where $B=5$ and benchmarks include AlpacaEval 2, IFEval, MT-Bench-101, GSM8K, and MMLU. The $x$-axis is total pipeline kWh (sum of stages), $y$-axis is $Q$, and each configuration is run 2–3 times with mean reported. Since CO₂e is linearly proportional to kWh under fixed grid factors, the same plot can be read as an emissions-quality frontier.
    - **Design Motivation**: Benchmark tables alone do not reveal which (pipeline, scale) combinations are strictly dominated in the energy-quality plane. Explicit Pareto plots directly inform practitioners which configurations are "obviously suboptimal" and wasteful.

3. **Teacher Amortization and Closed-Form Break-Even Threshold**:

    - **Function**: Quantifies when distillation overtakes baseline SFT in end-to-end energy, as teacher outputs (cached logits/synthetic datasets) are reused by $N$ students/hyperparameter seeds.
    - **Mechanism**: For each KD/synthetic SFT curve, per-student average energy is $E_{\text{teacher}}/N + E_{\text{student}}^{\text{distill}}$. The break-even point with baseline is $N^* = \dfrac{E_{\text{teacher}}}{E_{\text{student}}^{\text{baseline}}-E_{\text{student}}^{\text{distill}}}$. Similarly, for inference, $T^* = \dfrac{E_{\text{extra-train,kWh}}\cdot 3{,}600{,}000}{j_{\text{ref}}-j_{\text{student}}}$ tells users how many inference tokens are needed to recoup the extra training energy.
    - **Design Motivation**: Whether distillation "saves energy" is not an intrinsic property of KD or synthetic SFT, but a workflow issue determined by reuse count. Expressing it as a simple fraction with the denominator as "baseline minus distillation student training energy" allows threshold recalculation for new hardware/model families, providing a reuse-before-regenerate design rule.

### Loss & Training

The KD objective is the classic Hinton-style mixture: $\mathcal{L}_{\text{KD}}(\theta_s) = \alpha\,\mathrm{CE}(y_{\mathrm{hard}}, p_s) + (1-\alpha)\,T^2\,\mathrm{KL}(p_t^{(T)} \,\|\, p_s^{(T)})$, with default $\alpha=0.5$, $T=1$; sensitivity sweeps $T \in \{1, 2, 4\}$, $\alpha \in \{0.3, 0.5, 0.8\}$. Synthetic SFT uses pure autoregressive $\mathcal{L}_{\text{SFT}}(\theta_s; x, y) = -\sum_{t=1}^s \log p_{\theta_s}(y_t \mid x, y_{<t})$, with teacher outputs generated once via nucleus sampling and reused across students. Sweeps include max_new_tokens $\in \{256, 512, 1024\}$ and prompt counts 7000 vs 3500. All regimes share the same batch/scheduler/early stopping rules to ensure energy differences are attributable solely to pipeline structure and teacher presence.

## Key Experimental Results

### Main Results
Teacher 32B → students 1B/7B/13B; end-to-end energy, Joule/token, and relative teacher quality retention $Q$ are averaged across three datasets, repeated 2–3 times.

| Pipeline | Scale | $E$ (kWh) | J/token | $Q$ | Notes |
|---|---|---|---|---|---|
| Baseline SFT | 1B | 7.00 | 0.84 | 0.69 | Lowest energy |
| Baseline SFT | 7B | 19.50 | 2.34 | 0.90 | Pareto-optimal for 7B/13B |
| Baseline SFT | 13B | 34.60 | 4.15 | 0.99 | Highest quality |
| KD | 1B | 16.90 | 2.03 | 0.70 | $\sim 2.4\times$ more energy than 1B SFT |
| KD | 13B | 42.50 | 5.10 | 0.82 | Dominated by baseline 13B |
| Synthetic SFT | 13B | 40.70 | 4.88 | 0.85 | Teacher generation dominates |

### Ablation Study
Stage-wise (kWh) breakdown of key amortization:

| Pipeline | Student Scale | Data Preprocessing | Teacher Side | Student Training | Evaluation |
|---|---|---|---|---|---|
| Baseline SFT | 1B / 13B | 0.37 / 0.37 | – | 6.30 / 33.15 | 0.33 / 1.08 |
| KD | 1B / 13B | 0.37 / 0.37 | 11.00 (logit cache) | 5.20 / 30.05 | 0.33 / 1.08 |
| Synthetic SFT | 1B / 13B | 0.37 / 0.37 | 10.60 (synthetic gen) | 5.35 / 28.65 | 0.33 / 1.08 |

Reuse thresholds: For KD, $N^*$ is about 10 / 5–6 / 4 for 1B/7B/13B; for synthetic SFT, about 11 / 6 / 2–3.

### Key Findings
- Teacher-side cost is the "invisible hand" shifting the distillation curve rightward—when run once, it causes KD/synthetic SFT to be strictly Pareto-dominated by baseline SFT at 7B/13B.
- The smaller the student, the harder it is to amortize the teacher; larger students break even faster, making "reuse-before-regenerate" most critical for small students.
- Distillation student training kWh is always lower than baseline of the same size, due to faster convergence (soft label supervision → earlier stopping), not because "distillation runs more energy-efficiently on GPU."
- In KD hyperparameters, $T$ is a secondary knob, while $\alpha$ dominates the energy-quality trade-off; some $(T,\alpha)$ combinations are Pareto-dominated, consuming more energy for negative returns.
- In synthetic SFT, max_new_tokens is the main driver of nonlinear energy growth; beyond moderate lengths, marginal returns diminish, so prompt count and length should be reduced before expanding generation.

## Highlights & Insights
- Transforms the marketing claim "is distillation really energy-saving" into actionable engineering math: a one-line break-even formula $N^* = E_{\text{teacher}}/(E^{\text{baseline}}_{\text{student}} - E^{\text{distill}}_{\text{student}})$ tells teams "we need to distill for at least 6 students to break even."
- True greenness is not "using a smaller model" but "versioning and registering teacher outputs as shared infrastructure"—a directly actionable workflow recommendation for enterprise R&D.
- The combination of NVML time-series integration + CodeCarbon CPU estimation provides an open-source harness for future research, reusable for energy auditing in quantization, pruning, LoRA, and other post-training tasks.

## Limitations & Future Work
- All experiments use a single H100 card and OLMo-2 model family; J/token will likely drift significantly on multi-GPU/TPU/A100, requiring break-even thresholds to be recalculated.
- Teacher is fixed at 32B; teacher scale is not swept—smaller teachers may significantly lower the break-even, allowing KD to break even with fewer reuses.
- Task coverage is limited to instruction/math/code supervised tasks; does not address safety alignment, multilingual, or long-context deployment targets. CO₂e absolute values will vary greatly under different PUE/grid assumptions.
- The $T^*$ formula for inference amortization is still at the "same scale" level; only when distilled small models can truly replace large models at equal quality will the formula provide practical guidance for inference energy savings.

## Related Work & Insights
- **vs Schwartz et al. Green AI**: While they advocated for energy as an evaluation metric, this work implements that philosophy in the concrete post-training subfield of distillation, providing a reproducible protocol rather than a position paper.
- **vs Rafat et al. 2023 / Yuan et al. 2024**: The former focuses on carbon cost of CNN KD, the latter compares inference energy of pre-distilled NLP models—both treat teacher as sunk cost. This work explicitly treats teacher forward as a first-class cost, reaching the opposite conclusion that "distillation is more energy-consuming for small students."
- **vs CodeCarbon / Experiment Impact Tracker**: These are general-purpose estimators; this work layers NVML direct sampling + explicit stage boundaries on top, absorbing estimation error and enabling pipeline-level Pareto plotting.

## Rating
- Novelty: ⭐⭐⭐⭐ Not a new algorithm, but the first to make the "teacher-side sunk cost" of distillation quantifiable and reproducible as a break-even framework; novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 3 pipelines × 3 student scales × 3 datasets + KD/synthetic SFT hyperparameter sweeps; solid 2000 GPU-hours investment.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-integrated formulas, tables, and Pareto plots; practical recommendations are directly actionable.
- Value: ⭐⭐⭐⭐⭐ Debunks the common "distillation is greener" narrative and provides an open-source harness for ongoing industry energy audits, useful for both policy and engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] STORM: End-to-End Referring Multi-Object Tracking in Videos](../../CVPR2026/video_understanding/storm_referring_multi_object_tracking.md)
- [\[CVPR 2025\] WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild](../../CVPR2025/video_understanding/wilor_end-to-end_3d_hand_localization_and_reconstruction_in-the-wild.md)
- [\[ECCV 2024\] OneTrack: Demystifying the Conflict Between Detection and Tracking in End-to-End 3D Trackers](../../ECCV2024/video_understanding/onetrack_demystifying_the_conflict_between_detection_and_tracking_in_end-to-end_.md)
- [\[ICML 2026\] Token-Efficient Change Detection in LLM APIs](token-efficient_change_detection_in_llm_apis.md)
- [\[ICML 2026\] OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models](omnisift_modality-asymmetric_token_compression_for_efficient_omni-modal_large_la.md)

</div>

<!-- RELATED:END -->
