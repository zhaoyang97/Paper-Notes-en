---
title: >-
  [Paper Note] Turning Drift into Constraint: Robust Reasoning Alignment in Non-Stationary Multi-Stream Environments
description: >-
  [ICML 2026][Medical Imaging][Multi-source alignment] This paper reinterprets the reasoning "drift" between multiple MLLMs as negative sample constraints in DPO. By using Plackett-Luce preference loss to simultaneously su…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Multi-source alignment"
  - "Concept drift"
  - "Preference optimization"
  - "Chest X-ray diagnosis"
  - "Plackett-Luce"
date: 2026-05-08
content_hash: 546de863c96b7b7c
---

# Turning Drift into Constraint: Robust Reasoning Alignment in Non-Stationary Multi-Stream Environments

**Conference**: ICML 2026  
**arXiv**: [2510.04142](https://arxiv.org/abs/2510.04142)  
**Code**: https://github.com/XiaoyuYoung/APO (Available)  
**Area**: Medical Imaging / Multimodal VLM / Alignment RLHF  
**Keywords**: Multi-source alignment, Concept drift, Preference optimization, Chest X-ray diagnosis, Plackett-Luce

## TL;DR
This paper reinterprets the reasoning "drift" between multiple MLLMs as negative sample constraints in DPO. By using Plackett-Luce preference loss to simultaneously suppress divergent trajectories from $N$ source models, a 7B student model exceeds all source teachers in chest X-ray classification and report generation tasks using only 10% of MIMIC-CXR without ground-truth reports.

## Background & Motivation

**Background**: Using multiple large models as reasoning teachers to let a student model distill multiple CoT trajectories is a standard approach in multi-source alignment and "collective intelligence." In specialized fields like medical QA, leveraging multiple complementary teachers is the default recipe.

**Limitations of Prior Work**: The authors found that reasoning distributions of different source MLLMs are inherently divergent—e.g., Qwen-VL-Max tends to be precise and concise, while GPT-4o tends to be high-recall and wordy. Directly concatenating these heterogeneous trajectories for SFT prevents the student from automatically absorbing the strengths of each; instead, it learns all biases indiscriminately, leading to hallucinations and semantic inconsistency.

**Key Challenge**: Diversity among source models is both a benefit (broader coverage) and a risk (conflicts). Existing works treat conflicts as noise to be averaged out, but these conflict regions actually contain the most informative "decision boundaries." Averaging erases this information.

**Goal**: To let the student model learn a robust reasoning manifold under a non-stationary multi-stream environment where source reasoning trajectories constantly drift and no ground-truth supervision is available. It also aims to prove that such drift can be explicitly utilized rather than just treated as noise.

**Key Insight**: The evolution of multi-source reasoning is mapped into the concept drift theoretical framework—treating the autoregressive steps of CoT as the "time axis." Thus, the divergence between models becomes a non-stationary environment. From this perspective, divergent regions define "what should be avoided."

**Core Idea**: Use the consensus among source models as positive samples and the divergent trajectories of each source as negative samples. Extend DPO to the Plackett-Luce multi-negative sample form, turning drift from noise into "active unlearning supervision signals."

## Method

### Overall Architecture
The APO framework consists of two stages. Stage 1 is Supervised Bootstrapping with Consensus Synthesis: supervised distillation is performed using all source reasoning trajectories to project the target policy $\pi_\theta$ into the union of source capabilities, resulting in $\hat{\pi}_{st}$. Then, $\hat{\pi}_{st}$ acts as an in-context aggregator, taking $N$ source trajectories $\mathcal{T}=\{\tau^1,\ldots,\tau^N\}$ for the same problem as context to generate a self-consistent consensus trajectory $t^+ \sim \hat{\pi}_{st}(\cdot|v,l,\text{Context}=\mathcal{T})$. Stage 2 is Constraint-Aware Optimization: $t^+$ is used as the positive sample and the $N$ original source trajectories are used as negative samples for Plackett-Luce preference optimization. At inference time, only the final $\pi_\theta$ is used.

### Key Designs

1.  **Modeling multi-stream reasoning under concept drift**:
    -   Function: Formalizes the divergence phenomenon in multi-teacher reasoning as a non-stationary stochastic process, providing a theoretical explanation for why simple distillation fails.
    -   Mechanism: Assumes $N$ source models generate CoT conditionally independently, factorizing the joint distribution into $P_j(\mathcal{S}_j)=\prod_{u=1}^N P(t_{<j}^u|v,l) \cdot P(z_j^u|t_{<j}^u,v,l)$, where the former is cumulative historical divergence and the latter is instantaneous drift. Concept drift exists when $P_j(\mathcal{S}) \neq P_{j+\Delta}(\mathcal{S})$, meaning the supervision labels themselves are drifting.
    -   Design Motivation: Traditional distillation assumes teachers provide stable ground-truth; this work proves teachers develop non-stationary disagreements as reasoning progresses, meaning naive SFT will inherit all biases.

2.  **Consensus Synthesis via In-Context**:
    -   Function: Automatically constructs a "preferred trajectory" $t^+$ as a positive anchor for preference optimization without ground-truth labels.
    -   Mechanism: The bootstrapped $\hat{\pi}_{st}$ has absorbed the union of source knowledge but still carries drift. By feeding $N$ source trajectories as context to $\hat{\pi}_{st}$, it acts as a "weighted aggregator"—retaining tokens supported by multiple sources and filtering out incoherent parts lacking cross-model support. This utilizes in-context learning for implicit voting.
    -   Design Motivation: Replaces expensive manual annotation. Consensus is not a simple token-level majority vote but a semantic trajectory-level refinement generated by the student, allowing unsupervised iteration.

3.  **APO Loss with Plackett-Luce Multi-Negative Samples**:
    -   Function: Generalizes DPO from binary pairwise to a multi-constraint form with one positive and $N$ negatives to suppress drift patterns from all source models.
    -   Mechanism: Uses $\hat{\pi}_{st}$ as the reference policy and defines implicit reward $r(v,l,t)=\beta \log \frac{\pi_\theta(t|v,l)}{\hat{\pi}_{st}(t|v,l)}$. The preference probability is extended to $P(t^+ \succ \mathcal{T}|v,l)=\frac{\exp(r(v,l,t^+))}{\exp(r(v,l,t^+))+\sum_{u=1}^N \exp(r(v,l,\tau^u))}$. The loss is $-\mathbb{E}[\log P(t^+ \succ \mathcal{T}|v,l)]$. The optimization pushes the probability of $t^+$ up while simultaneously suppressing every $\tau^u$.
    -   Design Motivation: Standard DPO handles only one pair at a time, but source drift is an $N$-way conflict. The Plackett-Luce form treats the entire set of negative samples as competing hypotheses, making "active unlearning of $N$ biases" a first-order objective, which is more efficient and fits the geometric intuition of drift-as-constraint.

### Loss & Training
The two stages are trained serially: Stage 1 is KL-minimization SFT $q^* = \arg\min_q \sum_u \mathbb{D}_{\text{KL}}(\pi_u || q)$, and Stage 2 is the APO objective. The model used is Qwen2.5-VL 7B, with 1 epoch per stage and batch size = 2. The CXR-MAX dataset uses only 1/10 of MIMIC-CXR (approx. 170k multi-teacher reasoning trajectories, 14 chest diseases) and does be not use radiologist reports, emphasizing supervision solely from multi-teacher drift.

## Key Experimental Results

### Main Results

| Dataset | Task | Metric | Ours (7B) | Prev. SOTA | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MS-CXR-T | Multi-label Class. (Avg 5) | Top-1 Acc | 0.78 | 0.69 (CoCa-CXR) | +0.09 |
| MS-CXR-T | Pneumothorax | Top-1 Acc | 0.96 | 0.73 | +0.23 |
| MS-CXR-T | Consolidation | Top-1 Acc | 0.84 | 0.70 | +0.14 |
| MIMIC-CXR | Report Generation | BLEU-1 | 0.56 | 0.43 (CPO) | +0.13 |
| MIMIC-CXR | Report Generation | ROUGE-L | – | 0.42 (CPO) | Improvement |

Note: Ours uses 10% data without radiologist reports; baselines use full data with reports.

### Ablation Study

| Configuration | Key Findings | Description |
| :--- | :--- | :--- |
| Supervised Bootstrap Only | Significant hallucinations | Inherits source biases, confirming Observation 1.2. |
| Bootstrap + DPO (Pairwise) | Partial gain but inferior | Shows the necessity of Plackett-Luce multi-negative constraints. |
| Full APO (PL Multi-Negative) | Avg 0.78 | Drift-as-constraint is more robust than consensus training alone. |
| Source Teachers | Lower than Student 7B | Student exceeds teachers, proving the effect of ensemble + reverse constraint. |

### Key Findings
- **Pneumothorax Lead (+0.23)**: Pleural lines for this condition are subtle; source models are uncertain and show maximum drift. By treating uncertainty regions as negative constraints, APO sharpens sensitivity to key visual cues.
- **Edema Slightly Lower**: APO treats high-variance drift regions as areas to "avoid," leading to conservative behavior and sacrificing some recall for safety.
- **7B Student Exceeds Source Teachers (including GPT-4o, Qwen-VL-Max)**: Proves the combination of consensus and explicit unlearning of drift is stronger than the annotation quality of any single teacher.

## Highlights & Insights
- **Drift-as-constraint Perspective**: Flips the "teacher conflict" problem from a headache to an "explicit negative constraint," solving both unsupervised and robustness challenges.
- **Natural Progression to Plackett-Luce**: While DPO requires positive-negative pairs, multi-source scenarios are inherently $1:N$. APO is the first to bridge this theory to multi-teacher distillation.
- **Transferable Self-supervised Alignment**: This framework can be applied to any scenario where "multiple teachers disagree but gold labels are missing," such as multi-LLM judge evaluation, cross-model reward synthesis, or multi-retriever ranking.

## Limitations & Future Work
- **Dependency on Consensus Extractability**: If teachers have zero consensus (extremely high-variance tasks), the in-context $t^+$ becomes unreliable, causing training signals to collapse.
- **Equal Weighting in Plackett-Luce**: Currently all sources are treated equally, whereas GPT-4o and smaller models have different reliability. Future work could weight negative samples by dynamic confidence.
- **Domain Focus**: Whether it holds for broader multi-source reasoning (math, code) remains to be verified.
- **Benchmark Update**: CXR-MAX depends on current MLLM reasoning; as models upgrade, drift patterns change, requiring the benchmark to be updated.

## Related Work & Insights
- **vs DPO (Rafailov 2023)**: DPO uses static external preference labels; APO automatically constructs pairs, uses PL multi-negative samples, and targets active unlearning.
- **vs WeakLM Distillation / FUSE-style Multi-teacher**: These either average or pick the strongest teacher. APO leverages the divergent regions between teachers as training signals.
- **vs Self-Refine / Self-consistency**: Self-consistency only performs majority voting at inference; APO moves this to the RL/preference learning stage and utilizes "minority" trajectories as constraints rather than discarding them.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Perspective flip of drift-as-constraint + DPO to Plackett-Luce extension.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong comparisons on MS-CXR-T and report generation, though ablation could be more granular.
- Writing Quality: ⭐⭐⭐⭐⭐ Seamless transition between theory and method; formulas correspond well with observations.
- Value: ⭐⭐⭐⭐⭐ Provides direct inspiration for multi-teacher distillation, unsupervised alignment, and medical VQA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MDP-GRPO: Stabilized Group Relative Policy Optimization for Multi-Constraint Instruction Following](../../ACL2026/medical_imaging/mdp-grpo_stabilized_group_relative_policy_optimization_for_multi-constraint_inst.md)
- [\[CVPR 2026\] Robust Multi-Source Covid-19 Detection in CT Images](../../CVPR2026/medical_imaging/robust_multi-source_covid-19_detection_in_ct_images.md)
- [\[ACL 2026\] MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning](../../ACL2026/medical_imaging/multidx_a_multi-source_knowledge_integration_framework_towards_diagnostic_reason.md)
- [\[ICML 2026\] SynerMedGen: Synergizing Medical Multimodal Understanding with Generation via Task Alignment](synermedgen_synergizing_medical_multimodal_understanding_with_generation_via_tas.md)
- [\[ICML 2026\] Evidential Reasoning Advances Interpretable Real-World Disease Screening](evidential_reasoning_advances_interpretable_real-world_disease_screening.md)

</div>

<!-- RELATED:END -->
