---
title: >-
  [Paper Note] Turning Drift into Constraint: Robust Reasoning Alignment in Non-Stationary Multi-Stream Environments
description: >-
  [ICML 2026][Reinforcement Learning][Multi-source alignment] This work reinterprets the reasoning "drift" among multiple MLLMs as negative sample constraints in DPO…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Multi-source alignment"
  - "concept drift"
  - "preference optimization"
  - "chest X-ray diagnosis"
  - "Plackett-Luce"
date: 2026-05-08
content_hash: 2b60e8c4c7d03081
---

# Turning Drift into Constraint: Robust Reasoning Alignment in Non-Stationary Multi-Stream Environments

**Conference**: ICML 2026  
**arXiv**: [2510.04142](https://arxiv.org/abs/2510.04142)  
**Code**: https://github.com/XiaoyuYoung/APO (available)  
**Area**: Medical Imaging / Multimodal VLM / Alignment RLHF  
**Keywords**: Multi-source alignment, concept drift, preference optimization, chest X-ray diagnosis, Plackett-Luce

## TL;DR
This work reinterprets the reasoning "drift" among multiple MLLMs as negative sample constraints in DPO, using Plackett-Luce preference loss to simultaneously suppress the divergent trajectories of N source models. As a result, a 7B student model, without ground-truth reports and using only 10% of MIMIC-CXR, surpasses all source teachers in chest X-ray classification and report generation tasks.

## Background & Motivation

**Background**: Using multiple large models as reasoning teachers and distilling multiple CoT trajectories into a single student model is the standard approach in multi-source alignment and "collective intelligence." In specialized domains like medical QA, leveraging complementary teachers is almost a default recipe.

**Limitations of Prior Work**: The authors observe that the reasoning distributions of different source MLLMs are inherently divergent—for example, Qwen-VL-Max tends to be precise and concise, while GPT-4o favors higher recall and verbosity. Directly concatenating these heterogeneous trajectories for SFT leads the student to inherit all biases, resulting in hallucinations and semantic inconsistency.

**Key Challenge**: The diversity among source models is both a benefit (broader coverage) and a risk (mutual conflict). Existing work treats conflicts as noise to be averaged out, but these conflict regions actually contain the most informative "decision boundaries." Averaging erases this information.

**Goal**: In the absence of ground-truth supervision and with continuously drifting source model reasoning trajectories, enable the student model to learn a robust reasoning manifold; further, demonstrate that such drift can be explicitly leveraged rather than merely treated as noise.

**Key Insight**: Frame the evolution of multi-source reasoning within the concept drift theoretical framework—mapping the autoregressive steps of CoT to the "time axis" in drift theory, so divergence among models becomes a non-stationary multi-stream environment. From this perspective, divergent regions define "what should be avoided."

**Core Idea**: Use the consensus among source models as positive samples and each source's divergent trajectory as negative samples, extending DPO to a Plackett-Luce multi-negative-sample form, turning drift from noise into an "active unlearning supervision signal."

## Method

### Overall Architecture
The APO framework consists of two stages. The first stage, Supervised Bootstrapping with Consensus Synthesis, uses all source models' reasoning trajectories for supervised distillation, projecting the target model $\pi_\theta$ into the union of source capabilities to obtain $\hat{\pi}_{st}$. Then, $\hat{\pi}_{st}$ acts as an in-context aggregator: given N source trajectories $\mathcal{T}=\{\tau^1,\ldots,\tau^N\}$ for the same question, the model generates a coherent consensus trajectory $t^+ \sim \hat{\pi}_{st}(\cdot|v,l,\text{Context}=\mathcal{T})$. The second stage, Constraint-Aware Optimization, uses $t^+$ as the positive sample and the N original source trajectories as negative samples for Plackett-Luce preference optimization. During inference, only the final $\pi_\theta$ is used; source teachers are no longer needed.

### Key Designs

1. **Multi-Stream Reasoning Modeling from the Concept Drift Perspective**:

    - **Function**: Formalizes the divergence in multi-teacher reasoning as a non-stationary stochastic process, providing a theoretical explanation for "why simple distillation fails."
    - **Mechanism**: Assumes N source models independently generate CoT, factorizing the joint distribution as $P_j(\mathcal{S}_j)=\prod_{u=1}^N P(t_{<j}^u|v,l) \cdot P(z_j^u|t_{<j}^u,v,l)$, where the former is cumulative historical divergence and the latter is instantaneous drift at the current step. When $P_j(\mathcal{S}) \neq P_{j+\Delta}(\mathcal{S})$, concept drift occurs, meaning the supervision signal seen by the student is itself drifting.
    - **Design Motivation**: Traditional distillation assumes teachers provide stable ground-truth; here, it is shown that teachers diverge non-stationarily as reasoning progresses, so naive SFT inevitably inherits all biases, necessitating a new framework.

2. **Consensus Synthesis via In-Context Consensus Extraction**:

    - **Function**: Automatically constructs a "preferred trajectory" $t^+$ as the positive anchor for subsequent preference optimization, without ground-truth labels.
    - **Mechanism**: The bootstrapped $\hat{\pi}_{st}$ has absorbed the union of source knowledge but still contains drift. Feeding the N source trajectories for the same sample as context to $\hat{\pi}_{st}$, it acts as a "weighted aggregator"—retaining tokens supported by multiple sources and filtering out incoherent parts lacking cross-model support. This leverages in-context learning as implicit voting.
    - **Design Motivation**: Replaces costly manual annotation. The consensus is not a simple majority vote at the token level, but a semantically refined trajectory generated by the student itself, enabling unsupervised iteration.

3. **Plackett-Luce Multi-Negative-Sample APO Loss**:

    - **Function**: Generalizes DPO from binary pairwise to one-positive N-negative multi-constraint form, simultaneously suppressing all source model drift patterns.
    - **Mechanism**: Uses the bootstrapped $\hat{\pi}_{st}$ as the reference policy, defining implicit reward $r(v,l,t)=\beta \log \frac{\pi_\theta(t|v,l)}{\hat{\pi}_{st}(t|v,l)}$; preference probability is extended to $P(t^+ \succ \mathcal{T}|v,l)=\frac{\exp(r(v,l,t^+))}{\exp(r(v,l,t^+))+\sum_{u=1}^N \exp(r(v,l,\tau^u))}$; the final loss is $-\mathbb{E}[\log P(t^+ \succ \mathcal{T}|v,l)]$. The optimization pushes up the probability of $t^+$ while simultaneously pushing down each $\tau^u$.
    - **Design Motivation**: Standard DPO only uses one positive-negative pair at a time, while source drift is inherently N-to-N multi-modal conflict; the Plackett-Luce form allows the entire set of negatives to be treated as competing hypotheses, making "actively forgetting N biases" a first-class training objective—more efficient and geometrically intuitive than pairwise DPO.

### Loss & Training
Two-stage sequential training: Stage 1 is SFT minimizing KL divergence $q^* = \arg\min_q \sum_u \mathbb{D}_{\text{KL}}(\pi_u || q)$; Stage 2 uses the above APO objective. The model is Qwen2.5-VL 7B, each stage runs for only 1 epoch, batch size = 2. The CXR-MAX dataset uses only 1/10 of MIMIC-CXR (about 170,000 multi-teacher reasoning trajectories, 14 chest disease types), without radiologist reports, emphasizing supervision purely from multi-teacher drift.

## Key Experimental Results

### Main Results

| Dataset | Task | Metric | Ours 7B | Prev. SOTA | Gain |
|---------|------|--------|---------|------------|------|
| MS-CXR-T | Multi-label classification (5-class avg) | Top-1 Acc | 0.78 | 0.69 (CoCa-CXR) | +0.09 |
| MS-CXR-T | Pneumothorax | Top-1 Acc | 0.96 | 0.73 | +0.23 |
| MS-CXR-T | Consolidation | Top-1 Acc | 0.84 | 0.70 | +0.14 |
| MIMIC-CXR | Report generation | BLEU-1 | 0.56 | 0.43 (CPO) | +0.13 |
| MIMIC-CXR | Report generation | ROUGE-L | – | 0.42 (CPO) | Gain |

Note: This work uses only 10% of the data and no radiologist reports, while baselines use the full dataset and reports.

### Ablation Study

| Configuration | Key Phenomenon | Description |
|---------------|---------------|-------------|
| Supervised Bootstrap only | Inherits source bias, significant hallucination | Validates "naive distillation = learning all biases" (Observation 1.2) |
| Bootstrap + DPO (pairwise) | Partial improvement but less than multi-negative | Shows necessity of Plackett-Luce multi-negative constraints |
| Full APO (PL multi-negative) | Avg 0.78 | Drift-as-constraint is more stable than consensus-only training |
| Source teachers themselves | Lower than student 7B | Student surpasses teachers, proving the combined effect of ensemble + reverse constraint |

### Key Findings
- **Significant lead on Pneumothorax (+0.23)**: This disease features very subtle pleural lines; individual source models are uncertain and diverge most. APO treats these uncertain regions as negative constraints, sharpening sensitivity to key visual cues.
- **Slightly lower on Edema**: High-variance drift regions are treated by APO as "to be avoided," making the model conservative and trading off some recall for safety; the authors acknowledge this trade-off.
- **7B student surpasses all source teachers (including GPT-4o, Qwen-VL-Max)**: Indicates that consensus plus explicit unlearning of drift is indeed stronger than any single teacher's "annotation quality."

## Highlights & Insights
- **The drift-as-constraint perspective is ingenious**: It flips the troublesome "teacher disagreement" in multi-teacher distillation from "how to reconcile" to "explicit negative constraint," simultaneously solving unsupervised and robustness challenges.
- **DPO to Plackett-Luce is a natural progression**: DPO inherently requires positive-negative pairs, while multi-source scenarios are naturally 1:N preferences; the PL extension is an almost "should be this way" generalization, but the authors are the first to apply this theory to multi-teacher distillation.
- **Self-supervised alignment is transferable**: Any scenario with "multiple teachers disagreeing but lacking gold labels" (e.g., multiple LLM judges, cross-model reward synthesis, multi-retriever ranking) can use this framework—consensus as positive, individual divergence as N negatives.

## Limitations & Future Work
- **Depends on extractability of consensus**: When teachers' trajectories have almost no consensus (extremely high-variance tasks), the in-context extracted $t^+$ itself becomes unreliable, causing APO's training signal to collapse.
- **All sources treated equally in Plackett-Luce loss**: In reality, GPT-4o and smaller models have different reliability; future work could weight negatives or adjust dynamically by confidence.
- **Experiments focus on chest X-rays**: Whether this holds for broader multi-source reasoning tasks (math, code) remains to be validated.
- **CXR-MAX dataset depends on current MLLM reasoning**: As MLLMs evolve, drift patterns will change, so the benchmark may require continual updates.

## Related Work & Insights
- **vs DPO (Rafailov 2023)**: DPO uses static external preference annotations for pairwise comparison; APO automatically constructs preference pairs, uses PL multi-negatives, and explicitly targets active unlearning—extending DPO in three dimensions.
- **vs WeakLM distillation / FUSE-style multi-teacher**: These either average or select the strongest teacher; APO does the opposite—specifically leveraging divergence among teachers as training signal.
- **vs Self-Refine / self-consistency voting**: Self-consistency only does majority vote at inference, not changing model parameters; APO moves this idea to RL/preference learning and uses "minority" trajectories as constraints rather than discarding them.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The drift-as-constraint perspective and DPO → Plackett-Luce extension are highly innovative
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad disease comparison on MS-CXR-T and multiple report generation metrics, though ablation could be more detailed
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical exposition and methodological transitions are seamless, with formulas and observations reinforcing each other
- Value: ⭐⭐⭐⭐⭐ Directly inspiring for multi-teacher distillation, unsupervised alignment, and medical VQA; CXR-MAX is also a rare resource

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning](tracking_drift_variation-aware_entropy_scheduling_for_non-stationary_reinforceme.md)
- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](../../NeurIPS2025/reinforcement_learning/forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)
- [\[ICML 2026\] Safe Reinforcement Learning with Preference-Based Constraint Inference](safe_reinforcement_learning_with_preference-based_constraint_inference.md)
- [\[ICML 2026\] Turning Bias into Bugs: Bandit-Guided Style Manipulation Attacks on LLM Judges](turning_bias_into_bugs_bandit-guided_style_manipulation_attacks_on_llm_judges.md)
- [\[ICML 2026\] Interaction-Breaking Adversarial Learning Framework for Robust Multi-Agent Reinforcement Learning](interaction-breaking_adversarial_learning_framework_for_robust_multi-agent_reinf.md)

</div>

<!-- RELATED:END -->
