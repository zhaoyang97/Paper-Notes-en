---
title: >-
  [Paper Note] DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning
description: >-
  [NeurIPS 2025][Reasoning][Process Reward Model] DreamPRM is proposed to automatically learn domain weights for multimodal reasoning datasets via bi-level optimization, addressing the data quality imbalance in PRM training. It achieves 85.2% top-1 accuracy on the MathVista leaderboard using the o4-mini model.
tags:
  - "NeurIPS 2025"
  - "Reasoning"
  - "Process Reward Model"
  - "multimodal reasoning"
  - "domain reweighting"
  - "bi-level optimization"
  - "test-time scaling"
date: 2026-05-08
content_hash: b716b8fe28c45940
---

# DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2505.20241](https://arxiv.org/abs/2505.20241)  
**Code**: [GitHub](https://github.com/coder-qicao/DreamPRM)  
**Area**: LLM Reasoning
**Keywords**: Process Reward Model, multimodal reasoning, domain reweighting, bi-level optimization, test-time scaling

## TL;DR
DreamPRM is proposed to automatically learn domain weights for multimodal reasoning datasets via bi-level optimization, addressing the data quality imbalance in PRM training. It achieves 85.2% top-1 accuracy on the MathVista leaderboard using the o4-mini model.

## Background & Motivation
**Background**: Process Reward Models (PRMs) guide LLM reasoning through fine-grained evaluation of intermediate steps in reasoning chains, achieving notable success in the text domain. Extending PRMs to multimodal LLMs (MLLMs) is a natural next step.

**Limitations of Prior Work**: Multimodal reasoning spans a broader task spectrum (science, geometry, charts, commonsense, etc.), leading to more severe train-test distribution shift and greater generalization difficulty. While large-scale, diverse datasets are necessary to ensure coverage, existing multimodal reasoning datasets suffer from severe quality imbalance—many contain uninformative modalities or overly simple questions.

**Key Challenge**: Training a generalizable PRM requires data covering multiple domains, yet naively mixing all datasets degrades PRM performance due to noise introduced by low-quality data.

**Goal**: To automatically assign appropriate weights to multimodal reasoning datasets of varying quality, producing a more generalizable multimodal PRM.

**Key Insight**: Drawing inspiration from domain reweighting techniques in pretraining (e.g., DoReMi), the paper introduces domain reweighting into PRM training and designs an aggregation function loss tailored to the reasoning setting.

**Core Idea**: Bi-level optimization is used to automatically learn dataset weights, directing the PRM to focus on high-quality reasoning samples while discounting noisy data.

## Method

### Overall Architecture
DreamPRM adopts a Bi-Level Optimization (BLO) framework:
- **Lower-level optimization**: Trains PRM parameters $\phi$ using a weighted loss over multiple training domains.
- **Upper-level optimization**: Evaluates the PRM on a separate meta-learning dataset and updates domain weights $\alpha$ via an aggregation function loss.

### Key Designs
1. **Domain-Weighted PRM Training (Lower-level)**: The PRM training loss is a weighted sum of per-domain losses: $\mathcal{L}_{tr} = \sum_{k=1}^K \alpha_k \mathcal{L}_{tr}(\mathcal{D}_k, \phi)$. Per-domain losses use Monte Carlo-estimated process supervision signals (obtained by rolling out multiple completions and comparing answer correctness rates) to train the PRM to predict the correctness probability of each step.
2. **Aggregation Function Loss (Upper-level)**: A meta-loss is designed to be consistent with inference-time behavior. Rather than evaluating single-step predictions directly, the PRM's per-step predictions are first transformed into a score for the entire reasoning chain via the aggregation function $\mathcal{A}(p) = \sum_i \log \frac{p_i}{1-p_i}$, and an MSE loss is then computed against correctness labels. This bridges the gap between training and inference.
3. **Multi-Stage Reasoning Prompting**: Drawing on LLaVA-CoT's structured thinking, the MLLM is prompted to reason in five stages: restating the problem → collecting evidence from the image → identifying relevant background knowledge → reasoning based on evidence → summarizing and drawing conclusions.
4. **Data Organization**: Fifteen training domains cover four broad categories—science, charts, geometry, and commonsense—with MMMU serving as the meta-learning dataset.

### Loss & Training
- **Lower-level**: MSE loss between PRM predictions and Monte Carlo-estimated process supervision signals.
- **Upper-level**: MSE loss between aggregation function outputs (after sigmoid) and answer correctness labels.
- The upper-level is updated once every 5 lower-level steps (unroll steps = 5).
- AdamW optimizer; lower-level lr = 5e-7, upper-level lr = 0.01.
- Total training: 10,000 iterations; approximately 10 hours on a single A100.
- PRM backbone: Qwen2-VL-2B-Instruct; reasoning MLLM: InternVL-2.5-8B-MPO.

## Key Experimental Results

### Main Results
Using InternVL-2.5-8B-MPO as the base model, accuracy (%) on five multimodal reasoning benchmarks:

| Method | WeMath | MathVista | MathVision | MMVet | MMStar |
|--------|--------|-----------|------------|-------|--------|
| Base (zero-shot) | 51.7 | 65.4 | 20.4 | 55.9 | 58.9 |
| Self-consistency | 56.4 | 67.1 | 20.7 | 57.4 | 59.6 |
| ORM | 56.9 | 65.3 | 20.5 | 55.9 | 60.1 |
| Vanilla PRM | 54.2 | 67.2 | 20.6 | 58.9 | 60.8 |
| CaR-PRM | 54.7 | 67.5 | 21.0 | 60.6 | 61.1 |
| s1-PRM | 57.1 | 65.8 | 20.2 | 60.1 | 60.4 |
| **DreamPRM** | **57.4** | **68.9** | **22.1** | **61.4** | **62.3** |

MathVista leaderboard: o4-mini + DreamPRM achieves **85.2%** (top-1), surpassing VL-Rethinker, Kimi-k1.6, OpenAI o1, and others.

### Ablation Study

| Ablation Configuration | Impact |
|------------------------|--------|
| Remove bi-level optimization (BLO) | MathVista −3.5%, MMStar −3.4% |
| Remove aggregation function loss (AFL) | WeMath −1.1%, consistent 1–2% drop |
| Remove structured thinking (ST) | MathVision −1.8% |

### Key Findings
- **Domain weight convergence**: M3CoT and FigureQA receive the highest weights (~1.5), while AI2D and IconQA receive the lowest (<0.8)—high-weight datasets require deeper reasoning, whereas low-weight datasets are overly simple.
- **Scalability**: Performance increases monotonically as the number of CoT candidates grows from 2 → 4 → 8, demonstrating that DreamPRM reliably selects high-quality candidates from larger pools.
- **Cross-model generalization**: A PRM trained on InternVL consistently improves performance when applied to GPT-4.1-mini and o4-mini.

## Highlights & Insights
- **First domain-reweighting framework for multimodal PRMs**: The paper innovatively transfers domain reweighting from pretraining to PRM training.
- **Elegant aggregation function loss design**: The upper-level optimization objective fully aligns with the inference-time usage of the PRM, effectively closing the training–inference gap.
- **Strong empirical performance**: MathVista leaderboard top-1, achieved at low computational cost (single GPU, 10 hours).
- **Cross-model generalization**: A PRM trained on a smaller model can enhance reasoning in larger models, offering high practical value.
- **Interpretable learned domain weights**: The weight distribution aligns intuitively with the difficulty and quality of each dataset.

## Limitations & Future Work
- The PRM backbone is relatively small (Qwen2-VL-2B); whether larger PRMs yield further gains remains to be validated.
- The meta-learning dataset is fixed as MMMU; sensitivity to this choice is not thoroughly analyzed.
- Domain weights operate at the dataset level; finer-grained sample-level reweighting may yield additional improvements.
- Evaluation is limited to the best-of-N inference paradigm; more complex search strategies such as MCTS are not explored.
- Inference requires generating multiple CoT candidates (8 in total), increasing computational cost approximately eightfold compared to zero-shot inference.

## Related Work & Insights
- DoReMi and DOGE on domain reweighting in pretraining provide direct inspiration for this work.
- Monte Carlo process supervision methods from Math-Shepherd and OmegaPRM form the foundation of the PRM training approach.
- Compared to heuristic data selection methods such as CaR-PRM and s1-PRM, the automated domain reweighting strategy demonstrates superior performance.
- This work is a valuable reference for test-time scaling research: PRM quality is a critical bottleneck for the effectiveness of test-time scaling.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of domain reweighting and bi-level optimization is not entirely novel, but its application to multimodal PRM training is a first, and the aggregation function loss design is noteworthy.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluation spans 5 benchmarks with multiple baselines, ablation studies, domain weight analysis, cross-model generalization, and leaderboard validation—highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ The structure is clear and mathematical derivations are complete, though some notation is dense.
- **Value**: ⭐⭐⭐⭐ Strong empirical results with MathVista top-1; the method is directly applicable to other PRM training scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[NeurIPS 2025\] Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning](stop_summation_minform_credit_assignment_is_all_process_rewa.md)
- [\[NeurIPS 2025\] Know What You Don't Know: Uncertainty Calibration of Process Reward Models](know_what_you_dont_know_uncertainty_calibration_of_process_reward_models.md)
- [\[NeurIPS 2025\] SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)
- [\[NeurIPS 2025\] ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs](reasonfluxprm_trajectoryaware_prms_for_long_chainofthought_r.md)

</div>

<!-- RELATED:END -->
