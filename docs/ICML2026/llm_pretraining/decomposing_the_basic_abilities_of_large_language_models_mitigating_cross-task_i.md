---
title: >-
  [Paper Note] Decomposing the Basic Abilities of Large Language Models: Mitigating Cross-Task Interference in Multi-Task Instruct-Tuning
description: >-
  [ICML 2026][LLM Pretraining][Multi-Task Instruct-Tuning] This paper addresses the issue of cross-task gradient conflict in multi-task instruct-tuning by proposing Badit: first…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Multi-Task Instruct-Tuning"
  - "Cross-Task Interference"
  - "SVD-LoRA"
  - "Spherical Clustering"
  - "Orthogonal Basic Abilities"
date: 2026-05-08
content_hash: d769944973bcdeba
---

# Decomposing the Basic Abilities of Large Language Models: Mitigating Cross-Task Interference in Multi-Task Instruct-Tuning

**Conference**: ICML 2026  
**arXiv**: [2605.05676](https://arxiv.org/abs/2605.05676)  
**Code**: https://github.com/wangbing1416/BADIT  
**Area**: LLM Efficiency / Multi-Task Fine-Tuning / LoRA / MoE  
**Keywords**: Multi-Task Instruct-Tuning, Cross-Task Interference, SVD-LoRA, Spherical Clustering, Orthogonal Basic Abilities

## TL;DR
This paper addresses the issue of cross-task gradient conflict in multi-task instruct-tuning by proposing Badit: first, SVD is used to decompose pretrained weights into a set of naturally orthogonal, high-singular-value LoRA "basic ability" experts; then, during training, spherical K-means is used to dynamically orthogonally group rank-1 components, shifting the traditional paradigm of "parameter isolation by task" to "decoupling by basic ability." On six LLMs, Badit achieves an average improvement of 2.68 Rouge over GainLoRA.

## Background & Motivation
**Background**: The strong multi-task capabilities of current LLMs mainly rely on multi-task instruct-tuning, but multi-task training inevitably leads to *cross-task interference*—different tasks generate opposing gradients on shared parameters, overwriting each other. Existing solutions fall into two categories: (1) task-specific neuron/parameter selection (e.g., Leng & Xiong 2025), using gradient attribution to identify a parameter subset unique to each task and only training those; (2) MoE-style expert isolation (LoRAMoE, OLoRA, etc.), assigning independent LoRA experts to each task and imposing orthogonality constraints.

**Limitations of Prior Work**: Empirical analysis on 15 SuperNI tasks (Fig. 1, Fig. 2) reveals that both approaches fail: the vast majority of "task-activated neurons/parameters" or "experts routed in MoE" are **activated by multiple tasks simultaneously**—some neurons are even shared by all 15 tasks. In other words, the so-called "task-specificity" is an illusion; cross-task interference is never truly eliminated.

**Key Challenge**: Using "task" as the isolation granularity is inherently flawed—tasks are surface-level labels, not functionally distinct units at the parameter level. Forcibly grouping by task leads to parameter sharing due to overlapping abilities among tasks.

**Goal**: (1) Identify a more fundamental, truly separable unit of ability than "task"; (2) Design an MoE structure that maintains orthogonality among these units throughout training.

**Key Insight**: Fig. 1 also reveals a reverse signal—"certain neurons/parameters are consistently co-activated by multiple tasks," and these co-activation sets recur, forming a small number of "base groups." The authors propose a metaphor: LLMs encode several **orthogonal basic abilities**, and each task is a linear combination of these abilities. Therefore, parameters should be isolated by "basic ability" rather than by "task."

**Core Idea**: Use SVD to decompose pretrained weights into multiple naturally orthogonal LoRA experts (each corresponding to a basic ability), and during training, use spherical clustering to continually regroup rank-1 components, enforcing orthogonality among expert gradient directions.

## Method

### Overall Architecture
Badit rewrites each LLM weight matrix $\mathbf{W}^0\in\mathbb{R}^{m\times n}$ as $\mathbf{W}^0 = \sum_{k=1}^{K}\alpha_k \mathbf{A}_k\mathbf{B}_k + \widehat{\mathbf{W}}$: $K$ LoRA experts cover the top $rK$ singular values (each with rank $r$), and the residual $\widehat{\mathbf{W}}$ contains the remaining small singular values. Each expert is interpreted as a "basic ability" expert, with a learnable routing coefficient $\alpha_k$; the residual is frozen throughout fine-tuning. The process consists of two steps: **BAD** provides the initial orthogonal decomposition, and **DOG** maintains orthogonality during training. The final routing weights allow the model to adaptively mix these basic abilities per task.

### Key Designs

1. **Basic Ability Decomposition (BAD)**:

    - **Function**: For each pretrained weight $\mathbf{W}^0$, perform SVD, take the top $rK$ singular values/vectors, and split them into $K$ rank-$r$ LoRA experts as the initialization of basic abilities.
    - **Mechanism**: $\mathbf{U}_{[:rK]}\boldsymbol{\Sigma}_{[:rK]}\mathbf{V}_{[:rK]}^\top$ is partitioned column-wise into $K$ segments; the $k$-th segment constructs $\mathbf{A}_k = \mathbf{U}_{[r(k-1):rk]}\,\mathrm{diag}(\sqrt{\boldsymbol{\Sigma}_{[r(k-1):rk]}})$ and similarly for $\mathbf{B}_k$; the remaining low singular values form the residual, which is frozen. The paper proves (Appendix A) that these $K$ LoRA experts are **naturally pairwise orthogonal** at initialization—SVD's left/right singular vectors are in different subspaces, so each expert corresponds to a completely non-overlapping "ability direction." Each expert is assigned a learnable routing $\alpha_k$ (initialized to 1) for MoE compatibility.
    - **Design Motivation**: Compared to traditional LoRA (zero matrix initialization) and traditional MoE (random expert-task assignment), BAD's key insight is that "basic abilities are not assigned post hoc, but are inherent directions in pretrained weights." SVD provides an orthogonal basic ability dictionary without training, shifting the burden of "expert differentiation" from training to initialization.

2. **Dynamically Orthogonal Grouping (DOG)**:

    - **Function**: Gradient updates during training destroy expert orthogonality. DOG periodically regroups the $rK$ rank-1 components so that the new $K$ LoRA experts' gradient directions are again orthogonal.
    - **Mechanism**: Each rank-1 component's concatenated gradient $[\mathbf{a}_i;\mathbf{b}_i^\top]$ is normalized to $\widehat{\mathbf{g}}_i$. The goal is to find a 0/1 assignment matrix $\boldsymbol{\Pi}\in\{0,1\}^{rK\times K}$ (each new expert gets exactly $r$ components), maximizing "intra-expert gradient similarity and inter-expert orthogonality": $\max_{\boldsymbol{\Pi}}\sum_k \|\sum_i \pi_{i,k}\widehat{\mathbf{g}}_i\|^2$. This is solved iteratively: (1) Spherical K-means gives an initial assignment; (2) compute each cluster centroid $\mathbf{c}_k^{(\tau)}=\sum_i \pi_{i,k}^{(\tau)}\widehat{\mathbf{g}}_i$, perform SVD on the centroid matrix $\mathbf{C}^{(\tau)}=\mathbf{U}_c\boldsymbol{\Sigma}_c\mathbf{V}_c^\top$ to get $\mathbf{Q}^{(\tau)}=\mathbf{U}_c\mathbf{V}_c^\top$, which are the enforced orthogonal target directions; (3) update $\boldsymbol{\Pi}$ by solving an integer assignment problem with the constraint $\sum_i \pi_{i,k}=r$ based on $\langle \widehat{\mathbf{g}}_i, \mathbf{q}_k^{(\tau)}\rangle$, iterating up to 10 steps or until convergence. Appendix C proves that MoE output is mathematically invariant before and after regrouping—crucial to avoid model disturbance from each regrouping.
    - **Design Motivation**: SVD initialization orthogonality only holds at $t=0$; gradients quickly drift (Fig. 4 shows LoRAMoE's inter-expert angles deviate from 90° over time). DOG's "cluster then orthogonalize" approach avoids the pitfalls of directly penalizing gradient orthogonality (which distorts loss geometry), instead recasting orthogonality as a discrete optimization over direction grouping—maintaining end-to-end trainability and stably locking inter-expert angles near 90°, intra-expert angles near 60°.

### Loss & Training
Standard SFT loss $\mathcal{L}(\mathcal{F}(\mathbf{x};\boldsymbol{\theta}), y)$ is used, with no additional orthogonality regularization. Number of experts $K=8$, rank $r$ matches LoRAMoE. DOG is triggered periodically during training (clustering and integer optimization on CPU). Two evaluation paradigms: mixed training (15 tasks jointly) and sequential training (tasks trained in sequence, averaged over 5 orders, focusing on forgetting rate).

## Key Experimental Results

### Main Results
On 15 SuperNI tasks and 6 LLMs (Qwen3-8B/4B, Llama3-8B/3B, Gemma2-9B/2B), five baselines are compared. The table below summarizes Qwen3-8B and Llama3-8B results for both mixed and sequential paradigms:

| Model / Setting | Method | Mixed Rouge↑ | Seq Rouge↑ | Seq Forget Rate↓ | Seq Backward↑ |
|------|------|------|------|------|------|
| Qwen3-8B | LoRA | 54.22 | 47.08 | 9.21 | -8.11 |
| Qwen3-8B | LoRAMoE | 54.64 | 48.07 | 8.47 | -6.23 |
| Qwen3-8B | GainLoRA (SOTA) | 54.33 | 48.44 | 8.96 | -6.42 |
| Qwen3-8B | **Badit** | **55.87** | **50.86** | **6.95** | **-5.43** |
| Llama3-8B | LoRA | 52.58 | 44.88 | 12.41 | -10.23 |
| Llama3-8B | GainLoRA | 52.71 | 45.04 | 12.70 | -10.94 |
| Llama3-8B | **Badit** | **54.75** | **48.83** | **8.57** | **-3.49** |

Badit outperforms GainLoRA by an average of **2.68 Rouge** across 6 LLMs, with the best forgetting rate and backward transfer. On Llama3-8B, forget rate drops from over 12 to 8.57, and backward improves from -10 to -3.49, indicating catastrophic forgetting is significantly mitigated in sequential scenarios.

### Ablation Study
The authors ablate BAD and DOG to assess their individual contributions (Δ is the total drop relative to full Badit):

| Model | Config | Seq Rouge | Seq Forget | Mixed Rouge | Δ |
|------|------|-----------|------------|-------------|---|
| Qwen3 | Badit (full) | 50.86 | 6.95 | 55.87 | - |
| Qwen3 | w/o BAD | 50.07 | 7.34 | 55.02 | 2.03 |
| Qwen3 | w/o DOG | 49.70 | 8.20 | 54.85 | 3.43 |
| Qwen3 | w/o BAD & DOG | 48.07 | 8.47 | 54.64 | 5.54 |
| Llama3 | Badit | 48.83 | 8.57 | 54.75 | - |
| Llama3 | w/o BAD | 47.92 | 9.30 | 54.02 | 2.37 |
| Llama3 | w/o DOG | 47.33 | 9.74 | 53.74 | 3.68 |

### Key Findings
- **DOG is more important than BAD**: SVD initialization alone (w/o DOG) leads to a larger drop than dynamic grouping alone (w/o BAD), indicating that "maintaining orthogonality throughout training" is more critical than "good initialization"; initial orthogonality is diluted after a few epochs.
- **Basic ability hypothesis directly validated by Fig. 4**: Badit's inter-expert gradient angles remain close to 90°, intra-expert angles around 60° (about 20° lower than LoRAMoE), confirming "inter-expert orthogonality, intra-expert consistency."
- **Acceptable overhead**: Table 3 shows Badit's total training time is about $1.22\times$ that of LoRAMoE, with the extra cost mainly from DOG's spherical K-means and integer optimization (run on CPU); BAD may even accelerate convergence.

## Highlights & Insights
- **Shifting isolation granularity from "task" to "ability"**: This is a conceptual leap. Traditional MoE assumes each task corresponds to an expert; the authors argue that tasks are linear combinations of abilities, and abilities are the atomic units. This perspective directly explains why task-based isolation always fails—tasks overlap in ability space.
- **Using SVD as a "training-free expert differentiator"**: The challenge of "expert differentiation," which typically requires special loss terms, is shifted to a pure linear algebra operation (SVD), which is both efficient and provably orthogonal—an advance beyond PiSSA.
- **DOG's "cluster then orthogonalize" approach is transferable**: Directly penalizing gradient orthogonality often destabilizes training; DOG uses K-means to group "nearby directions" and then orthogonalizes the groups, followed by discrete assignment—this "soft grouping + hard orthogonalization" can be generalized to any scenario requiring direction decoupling (continual learning, multi-domain adaptation, vision MoE, etc.).
- **Clustering + integer assignment + invariance proof**: Appendix C proves MoE output is invariant before and after regrouping, which is crucial; otherwise, dynamic rearrangement would "swap heads" each time. This "surgical intervention on representations while maintaining end-to-end equivalence" is a valuable design.

## Limitations & Future Work
- The authors acknowledge DOG introduces a 1.22× training cost, but do not provide scaling curves for larger batch sizes or $K$; this overhead may increase in industrial settings.
- Spherical K-means and integer assignment run on CPU, creating a synchronization bottleneck for GPU-bound training; more efficient GPU implementations or approximate algorithms are needed.
- The assumption that "basic ability = SVD principal singular direction" is mathematically elegant, but whether these abilities correspond to interpretable semantics (e.g., arithmetic, reading comprehension) is not further validated—there may be a mismatch between "high singular value direction ≠ semantic basic ability."
- Only validated on 15-task SuperNI; whether larger $K$ is needed for more tasks, cross-lingual/cross-modal settings, or if abilities saturate, remains open.
- The residual $\widehat{\mathbf{W}}$ is completely frozen, discarding small singular value directions, which may be a hidden bottleneck for certain long-tail tasks; lightweight adaptation for the residual could be considered.

## Related Work & Insights
- **vs LoRAMoE / OLoRA**: These use randomly initialized LoRA experts and route by task, maintaining orthogonality via training penalties; Badit derives experts from SVD and maintains orthogonality via periodic regrouping. Advantages: orthogonal from the start, controllable throughout training; disadvantage: DOG adds ~22% training time.
- **vs PiSSA**: PiSSA is single-expert SVD-LoRA, taking only the top singular values; Badit segments the SVD spectrum into $K$ parts for multiple experts, moving from single-expert acceleration to multi-expert decoupling.
- **vs GainLoRA**: GainLoRA is also a SOTA MoE approach but still allocates parameters by task; the authors empirically refute its implicit "task separability" assumption, achieving a stable 2.68 Rouge improvement.
- **Insights**: This two-stage approach—using algebraic structure (SVD/PCA) for training-free decoupled initialization, then maintaining properties dynamically—can be transferred to continual learning, domain adaptation, multi-modal expert routing, and other tasks requiring subspace decoupling.

## Rating
- Novelty: ⭐⭐⭐⭐ Shifting "isolation granularity" from task to basic ability is an interesting conceptual change; the SVD + dynamic clustering combination is also relatively new, though building blocks like PiSSA and OLoRA already exist—this is "putting the right blocks together in the right place."
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 LLMs × 15 tasks × 2 training paradigms × 5 seeds, with broad coverage; includes auxiliary analyses such as gradient angle dynamics and time overhead.
- Writing Quality: ⭐⭐⭐⭐ Motivation is introduced via counterintuitive findings in Fig. 1/2, with a clear logical chain; method section is formula-dense but well-structured.
- Value: ⭐⭐⭐⭐ Multi-task instruct-tuning is a core stage in LLM post-training; an average 2.68 Rouge improvement is directly attractive for industrial training pipelines, and code is open-sourced; the extra 22% training time is a cost to weigh for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)
- [\[ICML 2026\] On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](on_training_large_language_models_for_long-horizon_tasks_an_empirical_study_of_h.md)
- [\[NeurIPS 2025\] Quantifying Task-Relevant Representational Similarity Using Decision Variable Correlation](../../NeurIPS2025/llm_pretraining/quantifying_task-relevant_representational_similarity_using_decision_variable_co.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)

</div>

<!-- RELATED:END -->
