---
title: >-
  [Paper Note] Scientific Logicality Enriched Methodology for LLM Reasoning: A Practice in Physics
description: >-
  [ICML2026][LLM Reasoning][Scientific Logicality] This paper presents the first systematic study of "logicality" in LLM scientific reasoning. It proposes a three-dimensional evaluation metric—"Logical Fidelity / Causal Co…
tags:
  - "ICML2026"
  - "LLM Reasoning"
  - "Scientific Logicality"
  - "Logicality Evaluation"
  - "SFT Data Selection"
  - "Physics Reasoning"
  - "PhysLogic"
date: 2026-05-08
content_hash: 08ab94b3534b2c9a
---

# Scientific Logicality Enriched Methodology for LLM Reasoning: A Practice in Physics

**Conference**: ICML2026  
**arXiv**: [2605.17104](https://arxiv.org/abs/2605.17104)  
**Code**: https://github.com/ScienceOne-AI/PhysLogic  
**Area**: LLM Reasoning / Scientific Reasoning / Physics QA  
**Keywords**: Scientific Logicality, Logicality Evaluation, SFT Data Selection, Physics Reasoning, PhysLogic

## TL;DR
This paper presents the first systematic study of "logicality" in LLM scientific reasoning. It proposes a three-dimensional evaluation metric—"Logical Fidelity / Causal Connection / Inferential Progress"—and constructs two SFT data sampling methods based on these metrics: Reasoning Style Transfer (RST) and Logic-Distill. These methods significantly improve both the logicality and answer accuracy of 7B models on the self-developed PhysLogic benchmark and three public physics benchmarks.

## Background & Motivation

**Background**: Current research applying LLMs to scientific Q&A primarily focuses on "scaling data + scaling long CoT." By collecting large-scale math/physics/chemistry corpora containing long-chain thoughts for SFT or RL on reasoning models like DeepSeek-R1 and o1, these studies use final accuracy on QA benchmarks such as GPQA, SciBench, and PhysReason as the sole evaluation metric.

**Limitations of Prior Work**: The authors observe that while the Chain-of-Thought (CoT) of models like R1 on physics problems is lengthy, it is often a patchwork of "recall + repetition + self-reflection." It lacks the rigorous "logical chain" used by professionals, which involves problem formalization, model construction, evidence generation, evidence evaluation, and conclusion derivation. Figure 1 contrasts the CoT of R1 with a physicist's solution for the same problem, showing visible differences.

**Key Challenge**: The essence of scientific reasoning (logicality)—a set of concepts/methods/principles ensuring valid inference steps and reliable conclusions—is completely lost in existing "end-to-end NLP task" modeling. Relying solely on final answer correctness neither explains where a CoT fails nor guides training to improve reasoning quality.

**Goal**: (1) Establish a quantifiable scientific logicality evaluation method; (2) Construct high-logicality SFT data based on these metrics; (3) Verify whether stronger logicality translates into better problem-solving performance.

**Key Insight**: Drawing from Fischer et al.'s definition of "cognitive activities" in scientific inquiry, the authors decompose the solution process of a scientific problem into several "logical nexuses" $\mathcal{N}=\{\nu_1,\dots,\nu_n\}$ assigned with importance weights $\mathcal{W}=\{w_1,\dots,w_n\}$. The model's CoT is segmented into a sentence-level sequence $\mathcal{R}=\{r_1,\dots,r_m\}$. Both are embedded into a vector space using a sentence encoder, converting "logicality" into computable geometric relationships.

**Core Idea**: A similarity matrix $M\in\mathbb{R}^{n\times m}$ between "sentences vs. logical nodes" is used to simultaneously characterize *content coverage, causal order, and forward progress*, using this three-dimensional score as a signal for SFT data filtering.

## Method

### Overall Architecture
The methodology follows two main tracks. The first is "Evaluation": Given a scientific problem with ground-truth logical nodes $(\mathcal{N},\mathcal{W})$ and a model's reasoning $\mathcal{R}$, both are encoded into $V_\mathcal{N},V_\mathcal{R}$ using all-MiniLM-L6-v2 to compute a cosine similarity matrix $M$, which outputs three scores $\mathcal{F},\mathcal{O},\mathcal{P}$. The second is "Data": 380,000 physics papers are crawled from arXiv and journals $\rightarrow$ filtered to 118,000 by R1 to remove reviews/tools $\rightarrow$ R1 generates $(Q,R,A,\mathcal{N},\mathcal{W})$ quintuplets from derivation chains via multi-turn dialogue (rejection sampling ensures $A'=A$, max 5 retries) $\rightarrow$ 864 items are kept for the PhysLogic benchmark, while the remaining 80k + 40k follow two sampling strategies for SFT. Finally, full-parameter SFT is performed using LlamaFactory on 8×H100 across three backbones: Llama-3.1-8B, Qwen2.5-7B-Instruct, and DeepSeek-R1-Distill-Qwen-7B (lr $5\times10^{-6}$, cosine, 2 epochs, cutoff 32768), followed by closed-loop evaluation on PhysLogic and 3 public benchmarks.

### Key Designs

1.  **3D Scientific Logicality Metrics $\mathcal{F},\mathcal{O},\mathcal{P}$**:
    - **Function**: Decomposes "CoT logic" into independently computable components: content, order, and progress.
    - **Mechanism**: Greedy one-to-one matching on $M$ yields a set of pairs $\mathcal{C}$ (threshold $\tau$). **Logical Fidelity** is the harmonic mean $\mathcal{F}=2\pi\rho/(\pi+\rho)$ of weighted recall $\rho=\sum_{(i,j)\in\mathcal{C}} w_i M_{ij}/\sum_k w_k$ and precision $\pi=|\mathcal{C}|/m$. **Causal Connection** calculates a "semantic centroid" $P_i=\sum_j j\cdot M_{ij}/\sum_j M_{ij}$ for each $\nu_i$, then computes the weighted proportion $\mathcal{O}$ of nexus pairs satisfying the ground-truth order. **Inferential Progress** represents each step $r_j$ as a similarity vector $\vec{S_j}$ to all nexuses, defining novelty as $1-\max_{k<j}\cos(\vec{S_j},\vec{S_k})$, with $\mathcal{P}$ being the average novelty along the path.
    - **Design Motivation**: Traditional accuracy only judges the final answer, and process metrics (e.g., ProcessBench) only look at local correctness; neither distinguishes "correct answer via flawed reasoning" from "correct and rigorous reasoning." The 3D decomposition attributes failures to "missing steps," "inverted order," or "self-looping," providing interpretable multi-objective signals for data filtering.

2.  **Reasoning Style Transfer (RST) Data Sampling**:
    - **Function**: Translates "discrete logical nodes + weights" into a coherent, first-person, scientist-style CoT with `<think>` tags, paired with the original question and answer as SFT samples.
    - **Mechanism**: A powerful reasoning LLM $\mathcal{L}$ performs style transfer: $R'=\mathcal{L}(Q,\mathcal{N},\mathcal{W})$, resulting in samples $\{Q,R',A\}$. Note that answer $A$ comes from the original paper, not model generation.
    - **Design Motivation**: Direct distillation from R1's native CoT (Direct-Distill) risks inheriting bad habits like "repetition + self-doubt." Conversely, simply listing nexuses as bullets lacks the flow of natural CoT. RST yields samples with "paper-logic skeletons and strong-model skin," ensuring logical rigor while maintaining a natural thinking style. Table 5 shows this achieves the highest in-domain logicality and accuracy across all three backbones.

3.  **Logic-Distill: Filtering Strong Model Native CoT with 3D Scores**:
    - **Function**: Ranks R1's own generated CoTs by a "scientific logicality total score" $\mathcal{S}$ without relying on ground-truth nexus rewriting, keeping the top-$\kappa$ percentile for SFT.
    - **Mechanism**: $\mathcal{L}$ reasons on $Q$ to get $(R',A')$ and calculates $\pi,\rho,\mathcal{O},\mathcal{P}$ for $R'$. To eliminate scale differences, scores are z-score normalized and passed through a sigmoid: $\tilde X=\sigma((X-\mu_X)/\sigma_X)$. These are fused into $\mathcal{S}=\delta_\mathcal{F}\cdot 2\tilde\pi\tilde\rho/(\tilde\pi+\tilde\rho)+\delta_\mathcal{O}\tilde{\mathcal{O}}+\delta_\mathcal{P}\tilde{\mathcal{P}}$, where $D=\mathrm{Top}_\kappa(D_{\text{full}}, \mathrm{key}=\mathcal{S})$.
    - **Design Motivation**: RST requires ground-truth nexuses, which is computationally expensive. Logic-Distill treats the 3D metrics as weak supervision signals for sample selection without re-generation. This selects the ~40k "accidentally logically rigorous" CoTs from a massive pool, approaching full-scale distillation performance with half the data, making it highly scalable to other disciplines.

### Loss & Training
Standard SFT cross-entropy is used without auxiliary losses; logicality is injected solely through data selection. Training utilizes LlamaFactory for full-parameter fine-tuning with BF16 + DeepSpeed ZeRO-3 + FlashAttention-2 + gradient checkpointing. Per-device batch=1, grad accum=2, cutoff 32768, 2 epochs, seed 42, and 0.03 warmup on 8×H100.

## Key Experimental Results

### Main Results (In-domain, PhysLogic benchmark)
Improvement of RST over the strongest baseline across three backbones (averages of $\mathcal{F},\mathcal{O},\mathcal{P}$ and final Acc):

| Backbone | Best Baseline | Avg Logicality Δ | Acc Δ |
|----------|---------------|------------------|-------|
| Llama-3.1-8B | MegaScience (42.35 / 31.02) | **+3.59** → 45.94 | **+13.65** → 44.67 |
| Qwen2.5-7B-Instruct | MegaScience (43.12 / 39.81) | **+1.94** → 45.06 | **+3.01** → 42.82 |
| R1-Distill-Qwen-7B | SCP-116k (42.68 / 46.30) | **+3.30** → 45.98 | **+1.15** → 47.45 |

Average accuracy on three Out-of-Domain (OOD) public physics benchmarks (GPQA-physics / SciBench-physics / PhysReason):

| Backbone | Best Baseline | Ours Logic-Distill (40k) | Ours RST (80k) |
|----------|---------------|--------------------------|----------------|
| Llama-3.1-8B | SCP-116k 35.08 | 35.14 | 30.98 |
| Qwen2.5-7B-Instruct | SCP-116k 34.72 | **45.04** (+10.32) | 41.41 |
| R1-Distill-Qwen-7B | SCP-116k 47.34 | **53.42** (+6.08) | 52.26 |

### Ablation Study (Contribution of components in Logic-Distill)

| Configuration | Llama Logic/Acc | Qwen Logic/Acc | R1-7B Logic/Acc |
|---------------|-----------------|----------------|-----------------|
| Logic-Distill (full) | 45.50 / 36.54 | 42.78 / 44.02 | 44.14 / 49.73 |
| w/o $\mathcal{F}$ | -1.60 / -2.69 | -2.75 / -3.43 | -2.56 / -4.65 |
| w/o $\mathcal{O}$ | -1.85 / **-5.23** | -4.43 / **-5.77** | -2.76 / **-11.05** |
| w/o $\mathcal{P}$ | -1.44 / -2.82 | -2.01 / -5.30 | -2.45 / -4.55 |
| Random Sampling | -1.83 / -4.61 | -1.03 / -15.25 | -3.90 / -7.95 |

### Key Findings
- Among the three dimensions, **removing Causal Connection $\mathcal{O}$ caused the most severe Acc drop** (Llama -5.23, Qwen -5.77, R1-7B -11.05), indicating that "step order" is the most critical process signal for physics problem-solving.
- Third-party consistency verification (Table 3): $\mathcal{F},\mathcal{O},\mathcal{P}$ show Pearson correlations of 0.69–0.83 ($p < 0.001$) with human physics experts and GPT-5 scores, proving the metrics are not arbitrary.
- Significant gaps in 3D metrics exist between correct and incorrect answer groups for the same problem (Table 4; Avg. 43.9 vs 39.7, $p < 0.001$), empirically validating "higher logicality $\rightarrow$ higher accuracy."
- **Data Efficiency**: Logic-Distill achieved the best OOD results on Qwen and R1-7B using only 40k samples, outperforming the 80k RST. This suggests that "selection precision" yields higher returns than "selection volume" in scientific reasoning.
- On PhysReason, the SFT 7B model surpassed 14B/32B counterparts and even topped the average logicality leaderboard for closed-source models (Appendix Figures 5–8).

## Highlights & Insights
- **Geometric logic visualization is ingenious**: Using sentence embeddings and a similarity matrix to unify "coverage, order, and progress" into a single $M$ provides an interpretable and low-cost evaluation paradigm (using only all-MiniLM-L6-v2), replacing symbolic logic checks with embedding space geometry.
- **Process metrics serve as both evaluation and data selection signals**: Logic-Distill uses $\mathcal{F},\mathcal{O},\mathcal{P}$ as a weak-supervision ranker. This equates to "the ability to pick correctly solved problems without knowing the solution," providing a template for process-aware data curation transferable to math, chemistry, biology, and code.
- **Causal Connection is an underrated process signal**: Ablations show removing $\mathcal{O}$ hurts accuracy more than removing $\mathcal{F}$. This implies many physics errors stem from "disordered steps or reversed causality" rather than "missing steps," suggesting a priority direction for process-reward or step-level RL.
- **Discipline-specific weights $\mathcal{W}$** explicitly model which nexuses are more important, avoiding the simplification of "equal weight for every step" and offering potential for cross-disciplinary generalization.

## Limitations & Future Work
- The evaluation's ground truth $(\mathcal{N},\mathcal{W})$ is automatically generated by DeepSeek-R1, creating a potential "self-scoring bias." although validated on 200 samples by experts, this is small compared to the 80k training set.
- Sentence embeddings + cosine similarity struggle to distinguish "visually similar but symbolically wrong" cases (e.g., $E=mc^2$ vs $E=mc^3$). Fine-grained correctness in physics reasoning still requires symbolic or numerical verification.
- The work is limited to physics, with training data sourced entirely from arXiv "derivation chains," limiting coverage of experimental, engineering, or non-deductive problems (e.g., experiment design).
- RST and Logic-Distill were not directly compared with RL approaches (PRM, GRPO). A natural future direction is using the 3D metrics as process rewards for direct RL optimization.

## Related Work & Insights
- **vs Direct-Distill / MegaScience / SCP-116k / Sci-Instruct**: These baselines focus on "more and longer CoT." This paper prioritizes "which CoTs are actually logical," significantly outperforming them with equal or half the data volume.
- **vs PhysReason / PRISM-PHYSICS**: Existing physics benchmarks focus on "step correctness." PhysLogic is the first to cover "steps, order, and progress" across high school to graduate levels and four problem types (MCQ, expression, numerical, proof).
- **vs ProcessBench / PRM**: While those works label 0/1 for each math step, this study treats "process quality" as a continuous multi-dimensional signal and uses it for data selection, offering an executable paradigm for process-aware data curation.

## Rating
- Novelty: ⭐⭐⭐⭐ 3D logicality metrics + reusing the evaluator as a data filter is quite novel for scientific reasoning; construction leans towards engineering integration.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered 3 backbones, in-domain + 3 OOD benchmarks, third-party consistency, full ablations, and data scale comparisons.
- Writing Quality: ⭐⭐⭐⭐ Consistent notation; Figures 1-3 clearly explain motivation and metrics. Slightly formula-dense, making it a bit high-threshold for non-physics readers.
- Value: ⭐⭐⭐⭐ Provides interpretable, reusable process signals for scientific LLM training. Open-sourcing PhysLogic and its training set will benefit the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Scaling-Aware Adapter for Structure-Grounded LLM Reasoning](scaling-aware_adapter_for_structure-grounded_llm_reasoning.md)
- [\[ICLR 2026\] Nudging the Boundaries of LLM Reasoning](../../ICLR2026/llm_reasoning/nudging_the_boundaries_of_llm_reasoning.md)
- [\[ICML 2026\] R2-Router: A New Paradigm for LLM Routing with Reasoning](r2-router_a_new_paradigm_for_llm_routing_with_reasoning.md)
- [\[ICML 2026\] Are Tools Always Beneficial? Learning to Invoke Tools Adaptively for Dual-Mode Multimodal LLM Reasoning](are_tools_always_beneficial_learning_to_invoke_tools_adaptively_for_dual-mode_mu.md)
- [\[ICML 2026\] Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning](beyond_two-stage_training_cooperative_sft_and_rl_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->
