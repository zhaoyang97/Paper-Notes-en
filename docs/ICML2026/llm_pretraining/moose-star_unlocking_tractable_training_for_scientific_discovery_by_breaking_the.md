---
title: >-
  [Paper Note] MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier
description: >-
  [ICML 2026][LLM Pretraining][Hypothesis generation] MOOSE-Star decomposes the problem of "training an LLM to directly generate scientific hypotheses"—originally a search in an $\mathcal{O}(N^k)$ combinatorial space—into…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Hypothesis generation"
  - "inspiration retrieval"
  - "hierarchical search"
  - "tractable training"
  - "TOMATO-Star"
date: 2026-05-08
content_hash: 006179cc3ee3f993
---

# MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier

**Conference**: ICML 2026  
**arXiv**: [2603.03756](https://arxiv.org/abs/2603.03756)  
**Code**: https://github.com/ZonglinY/MOOSE-Star (Available)  
**Area**: LLM Reasoning / Scientific Discovery / Decomposed Training  
**Keywords**: Hypothesis generation, inspiration retrieval, hierarchical search, tractable training, TOMATO-Star

## TL;DR
MOOSE-Star decomposes the problem of "training an LLM to directly generate scientific hypotheses"—originally a search in an $\mathcal{O}(N^k)$ combinatorial space—into two sequential subtasks: "inspiration retrieval + hypothesis synthesis." By layering hierarchical tree retrieval, bounded composition, and motivation planning, the optimal complexity is reduced from exponential to $\mathcal{O}(\log N)$. The authors also release the TOMATO-Star dataset, which includes 108,717 papers with decomposed annotations.

## Background & Motivation
**Background**: Most work on LLMs for scientific discovery focuses on "how to use LLMs at inference time" or "fine-tuning with external feedback" (e.g., reviewer feedback, rule-based scores, or reward alignment with data). Modeling and training the core conditional distribution $P(\text{hypothesis}\mid\text{background})$ directly remains largely unexplored.

**Limitations of Prior Work**: The authors theoretically demonstrate that directly training $P(h\mid b)$ implicitly requires identifying the correct sequence of $k$ inspirations within a global scientific literature corpus of $N\approx 10^7$. The search space is $\mathcal{O}(N^k)$ (e.g., $\approx 10^{21}$ for $N=10^7, k=3$), creating a "combinatorial complexity wall" that makes end-to-end training mathematically ill-posed.

**Key Challenge**: Either abandon the direct modeling of $P(h\mid b)$ (the current feedback-based route) or confront combinatorial complexity (infeasible). Both paths are problematic.

**Goal**: To directly model $P(h\mid b)$ while reducing training complexity to a level manageable by modern computing power, and to provide a reproducible dataset and open-source code.

**Key Insight**: Leveraging the probability decomposition theorem from MOOSE-Chem, $P(h\mid b)\approx \prod_j P(i_j\mid b,h_{j-1},\mathcal{I})\cdot P(h_j\mid b,h_{j-1},i_j)$ is viewed as a sequence of "inspiration retrieval + incremental synthesis." This decomposition, previously used only for inference, is upgraded in this work to a trainable objective.

**Core Idea**: The untrainable $\mathcal{O}(N^k)$ problem is reduced to trainable $\mathcal{O}(k\cdot N)$ sequential subtasks. Subsequently, hierarchical tree search, bounded composition, and motivation planning are used to further compress the retrieval complexity from $\mathcal{O}(N)$ to $\mathcal{O}(\log N)$.

## Method

### Overall Architecture
The training framework consists of three layers: (1) **Data**: 108,717 open papers from 2020–2025 are deconstructed using R1 / R1-distill-Qwen into (research background $b$, hypothesis $h$, inspirations $\{i_j\}$) triplets. $h$ is split into $\Delta h_1,\ldots,\Delta h_k$, each structured into (Motivation, Mechanism, Methodology) layers. (2) **Model**: $P(h\mid b)$ is split into "Inspiration Retrieval (IR)" and "Hypothesis Composition (HC)" tasks for Rejection Sampling Fine-Tuning (RFT). IR uses a pool of 1 positive and 14 hard negative samples; HC uses a rubric-based evaluator for rejection sampling. (3) **Inference**: The literature corpus is organized into a semantic retrieval tree. Motivation variables are used to dynamically prune irrelevant subtrees. HC is trained within a bounded tolerance radius $M$ to ensure robustness against retrieval errors.

### Key Designs

1.  **Decomposed Sequential Training (IR + HC)**:
    - **Function**: Replaces "end-to-end learning of $P(h\mid b)$" with "learning retrieval followed by learning incremental synthesis," repeated $k$ times.
    - **Mechanism**: Following the chain rule, $P(h\mid b)\approx \prod_{j=1}^{k} P(i_j\mid b,h_{j-1},\mathcal{I})\cdot P(h_j\mid b,h_{j-1},i_j)$ is decomposed. The IR task involves generatively selecting the most relevant paper from 15 candidates (input: title + abstract; output: CoT reasoning + selection). The HC task involves writing the incremental hypothesis $\Delta h_j$ given the ground-truth $i_j$. Both utilize teacher-based RFT. The overall complexity becomes $\mathcal{O}(k\cdot(N+1))$ instead of $\mathcal{O}(N^k)$.
    - **Design Motivation**: Converting an exponential Cartesian product into $k$ linear sums is the key step to bringing the untrainable problem into a trainable domain. IR and HC are clear, supervised, and measurable tasks, making them more stable than globally scoring $h$.

2.  **Bounded Composition**:
    - **Function**: Ensures the HC model is robust when the retrieved inspiration is not exactly the ground-truth $i^*$.
    - **Mechanism**: A semantic tolerance neighborhood $\mathcal{I}_{i^*}\subset\mathcal{I}$ of size $M$ centered at $i^*$ is defined. During training, "approximate inspirations" are sampled from this neighborhood to feed the HC model, teaching it to synthesize valid $\Delta h_j$ from adjacent concepts. This relaxes the retrieval precision requirement from "1/N exact match" to "1/(N/M) fuzzy match," further compressing the effective search space for IR.
    - **Design Motivation**: Even if hierarchical tree retrieval reaches $\mathcal{O}(\log N)$, the final layer may not be perfectly accurate. Bounded composition explicitly models "retrieval error" into the training distribution, similar to noise-aware training, preventing pipeline collapse under real-world noise.

3.  **Motivation-Guided Hierarchical Search**:
    - **Function**: Replaces "linear scanning of $N$ papers" with "top-down $\log N$ steps through a semantic tree," using motivation variables for pruning.
    - **Mechanism**: Literature is clustered into a semantic retrieval tree. At each step, the most relevant branch is selected among children nodes; ideally, search depth is $\mathcal{O}(\log N)$. An explicit motivation variable $m$ (from the Motivation layer of $\Delta h$) is attached to the background, acting as a "generative root" to dynamically prune subtrees irrelevant to the current target, reducing the searchable space from $N$ to $N_m\ll N$.
    - **Design Motivation**: A semantic tree alone only reduces retrieval steps, but "which subtree to search" remains open. The motivation variable provides a generative direction control signal, allowing the model to truly scale during inference.

### Loss & Training
Both IR and HC move through Rejection Sampling Fine-Tuning (RFT) with CoT supervision: for each sample, $N$ CoT trajectories are sampled, and low-quality ones are filtered via a rubric evaluator. The HC rubric checks the Motivation, Mechanism, and Methodology layers. The TOMATO-Star dataset was archived only after passing four automated quality checks: necessity, sufficiency, mutual exclusivity, and non-redundancy.

## Key Experimental Results

### Main Results

| Dimension | Configuration | Highlight |
|------|------|---------|
| Data Scale | 108,717 open papers, 38,400 GPU hours | Training: 2020-09 to 2025; Test: 2025-10 (No temporal leakage) |
| Complexity (Worst → Best) | $\mathcal{O}(N^k)$ → $\mathcal{O}(\log N)$ | Compressed via IR/HC decomposition + tree retrieval + motivation pruning |
| Test-time scaling | brute-force vs. MOOSE-Star | Brute-force hits the "complexity wall" quickly in multi-inspiration tasks; MOOSE-Star success rate rises with computation budget |
| Inspiration Hit Rate | IR vs. random/nearest neighbor | Generative selection + CoT supervision significantly outperforms baselines (see § F) |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| W/O Bounded Composition ($M=1$) | HC sensitivity to noise increases | Validates the need for synthesis under imperfect retrieval |
| W/O Motivation Variable | Tree search path lengthens | Motivation is key to effective pruning efficacy |
| End-to-end $P(h\mid b)$ (baseline) | Fails to converge | § 7.1 shows distillation of reasoning traces directly for $b\to h$ is infeasible |
| Brute-force test-time sampling | Hits "complexity wall" | Highlights the scaling capability of the hierarchical search in MOOSE-Star |

### Key Findings
- The failure of direct $P(h\mid b)$ training stems from the implicit combinatorial search space being too large, not from a lack of data or model size. This serves as a fundamental critique of the "feedback-driven discovery" route.
- Decomposing an exponential problem into hierarchical, tolerant, and pruned steps is a transferable paradigm: solving one complexity bottleneck at a time allows reduction from $N^k$ to $\log N$.
- The structure of TOMATO-Star—(b, h, i) + (Motivation, Mechanism, Methodology)—represents an upgrade in discovery dataset design, moving beyond "abstract-style hypotheses."
- Strict temporal splitting (test sets after 2025-10) prevents pre-training contamination, a practice recommended for future LLM evaluation.

## Highlights & Insights
- Provides a rigorous complexity argument ($\mathcal{O}(N^k)$ vs $\mathcal{O}(\log N)$) for why $P(h|b)$ is hard to train, upgrading "LLM for science" from engineering papers to a theoretically grounded research direction.
- Reinterpreting inference-time probability decomposition as a training objective is the most significant leap since MOOSE-Chem, conceptually similar to breaking the Bellman equation into TD-updates in RL.
- Bounded Composition relaxes the "retrieval-synthesis" alignment to "neighborhood tolerance," an engineering posture toward real-world retrieval noise that aligns with RAG practices.
- By providing 108k structured papers, a full training/inference suite, and models, the work sets a high bar for reproducibility in scientific discovery.

## Limitations & Future Work
- The system still relies on "author citations = ground-truth inspirations," which may bias toward explicitly cited influences and ignore "uncited but profoundly influential" inspirations.
- The 1 positive + 14 negative IR setting is a limited approximation; the real library is much larger. Hierarchical search lacks a self-correction mechanism if the root node is misidentified.
- The tolerance radius $M$ in Bounded Composition is a hyperparameter; too small leads to exact match issues, while too large risks uncontrolled generalization. No systematic strategy for choosing $M$ is provided.
- Validation was primarily in biology, chemistry, and medicine. Effectiveness in fields like ML/CS, with denser citation structures and shorter inspiration chains, remains to be verified.

## Related Work & Insights
- **vs MOOSE-Chem (Yang et al., 2025b)**: MOOSE-Chem used probability decomposition only at inference. This work converts that decomposition into a training objective, a key transition from "inference tool" to "training paradigm."
- **vs Feedback-driven training**: Prior works fine-tune hypotheses based on reviewer or data feedback without addressing the core $P(h\mid b)$ distribution. This is the first work to train that distribution directly.
- **vs O'Neill et al. (2025)**: Attempts to model $P(h\mid b)$ via distillation, which § 7.1 of this paper argues is infeasible due to the difficulty of replicating reasoning traces.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Rigorous argument of combinatorial complexity as the root cause of training failure, providing an executable roadmap to $\log N$. Rare mix of theoretical and engineering novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid data scale and GPU investment (38,400 A800 hours), though comparative experiments lean qualitative and lack some hard benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐ Clear complexity derivations and logical causality; well-explained transitions across complexity levels.
- **Value**: ⭐⭐⭐⭐⭐ Provides the framework, dataset (108k TOMATO-Star), code, and models, making it a de facto baseline for "LLM-for-discovery."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Breaking the Gradient Barrier: Unveiling Large Language Models for Strategic Classification](../../NeurIPS2025/llm_pretraining/breaking_the_gradient_barrier_unveiling_large_language_models_for_strategic_clas.md)
- [\[ICLR 2026\] RECON: Robust symmetry discovery via Explicit Canonical Orientation Normalization](../../ICLR2026/llm_pretraining/recon_robust_symmetry_discovery_via_explicit_canonical_orientation_normalization.md)
- [\[ICML 2026\] Annotations Mitigate Post-Training Mode Collapse](annotations_mitigate_post-training_mode_collapse.md)
- [\[ICML 2026\] On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](on_training_large_language_models_for_long-horizon_tasks_an_empirical_study_of_h.md)
- [\[NeurIPS 2025\] Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](../../NeurIPS2025/llm_pretraining/breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)

</div>

<!-- RELATED:END -->
