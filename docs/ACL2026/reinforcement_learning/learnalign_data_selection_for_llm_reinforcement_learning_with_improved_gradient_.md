---
title: >-
  [Paper Note] LearnAlign: Data Selection for LLM Reinforcement Learning with Improved Gradient Alignment
description: >-
  [ACL 2026][Reinforcement Learning][RLVR] Focusing on data selection for RLVR post-training, this paper proposes LearnAlign—employing "gradient alignment" as a representativeness metric and "success rate $V(\xi)=p(1-p)$"…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "RLVR"
  - "GRPO"
  - "Data Selection"
  - "Gradient Alignment"
  - "Data Learnability"
  - "Proximal Development Zone"
date: 2026-05-08
content_hash: 3c6ec9a3940b7d66
---

# LearnAlign: Data Selection for LLM Reinforcement Learning with Improved Gradient Alignment

**Conference**: ACL 2026  
**arXiv**: [2506.11480](https://arxiv.org/abs/2506.11480)  
**Code**: TBD  
**Area**: Reinforcement Learning / Data Selection / LLM Reasoning  
**Keywords**: RLVR, GRPO, Data Selection, Gradient Alignment, Data Learnability, Proximal Development Zone

## TL;DR
Focusing on data selection for RLVR post-training, this paper proposes LearnAlign—employing "gradient alignment" as a representativeness metric and "success rate $V(\xi)=p(1-p)$" as learnability weights to eliminate response length bias. Using only 1000 samples (approx. 6%), it achieves performance near full-dataset training across 5 reasoning benchmarks (42.4% vs 44.9%), and even surpasses full training on GSM8K (77.5%) using only 13.4% of the data (77.0% for full).

## Background & Motivation

**Background**: RLVR (Reinforcement Learning with Verifiable Rewards) has become the standard post-training scheme for reasoning LLMs like OpenAI o1, DeepSeek-R1, and Kimi k1.5—utilizing rule-verifiable rewards (e.g., math answer correctness, code passing) as supervision signals via GRPO/PPO. However, RLVR is characterized by low data efficiency and expensive training costs.

**Limitations of Prior Work**: Most existing data selection methods are designed for SFT (INSTAG, ALPAGASUS, IFD, LESS, SelectIT, Nuggets, etc.), prioritizing "high quality" as high difficulty or low perplexity. Recent RLVR-related works (LIMR, 1-shot RLVR) prove that a few samples are sufficient, but their selection phase requires full training for several epochs, resulting in **extremely high evaluation costs that defeat the purpose of "selection."**

**Key Challenge**: The objective of SFT is to maximize target likelihood, making "the harder, the better" a viable heuristic. In contrast, the RLVR objective is reward maximization; **only samples with difficulty matching the current policy's capability can generate learning signals**. Samples that are too simple ($p\approx 1$) or too difficult ($p\approx 0$) yield no learning utility for RLVR. Directly applying SFT selection methods to RLVR often yields results inferior to random sampling.

**Goal**: (1) Identify an efficient (not requiring full training), interpretable, and quantifiable data selection criterion for RLVR; (2) Address two long-standing issues in gradient-based methods—the "short response bias" of gradient magnitude and the loss of magnitude information in cosine similarity; (3) Verify if small data amounts can match or exceed full-set training on GSM8K and DAPO-MATH-17K.

**Key Insight**: The authors draw inspiration from LESS (gradient alignment) in the SFT era and the "Zone of Proximal Development" (ZPD, Vygotsky) in pedagogy: selection must ensure both that samples align with the full training distribution (gradient direction similarity) and that they reside at the policy's capability boundary (maximum potential at $p\approx 0.5$).

**Core Idea**: Construct a **learnability-weighted gradient vector** $\mathbf{V}(\xi_i) = \frac{\nabla \mathcal{J}}{\|\nabla \mathcal{J}\|} \cdot V(\xi_i)$, where $V(\xi_i) = p(1-p)$ quantifies "learning potential" using the success rate. Multiplying this by unit gradients eliminates length bias while retaining alignment information. The LearnAlign Score $S_{ij}$ then calculates pairwise learnability and representativeness, choosing top-N samples by row-averaging.

## Method

### Overall Architecture

LearnAlign is a 4-step lightweight selection workflow that **does not require full training on the dataset**:

1. **Warmup Training**: Perform RLVR on a small random subset $\mathcal{D}_{\text{warmup}}$ to obtain an initial policy $\bm{\theta}_s$, avoiding high gradient noise during cold starts (300 samples for GSM8K, 1000 for DAPO-MATH-17K).
2. **Learnability Estimation**: For each sample $\xi_i$, perform $G=8$ rollouts under $\pi_{\bm{\theta}_s}$. Calculate success rate $p_i = \frac{1}{G}\sum_g \mathbb{I}(\mathbf{y}_g = \mathbf{y}^*)$ and derive $V(\xi_i) = p_i(1-p_i)$.
3. **Gradient Information Estimation + Random Projection**: Calculate the GRPO gradient $\nabla \mathcal{J}_{\text{GRPO}}$ for each sample at $\bm{\theta}_s$. Following the LESS approach, use Johnson-Lindenstrauss random projection $\Gamma$ to compress these into low-dimensional vectors $\phi(\bm{\theta}; \xi) = \Gamma^\top \nabla \mathcal{J}$ to avoid memory overhead.
4. **LearnAlign Score Calculation + Top-N Selection**: Calculate pairwise $S_{ij} = V(\xi_i)V(\xi_j) \cdot \cos(\phi(\xi_i), \phi(\xi_j))$, sort by the row average $\text{Avg}_i = \frac{1}{n}\sum_j S_{ij}$, and select the top $N$ samples for RLVR training.

### Key Designs

1. **Learnability Measure $V(\xi) = p(1-p)$ (ZPD Principle)**:
    - **Function**: Formalizes "the learning value of a sample given the current policy" as a scalar between $[0, 0.25]$, peaking at $p=0.5$ (the capability boundary).
    - **Mechanism**: Samples $G$ rollouts to compute $p$. $p$ represents the mastery probability and $1-p$ the room for improvement; the product $p(1-p)$ measures "expected learnability." The authors provide theoretical justification (information gain and policy gradient variance minimization) in Appendix B/C.
    - **Design Motivation**: (a) Replaces length-biased gradient magnitude—decoupling success rate from response length; (b) Aligns with educational psychology ZPD: weights for too easy ($p\to 1$) and too hard ($p\to 0$) samples approach zero; (c) Provides a "difficulty-capability match" mechanism absent in SFT methods.

2. **Learnability-Weighted Gradient Vector $\mathbf{V}(\xi_i) = \hat{\nabla}\mathcal{J} \cdot V(\xi_i)$**:
    - **Function**: Orthogonally combines "representativeness" (gradient alignment) and "learnability" (success rate) in a single vector space, calculating paired scores via inner products.
    - **Mechanism**: GRPO gradients are first normalized $\hat{\nabla}\mathcal{J}_i = \nabla \mathcal{J}_i / \|\nabla \mathcal{J}_i\|$ (eliminating magnitude bias) and then re-weighted by $V(\xi_i)$. The resulting $S_{ij} = V(\xi_i)V(\xi_j) \cdot \cos(\hat{\nabla}_i, \hat{\nabla}_j)$ peaks when learnable samples have highly aligned gradient directions.
    - **Design Motivation**: (a) Pure cosine similarity discards magnitude (failing to distinguish "more valuable" samples); (b) Pure inner products suffer from length bias (longer responses naturally have smaller gradients). Weighting by $V$ preserves the benefits of both—direction via cosine and magnitude via learnability.

3. **Random Projection of GRPO Gradients + Warmup Checkpoint**:
    - **Function**: Reduces the computational cost of RLVR data selection—avoiding full-set training (overcoming bottlenecks of LIMR/1-shot RLVR) and massive memory usage.
    - **Mechanism**: (a) Warmup provides a stable reference point $\bm{\theta}_s$; (b) Calculates the GRPO gradient for each sample on $\bm{\theta}_s$ in a single forward/backward pass; (c) Employs JL projection $\Gamma$ to compress gradients to a few thousand dimensions for inner product calculations.
    - **Design Motivation**: Migrates the valid JL projection technique from SFT (LESS) to the RLVR scenario. Table 4 indicates that selection time is reduced to levels comparable with SFT selection.

### Loss & Training

Standard GRPO is applied throughout without modification (KL coefficient $\beta=0.04$, clip $\epsilon=0.2$, lr $1\times 10^{-6}$, $G=8$ rollouts at temperature 1.0, batch size 48/64). During DAPO training, gradients are computed using only 1 correct rollout (following Lin et al. 2025 for acceleration).

## Key Experimental Results

### Main Results

**GSM8K Data Selection** (Qwen2.5-1.5B-Instruct, selection from the GSM8K training set):

| Method | 100 | 500 | 1,000 | 2,000 |
|------|-----|-----|-------|-------|
| Base (no RLVR) | 55.7 | – | – | – |
| Full data (~7.5K) | 77.0 | – | – | – |
| Random | 73.1 | 75.1 | 75.6 | 75.5 |
| IFD (SFT Selection) | 72.0 | 76.0 | 75.6 | 75.4 |
| Token Length | 72.3 | 74.4 | 76.2 | 75.6 |
| LIMR (RLVR baseline) | 74.2 | 76.2 | 76.1 | 76.7 |
| **LearnAlign** | **74.8** | **76.4** | **77.5** | **78.3** |

1000 samples already match full training (77.5% vs 77.0%), and 2000 samples **exceed** the full data performance by 1.3 points.

**DAPO-MATH-17K → 5 benchmarks** (1000 samples selected, Qwen2.5-7B):

| Method | GSM8K | MATH500 | AMC2023 | AIME2024 | CRUX | Avg. |
|------|-------|---------|---------|----------|------|------|
| Base | 26.4 | 67.2 | 18.1 | 16.7 | 25.1 | 30.7 |
| Full 17K | 89.8 | 76.4 | 47.0 | 30.0 | 51.1 | 58.9 |
| Random 1K | 81.1 | 65.0 | 30.1 | 23.3 | 40.8 | 48.1 |
| SelectIT 1K | 85.4 | 67.0 | 32.7 | 26.7 | 41.5 | 50.7 |
| LIMR 1K | 84.2 | 61.6 | 27.1 | 16.7 | 39.9 | 45.9 |
| **LearnAlign 1K** | **88.3** | **70.4** | **35.4** | **30.0** | **44.0** | **54.6** |

Ours achieves an average score of 54.6 using only 5.9% of the data (vs 58.9 for full), outperforming SelectIT by 3.9 points and matching full training performance on AIME2024 (30.0).

### Ablation Study

| Configuration | GSM8K (1K, 1.5B) | GSM8K (2K, 1.5B) | GSM8K (1K, 3B) | MATH500 (1K, 3B) | AMC2023 (1K, 3B) |
|------|------------------|------------------|----------------|-------------------|-------------------|
| **Full LearnAlign** | **77.5** | **78.3** | **79.3** | **60.2** | **28.3** |
| w/o warmup | 76.6 | 76.6 | 76.7 | 58.2 | 26.1 |
| w/o learnability $V$ | 75.6 | 76.7 | 77.5 | 58.4 | 28.3 |
| w/ feature similarity | 75.7 | 76.6 | 79.1 | 57.6 | 27.5 |

Both components are essential: removing $V$ decreases performance by an average of 1.4 points (supporting the ZPD hypothesis), and skipping warmup leads to significant degradation due to noise in cold-start gradients.

### Key Findings
- **Traditional SFT selection is ineffective or harmful for RLVR**: IFD, Top-PPL, and Token Length often perform worse than random sampling; this is because SFT focuses on absolute difficulty while RLVR requires capability matching.
- **Small data outperforms full data**: 2000 LearnAlign samples (78.3%) > 7.5K full data (77.0%) on GSM8K, indicating that effective signals are concentrated in "ZPD" samples while others are redundant or noisy.
- **Strong cross-domain generalization**: Models trained on DAPO-MATH perform well on OOD datasets (AMC2023/AIME2024) and different domains (CRUX code tasks).
- **Significant response length bias**: Pure gradient inner products favor shorter responses with lower performance; $V$ weighting selects mid-range lengths and improves performance.
- **Warmup is non-negotiable**: Warmup stabilizes gradient estimates to accurately reflect learning potential.
- **Efficiency advantage**: LearnAlign significantly reduces selection time compared to LIMR and 1-shot RLVR.

## Highlights & Insights
- **"$V(\xi) = p(1-p)$" elegantly translates ZPD into RLVR**: A single formula simultaneously eliminates length bias, quantifies learnability, and provides rational magnitude for gradient vectors.
- **The "warmup + projected gradient" combo successfully ports LESS to RLVR**: By substituting the SFT gradient with the GRPO gradient and using a warmup checkpoint, the paper adapts SFT data selection infrastructure for RLVR.
- **Valuable negative results regarding SFT selection**: Publicly debunking the suitability of SFT selection tools for RLVR saves the community significant trial-and-error costs.
- **Untapped data efficiency frontier**: The fact that small subsets outperforom full datasets suggests massive redundancy in human-annotated reasoning data for RL purposes.

## Limitations & Future Work
- **Dependency on warmup accuracy**: Noisy success rate estimates from insufficient warmup or few rollouts can bias $V(\xi)$. Dynamic or iterative updates might be necessary.
- **Domain restriction**: Valid only for tasks with verifiable rewards (math/code). Open-domain RLVR lacks a clear definition for $p$.
- **Projected gradient cost for large models**: While cheaper than training, calculating 17K GRPO gradients for 70B+ models remains storage-intensive.
- **Static selection approach**: $p$ evolves during training (samples at $p=0.5$ will eventually reach $p\to 1$), suggesting a need for online or curriculum-based LearnAlign.
- **Lack of analysis on "harmful" samples**: While some samples decrease performance, the paper does not define what constitutes a "noisy sample" in RLVR.

## Related Work & Insights
- **vs LIMR / 1-shot RLVR**: Both find that small data is sufficient, but LearnAlign avoids full-set training, making it a more practical selection tool.
- **vs LESS (Xia et al., 2024)**: Inherits the JL projection and gradient alignment framework but addresses LESS's weaknesses (length bias and lost magnitude) using $V(\xi)$.
- **vs SelectIT / IFD**: These target SFT alignment. Ours provides negative results for these on RLVR and offers the ZPD explanation for the discrepancy.
- **vs Curriculum Learning**: LearnAlign grounds curriculum concepts ("medium-difficulty first") into a computable $p(1-p)$ metric within a gradient alignment framework.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Combining gradient alignment with ZPD learnability is a clear innovation for RLVR.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extensive testing across 5 benchmarks, 3 models, and multiple baselines.
- **Writing Quality**: ⭐⭐⭐⭐ Clear mathematical derivations and comprehensive ablations.
- **Value**: ⭐⭐⭐⭐⭐ High practical impact for optimizing RLVR training costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Single-Rollout Hidden-State Dynamics for Training-Free RLVR Data Selection](../../ICML2026/reinforcement_learning/single-rollout_hidden-state_dynamics_for_training-free_rlvr_data_selection.md)
- [\[ICCV 2025\] RL-Selector: Reinforcement Learning-Guided Data Selection via Redundancy Assessment](../../ICCV2025/reinforcement_learning/reinforcement_learning-guided_data_selection_via_redundancy_assessment.md)
- [\[ACL 2026\] Efficient Hyperparameter Optimization for LLM Reinforcement Learning](efficient_hyperparameter_optimization_for_llm_reinforcement_learning.md)
- [\[ACL 2026\] CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning](ce-gppo_coordinating_entropy_via_gradient-preserving_clipping_policy_optimizatio.md)
- [\[ACL 2026\] Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with Constraints](deliberative_searcher_improving_llm_reliability_via_reinforcement_learning_with_.md)

</div>

<!-- RELATED:END -->
