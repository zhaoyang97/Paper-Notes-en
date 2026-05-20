---
title: >-
  [Paper Note] Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation
description: >-
  [AAAI 2026][Optimization][Neural Combinatorial Optimization] This paper proposes EvoReal, a framework that employs LLM-driven evolutionary search to generate synthetic VRP instances structurally aligned with real-world d…
tags:
  - "AAAI 2026"
  - "Optimization"
  - "Neural Combinatorial Optimization"
  - "Vehicle Routing Problem"
  - "LLM-Guided Evolution"
  - "Data Generator"
  - "Progressive Fine-Tuning"
  - "Generalization"
date: 2026-05-08
content_hash: d76ad8588b251b89
---

# Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation

**Conference**: AAAI 2026
**arXiv**: [2511.10233](https://arxiv.org/abs/2511.10233)  
**Authors**: Jianghan Zhu, Yaoxin Wu, Zhuoyi Lin, Zhengyuan Zhang, Haiyan Yin, Zhiguang Cao, Senthilnath Jayavelu, Xiaoli Li
**Code**: [GitHub](https://github.com/HenryZhu1029/EvoReal)  
**Area**: Optimization
**Keywords**: Neural Combinatorial Optimization, Vehicle Routing Problem, LLM-Guided Evolution, Data Generator, Progressive Fine-Tuning, Generalization

## TL;DR

This paper proposes EvoReal, a framework that employs LLM-driven evolutionary search to generate synthetic VRP instances structurally aligned with real-world distributions, and then adapts pretrained neural solvers to real benchmarks via a two-stage progressive fine-tuning strategy. EvoReal substantially outperforms existing neural solvers on TSPLib (1.05% gap) and CVRPLib (2.71% gap).

## Background & Motivation

### State of the Field
The Vehicle Routing Problem (VRP) is a classical NP-hard combinatorial optimization problem widely encountered in logistics scheduling and public transportation. In recent years, neural combinatorial optimization (NCO) methods based on deep reinforcement learning and attention mechanisms have achieved remarkable success on synthetic VRP instances, with representative works including POMO (Kwon et al. 2020) and LEHD (Luo et al. 2023). However, the training data for these methods is almost entirely drawn from uniformly sampled synthetic instances, resulting in a significant distribution shift relative to real-world routing problems.

### Limitations of Prior Work

**Synthetic-to-real generalization gap**: Existing NCO models suffer severe performance degradation on real-world benchmarks such as TSPLib and CVRPLib. For instance, the pretrained POMO achieves an optimality gap as high as 64.84% on large-scale (1,000–5,000 node) TSPLib instances, rendering it practically unusable.

**Scale limitations of LLM-based solvers**: LLM-based methods such as FunSearch and ReEvo excel at heuristic design but are constrained by context length, making them ineffective on medium- and large-scale instances exceeding 50 nodes.

**Limited effectiveness of direct fine-tuning**: Directly fine-tuning pretrained models on real benchmark instances yields only marginal improvements, as the limited quantity and complex distribution of real instances prevent the model from learning sufficient structural priors.

**Insufficient study of cross-distribution generalization**: Prior work has primarily focused on cross-scale generalization (small to large), with comparatively little attention to cross-distribution generalization (uniform to diverse real-world distributions).

### Core Idea
Rather than having LLMs directly generate solving heuristics or solutions—which faces scale bottlenecks—EvoReal leverages LLMs' code generation and reasoning capabilities to evolve data generators that produce synthetic training data structurally aligned with the real world. This represents a key perspective shift: from "LLM as solver" to "LLM as data factory designer," thereby indirectly enhancing the generalization capability of neural solvers in real-world settings.

## Method

### Overall Architecture
EvoReal comprises two core components: (1) LLM-driven generator evolution and (2) two-stage progressive fine-tuning. The overall workflow proceeds as follows: real benchmarks are split into validation and test sets; the validation set is grouped by distribution type; the LLM evolution module searches for the optimal data generator for each distribution type; and the evolved generators are used to produce synthetic data for two-stage fine-tuning of the pretrained model.

### Modular Generator Design
Taking TSP as an example, 48 of the 70 TSPLib instances are selected as the validation set and 22 as the test set. The 48 validation instances are categorized into three structural pattern groups (S1, S2, S3), each corresponding to one generator. Each generator is described by three components: a function signature definition, a generation code implementation, and a fitness score. The evolutionary objective is to minimize the divergence between the synthetic and real distributions.

### LLM-Driven Evolutionary Search
Building upon the ReEvo framework with modifications, the procedure is as follows:

1. **Population initialization**: The LLM is prompted to generate $N$ initial generators via an initialization prompt that includes a seed generator blueprint and design guidance, providing the LLM with prior knowledge about the target distribution.
2. **Population expansion**: New generator individuals are produced via four prompting strategies (crossover, mutation, short-term reflection, and long-term reflection). Short-term reflection targets improvements based on the relative performance difference between two parent generators; long-term reflection distills accumulated short-term reflections into higher-level design insights.
3. **Surrogate evaluation**: Each new generator fine-tunes the model for only a small number of epochs (e.g., 10 epochs for LEHD), and the best average gap on the validation set serves as the fitness score. This low-fidelity evaluation strategy substantially accelerates evolutionary search.
4. **Rank-based selection**: The population is ranked by fitness score and converted to selection probabilities. Compared to ReEvo, a rank-based selection mechanism is introduced to improve elite generator survival rates, and the selection step is deferred until after mutation to balance exploration and exploitation.
5. **Termination criterion**: Evolution stops when the number of evaluated generators reaches the upper limit, or when no improvement is found over $m$ consecutive iterations.

### Progressive Fine-Tuning Strategy
After evolution, a two-stage fine-tuning procedure is applied. The core idea is progressive adaptation from easy (synthetic data) to hard (real instances):

**Stage 1 (Synthetic data alignment)**: The three optimal generators collaboratively generate large-scale synthetic data (fixed at 100 nodes) to fine-tune the pretrained model. The goal is to enable the model to acquire preliminary structural priors aligned with diverse TSPLib-style node distribution patterns. The number of training epochs greatly exceeds that of the surrogate evaluation phase, and performance on the 48 validation instances is continuously monitored.

**Stage 2 (Real instance adaptation)**: Starting from the best model obtained in Stage 1, the model is further fine-tuned directly on real benchmark instances. Because the model has already learned rich distributional structure in Stage 1, it can more effectively absorb the complex patterns present in real instances, achieving a smooth transition from synthetic to real data.

This two-stage design not only bridges the distribution gap but also simultaneously addresses the scale gap, since the synthetic data in Stage 1 and the real instances in Stage 2 differ in both scale and distribution, creating a gradient of increasing difficulty.

## Key Experimental Results

### TSPLib Performance Comparison (70 instances)

| Method | [0,200) Gap | [200,500) Gap | [500,1000) Gap | [1000,5000) Gap | Overall Gap |
|--------|------------|--------------|---------------|----------------|------------|
| LKH-3 | 0.00% | 0.00% | 0.00% | 0.03% | 0.01% |
| ORTools | 1.68% | 3.31% | 3.69% | 4.41% | 3.06% |
| POMO (x8 aug) | 4.63% | 10.46% | 26.31% | 64.84% | 26.66% |
| LEHD (RRC-50) | 0.33% | 0.66% | 2.13% | 6.47% | 2.48% |
| ELG (x8 aug) | 1.15% | 3.98% | 8.73% | 11.36% | 5.62% |
| **POMO (ours, x8)** | **1.14%** | **2.16%** | **5.76%** | **32.43%** | **11.59%** |
| **LEHD (ours, RRC-50)** | **0.30%** | **0.35%** | **0.73%** | **2.54%** | **1.05%** |

LEHD (ours) outperforms all neural solvers and ORTools across all scale ranges, reducing the overall gap from 2.48% to 1.05%. POMO (ours) reduces the large-scale gap from 64.84% to 32.43%, a reduction of more than half.

### CVRPLib Performance Comparison (100 SetX instances)

| Method | [0,200) Gap | [200,500) Gap | [500,1002) Gap | Overall Gap |
|--------|------------|--------------|---------------|------------|
| HGS-CVRP | 0.01% | 0.06% | 0.24% | 0.11% |
| ORTools | 2.14% | 4.15% | 5.02% | 4.01% |
| POMO (x8 aug) | 6.75% | 14.95% | 41.15% | 21.53% |
| LEHD (RRC-50) | 4.91% | 4.73% | 8.27% | 5.90% |
| ELG (x8 aug) | 4.51% | 5.52% | 7.80% | 6.03% |
| **POMO (ours, x8)** | **3.52%** | **4.60%** | **6.20%** | **4.87%** |
| **LEHD (ours, RRC-50)** | **2.56%** | **2.59%** | **3.00%** | **2.71%** |

On CVRPLib, LEHD (ours) exhibits a gap variance of only 0.44% across scale ranges (2.56%–3.00%), demonstrating that the progressive fine-tuning strategy effectively narrows the performance gap between small- and large-scale instances.

## Highlights & Insights

- **Data-centric innovation**: Rather than modifying model architectures or designing new heuristics, EvoReal uses LLM-evolved data generators to bridge the distribution gap—a novel and practical approach. EvoReal can serve as a plug-and-play module applicable to arbitrary NCO solvers.
- **Clever exploitation of LLM capabilities**: The framework circumvents LLMs' context length limitations on large instances by having LLMs write generator code (short text) rather than directly solving instances (long text), maximizing LLM strengths.
- **Progressive fine-tuning outperforms direct fine-tuning**: Ablation studies show that skipping Stage 1 and fine-tuning directly on real data is substantially inferior to the two-stage strategy (POMO large-scale gap: 44.85% vs. 36.48%), confirming that the distributional priors provided by synthetic data are critical.
- **Evolved generators outperform naive distributions**: Compared to simple distributions such as Beta, exponential, and Gaussian mixture, evolved generators achieve lower gaps from the outset and converge to superior solutions, validating that LLMs can capture complex structural patterns in real data.
- **Negligible inference overhead**: Since the model architecture is unchanged, inference time after fine-tuning is essentially identical to that of the original model.

## Limitations & Future Work

- **High evolutionary search cost**: Generator evolution requires multiple surrogate evaluations, each involving short-term fine-tuning, resulting in non-trivial computational overhead and reliance on high-end LLMs such as o3.
- **Only two backbone models validated**: Experiments are conducted only on POMO and LEHD; applicability to other architectures such as BQ and ELG remains unknown.
- **Distribution grouping requires domain knowledge**: The manual categorization of TSP validation instances into three structural groups requires expert design.
- **Large-scale instances remain challenging**: POMO (ours) still achieves a gap of 32.43% on 1,000–5,000 node instances, far behind traditional solvers.
- **Limited to VRP-family problems**: The framework has not been extended to other combinatorial optimization problems such as MIS or bin packing.
- **Risk of overfitting in Stage 2**: LEHD (ours) exhibits a higher gap on small-scale CVRP instances than on large-scale ones, potentially due to overfitting in Stage 2.

## Related Work & Insights

- **POMO / LEHD (original)**: NCO models pretrained on uniform distributions suffer substantial performance degradation when generalized to real benchmarks. EvoReal improves the performance of both by over 50% without modifying the architecture.
- **ELG (Gao et al. 2024)**: Improves generalization to TSPLib via integrated transferable local policies, achieving an overall gap of 5.62%. EvoReal-LEHD substantially outperforms this with a 1.05% gap.
- **CNF (Zhou et al. 2024)**: Joint training across multiple distributions and tasks achieves a TSPLib overall gap of 23.42%, indicating that training strategy adjustments alone are insufficient to bridge the real-world distribution gap.
- **ReEvo (Ye et al. 2024)**: An LLM-evolved heuristic solver constrained by instance scale. EvoReal inherits its evolutionary framework and redirects the approach toward evolving generators rather than solvers.
- **FunSearch (Romera-Paredes et al. 2024)**: A pioneering LLM-driven evolutionary search framework focused on function discovery rather than data generation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The perspective shift to "LLM-evolved data generators" constitutes the core contribution; the progressive fine-tuning strategy follows standard domain adaptation practice.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers both TSPLib and CVRPLib benchmarks with comprehensive ablations, though only two backbone models are evaluated.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, clearly motivated, and supported by intuitive figures.
- Value: ⭐⭐⭐⭐ — Provides a practical and generalizable adaptation solution for real-world deployment of NCO models; code is publicly available.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization](../../ICLR2026/optimization/rrnco_towards_real-world_routing_with_neural_combinatorial_optimization.md)
- [\[AAAI 2026\] Instance Generation for Meta-Black-Box Optimization through Latent Space Reverse Engineering](instance_generation_for_meta-black-box_optimization_through_latent_space_reverse.md)
- [\[NeurIPS 2025\] Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees](../../NeurIPS2025/optimization/rethinking_neural_combinatorial_optimization_for_vehicle_routing_problems_with_d.md)
- [\[AAAI 2026\] PEOAT: Personalization-Guided Evolutionary Question Assembly for One-Shot Adaptive Testing](peoat_personalization-guided_evolutionary_question_assembly_for_one-shot_adaptiv.md)
- [\[NeurIPS 2025\] A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning](../../NeurIPS2025/optimization/a_theoretical_study_on_bridging_internal_probability_and_sel.md)

</div>

<!-- RELATED:END -->
