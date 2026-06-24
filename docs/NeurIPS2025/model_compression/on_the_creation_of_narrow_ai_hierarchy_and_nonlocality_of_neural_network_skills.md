---
title: >-
  [Paper Note] On the Creation of Narrow AI: Hierarchy and Nonlocality of Neural Network Skills
description: >-
  [NeurIPS 2025][Model Compression][Narrow AI] This paper investigates two fundamental challenges in creating narrow AI systems: the hierarchical dependencies among tasks require that certain narrow skills can only be learned effectively when trained on broad distributions; and the nonlocality of skills makes it impossible to precisely separate desired from undesired capabilities via pruning—yet pruning followed by recovery fine-tuning still outperforms both distillation and tr…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Narrow AI"
  - "Model Pruning"
  - "Knowledge Distillation"
  - "Curriculum Learning"
  - "Skill Nonlocality"
date: 2026-05-08
content_hash: b7ae360e31ca67e5
---

# On the Creation of Narrow AI: Hierarchy and Nonlocality of Neural Network Skills

**Conference**: NeurIPS 2025
**arXiv**: [2505.15811](https://arxiv.org/abs/2505.15811)  
**Code**: [GitHub](https://github.com/ejmichaud/narrow)  
**Area**: Model Compression
**Keywords**: Narrow AI, Model Pruning, Knowledge Distillation, Curriculum Learning, Skill Nonlocality

## TL;DR

This paper investigates two fundamental challenges in creating narrow AI systems: the hierarchical dependencies among tasks require that certain narrow skills can only be learned effectively when trained on broad distributions; and the nonlocality of skills makes it impossible to precisely separate desired from undesired capabilities via pruning—yet pruning followed by recovery fine-tuning still outperforms both distillation and training from scratch.

## Background & Motivation

The strongest AI systems today are general-purpose foundation models—the best math models simultaneously understand Roman history and cooking. This generality raises concerns along two axes:

- **Efficiency**: Models deployed as coding assistants carry substantial irrelevant knowledge; creating smaller, specialized networks could dramatically reduce inference costs.
- **Safety**: Narrow systems may reduce CBRN risks, are more amenable to mechanistic interpretability and formal verification, and an ecosystem of narrow "tool AIs" may be safer than a single general-purpose AI.

The authors focus on two fundamental questions:

**When can a narrow model be trained from scratch?** On hierarchically structured data, certain composite skills may *necessarily* require broad-distribution training (to first acquire atomic skills) before they can be efficiently learned—a strong curriculum learning effect.

**Can pruning convert a broad model into a narrow one?** Neural representations are distributed; skills are not confined to specific prunable components (e.g., individual neurons), making "surgical pruning" fundamentally challenging.

## Method

### Overall Architecture

The authors design a synthetic task (CMSP) to systematically study curriculum learning and pruning behavior, then validate findings on MNIST and LLMs (Llama series), comparing three approaches to creating narrow models: pruning, distillation, and training from scratch.

### Key Designs

1. **Compositional Multi-task Sparse Parity (CMSP)**: Extends the standard Multi-task Sparse Parity (MSP) with two modifications: (a) index sets for different subtasks are disjoint, $I_i \cap I_j = \emptyset$; (b) multiple control bits may be simultaneously ON, in which case the label is the parity of the union of corresponding index sets. This creates a hierarchical dependency between atomic and composite tasks—composite tasks can logically be computed by combining features of atomic tasks. Key finding: when trained on a broad distribution including atomic tasks, 27/40 networks converge on composite tasks within $2 \times 10^8$ samples; when trained on composite tasks alone, **0/40 networks** converge within $2 \times 10^9$ samples.

2. **Pruning and Nonlocality Analysis**: For trained CMSP networks, each neuron is assigned an ablation score $s_g = |\mathbb{E}[L(f(x;\theta)) - L(f(x;\theta_g^*))]|$, and greedy pruning proceeds from lowest to highest score. Analysis reveals: (a) skills are nonlocal—relevant connections are distributed throughout the network with no apparent structure; (b) skills are entangled—even after pruning to maximally preserve tasks $\{0,1,2\}$, tasks $\{3,4,5\}$ can be recovered with minimal fine-tuning, indicating shared neurons between the two task sets.

3. **Group Lasso Regularization**: A group sparsity regularizer $R(\theta) = \sum_{g \in G} \sqrt{\sum_{i \in g} \theta_i^2}$ (L1 norm of L2 norms) is applied during additional fine-tuning on narrow-domain data. This simultaneously achieves two goals: (a) concentrating target-skill features into fewer neurons, enabling more aggressive pruning; (b) forgetting unwanted skills. After regularization, tasks $\{3,4,5\}$ **cannot be revived by any amount of recovery training** while performance on $\{0,1,2\}$ is maintained—achieving robust forgetting.

### Loss & Training

- Pruning: $s_g = |\mathbb{E}[L(f(x;\theta)) - L(f(x;\theta_g^*))]|$, or linear-approximation attribution score $\hat{s}_g = |\sum_{i \in g} \frac{\partial L}{\partial \theta_i}(-\theta_i)|$.
- Group Lasso training: $\min_\theta \mathbb{E}[L(f(x;\theta)) + \lambda R(\theta)]$, with $\lambda = 10^{-3}$ (CMSP) or $5 \times 10^{-4}$ to $10^{-3}$ (LLM).
- Distillation: minimizing KL divergence between student and teacher output distributions, with temperature $T=20$ (MNIST) or $T=2$ (LLM).

## Key Experimental Results

### Main Results — CMSP Curriculum Learning Effect

| Training Distribution | Batch/Seeds | Composite Task Convergence Rate | Convergence Samples |
|---|---|---|---|
| Atomic + Composite (broad) | 2000/subtask, 40 seeds | 27/40 | ~2×10⁸ |
| Composite only (narrow) | 2000, 40 seeds | **0/40** | >2×10⁹, no convergence |

### Main Results — LLM Narrowing (Python Documentation)

| Method | Data Efficiency | Key Observation |
|---|---|---|
| Train Llama architecture from scratch | Worst | Requires large data and parameter count to reach target |
| Distillation (Llama-3.1-8B → small model) | Moderate | Better than from scratch, worse than pruning |
| Prune Llama-3.2-1B + recovery fine-tuning | **Best** | Pareto-dominates on data–parameter frontier |

### Forgetting Experiments

| Method | Sparsity | CounterFact | AI2-ARC | WMDP-Bio | WMDP-Cyber |
|---|---|---|---|---|---|
| Baseline (no pruning) | 0% | 0.18→0.49 | 0.65→0.67 | 0.52→0.62 | 0.35→0.59 |
| Random pruning | 30% | 0.00→0.50 | 0.24→0.25✓ | 0.22→0.27✓ | 0.27→0.27✓ |
| Attribution pruning | 30% | 0.12→0.65 | 0.35→0.69 | 0.36→0.56 | 0.27→0.55 |
| Group Lasso | 30% | 0.06→0.47 | 0.25→0.25✓ | 0.24→0.31✓ | 0.26→0.27✓ |
| Random pruning | 80% | 0.00→0.42 | 0.26→0.25✓ | 0.26→0.28✓ | 0.23→0.26✓ |

(✓ indicates post-fine-tuning score is ≥10% below baseline, i.e., successful forgetting)

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Attribution vs. random pruning (LLM) | Identical post-recovery performance | Skills are highly distributed; attribution advantage vanishes |
| MNIST: pruning vs. distillation vs. from scratch | Pruning is Pareto-superior | Advantage is especially pronounced at high neuron counts |
| Group Lasso at different λ | λ = 0.001–0.008 | Sensitive to hyperparameter but effect is consistent |

### Key Findings

- **Surprising finding**: On LLMs, random pruning performs identically to attribution pruning after recovery training. This is consistent with Bricken et al. [2023], who found that monosemantic features are highly distributed across large numbers of dimensions.
- Group Lasso exhibits stronger differential forgetting on CMSP than on LLMs, possibly because CMSP tasks are more independent.
- Pruning is significantly more data-efficient than distillation on LLMs (achieving the same target performance with less data), particularly when a pretrained model is already available.

## Highlights & Insights

- **Elegant CMSP task design**: Disjoint index sets and combinatorial control bits cleanly induce hierarchical dependencies, making CMSP an excellent testbed for studying curriculum learning.
- **A partial explanation for why general models dominate even on narrow tasks**: The learning of certain composite skills depends on a curriculum effect in which the constituent atomic skills must be acquired first.
- **The unexpected finding that random pruning ≈ attribution pruning** profoundly reveals the impact of feature distributedness on the design of pruning strategies.

## Limitations & Future Work

- CMSP is a synthetic task; the generality of its strong curriculum effects in real-world settings remains unclear.
- Hyperparameters for LLM experiments are not fully optimized; the advantage of pruning over training from scratch may partly reflect experimental design differences.
- Only simple greedy attribution pruning is employed; more sophisticated strategies (e.g., sparse autoencoder-guided pruning) may yield different outcomes.
- The safety motivation is compelling, but the failure of attribution pruning to achieve forgetting at 30% sparsity indicates that more refined methods are still needed.

## Related Work & Insights

- Complements Cloud et al. (2023)'s "gradient routing": gradient routing routes capabilities to specific components during training, whereas this paper achieves a similar effect post-training via regularization.
- Liu et al. (2024) study tasks analogous to CMSP and discover "domino" learning dynamics; this paper provides a more systematic analysis.
- Directly informs machine unlearning research in AI safety: simple pruning may suffice for capability forgetting, but attribution-based methods may be ineffective.

## Rating

- Novelty: ⭐⭐⭐⭐ Clever CMSP design; the finding that random pruning ≈ attribution pruning is genuinely surprising
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive three-tier validation from synthetic tasks to MNIST to LLMs
- Writing Quality: ⭐⭐⭐⭐ Clear structure and effective figures, though some experimental descriptions are slightly fragmented
- Value: ⭐⭐⭐⭐⭐ Provides systematic preliminary answers to fundamental questions about narrow AI (trainability and pruning feasibility), with implications for both efficiency and safety

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Beyond Student: An Asymmetric Network for Neural Network Inheritance](../../ICLR2026/model_compression/beyond_student_an_asymmetric_network_for_neural_network_inheritance.md)
- [\[NeurIPS 2025\] KINDLE: Knowledge-Guided Distillation for Prior-Free Gene Regulatory Network Inference](kindle_knowledge-guided_distillation_for_prior-free_gene_regulatory_network_infe.md)
- [\[NeurIPS 2025\] AI-Generated Video Detection via Perceptual Straightening](ai-generated_video_detection_via_perceptual_straightening.md)
- [\[NeurIPS 2025\] The Graphon Limit Hypothesis: Understanding Neural Network Pruning via Infinite Width Analysis](the_graphon_limit_hypothesis_understanding_neural_network_pruning_via_infinite_w.md)
- [\[AAAI 2026\] A Closer Look at Knowledge Distillation in Spiking Neural Network Training](../../AAAI2026/model_compression/a_closer_look_at_knowledge_distillation_in_spiking_neural_ne.md)

</div>

<!-- RELATED:END -->
