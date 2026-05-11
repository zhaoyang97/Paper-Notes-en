---
title: >-
  [Paper Note] DesignX: Human-Competitive Algorithm Designer for Black-Box Optimization
description: >-
  [NeurIPS 2025][Medical Imaging][Black-Box Optimization] This paper proposes DesignX, the first automated algorithm design framework that jointly learns two sub-tasks—optimizer workflow generation and dynamic hyperparamet…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Black-Box Optimization"
  - "Automated Algorithm Design"
  - "Dual-Agent Reinforcement Learning"
  - "MetaBBO"
  - "Transformer"
date: 2026-05-08
content_hash: de5adea6c770d4dd
---

# DesignX: Human-Competitive Algorithm Designer for Black-Box Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2505.17866](https://arxiv.org/abs/2505.17866)
**Code**: [GitHub](https://github.com/MetaEvo/DesignX)
**Area**: Medical Imaging / Optimization Algorithm Design
**Keywords**: Black-Box Optimization, Automated Algorithm Design, Dual-Agent Reinforcement Learning, MetaBBO, Transformer

## TL;DR
This paper proposes DesignX, the first automated algorithm design framework that jointly learns two sub-tasks—optimizer workflow generation and dynamic hyperparameter control—through dual Transformer agents pre-trained at scale on 10k synthetic problems. DesignX surpasses human-designed optimizers on both synthetic benchmarks and real-world tasks including protein docking, AutoML, and UAV path planning.

## Background & Motivation

**Background**: Black-box optimization (BBO) is a core problem in science and industry. Evolutionary computation (EC) is the dominant gradient-free paradigm, having produced a large family of variants—GA, DE, PSO, CMA-ES, etc.—over decades, each requiring expert-crafted adaptive operators and hyperparameter controllers.

**Limitations of Prior Work**:
- Manually redesigning optimizers for each new BBO problem does not scale.
- Although MetaBBO (Meta-Black-Box Optimization) introduces learning-based paradigms, existing methods learn only a **single sub-task**—either algorithm selection/workflow generation or hyperparameter control—and the separation leads to suboptimal designs.
- LLM-based approaches can generate algorithm code but likewise handle only one sub-task at a time.

**Key Challenge**: Algorithm design inherently involves two coupled sub-tasks (workflow structure + dynamic hyperparameters); optimizing them separately cannot achieve joint optimality.

**Key Insight**: Construct a modular algorithm space (Modular-EC) and a dual-agent RL system for end-to-end joint learning.

**Core Idea**: Agent-1 autoregressively generates valid optimizer workflows; Agent-2 dynamically controls hyperparameters. Both agents are meta-trained on a distribution of 10k problems through a cooperative training objective.

## Method

### Overall Architecture
The input to the framework is a feature vector characterizing a BBO problem instance (dimensionality, search range, ELA statistical features, etc.). Agent-1 (Transformer) autoregressively generates a valid optimizer workflow—selecting from 116 modules in Modular-EC—conditioned on the problem features. Agent-2 (Transformer) dynamically adjusts the hyperparameters of all controllable modules in response to real-time feedback during optimization. The two agents are jointly trained via a cooperative reward objective.

### Key Designs

1. **Modular-EC: Modular Algorithm Space**

    - **Function**: Decomposes EC optimizers into 10 module types (Initialization, Mutation, Crossover, Selection, Niching, …) with 116 module variants in total.
    - **Mechanism**: Each module has a unique 16-bit encoding and topology rules (defining valid successor modules), enabling autoregressive generation of valid workflows. Compared to its predecessor Modular-BBO (primarily targeting DE), Modular-EC adds ES/GA/PSO operators and the `Other_Update` module type.
    - **Design Motivation**: To unify decades of expert-designed algorithm components into a single encoding, providing the learning agent with a search space of **millions of possible workflows**.

2. **Agent-1: Workflow Generation**

    - **Function**: Given problem features $\mathcal{F}_p$ (13-dimensional, comprising 4 basic attributes and 9 ELA features), autoregressively samples a module sequence.
    - **Mechanism**: Based on a GPT-2 architecture, it ensures topological validity via masked softmax sampling:
      $$P(\mathcal{A}_p^{m+1} | \text{start}, \mathcal{A}_p^1, ..., \mathcal{A}_p^m) \sim \text{Softmax}(\text{mask}(\mathcal{A}_p^m) \odot (\mathcal{W}_\text{sample}^T \cdot H^{(m)}))$$
      The mask vector zeroes out probabilities of illegal modules according to the current module's topology rules.
    - **Design Motivation**: The Transformer's sequence modeling capacity is naturally suited to the ordered generation of workflows, and masked sampling guarantees that generated optimizers are always valid and executable.

3. **Agent-2: Dynamic Hyperparameter Control**

    - **Function**: At each optimization step, generates hyperparameter values for all controllable modules based on the observation $\mathcal{O}_t$ (a 9-dimensional progress feature vector).
    - **Mechanism**: Module IDs and observations are concatenated and encoded; another stack of GPT-2 blocks outputs the parameters of a normal distribution:
      $$\mu = \mathcal{W}_\mu^T \cdot H_{dec}, \quad \Sigma = \mathcal{W}_\Sigma^T \cdot H_{dec}$$
      Hyperparameters are then sampled from the predicted distribution: $C_t^m \sim \mathcal{N}(\mu^{(m)}, \Sigma^{(m)})$
    - **Design Motivation**: Hyperparameters directly govern the exploration–exploitation trade-off in EC optimizers; dynamic control allows adaptive adjustment across different stages of optimization.

4. **Cooperative Training Objective**

    - Agent-1 is trained with REINFORCE (delayed reward); Agent-2 is trained with PPO (dense reward).
    - Unified objective: $\mathcal{J}(\phi, \theta) = \mathbb{E}_{p \sim \mathcal{D}_{train}}[\sum_{t=1}^T r_t]$
    - Per-step reward: $r_t = \frac{f_p^{t-1,*} - f_p^{t,*}}{f_p^{0,*} - f_p^*}$ (normalized optimization progress)

### Loss & Training
- 12,800 synthetic problem instances (9,600 training / 3,200 test), constructed by combining 32 base functions through single/composition/hybrid schemes.
- Training runs for 6 days; the primary bottleneck is BBO simulation rather than neural network computation (optimization loops run on CPU).
- At inference time, DesignX requires approximately 5.5 s per problem, comparable to CMA-ES (5.0 s).

## Key Experimental Results

### Main Results: Synthetic Test Set (Selected)

| Problem | before 00 | 00s | 10s | after 20 | MetaBBO | **DesignX** |
|---------|-----------|-----|-----|----------|---------|-------------|
| F1 (50D, 30K FEs) | 6.60E+00 | 1.64E+00 | 1.27E+00 | 5.32E+00 | 2.80E+00 | **2.89E-01** |
| F2068 (20D) | 3.79E+01 | 2.32E+00 | 1.46E+01 | 1.65E+01 | 3.72E+01 | **5.16E-01** |
| F2390 (10D) | 3.93E+00 | 2.78E+00 | 6.34E+00 | 1.54E+00 | 2.04E+01 | **1.85E-03** |
| **Normalized Mean** | 2.94E-01 | 1.96E-01 | 1.54E-01 | 1.46E-01 | 1.32E-01 | **8.26E-02** |

DesignX ranks first on nearly all test instances; its normalized mean is 37% lower than that of the best MetaBBO baseline.

### Ablation Study

| Configuration | Description | Normalized Performance |
|--------------|-------------|----------------------|
| w/o A1+A2 | Random dual agents | Worst |
| w/o A1 | Train Agent-2 only | Poor |
| w/o A2 | Train Agent-1 only | Moderate |
| SBS | Static workflow | Poor |
| **DesignX** | **Dual-agent cooperation** | **Best** |

### Key Findings
- Agent-1 (workflow generation) contributes more to performance than Agent-2 (hyperparameter control), yet their cooperation yields results significantly superior to either sub-task alone.
- The design strategies learned by DesignX are interpretable: it favors composite mutation strategies for multimodal problems and population-reduction mechanisms for small search ranges.
- Notably, DesignX assigns negligible importance to the initialization strategy—a finding that may not align with conventional human intuition.
- DE-related modules are selected most frequently by DesignX, suggesting that DE operator combinations offer the strongest general-purpose utility.
- DesignX maintains its advantage on out-of-distribution real-world tasks including protein docking, AutoML, and UAV path planning.

## Highlights & Insights
- **First end-to-end framework to jointly learn both algorithm design sub-tasks**: unifying workflow generation and hyperparameter control breaks the single-sub-task bottleneck prevalent in the MetaBBO literature.
- **Masked Softmax for validity guarantees**: the topology rules combined with the mask mechanism elegantly ensure that autoregressively generated optimizer workflows are always valid and executable.
- **Interpretability analysis is valuable**: module importance factors and sub-module distribution analysis reveal non-trivial design principles learned by DesignX, providing reverse insights for human optimizer design.
- The paradigm of large-scale training on synthetic data followed by zero-shot transfer to real tasks is worth adopting more broadly.

## Limitations & Future Work
- Modular-EC currently supports only EC-family optimizers (DE/PSO/GA/ES) and does not cover other BBO paradigms such as Bayesian optimization.
- Training requires 6 days of CPU computation; scaling-law experiments are constrained by available compute.
- Under rank-based comparison, DesignX performance remains close to CMA-ES, indicating room for further improvement.
- Problem features are represented by only 13-dimensional ELA features, which may be insufficient to characterize high-dimensional or complex problems.
- Training is conducted only at the smallest model configuration (1-layer GPT-2); the potential of larger models and larger training sets remains largely unexplored.

## Related Work & Insights
- **vs. ConfigX**: ConfigX handles only DE hyperparameter control (single sub-task); DesignX extends this to dual sub-tasks and upgrades Modular-BBO to Modular-EC.
- **vs. ALDes**: ALDes performs workflow generation but not dynamic hyperparameter control; DesignX unifies both.
- **vs. GLHF**: GLHF simulates DE operators via gradient descent; DesignX directly learns module combinations through RL.
- **vs. LLM-based approaches**: LLMs generate algorithm code but process only one sub-task per query and incur high inference costs; DesignX achieves more efficient end-to-end design with a compact model.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First end-to-end dual-agent automated algorithm design framework, with innovations in both theoretical design and engineering implementation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3,200 synthetic tests + 3 real-world scenarios + ablation + scaling law + interpretability analysis—extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with rich visualizations, though the heavy notation requires repeated cross-referencing.
- Value: ⭐⭐⭐⭐⭐ Represents a paradigm-level advance for both MetaBBO and automated algorithm design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] From Black Box to Biomarker: Sparse Autoencoders for Interpreting Speech Models of Parkinson's Disease](from_black_box_to_biomarker_sparse_autoencoders_for_interpreting_speech_models_o.md)
- [\[ICLR 2026\] Knowledgeable Language Models as Black-Box Optimizers for Personalized Medicine](../../ICLR2026/medical_imaging/knowledgeable_language_models_as_black-box_optimizers_for_personalized_medicine.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[NeurIPS 2025\] Flow Density Control: Generative Optimization Beyond Entropy-Regularized Fine-Tuning](flow_density_control_generative_optimization_beyond_entropy-regularized_fine-tun.md)
- [\[NeurIPS 2025\] Is Sequence Information All You Need for Bayesian Optimization of Antibodies?](is_sequence_information_all_you_need_for_bayesian_optimization_of_antibodies.md)

</div>

<!-- RELATED:END -->
