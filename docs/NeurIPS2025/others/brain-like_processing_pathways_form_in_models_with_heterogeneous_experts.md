---
title: >-
  [Paper Note] Brain-Like Processing Pathways Form in Models With Heterogeneous Experts
description: >-
  [NeurIPS 2025][Mixture-of-Experts] Heterogeneous experts in Mixture-of-Experts models do not spontaneously form processing pathways. This paper proposes three brain-inspired inductive biases — routing cost…
tags:
  - "NeurIPS 2025"
  - "Mixture-of-Experts"
  - "heterogeneous experts"
  - "processing pathways"
  - "inductive biases"
  - "brain-inspired computation"
  - "cognitive tasks"
date: 2026-05-08
content_hash: d5cdc774af3e4a50
---

# Brain-Like Processing Pathways Form in Models With Heterogeneous Experts

**Conference**: NeurIPS 2025
**arXiv**: [2506.02813](https://arxiv.org/abs/2506.02813)
**Code**: [jackcook/mixture-of-pathways](https://github.com/jackcook/mixture-of-pathways)
**Area**: Others (Computational Neuroscience × MoE Architecture)
**Keywords**: Mixture-of-Experts, heterogeneous experts, processing pathways, inductive biases, brain-inspired computation, cognitive tasks

## TL;DR

Heterogeneous experts in Mixture-of-Experts models do not spontaneously form processing pathways. This paper proposes three brain-inspired inductive biases — routing cost, task-performance scaling, and expert dropout — that enable the model to develop a Mixture-of-Pathways architecture analogous to the brain's dynamic cortical–subcortical pathways.

## Background & Motivation

**Heterogeneous pathway organization in the brain**: The brain comprises numerous heterogeneous regions that dynamically organize into processing pathways (e.g., visual pathways, cognitive control networks) according to task demands, yet the mechanisms underlying pathway formation remain poorly understood.

**Insufficient expert specialization in MoE models**: Existing Mixture-of-Experts models (e.g., DeepSeek-MoE, Mixtral) are theoretically expected to form task-relevant expert pathways, but in practice exhibit limited expert specialization and fail to develop stable functional pathways.

**Limitations of multi-region interaction modeling**: Existing multi-brain-region models typically predefine fixed connectivity structures, precluding the study of how regions dynamically interact to form pathways; models that permit dynamic interaction, in turn, cannot perform standard cognitive tasks.

**The metabolic optimization hypothesis**: Minimizing metabolic cost is central to theories of brain structure and function, yet it has not been systematically leveraged to drive pathway formation in multi-expert models.

**New possibilities with heterogeneous MoE**: Heterogeneous MoE (HMoE) allows experts of varying sizes and types to coexist, providing a natural platform for studying how heterogeneous regions self-organize into pathways.

**Core Problem**: Do heterogeneous regions automatically form functional pathways, or are additional prior constraints required? Do the resulting pathways resemble the dynamic, adaptive pathways observed in the brain?

## Method

### Overall Architecture: Mixture-of-Pathways (MoP)

The model consists of three stacked HMoE layers, each containing three heterogeneous experts (a 16-neuron GRU, a 32-neuron GRU, and a skip connection) and a 64-neuron GRU router. The router dynamically determines each expert's contribution weight based on the input, with information propagating forward layer by layer. Three inductive biases are introduced on top of the baseline HMoE to yield the MoP model. The model is trained on 82 cognitive tasks from the Mod-Cog dataset, spanning difficulty levels from simple stimulus–response mappings to complex working memory tasks.

### Key Design 1: Routing Cost

- **Function**: A Learned Pathway Complexity (LPC) penalty is incorporated into the loss function, biasing the model toward simpler experts.
- **Mechanism**: LPC is defined as $LPC_i = \frac{1}{T_i}\sum_t^{T_i}\sum_j^{E} w_{i,j,t} s_j^2$, where $w$ denotes routing weights and $s_j^2$ denotes the square of the expert size (corresponding to a storage cost of $O(s_j^2)$). Adding LPC to the loss function penalizes the use of large experts.
- **Design Motivation**: Inspired by the brain's metabolic optimization theory, which posits that the brain minimizes energy expenditure. This constraint forces the model to recruit more complex experts only when necessary, giving rise to differentiated pathways matched to task difficulty.

### Key Design 2: Task Performance Scaling

- **Function**: The routing cost is divided by the current task's response loss $L_{response,i}$, yielding the final routing loss $\frac{\alpha \cdot LPC_i}{L_{response,i} + \epsilon}$.
- **Mechanism**: When the model performs poorly on a task (high loss), the routing cost is reduced, permitting the use of complex experts for learning; once a task is mastered (low loss), the routing cost is amplified, driving the model to transfer the task to simpler pathways.
- **Design Motivation**: This prevents the model from falling into a local optimum of minimizing routing cost by relying solely on skip connections without solving tasks. It is analogous to the brain's dynamic regulation of cognitive effort — allocating greater resources when acquiring new skills and shifting to more economical pathways once proficiency is achieved.

### Key Design 3: Stochastic Expert Dropout

- **Function**: During training, low-weight experts are randomly deactivated with probability $p_j$. When $w_j < \gamma = 0.1$, $p_j = \beta - \frac{\beta}{\gamma}w_j$; otherwise $p_j = 0$ (with $\beta = 0.8$).
- **Mechanism**: Experts with lower weights are deactivated with higher probability (up to 80%); experts contributing more than 10% are never deactivated.
- **Design Motivation**: Inspired by the stochastic nature of brain signal processing. Dropout prevents the model from relying on the marginal contributions of all experts simultaneously, forcing each pathway to become self-sufficient — i.e., removing experts outside the pathway does not substantially degrade performance.

### Loss & Training

The full loss function is:

$$L = L_{fix} + \sum_{i}^{\mathcal{T}} \left( L_{response,i} + \frac{\alpha \cdot LPC_i}{L_{response,i} + \epsilon} \right)$$

- $L_{fix}$: mean squared error during the fixation period, requiring zero output.
- $L_{response,i}$: cross-entropy loss during the task response period.
- $\alpha = 10^{-5}$: routing cost weighting hyperparameter.
- Optimizer: Schedule-Free AdamW (lr=0.01, betas=(0.9, 0.999)).
- Training: 10 epochs × 1,000 steps, each step with batch size 128 × 350 timesteps × 115 features.
- Training completes in approximately 1 hour on a single NVIDIA T4 GPU.

## Key Experimental Results

### Table 1: Validation of Three Pathway Formation Criteria

| Criterion | Metric | Baseline | MoP |
|-----------|--------|----------|-----|
| Consistency | Routing pattern correlation across 20 runs | 0.03 | **0.51** (p<0.0001) |
| Self-sufficiency | Accuracy after removing low-weight experts | 98.2%→16.4% | 85.8%→**74.4%** |
| Distinctness | Different task clusters use distinct expert combinations | Uniform distribution | **Power-law distribution** (p<0.0001) |

### Ablation Study — Effect of Design Choices on Brain-Like Pathways

| Model Variant | Accuracy | Difficulty–Complexity Correlation (Fig. 5) | Learning Dynamics Correlation (Fig. 6) |
|---------------|----------|--------------------------------------------|----------------------------------------|
| Baseline HMoE | 91.1% ± 8.9% | -0.01 | -0.49*** |
| **MoP (full model)** | **83.0% ± 15.5%** | **0.54***** | **0.31**** |
| Without dropout | 90.1% ± 8.9% | 0.55*** | 0.03 |
| α=1e-4 (too strong) | 69.0% ± 20.1% | -0.57*** | -0.37*** |
| α=1e-6 (too weak) | 89.7% ± 9.0% | 0.62*** | 0.18 |
| Without task embedding | 83.0% ± 14.2% | 0.58*** | 0.58*** |

**Key Findings**:

- The MoP model exhibits a **positive correlation** between task difficulty and pathway complexity (r=0.54), meaning harder tasks automatically recruit more complex experts — analogous to the brain's Multiple-Demand system.
- In terms of learning dynamics, difficult tasks are initially routed through complex pathways and gradually transferred to simpler ones (r=0.31), recapitulating the cortical-to-subcortical skill transfer observed in the brain.
- Expert dropout is critical for learning dynamics: removing it reduces the Fig. 6 correlation from 0.31 to 0.03.
- All three inductive biases are necessary, reflecting **interaction effects** among them.

## Highlights & Insights

1. **Elegant research paradigm**: The MoE architecture serves as an analogue of multi-region brain interactions, and routing weight analysis mirrors neural pathway research, bridging AI architecture design and computational neuroscience.
2. **Three-criterion framework**: The proposed criteria of consistency, self-sufficiency, and distinctness provide a quantitative toolkit for evaluating pathway formation and offer new analytical tools for studying MoE internal structure.
3. **Emergence of brain-like learning dynamics**: The model spontaneously exhibits the behavior of routing through complex pathways during learning and transferring to simpler pathways upon mastery, closely paralleling experimental observations of cortical–subcortical interactions in the brain.
4. **Implications for MoE research**: Complexity-aware routing loss can serve as a task-driven load-balancing strategy with potential relevance to MoE design in large-scale LLMs.

## Limitations & Future Work

1. **Small model scale**: The model comprises only three layers with three experts each; whether the routing cost strategy scales to large-scale LLMs (e.g., DeepSeek-MoE) remains unvalidated.
2. **Forward-pass only**: Recurrent or feedback connections are absent, precluding the modeling of the extensive loop structures prevalent in the brain.
3. **No biological counterpart for the router**: The router is not explicitly mapped to a specific brain structure (though thalamic nuclei are a candidate).
4. **Limited task complexity**: The 82 time-series cognitive tasks remain relatively simple compared to the challenges of real-world cognition.
5. **Pathway identification relies on multiple tests**: Three independent tests are currently used to determine whether pathways have formed, lacking a unified quantitative metric.
6. **Accuracy reduction**: The MoP model (83.0%) underperforms the baseline (91.1%) by approximately 8 percentage points, indicating that pathway formation comes at a performance cost.

## Related Work & Insights

- **Brain-inspired modularity**: Extends the spatially constrained metabolic work of Achterberg et al. from module formation to pathway formation.
- **Heterogeneous MoE**: Building on Wang et al. (HMoE) and Raposo et al. (Mixture of Depths), this paper is the first to investigate how dynamic pathways form among heterogeneous experts.
- **Multi-region RNNs**: Compared to the multi-region RNNs of Kozachkov et al., the key innovation here is task-context-driven dynamic routing.
- **A new perspective on MoE load balancing**: Routing complexity loss can be interpreted as a task-driven load-balancing mechanism, distinct from the frequency-based balancing strategy used in DeepSeek-MoE.
- **Broader implication**: Incorporating metabolic constraints into MoE is not only a neuroscientific modeling tool but may also serve as a practical strategy for improving the interpretability and efficiency of MoE models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first systematic study of the conditions for pathway formation in heterogeneous MoE; the combined design of three inductive biases is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The three-criterion framework is comprehensive, ablation studies are detailed, and comparisons with brain data are convincing.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The narrative logic is clear, and the progressive structure — from "baseline fails to form pathways" to "incremental addition of biases" to "comparison with the brain" — is elegantly constructed.
- **Value**: ⭐⭐⭐⭐ — Makes a direct contribution to computational neuroscience and offers inspiration for large-scale MoE design, though scalability requires further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Contextual Dynamic Pricing with Heterogeneous Buyers](contextual_dynamic_pricing_with_heterogeneous_buyers.md)
- [\[NeurIPS 2025\] Out-of-distribution Generalisation is Hard: Evidence from ARC-like Tasks](out-of-distribution_generalisation_is_hard_evidence_from_arc-like_tasks.md)
- [\[CVPR 2026\] Do Vision Models Perceive Illusory Motion in Static Images Like Humans?](../../CVPR2026/others/do_vision_models_perceive_illusory_motion_in_static_images_like_humans.md)
- [\[NeurIPS 2025\] Zebra: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding](zebra_towards_zero-shot_cross-subject_generalization_for_universal_brain_visual_.md)
- [\[NeurIPS 2025\] Radar: Benchmarking Language Models on Imperfect Tabular Data](radar_benchmarking_language_models_on_imperfect_tabular_data.md)

</div>

<!-- RELATED:END -->
