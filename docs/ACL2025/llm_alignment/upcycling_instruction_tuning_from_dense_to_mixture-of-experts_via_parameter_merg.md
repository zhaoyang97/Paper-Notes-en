---
title: >-
  [Paper Note] Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging
description: >-
  [ACL 2025][LLM Alignment][Mixture-of-Experts] This paper proposes UpIT (Upcycling Instruction Tuning), which leverages intermediate checkpoints from the instruction tuning process of a dense model as specialized experts, and achieves data-efficient and flexible dense-to-MoE upcycling through genetic algorithm-based expert expansion and router pre-optimization.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Mixture-of-Experts"
  - "Upcycling"
  - "Parameter Merging"
  - "Expert Diversity"
  - "Router Initialization"
date: 2026-05-08
content_hash: 2e42301cf669c9b3
---

# Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging

**Conference**: ACL 2025  
**arXiv**: [2410.01610](https://arxiv.org/abs/2410.01610)  
**Code**: None  
**Area**: RLHF Alignment  
**Keywords**: Mixture-of-Experts, Upcycling, Parameter Merging, Expert Diversity, Router Initialization

## TL;DR
This paper proposes UpIT (Upcycling Instruction Tuning), which leverages intermediate checkpoints from the instruction tuning process of a dense model as specialized experts, and achieves data-efficient and flexible dense-to-MoE upcycling through genetic algorithm-based expert expansion and router pre-optimization.

## Background & Motivation
- **Background**: MoE architectures significantly expand model capacity via sparse activation without increasing inference overhead, becoming an important direction for LLM efficiency improvements. Upcycling (converting dense models to MoE) is more efficient than training MoE from scratch.
- **Limitations of Prior Work**:
    - **Vanilla Upcycling** (duplicating FFN layers $\rightarrow$ large-scale post-training): Experts are initially homogeneous, requiring ~1T tokens or ~5M instruction data.
    - **Specialized Upcycling** (training domain experts first $\rightarrow$ assembling): Requires hundreds of billions of domain tokens, and the number of experts is inflexible.
- **Key Challenge**: How to **flexibly** convert a dense pre-trained model into a high-quality MoE instruction-following model with **very limited data**?
- **Key Insight**: Intermediate checkpoints from different training stages of instruction tuning naturally exhibit expertise in distinct domains (e.g., the checkpoint with the best MMLU performance differs from the one with the best GSM8K performance). These checkpoints are naturally suited as specialized experts!

## Method

### Overall Architecture
UpIT consists of four stages:
1. **Expert Preparation**: Fine-tune the dense model and regularly save checkpoints as experts.
2. **Expert Expansion**: Scale to an arbitrary number of experts using a genetic algorithm combined with parameter merging.
3. **Router Initialization**: Pre-optimize router vectors to ensure experts utilize their respective strengths.
4. **Model Upcycling**: Combine experts and routers for post-training.

### Key Designs
1. **Checkpoint-based Expert Preparation**

    - *Key Observation*: Checkpoints from different training epochs exhibit interleaved optimal performance across different benchmarks.
    - *Example*: The checkpoint at epoch 2 performs best on HellaSwag, while the checkpoint at epoch 0.25 performs best on MMLU.
    - Diverse experts can be obtained simply by saving checkpoints, without the need for carefully curated domain-specific data.
    - *Design Motivation*: Dramatically reduce the cost of obtaining experts, from "tens of billions of domain tokens" to "free intermediate products."

2. **Genetic Algorithm-driven Expert Expansion**

    - *Problem*: The number of intermediate checkpoints is fixed and may not match the target number of experts.
    - *Solution*: In each round, select the two most diverse experts as "parents."
    - Randomly allocate weights $\alpha, \beta$ ($\alpha + \beta = 1$).
    - Use DARE (Drop And REscale) to introduce "mutations" before merging.
    - New expert: $\mathbf{E}_{new} = \text{DARE}(\alpha \mathbf{E}_{j^*}, \beta \mathbf{E}_{k^*})$
    - *Crucial Step*: Selecting the two most dissimilar experts rather than random ones guarantees the diversity of new experts.

3. **Seed Data-based Router Pre-optimization**

    - *Problem*: Randomly initialized routers routing tokens to incorrect experts, undermining the pre-established expert uniqueness.
    - *Data Selection*: Randomly sample 1% of the training set (approx. 500-5000 samples), calculate the PPL of each sample across all experts, and assign each sample to the expert with the lowest PPL.
    - Auxiliary loss: $\mathcal{O}_i = \min_{\mathbf{E}_i}(\alpha \mathcal{L}_{lm} + (1-\alpha)\mathcal{L}_{aux})$
    - $\mathcal{L}_{aux} = \text{CrossEntropy}(\text{Sigmoid}(\mathbf{h}_{\mathbf{r}_i}), \mathbf{I})$: Maximizing the routing output probability for the specialty data of each expert.
    - Extremely low-cost pre-optimization requiring only 1% of the data and 4 epochs.

### Loss & Training
- **Router Initialization Stage**: $\alpha \mathcal{L}_{lm} + (1-\alpha)\mathcal{L}_{aux}$, where $\alpha=0.5$
- **Post-training Stage**: Standard causal LM loss + load balancing loss $\mathcal{L}_{load} = n \cdot \sum_i f_i \cdot P_i$
- **Training Schedule**: 4 epochs in total. The first 2 epochs are for expert preparation, and the remaining 2 epochs are for MoE post-training.
- **LoRA-based**: Learning rate (LR) of 2e-4; **FFN-based**: LR of 2e-5

## Key Experimental Results

### Main Results (LoRA-based, 8E Top-2)

| Method | HumanEval | GSM8K | MMLU | NQ | Avg. |
|------|-----------|-------|------|-----|------|
| LoRA Baseline | 22.56 | 45.72 | 49.33 | 14.99 | 47.53 |
| LoRAMoE_SFT | 28.66 | 49.81 | 50.54 | 20.55 | 49.99 |
| Self-MoE | 28.05 | 46.70 | 49.63 | 21.11 | 49.30 |
| **UpIT** | **35.37** | **49.51** | **50.31** | **24.52** | **52.21** |

### Main Results (FFN-based, 4E Top-2)

| Method | HumanEval | GSM8K | MMLU | NQ | Avg. |
|------|-----------|-------|------|-----|------|
| SFT Baseline | 26.22 | 29.19 | 33.93 | 8.42 | 31.31 |
| Upcycle_SFT | 23.17 | 33.97 | 38.90 | 15.18 | 37.60 |
| **UpIT** | **31.34** | **33.81** | **40.84** | **14.71** | **38.88** |

### Data Efficiency

| Configuration | Key Metrics | Note |
|------|---------|------|
| UpIT(8E) with 50K data | 47.13 avg | ≈ Level of LoRAMoE(8E) with 500K data |
| UpIT(16E) with 100K data | 49.18 avg | > LoRAMoE(16E) with 500K data |
| Data growth curve | Near-linear growth | Baseline methods exhibit a logarithmic curve (growth saturation) |

### Ablation Study

| Configuration | Avg. | Note |
|------|------|------|
| Full UpIT | 52.21 | - |
| w/o Router Initialization | 49.96 | -2.25, demonstrating the criticality of router pre-optimization |
| Router initialized with random data | 49.30 | Worse than no initialization (destructs diversity) |
| w/o Expert Expansion | 53.31 | Direct checkpoint usage works but is inflexible |
| Merging randomly selected parents | 52.41 | -0.90, proving the importance of selecting the most diverse parents |
| Checkpoints from the first half | 51.37 | Second half is better (continuous improvement in Math/Code) |

### Key Findings
- **Stunning Data Efficiency**: Matches the performance of conventional methods with 500K data using only 50K data (a 10x data efficiency improvement).
- **Scalable Expert Count**: UpIT gains stable improvements as the number of experts increases, whereas vanilla upcycling suffers performance degradation.
- **Routing Analysis**: UpIT's router accurately routes tokens from distinct domains to specific experts (e.g., HumanEval $\rightarrow$ Expert 4, MMLU $\rightarrow$ Expert 3), whereas LoRAMoE distributes them uniformly.
- **Higher Performance Ceiling**: With continued training epochs, UpIT maintains linear growth while baseline methods saturate.

## Highlights & Insights
- **Brilliant Core Insight**: Discovering that intermediate checkpoints exhibit distinct domain specialization—a natural yet highly valuable observation.
- **Design Consistency**: All designs across the four stages center around the core goal of "maintaining/enhancing expert diversity".
- **Engineering Friendly**: Introduces no requirement for extra training data, and router initialization requires only 1% data trained for 4 epochs.
- **Strong Generalization**: Effective in both LoRA-based and FFN-based paradigms.
- **Clever Application of Genetic Algorithm**: Selecting the most diverse "parents" for "parameter crossover + DARE mutation" to generate new experts, mimicking biological evolution.

## Limitations & Future Work
- Experiments are only conducted on Llama 2 7B and Sheared Llama 2.7B; the efficacy on larger models (e.g., 70B+) remains to be validated.
- The selection of the checkpoint saving interval (every 0.25 epoch) lacks thorough ablation.
- Router initialization relies on PPL for data selection, assuming low PPL equals expertise. However, the discriminative power of PPL may vary across different tasks.
- Contrastive studies against methods utilizing large-scale domain-specific data (e.g., Branch-Train-MiX) under similar data scales are missing.
- **Future Directions**: (1) Can the optimal checkpoint saving interval and count be automatically determined? (2) Can the number of experts or active experts be dynamically adjusted during inference?

## Related Work & Insights
- Relation to MoE Jetpack (Zhu et al., 2024b): MoE Jetpack also leverages checkpoints, but this work integrates them more systematically into a complete upcycling pipeline.
- Connection to Model Merging: Methods such as DARE and TIES-Merging are innovatively applied for MoE expert expansion.
- Comparison with Self-MoE (Kang et al., 2024): Self-MoE requires independently training specialized experts, while UpIT leverages naturally occurring diversities.
- Insight: Any scenario requiring the derivation of diverse modules from a single model can draw inspiration from the 'checkpoints as experts' approach.

## Rating
- Novelty: ⭐⭐⭐⭐ The insight of using intermediate checkpoint deviations as experts is novel and practical, and the genetic algorithm-based expansion is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive data-scale ablations, expert-count ablations, scale explorations, routing visualizations, and multiple ablation analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-defined synergy between Algorithms and Figures, and a natural transition from problem definition to methodology.
- Value: ⭐⭐⭐⭐⭐ Significantly lowers the data requirement for MoE training, resulting in a highly practical and reproducible methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Rethinking Table Instruction Tuning](rethinking_table_instruction_tuning.md)
- [\[ACL 2025\] JsonTuning: Towards Generalizable, Robust, and Controllable Instruction Tuning](jsontuning_towards_generalizable_robust_and_controllable_instruction_tuning.md)
- [\[ACL 2025\] Call for Rigor in Reporting Quality of Instruction Tuning Data](call_for_rigor_in_reporting_quality_of_instruction_tuning_data.md)
- [\[ACL 2025\] Beyond Similarity: A Gradient-based Graph Method for Instruction Tuning Data Selection](beyond_similarity_a_gradient-based_graph_method_for_instruction_tuning_data_sele.md)
- [\[ACL 2025\] Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric](measuring_data_diversity_for_instruction_tuning_a_systematic_analysis_and_a_reli.md)

</div>

<!-- RELATED:END -->
