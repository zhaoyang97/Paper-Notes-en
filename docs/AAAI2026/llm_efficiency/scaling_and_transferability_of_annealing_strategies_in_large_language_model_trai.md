---
title: >-
  [Paper Note] Scaling and Transferability of Annealing Strategies in Large Language Model Training
description: >-
  [AAAI 2026][LLM Efficiency][Learning Rate Annealing] This paper proposes a model-agnostic predictive framework that decomposes training loss into a forward-effect term (learning rate integral $S$), an annealing momentum term (Adam-style momentum integral $M$), and a model-size term $N$. It demonstrates that annealing strategies can be transferred from small models/small batches to large models/large batches, achieving a prediction MAPE below 2%.
tags:
  - "AAAI 2026"
  - "LLM Efficiency"
  - "Learning Rate Annealing"
  - "Scaling Law"
  - "Training Strategy"
  - "Transferability"
  - "MoE"
date: 2026-05-08
content_hash: 3f28d624b8c03657
---

# Scaling and Transferability of Annealing Strategies in Large Language Model Training

**Conference**: AAAI 2026
**arXiv**: [2512.13705](https://arxiv.org/abs/2512.13705)  
**Code**: [GitHub](https://github.com/xmed-lab/fm-annealing)  
**Area**: LLM Efficiency
**Keywords**: Learning Rate Annealing, Scaling Law, Training Strategy, Transferability, MoE

## TL;DR

This paper proposes a model-agnostic predictive framework that decomposes training loss into a forward-effect term (learning rate integral $S$), an annealing momentum term (Adam-style momentum integral $M$), and a model-size term $N$. It demonstrates that annealing strategies can be transferred from small models/small batches to large models/large batches, achieving a prediction MAPE below 2%.

## Background & Motivation

- **Background**: Learning rate schedules—particularly the annealing phase—significantly affect final performance in LLM training. Existing scaling laws (e.g., Chinchilla) primarily characterize the relationship between model size, training token count, and final loss, while neglecting training dynamics.
- **Limitations of Prior Work**: Identifying the optimal annealing strategy (annealing ratio, scheduler type, etc.) requires costly large-scale experiments. Even with identical total token counts, different training configurations yield substantially different loss curves, which existing frameworks cannot explain.
- **Key Challenge**: Prior forward–momentum scaling laws assume a fixed batch size; the forward term is not robust to batch size variation, and multiplicative momentum accumulation leads to numerical instability.
- **Key Insight**: The paper identifies training steps (rather than token count) as a more reliable unit for tracking loss, and reformulates the forward effect and annealing momentum in integral form to eliminate batch-size sensitivity and enable cross-configuration transfer.

## Method

### Overall Architecture

A unified loss prediction formula is proposed that decomposes training loss into three interpretable components:

$$L = \lambda_S \cdot S^{-\alpha_S} + \lambda_N \cdot N^{-\alpha_N} + \lambda_M \cdot M + L_0$$

where $S$ is the learning rate integral (forward effect), $N$ is the number of model parameters, and $M$ is the Adam-style annealing momentum integral.

### Key Designs

1. **Forward Effect Term $S$: Learning Rate Integral**

    - $S = \int_0^T \eta(t)\,dt$, integrating the learning rate over training steps.
    - Step-based tracking yields converging loss curves when batch size exceeds the critical batch size, unlike token-based tracking.
    - $S$ captures the cumulative "driving force" of the learning rate throughout training and serves as the primary driver of loss reduction.
    - The integral formulation is more robust to irregular step sizes than discrete summation.

2. **Annealing Momentum Term $M$: Adam-Style Momentum Integral**

    - Per-step momentum is computed using Adam-style first- and second-moment estimates.
    - After bias correction, momentum is accumulated as $M_t = M_{t-1} + \hat{m}_t / \sqrt{\hat{v}_t + \epsilon}$.
    - Compared to the multiplicative accumulation used in prior work (CMMT), the Adam-style formulation offers superior stability and generalization in transfer scenarios.
    - $M$ captures the "convergence effect" induced by learning rate decay during the annealing phase.

3. **Power-Law Relationship for Model Size Term $N$**

    - Follows the classical power-law form of scaling laws.
    - Fitting accuracy across model sizes is validated on both Dense (50M–1B) and MoE (100M–1.5B) models.
    - Combined with the forward and momentum terms, this enables annealing strategy prediction from small to large models.

4. **Transferability of the Optimal Annealing Ratio $R_\text{opt}$**

    - *Across learning rates*: $R_\text{opt}$ follows a power-law relationship with $\eta_\text{max}$, consistent across model sizes.
    - *Across model sizes*: $R_\text{opt}$ converges to the same value across different model scales.
    - *Across datasets*: $R_\text{opt}$ determined on the training set remains consistent on the validation set.
    - *Across training steps*: $R_\text{opt}$ follows a power-law relationship with $T$; longer training yields a smaller $R_\text{opt}$.

### Loss & Training

- All experiments use the AdamW optimizer with $\beta_1 = 0.9$ and $\beta_2 = 0.95$.
- Cosine and WSD (Warmup-Steady-Decay) schedulers are compared.
- Parameters fitted from the Cosine scheduler can predict loss curves under the WSD scheduler, and vice versa.
- The critical batch size follows a power law: $B_\text{opt}$ is inversely proportional to $L^{\alpha_B}$.

## Key Experimental Results

### Main Results

| Model Type | Parameter Range | Fitting MAPE | Cross-Scheduler Prediction MAPE |
|------------|----------------|-------------|----------------------------------|
| Dense | 50M–1B | <2% | 0.23%–0.80% |
| MoE | 100M–1.5B | <2% | 0.41%–0.72% |

### Ablation Study

| Transfer Dimension | Finding | Quantitative Result |
|--------------------|---------|---------------------|
| Across batch sizes | Step-based loss curves converge when $B > B_\text{opt}$ | Curves nearly overlap |
| Across learning rates | $R_\text{opt}$ follows power law with $\eta_\text{max}$ | Dense: $\alpha_\eta \approx 0.709$ |
| Across model sizes | $R_\text{opt}$ converges across scales | Consistent within same configuration |
| Across datasets | Train-to-validation transfer absolute error < 0.003 | Slight MAPE increase on validation set |
| Across training steps | $R_\text{opt}$ decreases as steps increase | 1B MoE: $\alpha_T \approx -0.946$ |

### Key Findings

- **Steps outperform tokens as the loss-tracking unit**: Within the critical batch size regime, step-based loss curves are more stable than token-based ones.
- **Adam-style momentum outperforms multiplicative accumulation**: It yields greater stability in cross-scheduler transfer and resolves numerical instability.
- **Cosine and WSD schedulers are mutually predictive**: Parameters fitted from one scheduler can predict the other with MAPE < 1%.
- **Longer training warrants a smaller annealing ratio**: Extended training allows the model to sufficiently explore the parameter space without premature annealing.
- **Larger models incur greater performance penalties from suboptimal annealing**: Model scale amplifies sensitivity to annealing strategy selection.

## Highlights & Insights

- **A paradigm shift from "tuning" to "prediction"**: The paper reformulates annealing strategy selection from an expensive empirical search into a mathematically predictable and transferable problem, offering direct cost savings for large-scale LLM training.
- **Elegant unification via integral formulation**: The learning rate integral and momentum integral cleanly capture the forward driving force and annealing convergence effect, respectively, with clear physical intuition.
- **First systematic study of annealing dynamics in MoE models**: The paper demonstrates that MoE and Dense models obey the same power-law framework, albeit with different coefficients.

## Limitations & Future Work

- The largest model evaluated contains 1.5B parameters; generalizability to models exceeding 100B remains to be verified.
- Only Cosine and WSD schedulers are evaluated; more complex scheduling strategies are not covered.
- Power-law estimates of the critical batch size may be inaccurate under extreme configurations.
- The effect of data mixture variation on annealing strategy is not considered.

## Related Work & Insights

- **vs. Chinchilla Scaling Law**: Chinchilla focuses on the optimal ratio of token count to model size while ignoring training dynamics; this paper incorporates training dynamics into the scaling framework.
- **vs. Tissue et al. Forward–Momentum Scaling Law**: Tissue et al. assume a fixed batch size and use multiplicative momentum accumulation; this paper eliminates batch-size dependency via integration and improves stability with Adam-style momentum.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The predictive framework for annealing strategies and multi-dimensional transferability analysis represent genuine contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive validation across batch size, scheduler, model size, dataset, and training steps.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear and the experimental design is systematic.
- **Value**: ⭐⭐⭐⭐⭐ — Directly actionable for improving LLM training efficiency with substantial savings in hyperparameter search costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scaling Large Vision-Language Model RL Training via Efficient Load Balancing](../../ICLR2026/llm_efficiency/scaling_large_vision-language_model_rl_training_via_efficient_load_balancing.md)
- [\[ICLR 2026\] Unlocking Full Efficiency of Token Filtering in Large Language Model Training](../../ICLR2026/llm_efficiency/unlocking_full_efficiency_of_token_filtering_in_large_language_model_training.md)
- [\[ICLR 2026\] UltraLLaDA: Scaling the Context Length to 128K for Diffusion Large Language Models](../../ICLR2026/llm_efficiency/ultrallada_scaling_the_context_length_to_128k_for_diffusion_large_language_model.md)
- [\[ICLR 2026\] ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding](../../ICLR2026/llm_efficiency/refusion_a_diffusion_large_language_model_with_parallel_autoregressive_decoding.md)
- [\[ICLR 2026\] Demystifying and Enhancing the Efficiency of Large Language Model Based Search Agents](../../ICLR2026/llm_efficiency/demystifying_and_enhancing_the_efficiency_of_large_language_model_based_search_a.md)

</div>

<!-- RELATED:END -->
