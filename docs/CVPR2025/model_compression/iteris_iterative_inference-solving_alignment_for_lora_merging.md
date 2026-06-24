---
title: >-
  [Paper Note] IterIS: Iterative Inference-Solving Alignment for LoRA Merging
description: >-
  [CVPR 2025][Model Compression][LoRA Merging] IterIS proposes an iterative inference-solving method for LoRA merging. By directly extracting the input features of the unified adapter (rather than using approximations) to establish a more accurate optimization objective, combined with regularization to reduce sample requirements to 1-5% of prior methods, and introducing adaptive weight balancing optimization, IterIS significantly outperforms baselines in LoRA merging across tex…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "LoRA Merging"
  - "Multi-task Models"
  - "Iterative Optimization"
  - "Parameter-Efficient Fine-Tuning"
  - "Multi-concept Customization"
date: 2026-05-08
content_hash: fcfe72900a0350c4
---

# IterIS: Iterative Inference-Solving Alignment for LoRA Merging

**Conference**: CVPR 2025  
**arXiv**: [2411.15231](https://arxiv.org/abs/2411.15231)  
**Code**: [https://github.com/HKUST-LongGroup/IterIS-merging](https://github.com/HKUST-LongGroup/IterIS-merging)  
**Area**: Diffusion Models  
**Keywords**: LoRA Merging, Multi-task Models, Iterative Optimization, Parameter-Efficient Fine-Tuning, Multi-concept Customization

## TL;DR
IterIS proposes an iterative inference-solving method for LoRA merging. By directly extracting the input features of the unified adapter (rather than using approximations) to establish a more accurate optimization objective, combined with regularization to reduce sample requirements to 1-5% of prior methods, and introducing adaptive weight balancing optimization, IterIS significantly outperforms baselines in LoRA merging across text-to-image diffusion models, vision-language models, and large language models.

## Background & Motivation

1. **Background**: LoRA is currently the most popular parameter-efficient fine-tuning method, often used to train task-specific LoRAs for different downstream tasks. When multi-task capabilities are required, LoRA merging can combine multiple LoRAs into a unified adapter without accessing the training data.

2. **Limitations of Prior Work**: Existing real-distribution-based LoRA merging methods (such as RegMean, Mix-of-Show) suffer from three key limitations: (1) **Crude assumption**: assuming that the input features of the unified adapter are identical to those of each task-specific LoRA, which deviates increasingly with network depth; (2) **Massive sample requirements**: a large number of unlabeled samples are typically required to ensure invertibility of the inner product matrix and enhance robustness; (3) **Imbalanced optimization**: differences in feature magnitudes across different tasks cause certain terms to dominate the direction of the solution in the optimization objective.

3. **Key Challenge**: Real-distribution methods employ a "lazy" approximation when establishing the optimization objective—specifically, using the input features of each individual LoRA to replace the input features of the unified adapter. While this approximation is acceptable in shallow layers, cumulative errors lead to severe performance degradation as design depth increases.

4. **Goal**: (1) Eliminate the crude assumption by using the actual input features of the unified adapter; (2) Reduce sample requirements; (3) Balance multi-task optimization.

5. **Key Insight**: Deep learning models have a directed acyclic graph (DAG) structure—if merging is performed layer-by-layer from shallow to deep layers, the input features of the unified adapter for subsequent layers can be directly obtained through inference once the preceding layers have been merged.

6. **Core Idea**: Iteratively "perform inference to obtain the true input features of the unified adapter $\rightarrow$ establish an accurate optimization objective $\rightarrow$ solve for a better unified adapter" to gradually approach the optimal solution.

## Method

### Overall Architecture
The input consists of $N$ task-specific LoRA-tuned models and a small number of unlabeled samples. The output is a unified model (pretrained model + merged adapter). Workflow: Initialization (using the input features of each LoRA as the initial value of $\tilde{X}_{nj}$) $\rightarrow$ Iteration (solving the closed-form solution layer-by-layer to update the adapter $\rightarrow$ performing inference to obtain new input features $\rightarrow$ updating the optimization objective) $\rightarrow$ Convergence output.

### Key Designs

1. **Iterative Inference-Solving Framework**:

    - Function: Gradually eliminate the approximation error caused by the crude assumption.
    - Mechanism: The optimization objective of the original method is $W^* = \arg\min_W \sum_i \|W_i^T X_i - W^T X_i\|_F^2$, where $X_i$ on the right side should be the input feature of the unified adapter $\tilde{X}_i$, but is approximated by the input feature of each individual LoRA $X_i$. IterIS modifies the optimization objective to $W^* = \arg\min_W \sum_i \lambda_i \|W_i^T X_i - W^T \tilde{X}_i\|_F^2$, which yields the closed-form solution: $W^* = (\sum_i \lambda_i \tilde{X}_i \tilde{X}_i^T)^{-1}(\sum_i \lambda_i \tilde{X}_i X_i^T W_i)$. Initially, $\tilde{X}_i = X_i$, and $\tilde{X}_i$ is updated through model inference with the current unified model after each iteration.
    - Design Motivation: Leveraging the property of the DAG structure, merging the preceding layers naturally changes the input distribution of the subsequent layers. By iteratively replacing the approximated input features with the real input features, convergence can theoretically be achieved in $J-1$ iterations (for a $J$-layer encoder).

2. **Efficient Regularization to Reduce Sample Requirements**:

    - Function: Maintain algorithm robustness while requiring only 1-5% of the samples.
    - Mechanism: Add regularization terms to the inner product matrices in the closed-form solution: $\tilde{X}_i\tilde{X}_i^T + \alpha\|\tilde{X}_i\tilde{X}_i^T\|_F \cdot I$ and $\tilde{X}_i X_i^T + \alpha\|\tilde{X}_i X_i^T\|_F \cdot I$. The regularization parameter $\alpha$ is typically set to $10^{-4}$ or smaller. Scaling the identity matrix by the Frobenius norm adaptively matches the regularization strength to the overall magnitude of the matrices.
    - Design Motivation: The original method requires a large number of samples to ensure numerical stability and invertibility of the inner product matrices. The regularization terms keep the matrix well-conditioned even with few-shot samples while preventing overfitting.

3. **Adaptive Weight Balancing**:

    - Function: Alleviate the imbalance in optimization objectives across different tasks.
    - Mechanism: Define the adaptive weight as $\lambda_i = \|W_i\|_F^2 \cdot \|W_i^T X_i\|_F^{-2}$. Intuitively, when the output feature magnitude $\|W_i^T X_i\|_F$ of a specific task is large, its weight is inversely reduced to prevent this task from dominating the optimization objective.
    - Design Motivation: Under uniform weights, tasks with large feature magnitudes will dominate the direction of the solution, causing performance degradation for other tasks. Adaptive weights balance the contribution of each term based on the ratio of LoRA weights to output feature magnitudes.

### Loss & Training
IterIS is a training-free method, with its core being the layer-by-layer and iteration-by-iteration update of closed-form solutions. Each iteration consists of one forward pass on all samples and the calculation of closed-form solutions for $J$ layers. The maximum number of iterations is set to 20 in the experiments. For text-to-image diffusion models, 50 input samples are used for inference. The entire workflow requires no annotated data or gradient computation.

## Key Experimental Results

### Main Results: Multi-Concept Customized Generation (7 Concept Pairs, Stable Diffusion v1.5)

| Metric | Custom Diffusion | Textual Inversion | Linear Merging | IterIS |
|------|-----------------|-------------------|----------------|--------|
| Image-align1↑ | 0.6881 | 0.6569 | 0.6811 | **0.6889** |
| Image-align2↑ | 0.7091 | 0.6887 | 0.6963 | **0.7124** |
| Text-align↑ | 0.6509 | 0.6363 | 0.6547 | **0.6800** |

### Multi-Style Caption Generation (BLIP + SentiCap Positive/Negative Sentiment)

| Method | Acc(pos,neg)↑ | CIDEr↑ | B-4↑ |
|------|--------------|--------|------|
| Linear merging | (0.522, 0.557) | 0.752 | 0.142 |
| RegMean | (0.624, 0.692) | 0.790 | 0.150 |
| **IterIS** | **(0.831, 0.781)** | **0.794** | **0.152** |

### Ablation Study
- Analysis on crude assumption: The Frobenius distance in the deep layers of the encoder shows that the approximation bias of RegMean increases sharply with depth, whereas IterIS completely eliminates this bias.
- Sample requirements: IterIS requires only 1-5% of the unlabeled samples compared to RegMean, while also reducing the running time accordingly.
- Weight balancing: The optimization terms $T_1$ and $T_2$ in RegMean differ vastly in magnitude; IterIS balances their ratio via adaptive weights.

### Key Findings
- Iteration typically converges within 20 steps, owing to the property of the DAG structure.
- IterIS slightly outperforms Custom Diffusion (which uses gradient training and data regularization) in multi-concept customization.
- In NLP multi-task merging, IterIS also significantly surpasses baselines like RegMean, demonstrating the generalizability of the method.
- The regularization term allows 50 samples to be sufficient (whereas RegMean requires thousands), drastically lowering the barrier for practical usage.

## Highlights & Insights
- **Inference-solving iterative paradigm**: This is a very clean approach—instead of settling for approximations, it directly obtains the actual input features via inference to build the optimization objective. By leveraging the hierarchical nature of the DAG structure, once the preceding layers are solved, the inputs to the deeper layers naturally become accurate.
- **Minimal sample requirement**: Only 1-5% of unlabeled samples are required, achieved through adaptive regularization. This makes LoRA merging far more practical in data-constrained scenarios (e.g., privacy protection, intellectual property).
- **High versatility**: The exact same method framework is applicable to LoRA merging in three distinct domains: diffusion models, VLMs, and LLMs, reflecting methodological clarity.

## Limitations & Future Work
- During iteration, a forward pass on all samples is required in each round, which may still incur computational overhead for exceptionally large models.
- The closed-form solution assumes a quadratic objective function, which may not be precise enough for LoRA merging of certain non-linear layers.
- The maximum iteration limit is set to 20 to prevent overfitting, but the optimal number of iterations may vary across different models.
- The design of adaptive weights is empirical and lacks guarantees of theoretical optimality.
- Future works to explore: extending to the merging of PEFT methods beyond LoRA (such as adapters or prompt tuning), or combining with gradient-based fine-tuning for hybrid optimization.

## Related Work & Insights
- **vs RegMean**: RegMean uses the same optimization framework but assumes $\tilde{X}_i = X_i$, whereas IterIS eliminates this approximation through iterative inference. In multi-style captioning, IterIS's positive/negative sentiment accuracy of (0.831, 0.781) vastly outperforms RegMean's (0.624, 0.692).
- **vs Mix-of-Show**: Mix-of-Show uses LBFGS to solve a similar optimization on diffusion models but suffers from the same crude assumption issue and requires gradient computation. IterIS is more efficient using closed-form solutions.
- **vs Linear Merging/Task Arithmetic**: These methods assume isotropic distribution of input features, and this oversimplification leads to poor performance. IterIS achieves the best performance among all baselines.

## Rating
- Novelty: ⭐⭐⭐⭐ The iterative inference-solving paradigm is elegant and simple. The analysis of the crude assumption is in-depth, and the solution is natural.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three domains: diffusion models, VLMs, and LLMs, with detailed ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear problem analysis (intuitive diagrams for the three limitations) and rigorous methodological derivation.
- Value: ⭐⭐⭐⭐ LoRA merging is a genuine demand in practical scenarios, and IterIS provides a significantly better solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging](../../ACL2025/model_compression/osrm_lora_merging_orthogonal.md)
- [\[ACL 2026\] Evolutionary Negative Module Pruning for Better LoRA Merging](../../ACL2026/model_compression/evolutionary_negative_module_pruning_for_better_lora_merging.md)
- [\[ICML 2025\] Make LoRA Great Again: Boosting LoRA with Adaptive Singular Values and Mixture-of-Experts Optimization Alignment](../../ICML2025/model_compression/make_lora_great_again_boosting_lora_with_adaptive_singular_values_and_mixture-of.md)
- [\[ACL 2026\] LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging](../../ACL2026/model_compression/lora_on_the_go_instance-level_dynamic_lora_selection_and_merging.md)
- [\[ICCV 2025\] CIARD: Cyclic Iterative Adversarial Robustness Distillation](../../ICCV2025/model_compression/ciard_cyclic_iterative_adversarial_robustness_distillation.md)

</div>

<!-- RELATED:END -->
