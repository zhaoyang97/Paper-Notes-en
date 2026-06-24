---
title: >-
  [Paper Note] A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models
description: >-
  [ACL 2025][LLM (Other)][Layer Freezing] By analyzing the transition traces of latent representations during LLM inference to compute the semantic deviation of each layer, combined with a derived scaling law formula to estimate each layer's contribution to reducing loss, this paper determines "which layers to fine-tune," achieving an efficient fine-tuning approach that is orthogonal to PEFT.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Layer Freezing"
  - "Fine-Tuning Efficiency"
  - "Semantic Deviation"
  - "Scaling Laws"
  - "Backpropagation Optimization"
date: 2026-05-08
content_hash: f39059f5e6349b31
---

# A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models

**Conference**: ACL 2025  
**arXiv**: [2406.11753](https://arxiv.org/abs/2406.11753)  
**Code**: None  
**Area**: LLM Fine-Tuning Efficiency  
**Keywords**: Layer Freezing, Fine-Tuning Efficiency, Semantic Deviation, Scaling Laws, Backpropagation Optimization

## TL;DR

By analyzing the transition traces of latent representations during LLM inference to compute the semantic deviation of each layer, combined with a derived scaling law formula to estimate each layer's contribution to reducing loss, this paper determines "which layers to fine-tune," achieving an efficient fine-tuning approach that is orthogonal to PEFT.

## Background & Motivation

Fine-tuning is a critical step for adapting pre-trained language models to downstream tasks, but full-parameter fine-tuning incurs massive computational costs. Existing works primarily focus on "**how to fine-tune**"—for example, Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA and Adapters reduce costs by decreasing the number of trainable parameters. However, an overlooked orthogonal problem is "**where to fine-tune**"—even with PEFT, if fine-tuning is performed across all layers, the computational overhead of backpropagation remains high.

**Key Challenge**: The contributions of different layers to downstream tasks are non-uniform. Earlier layers typically encode general linguistic features (syntax, lexicology), whereas latter layers encode more task-specific semantic information. Blindly fine-tuning all layers not only wastes computational resources but may also degrade generalization capability by excessively modifying existing general representations.

**Limitations of Prior Work**: Layer freezing is not a novel concept, but prior methods either rely on simple heuristics (e.g., freezing the first N layers) or require expensive search processes to determine the optimal freezing strategy. There is a lack of a **theoretically grounded framework** to automatically determine which layers are worth fine-tuning.

**Key Insight**: **Through semantic analysis of LLM inference**, this work leverages the transition traces of latent representations across layers to quantify the "semantic deviation" of each layer. It then estimates the fine-tuning gain of each layer using scaling laws to determine the optimal range of layers to fine-tune with minimal search cost.

## Method

### Overall Architecture

Given a pre-trained model and downstream task data, the method consists of three steps: (1) performing a single forward pass on downstream data to collect the transition traces of latent representations for each layer, (2) calculating the semantic deviation of each layer, and (3) estimating the gain of fine-tuning each layer in reducing the total loss using a derived scaling law formula to determine the optimal set of fine-tuning layers. Finally, backpropagation is executed only on the selected layers.

### Key Designs

1. **Transition Traces and Semantic Deviation**:

    - **Function**: Quantifying the degree of representation change after the input passes through each Transformer layer.
    - **Mechanism**: For layer $l$, the transition trace is defined as the difference between the input and output hidden states. Let $\mathbf{h}_l$ be the output of layer $l$, the semantic deviation of layer $l$ is defined as:

    $d_l = \|\mathbf{h}_l - \mathbf{h}_{l-1}\|$

      This deviation measures the "amount of modification" the layer applies to the representation. On downstream task data, if the average deviation of a layer is large, it indicates a significant "semantic gap" between the current parameters of that layer and the representations required by the downstream task—meaning this layer has a greater demand for fine-tuning.
    - **Design Motivation**: Intuitively, if a layer barely changes the representation (low deviation), the gain from fine-tuning it is minimal; conversely, layers with high deviation possess a larger optimization space.

2. **Scaling-Law-Based Layer Gain Estimation**:

    - **Function**: Deriving a formula to estimate the contribution of fine-tuning a specific layer to reducing the total loss.
    - **Mechanism**: Drawing on the concepts of neural scaling laws, the fine-tuning gain of each layer is modeled as a function of its semantic deviation. The relationship between the layer gain $G_l$ and the deviation $d_l$ is derived as:

    $G_l \propto f(d_l, \theta_l)$

      where $\theta_l$ represents the layer parameters. Through this formula, the fine-tuning gain of each layer can be **estimated** without actually performing fine-tuning, thereby selecting the layers with the highest gains for adaptation.
    - **Design Motivation**: To avoid brute-force search over all possible combinations of layer freezing ($2^L$ combinations), reducing the search cost to a single forward pass plus formula computation.

3. **Cost-Benefit Balance and Layer Selection**:

    - **Function**: Finding the optimal balance point between fine-tuning performance and computational cost.
    - **Mechanism**: Layers are sorted by their estimated gains $G_l$, and selected for fine-tuning from highest to lowest until the marginal gain falls below a threshold. This naturally forms an **adaptive layer selection strategy**—different tasks and models may require fine-tuning different numbers of layers at various positions.
    - **Design Motivation**: The strategy of freezing the first N layers is too coarse. This method allows non-contiguous selection (e.g., freezing layer 5 but fine-tuning layers 4 and 6), offering greater flexibility.

### Orthogonality to PEFT

Key insight: This method addresses "where to fine-tune," whereas PEFT addresses "how to fine-tune within selected layers." The two can be **used in combination**—first determining the layers that need fine-tuning using this method, and then applying PEFT methods such as LoRA on those layers, achieving a dual efficiency gain.

## Key Experimental Results

### Main Results

| Method | Dataset | Performance | Training FLOPs | Description |
|------|--------|------|----------|------|
| Full Fine-tuning | Multiple NLU/NLG | Baseline Performance | 100% | Full-parameter fine-tuning |
| Freeze first N layers | Multiple NLU/NLG | Slight drop | ~50-70% | Simple heuristic |
| Ours | Multiple NLU/NLG | Comparable to or exceeding full fine-tuning | ~40-60% | Semantic-aware selection |
| LoRA (All layers) | Multiple NLU/NLG | Close to full fine-tuning | ~30% parameters | Parameter-efficient but backpropagation across all layers |
| Ours + LoRA | Multiple NLU/NLG | Close to full fine-tuning | Dual savings | Orthogonal combination |

### Ablation Study

| Configuration | Performance Change | Compute Savings | Description |
|------|---------|---------|------|
| Freeze only low-deviation layers | Almost no loss | ~30-40% | Validates the effectiveness of the deviation metric |
| Freeze only high-deviation layers | Significant drop | - | Counter-validation: High-deviation layers are indeed important |
| Randomly freeze the same number of layers | Unstable, average drop | Same | Demonstrates that semantic-aware selection outperforms random selection |
| Different scaling law parameters | Robust | - | The method is insensitive to hyperparameters |

### Key Findings

- **High-deviation layers are concentrated in middle and latter layers**, which aligns with intuition—these layers encode more task-specific semantic information.
- While saving 40-60% of the backpropagation computation, the performance remains basically unaffected, with some tasks even showing slight gains.
- **Combining with LoRA** can further reduce computational overhead, validating their orthogonality.
- The optimal freezing strategies vary across different models and tasks, highlighting the necessity of **adaptive selection**.
- The fine-tuning gain of a layer does not perfectly align with simple layer depth—it is not always the case that deeper layers require more fine-tuning.

## Highlights & Insights

- **Filling a Research Gap**: Pioneeringly addresses "where to fine-tune" as an independent research problem, which complements PEFT's "how to fine-tune".
- **Combining Theory and Practice**: Instead of relying purely on empirical layer selection, this work provides theoretical support through semantic analysis and scaling law derivations.
- **Plug-and-Play**: As an orthogonal method, it can be combined with any existing fine-tuning strategy (full fine-tuning, LoRA, Adapter, etc.).
- **High Practicality**: Layer selection only requires a single forward pass, incurring minimal overhead.

## Limitations & Future Work

- The scaling law formula may contain hyperparameters that need to be fitted on a validation set, increasing the complexity of the method.
- The calculation of layer deviation is based on a forward pass snapshot, which may not fully reflect the dynamic changes in layer importance during the fine-tuning process.
- For extreme scenarios (such as cross-domain transfer or multi-task learning), the accuracy of estimating semantic deviation is yet to be verified.
- Experiments do not cover ultra-large models (e.g., 70B+), where the gains of layer freezing could be even more significant.

## Related Work & Insights

- **vs LoRA (Hu et al., 2022)**: LoRA focuses on efficiency in the parameter dimension (low-rank decomposition), while this method focuses on efficiency in the layer dimension (selective backpropagation), making them orthogonally complementary.
- **vs FreezeOut (Brock et al., 2017)**: FreezeOut progressively freezes layers based on training phases, lacking task adaptability; this method selects layers based on semantic analysis, making it more data-driven.
- **vs Layer-Drop (Fan et al., 2020)**: Layer-Drop randomly drops layers during inference to speed up inference, whereas this method freezes layers during training to accelerate training, targeting different objectives.
- **vs SmartFRZ (Llona et al., 2023)**: SmartFRZ decides which layers to freeze based on gradient information, requiring additional training overhead; this method only requires a forward pass.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ For the first time, systematically answers "where to fine-tune" from the perspective of semantic analysis, opening up a new research direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple models and datasets with ablation studies, but lacks experiments on ultra-large models.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and sound methodology derivation, but some theoretical derivations may not be intuitively easy to follow.
- Value: ⭐⭐⭐⭐⭐ Extremely high practical value; its orthogonality with PEFT offers broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)
- [\[ACL 2025\] Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models](refining_salience-aware_sparse_fine-tuning_strategies_for_language_models.md)
- [\[ACL 2025\] HFT: Half Fine-Tuning for Large Language Models](hft_half_fine-tuning_for_large_language_models.md)
- [\[ACL 2025\] Quantifying Semantic Emergence in Language Models](quantifying_semantic_emergence_in_language_models.md)
- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)

</div>

<!-- RELATED:END -->
