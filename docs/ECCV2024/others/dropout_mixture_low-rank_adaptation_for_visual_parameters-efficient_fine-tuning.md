---
title: >-
  [Paper Note] Dropout Mixture Low-Rank Adaptation for Visual Parameters-Efficient Fine-Tuning
description: >-
  [ECCV 2024][Parameter-Efficient Fine-Tuning] This paper proposes DMLoRA (Dropout-Mixture Low-Rank Adaptation), which balances accuracy and regularization by introducing a multi-branch up-and-down projection structure and progressively dropping out branches during training. Along with a two-stage learning scalar strategy to optimize the scaling coefficients of each layer, DMLoRA achieves SOTA performance on the VTAB-1k and FGVC visual fine-tuning benchmarks with zero additiona…
tags:
  - "ECCV 2024"
  - "Parameter-Efficient Fine-Tuning"
  - "Low-Rank Adaptation"
  - "Dropout Regularization"
  - "Visual Transformer"
  - "VTAB-1k"
date: 2026-05-08
content_hash: fbcc267789624402
---

# Dropout Mixture Low-Rank Adaptation for Visual Parameters-Efficient Fine-Tuning

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Model Fine-Tuning / Parameter-Efficient Fine-Tuning  
**Keywords**: Parameter-Efficient Fine-Tuning, Low-Rank Adaptation, Dropout Regularization, Visual Transformer, VTAB-1k

## TL;DR

This paper proposes DMLoRA (Dropout-Mixture Low-Rank Adaptation), which balances accuracy and regularization by introducing a multi-branch up-and-down projection structure and progressively dropping out branches during training. Along with a two-stage learning scalar strategy to optimize the scaling coefficients of each layer, DMLoRA achieves SOTA performance on the VTAB-1k and FGVC visual fine-tuning benchmarks with zero additional inference overhead.

## Background & Motivation

**Background**: Parameter-Efficient Fine-Tuning (PEFT) has become the mainstream paradigm for adapting large models to downstream tasks. LoRA (Low-Rank Adaptation) is one of the most representative methods, achieving parameter-efficient updates by adding a low-rank decomposition matrix $\Delta W = BA$ ($B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$) alongside the original weights. While widely validated in the NLP domain, LoRA has recently been applied to visual Transformer fine-tuning as well.

**Limitations of Prior Work**: When directly applying existing PEFT methods to different visual tasks, performance fluctuates significantly. For example, on the VTAB-1k benchmark, the performance of the same method varies greatly across natural, structured, and specialized image tasks. The authors attribute this instability to the insufficient robustness of existing PEFT methods—a single low-rank adaptation path might find a good gradient descent direction on some tasks but trap the model in poor local optima on others.

**Key Challenge**: PEFT methods must strike a balance between model capacity and regularization. Increasing adaptive parameters enhances model capacity but risks overfitting (especially on small datasets), while reducing parameters prevents overfitting but may lead to underfitting. Additionally, using a fixed scaling factor for all layers overlooks the varying importance of different layers for downstream tasks.

**Goal**: (1) How to provide a more robust gradient descent path for LoRA to improve stability across different vision tasks? (2) How to adaptively determine the optimal scaling factor for the LoRA modules of each layer?

**Key Insight**: The authors approach the problem from the perspectives of ensemble learning and dropout regularization. Multiple low-rank branches can be viewed as an ensemble of multiple weak learners. In the early stages of training, all branches are used to provide sufficient model capacity; as training progresses, branches are gradually dropped out to provide a regularization effect. This dynamic "expansion followed by contraction" training strategy adaptively balances capacity and regularization at different training stages.

**Core Idea**: Use multi-branch low-rank adaptation combined with gradual branch dropout to achieve a dynamic accuracy-regularization balance, along with two-stage learning of scaling factors to optimize the adaptation strength of each layer.

## Method

### Overall Architecture

The overall architecture of DMLoRA is a parameter-efficient fine-tuning method designed for pre-trained visual Transformers (such as ViT). In each attention layer of ViT, the single-branch structure of standard LoRA is replaced with a multi-branch structure, containing $K$ parallel up-projection and down-projection paths. During training, as epochs increase, branches are progressively dropped out with a certain probability, eventually converging to fewer branches. At inference time, the weights of all surviving branches are merged into the original weights, introducing no extra computational overhead. Meanwhile, a 2-Stage Learning Scalar (2S-LS) strategy is employed to determine the optimal scaling coefficient for each layer.

### Key Designs

1. **Multi-Branch Low-Rank Adaptation**:

    - **Function**: Provide multiple gradient descent paths for the model to enhance training robustness.
    - **Mechanism**: Extend the single up-projection matrix $B$ and down-projection matrix $A$ of standard LoRA into $K$ parallel branches $\{(B_1, A_1), (B_2, A_2), ..., (B_K, A_K)\}$. The rank of each branch can be designed as $r/K$ to keep the total parameter count constant, or the full rank $r$ can be used to increase model capacity. During forward propagation, $\Delta W = \sum_{k=1}^{K} B_k A_k$. The multi-branch structure allows the model more exploration directions in the parameter space, preventing it from getting trapped in a single local optimum.
    - **Design Motivation**: Similar to how an ensemble of multiple weak classifiers outperforms a single strong classifier in ensemble learning, the combination of multiple low-rank branches provides better adaptation than a single branch. Additionally, the multi-branch structure lays the foundation for subsequent gradual branch dropout.

2. **Gradual Branch Dropout**:

    - **Function**: Dynamically balance model capacity and regularization during training.
    - **Mechanism**: Keep all $K$ branches active in the early training stage to allow the model to fully learn downstream task features. As training progresses, the dropout probability of branches is gradually increased according to a predefined schedule function (such as linear or cosine schedule). The dropped branches participate in neither forward nor backward propagation during that iteration. In the later training stages, most branches are dropped out, providing a strong regularization effect to prevent overfitting. This process resembles a transition from "broad search" to "narrow fine-tuning."
    - **Design Motivation**: Adequate capacity is required in the early training stage to learn task features, where excessive regularization would lead to underfitting. In the later training stage, when the model has mostly converged, too many parameters would lead to overfitting, necessitating enhanced regularization. Gradual dropout implements this "loose-first, tight-later" adaptive training strategy. During inference, all branches can be merged into a single weight matrix, resulting in zero additional inference overhead.

3. **2-Stage Learning Scalar (2S-LS)**:

    - **Function**: Adaptively determine the optimal scaling factor $\alpha$ for each layer's DMLoRA module.
    - **Mechanism**: The scaling factor $\alpha$ in LoRA controls the influence strength of the adaptive update $\Delta W$ on the original weights. Traditional methods use the same $\alpha$ across all layers, but different layers have varying importance for different tasks. The 2S-LS strategy optimizes in two stages: the first stage trains the weight parameters of the DMLoRA modules using a large global scaling factor; the second stage freezes these weight parameters and treats the scaling factor $\alpha_l$ of each layer as a learnable parameter, optimizing them with a smaller learning rate. This allows the model to automatically discover which layers require larger adaptive updates and which layers are better off keeping the original weights.
    - **Design Motivation**: Experiments observe that different vision tasks exhibit varying dependencies on different ViT layers—natural image tasks depend more on shallow layers (low-level features), while structured tasks more on deep layers (high-level semantic features). A fixed scaling factor cannot accommodate this variance. The two-stage manner avoids the optimization difficulties of training both the weights and the scaling factors simultaneously.

### Loss & Training

Standard cross-entropy loss is used for classification fine-tuning. The training strategy is divided into two stages: Stage 1 trains all branch weights of DMLoRA with a globally uniform scaling factor while performing gradual branch dropout; Stage 2 freezes the branch weights and only optimizes the scaling factor $\alpha_l$ of each layer, with a learning rate set to one-tenth of Stage 1. During inference, the weights of each branch are merged into the original pre-trained weights according to the scaling factor: $W' = W + \frac{\alpha_l}{r} \sum_{k} B_k A_k$, introducing no extra latency.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (DMLoRA) | Prev. SOTA | Gain |
|---|---|---|---|---|
| VTAB-1k (Natural, 7 tasks) | Average Accuracy | SOTA | Second-best PEFT method | Gain |
| VTAB-1k (Specialized, 4 tasks) | Average Accuracy | SOTA | Second-best PEFT method | Gain |
| VTAB-1k (Structured, 8 tasks) | Average Accuracy | SOTA | Second-best PEFT method | Gain |
| VTAB-1k (Overall, 19 tasks) | Average Accuracy | SOTA | Second-best PEFT method | Consistent Lead |
| FGVC (5 fine-grained datasets) | Average Accuracy | SOTA | Second-best PEFT method | Gain |

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| Standard LoRA (single branch) | Baseline performance | Unstable on some tasks |
| Multi-branch without dropout | Improvement over single branch | Multiple paths help but may overfit |
| Multi-branch + gradual dropout | Further improvement | Dropout acts as regularization |
| Without 2S-LS (uniform scaling factor) | Lower than the full model | Inter-layer differentiated scaling is important |
| Full DMLoRA (multi-branch + dropout + 2S-LS) | Best | Three components work synergistically |
| Impact of branch number K | $K=4$ or $8$ is optimal | Too few lacks diversity, too many increases training overhead |

### Key Findings

- The multi-branch structure provides more robust gradient descent paths than single-branch LoRA, reducing performance fluctuations across tasks.
- Gradual dropout is key—keeping all branches active throughout or dropping them out too early is inferior to a gradual schedule.
- The optimal scaling factors differ significantly across layers, validating the necessity of layer-dependent adaptation.
- DMLoRA can be fully merged into the original weights during inference, proving the feasibility of "structured during training, zero overhead during inference."
- Stable performance gains are achieved across all three categories of VTAB-1k tasks (Natural/Specialized/Structured), verifying its robustness.

## Highlights & Insights

- The paradigm of "multi-branch during training, merging during inference" elegantly resolves the conflict between model capacity and efficiency.
- The gradual dropout concept is highly creative—scaling the application granularity of dropout regularization from individual neurons to structural branches.
- The design of the two-stage learning scaling factor transforms the inter-layer differentiated adaptation problem into a simple scalar optimization.
- The entire method introduces zero inference overhead, making it highly friendly for practical deployment.
- Comprehensive experimental design: VTAB-1k covers 19 different types of vision tasks.

## Limitations & Future Work

- The multi-branch structure during training increases GPU memory and computational cost, although there is no additional inference overhead.
- The schedule function of gradual dropout (linear/cosine) is predefined; adaptive scheduling based on validation performance could be considered.
- Validated only on ViT; could be extended to other architectures such as CNNs and Swin Transformers.
- The initialization strategy of branches might affect the final performance and has not been deeply analyzed.
- Combining gradual dropout with structural pruning could be considered to permanently remove unimportant branches after training.
- The scalability to larger models (e.g., ViT-H, ViT-G) remains to be verified.
- No experiments were conducted on combinations with other PEFT methods (e.g., Prefix Tuning, Adapter).

## Related Work & Insights

- **LoRA**: The pioneering work on low-rank adaptation, upon which this paper introduces a multi-branch structure and gradual dropout.
- **AdaLoRA**: Adaptive rank allocation for LoRA, sharing similarities with the inter-layer scaling factor optimization concept in this study.
- **DoRA**: Decomposes LoRA into magnitude and direction components for separate adaptation.
- **VPT (Visual Prompt Tuning)**: Achieves visual fine-tuning by incorporating learnable prompts.
- **SSF (Scale & Shift)**: Accomplishes parameter-efficient fine-tuning via scale and shift operations.
- **Dropout**: A classic regularization technique; this paper scales it from the element level to the branch level.
- **Insight**: The concept of gradual structural dropout can be extended to other multi-branch architectures, such as Mixture-of-Experts (progressively reducing the number of active experts during training).

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of multi-branch LoRA and gradual dropout offers some originality, though the individual components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluations across the 19 tasks of VTAB-1k and 5 FGVC datasets are highly thorough.
- Writing Quality: ⭐⭐⭐⭐ The method is clearly described, and the rationale is logically argued.
- Value: ⭐⭐⭐⭐ Provides a more robust variant for LoRA, offering valuable insights for the visual PEFT community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Functional Transform-Based Low-Rank Tensor Factorization for Multi-Dimensional Data Recovery](functional_transform-based_low-rank_tensor_factorization_for_multi-dimensional_d.md)
- [\[ICLR 2026\] Consistent Low-Rank Approximation](../../ICLR2026/others/consistent_low-rank_approximation.md)
- [\[CVPR 2026\] Basis-Oriented Low-rank Transfer for Few-Shot and Test-Time Adaptation](../../CVPR2026/others/basis-oriented_low-rank_transfer_for_few-shot_and_test-time_adaptation.md)
- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](../../NeurIPS2025/others/the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [\[ECCV 2024\] Synergy of Sight and Semantics: Visual Intention Understanding with CLIP](synergy_of_sight_and_semantics_visual_intention_understanding_with_clip.md)

</div>

<!-- RELATED:END -->
