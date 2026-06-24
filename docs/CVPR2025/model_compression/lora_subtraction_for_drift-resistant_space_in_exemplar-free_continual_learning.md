---
title: >-
  [Paper Note] LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning
description: >-
  [CVPR 2025][Model Compression][Continual Learning] LoRA-DRS proposes a "LoRA subtraction" operation—subtracting the LoRA weights of old tasks from the pre-trained weights before learning a new task to construct a Drift-Resistant Space (DRS). Subsequently, the LoRA of the new task is trained via gradient projection within this space, combined with an augmented triplet loss to enhance plasticity. This approach achieves SOTA performance in exemplar-free continual learning…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Continual Learning"
  - "Feature Drift"
  - "LoRA"
  - "Exemplar-Free Continual Learning"
  - "Drift-Resistant Space"
date: 2026-05-08
content_hash: a0f6b72495b2510a
---

# LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.18985](https://arxiv.org/abs/2503.18985)  
**Code**: [https://github.com/scarlet0703/LoRA-Sub-DRS](https://github.com/scarlet0703/LoRA-Sub-DRS)  
**Area**: Model Compression / Continual Learning  
**Keywords**: Continual Learning, Feature Drift, LoRA, Exemplar-Free Continual Learning, Drift-Resistant Space

## TL;DR

LoRA-DRS proposes a "LoRA subtraction" operation—subtracting the LoRA weights of old tasks from the pre-trained weights before learning a new task to construct a Drift-Resistant Space (DRS). Subsequently, the LoRA of the new task is trained via gradient projection within this space, combined with an augmented triplet loss to enhance plasticity. This approach achieves SOTA performance in exemplar-free continual learning, with a particularly significant advantage in long task sequences.

## Background & Motivation

**Background**: Continual learning requires models to learn new categories without forgetting old knowledge. Replay-based methods mitigate forgetting by storing old data, which is infeasible under privacy and storage constraints. Therefore, Exemplar-Free Continual Learning (EFCL) has gained attention. Recently, methods combining Pre-trained Models (PTMs) and Parameter-Efficient Fine-Tuning (PEFT) have become mainstream.

**Limitations of Prior Work**: Simply utilizing LoRA or Prompt Tuning is insufficient to address catastrophic forgetting, as feature drift continuously intensifies with the accumulation of tasks. Existing methods (e.g., InfLoRA, Adam-NSCL) attempt to construct a feature subspace that "does not interfere with old tasks." However, they rely on stored static features or statistics of old tasks to calculate the subspace. This information becomes increasingly outdated as the number of tasks increases, failing to capture the dynamic evolution of the feature space.

**Key Challenge**: Under the EFCL setting, access to past task data is unavailable, yet understanding the current state of the past tasks' feature space is required to construct an effective protective subspace. The "outdatedness" of old task statistics leads to significant performance degradation of these methods in long task sequences.

**Goal**: How to construct a training space that dynamically adapts to feature evolution without preserving old task data or features, thereby minimizing the interference of new task learning on old tasks?

**Key Insight**: Drawing inspiration from research on "task vectors"—subtracting a task vector can "forget" task-specific knowledge. If the model is made to "forget" old tasks before processing new task data, the resulting feature space naturally reflects directions that "do not contain old task knowledge." Learning new tasks in this space will not interfere with old knowledge.

**Core Idea**: Construct the drift-resistant space by subtracting the LoRA weights of old tasks from the pre-trained weights, and constrain the gradient update direction of the new task within this space to resolve feature drift at the parameter space level.

## Method

### Overall Architecture

LoRA-DRS is built upon a frozen pre-trained ViT-B/16. For each new task $t$, the learning process is divided into two stages: **Stage 1** (DRS Construction)—The modified weight $\tilde{W_t} = W_0 - \sum_{j=1}^{t-1} B_j A_j$ is calculated via LoRA subtraction. Processing new task data through the modified model yields the input features for each layer, and the SVD of their covariance matrix produces the DRS projection matrix. **Stage 2** (Training in DRS)—A new LoRA branch $\Delta W_t = B_t A_t$ is deployed, and gradients are projected onto the DRS during training to update the parameters, combined with an augmented triplet loss to enhance plasticity. The LoRA branches of old tasks and the pre-trained weights remain frozen throughout.

### Key Designs

1. **LoRA Subtraction (LoRA-):**

    - **Function**: Removes the influence of old tasks in the parameter space, presenting a foundation for constructing the DRS.
    - **Mechanism**: Inspired by the task vector theory—the task vector $V_{t-1} = W_{t-1} - W_0 = \sum_{j=1}^{t-1} B_j A_j$ represents the parameter changes from pre-training to learning old tasks. Adding its negation to the pre-trained weights yields $\tilde{W_t} = W_0 - V_{t-1} = W_0 - \sum_{j=1}^{t-1} B_j A_j$. Under the LoRA setting, this operation is extremely elegant—it does not require storing old task features, but only the existing LoRA weights. Processing the new task data with the modified model produces the input features for each layer $\tilde{X_t^l}$, from which the uncentered covariance matrix $\tilde{\mathcal{X}_t^l} = \frac{1}{n_t} (\tilde{X_t^l})^\top \tilde{X_t^l}$ is calculated.
    - **Design Motivation**: Existing methods (e.g., InfLoRA, Adam-NSCL) use static features of old tasks to construct subspaces, but these features become outdated during continual learning. LoRA subtraction avoids reliance on old data/features by directly operating in the parameter space, naturally reflecting the current model state.

2. **Gradient Projection in Drift-Resistant Space (DRS):**

    - **Function**: Constrains the parameter update directions of the new task, minimizing interference with old tasks.
    - **Mechanism**: Perform SVD on the covariance matrix $\tilde{\mathcal{X}_t^l}$ obtained from LoRA subtraction, taking the eigenvectors corresponding to the top $k$ largest eigenvalues to form the projection matrix $P_t^l = (U_t^l)_k$. The choice of $k$ is based on a cumulative variance ratio threshold $\varepsilon$. During training, gradients at each step are projected onto the DRS: $\Delta w_{t,s}^l = P_t^l (P_t^l)^\top g_{t,s}^l$, ensuring parameter updates occur only along DRS directions. For the first task, no projection is performed, and original gradients are directly used.
    - **Design Motivation**: The DRS is defined by the features of the new task "after removing the influence of old tasks." Gradient updates within this space naturally avoid the feature directions of old tasks, achieving protection of old knowledge without explicit access to old task features.

3. **Augmented Triplet Loss (ATL):**

    - **Function**: Enhances model plasticity—enabling effective learning of discriminative features for new categories under DRS constraints.
    - **Mechanism**: The standard triplet loss is defined as $\mathcal{L}_{TL} = \max(0, e_{ap} - e_{an} + \epsilon)$, where the positive sample distance $e_{ap}$ takes the farthest positive sample (hardest positive), and the negative sample distance $e_{an}$ considers both different classes in the current task and the prototypes of old tasks. The total loss is formulated as $\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{TL}$. By incorporating old task prototypes into the negative samples, the distinction between new and old categories is enhanced.
    - **Design Motivation**: Although gradient projection guarantees stability (no interference with old tasks), it limits model plasticity (the ability to learn new tasks). ATL compensates for this loss of plasticity by enlarging inter-class distances under the constraints of the DRS.

### Loss & Training

The total loss is $\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{TL}$. A ViT-B/16-IN21K pre-trained model is used with the Adam optimizer ($\beta_1=0.9, \beta_2=0.999$) and a batch size of 128. On ImageNet-R, training is conducted for 50 epochs per task, and on CIFAR-100 for 20 epochs. LoRA is integrated into the Key and Value projection matrices of the ViT attention modules. During evaluation, a nearest class mean (NCM) classifier is used, extracting features by merging all task LoRA weights.

## Key Experimental Results

### Main Results

ImageNet-R (200 classes, average over 5 random seeds):

| Method | 10 tasks ACC | 25 tasks ACC | 50 tasks ACC |
|------|-------------|-------------|-------------|
| LoRA-FT | 74.54 | 56.80 | 44.89 |
| CODA-Prompt | 72.15 | 63.86 | 48.89 |
| InfLoRA | 74.95 | 69.09 | 60.49 |
| EASE | 75.94 | 72.69 | 68.54 |
| Adam-NSCL | 72.24 | 62.04 | 49.82 |
| **LoRA-DRS** | 74.74 | **74.19** | **72.12** |

CIFAR-100:

| Method | 10 tasks ACC | 25 tasks ACC | 50 tasks ACC |
|------|-------------|-------------|-------------|
| EASE | 75.94 | 72.69 | 68.54 |
| InfLoRA | 86.44 | 77.51 | 56.65 |
| **LoRA-DRS** | 87.06 | **84.10** | **78.32** |

### Ablation Study

| Configuration | 25 tasks ACC | Description |
|------|-------------|------|
| Full LoRA-DRS | 74.19 | Full model |
| w/o DRS (Direct Gradient) | ~67 | Without gradient projection, severe feature drift |
| w/o ATL | ~72 | Without triplet loss, degraded plasticity |
| w/o LoRA- (Construct subspace with original weight) | ~71 | Without subtracting old LoRA, degraded DRS quality |

### Key Findings

- **Significant advantage in long task sequences**: At 50 tasks, LoRA-DRS outperforms EASE by 3.58% and InfLoRA by 11.63%, although it is slightly lower than EASE at 10 tasks. This indicates the core value of DRS lies in solving the "outdatedness of old statistics", where longer sequences yield more pronounced advantages.
- Feature drift curves (Fig. 1) show that LoRA-DRS is the only method maintaining low and stable drift throughout the entire task sequence.
- BWT metrics consistently outperform other methods, proving significantly better backward transfer (retention of old task performance).
- The simplicity of LoRA subtraction is impressive—requiring only elementary arithmetic operations on existing LoRA weights.

## Highlights & Insights

- **"Subtraction is Forgetting" Core Insight**: Utilizing the property of task vectors, parameter subtraction allows the model to "forget" old tasks before processing new data, so that the obtained features naturally indicate a "safe learning direction." This concept is extremely simple yet highly effective. It may also be useful in other knowledge management scenarios (e.g., model editing, machine unlearning).
- **Avoiding Outdated Static Statistics**: Unlike InfLoRA and Adam-NSCL, which rely on storing historical features/gradients, LoRA-DRS dynamically obtains the DRS via parameter operations without storing any historical data/statistics, fundamentally avoiding information outdatedness. This is the key reason for its significant advantage in long task sequences.
- **Decoupled Design of Stability and Plasticity**: DRS ensures stability (no interference with old tasks), while ATL enhances plasticity (maximizing inter-class discrimination within the constrained space). The two components have clear division of labor and complement each other effectively.

## Limitations & Future Work

- Under short sequences (10 tasks), performance is slightly below EASE (74.74 vs 75.94), suggesting that DRS construction might be overly conservative with fewer tasks.
- The choice of LoRA rank affects DRS quality, but the paper lacks a detailed sensitivity analysis regarding rank.
- Currently validated only on classification tasks; effectiveness on more complex tasks such as detection and segmentation remains to be explored.
- Theoretical analysis of LoRA subtraction is relatively weak—why is the feature space after subtracting old LoRA "safe"? A more rigorous theoretical proof is lacking.

## Related Work & Insights

- **vs InfLoRA**: InfLoRA utilizes gradient information of old tasks to design the subspace of the LoRA projection matrix, but gradient information gradually becomes outdated in long task sequences. LoRA-DRS avoids this problem via parameter subtraction, outperforming it by 11.63% at 50 tasks.
- **vs EASE**: EASE employs a semantic-guided prototype relation strategy, behaving well in short sequences but degrading in long ones. LoRA-DRS provides more stable drift control over long sequences.
- **vs Adam-NSCL**: Adam-NSCL optimizes in the approximate null space of old task input features, but static features make the null space increasingly inaccurate. LoRA-DRS completely avoids dependency on old features.
- **Extension of the Task Vector Concept**: This work cleverly applies the "subtraction = forgetting" property of task vectors to construct DRS in continual learning, representing an important application of task arithmetic in CL.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The concept of LoRA subtraction is simple and elegant; the logical chain "subtraction = forgetting -> forgetting = protection" is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across multiple datasets and task lengths, averaged over 5 random seeds, although some analytical experiments are lacking.
- Writing Quality: ⭐⭐⭐⭐ The methodological workflow is clear, and the visualization of feature drift in Fig. 1 is intuitive and powerful.
- Value: ⭐⭐⭐⭐⭐ Addresses the core pain point in EFCL, showing prominent advantages in long tasks with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)
- [\[NeurIPS 2025\] REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning](../../NeurIPS2025/model_compression/rep_resource-efficient_prompting_for_rehearsal-free_continual_learning.md)
- [\[ICML 2026\] Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning](../../ICML2026/model_compression/task-driven_subspace_decomposition_for_knowledge_sharing_and_isolation_in_lora-b.md)
- [\[ICCV 2025\] PLAN: Proactive Low-Rank Allocation for Continual Learning](../../ICCV2025/model_compression/plan_proactive_low-rank_allocation_for_continual_learning.md)
- [\[ICML 2025\] Semantic Shift Estimation via Dual-Projection and Classifier Reconstruction for Exemplar-Free Class-Incremental Learning](../../ICML2025/model_compression/semantic_shift_estimation_via_dual-projection_and_classifier_reconstruction_for_.md)

</div>

<!-- RELATED:END -->
