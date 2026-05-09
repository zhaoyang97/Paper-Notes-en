---
title: >-
  [Paper Note] LCA: Local Classifier Alignment for Continual Learning
description: >-
  [ICLR 2026][LLM Evaluation][Class-Incremental Learning] This paper proposes Local Classifier Alignment (LCA), a loss function that simultaneously minimizes classification loss and loss sensitivity within local regions of class prototype Gaussian distributions. LCA addresses the classifier mismatch problem arising from incremental backbone merging in continual learning. Combined with an Incremental Merging (IM) strategy for PEFT modules, the method achieves an overall average accuracy of 85.6% across 7 benchmark datasets, substantially outperforming prior state-of-the-art methods.
tags:
  - ICLR 2026
  - LLM Evaluation
  - Class-Incremental Learning
  - Classifier Alignment
  - Model Merging
  - Robustness
  - Pre-trained Models
date: 2026-05-08
content_hash: d14d04d8ed50d8e4
---

# LCA: Local Classifier Alignment for Continual Learning

**Conference**: ICLR 2026
**arXiv**: [2603.09888](https://arxiv.org/abs/2603.09888)
**Code**: [GitHub](https://github.com/tung-tran-kyushu/LCA)
**Area**: Continual Learning
**Keywords**: Class-Incremental Learning, Classifier Alignment, Model Merging, Robustness, Pre-trained Models

## TL;DR

This paper proposes Local Classifier Alignment (LCA), a loss function that simultaneously minimizes classification loss and loss sensitivity within local regions of class prototype Gaussian distributions. LCA addresses the classifier mismatch problem arising from incremental backbone merging in continual learning. Combined with an Incremental Merging (IM) strategy for PEFT modules, the method achieves an overall average accuracy of 85.6% across 7 benchmark datasets, substantially outperforming prior state-of-the-art methods.

## Background & Motivation

**State of the Field**: Pre-trained model (PTM)-based class-incremental learning (CIL) represents the dominant paradigm in continual learning. PTMs provide powerful feature extraction capabilities that require only lightweight fine-tuning to adapt to new tasks; however, naive sequential fine-tuning still leads to catastrophic forgetting.

**Limitations of Prior Work**: (1) Fine-tuning only on the first task (e.g., APER) causes rapid performance degradation as the number of tasks grows and distribution shift accumulates; (2) per-task fine-tuning with backbone merging (e.g., EASE, MOS) yields strong aggregate performance but introduces a mismatch between the frozen task-specific classifiers and the merged backbone.

**Root Cause**: After multi-task merging, the backbone parameters shift, causing previously frozen task-specific classifiers to become misaligned with the new feature space, resulting in severe degradation of performance on old tasks. Since historical data cannot be revisited, directly retraining the classifiers is infeasible.

**Starting Point**: The paper generates synthetic samples using Gaussian class prototypes and re-aligns all classifier heads on these synthetic samples. The key innovation is to not only minimize classification loss, but also regularize the sensitivity of the loss to input perturbations, thereby achieving local robustness and better generalization.

**Related Work & Insights from Model Merging**: Works such as Task Arithmetic and TIES-Merge demonstrate that independently trained task-specific models can be combined via parameter merging to form a stronger unified model. This paper incorporates that insight into CIL by merging only the PEFT (LoRA) parameters, incurring minimal storage overhead.

**Theoretical Gap**: Existing CIL methods lack theoretical analysis to guide classifier alignment. This paper provides a decomposition theorem for test error, decomposing CIL performance into three controllable components: feature distribution shift, class loss, and robustness.

## Method

### Key Design 1: Incremental Merging (IM)

**Function**: Fine-tunes PEFT modules per task and then merges task vectors element-wise into a unified backbone.

**Mechanism**: Each new task initializes training from the previous merged result, maintaining proximity in parameter space. After training, the task vector $\tau_{\text{curr}} = \theta_{\text{peft}_i} - \theta_{\text{peft}_0}$ is computed and compared element-wise with the accumulated vector $\tau$, retaining the entry with larger absolute value:

$$
\tau^{(k)} \leftarrow \begin{cases} \tau_{\text{curr}}^{(k)} & \text{if } |\tau_{\text{curr}}^{(k)}| \geq |\tau^{(k)}| \\ \tau^{(k)} & \text{otherwise} \end{cases}
$$

The final merged result is $\theta_{\text{merged}} = \theta_{\text{peft}_0} + \alpha \cdot \tau$.

**Design Motivation**: (1) Retaining only the current accumulated vector and the new task vector avoids storing all historical parameters; (2) selecting the entry with the larger absolute value preserves the most salient task-specific updates; (3) initializing from the previous merged result maintains parameter-space continuity, which promotes stable merging (Li et al., 2025).

### Key Design 2: LCA Loss Function

**Function**: After backbone merging, LCA generates synthetic samples from class Gaussian prototypes and retrains all classifier heads using the LCA loss.

**Mechanism**: The LCA loss for each class $i$ is defined as:

$$
L_i = \underbrace{\mathbb{E}_{\boldsymbol{z} \sim \boldsymbol{D}_i}[\ell(h_t, \boldsymbol{z})]}_{\text{Classification Loss}} + \lambda \underbrace{\mathbb{E}_{\boldsymbol{z}, \boldsymbol{z}' \sim \boldsymbol{D}_i}[|\ell(h_t, \boldsymbol{z}) - \ell(h_t, \boldsymbol{z}')|]}_{\text{Loss Sensitivity Regularizer}}
$$

The total loss is the mean over all seen classes: $L(\boldsymbol{D}, h_t) = \frac{1}{C_t} \sum_{i=1}^{C_t} L_i$.

**Design Motivation**:
- The first term is the standard cross-entropy loss, ensuring correct classification of synthetic samples drawn near class prototypes.
- The second term measures the loss discrepancy between two random samples from the same class distribution, penalizing the classifier's sensitivity to small input variations and thereby flattening the loss surface within the prototype neighborhood.
- This local robustness is especially important because some Gaussian-sampled points may lie far from their own class prototype and close to others; the second term reduces the negative influence of such "harmful samples" during training.
- $\lambda$ controls the strength of the robustness penalty; $\lambda = 0.1$ yields consistently stable performance across all datasets.

### Key Design 3: Theoretical Error Decomposition

**Function**: Provides a theoretical analysis of classifier generalization in CIL by decomposing test error into three controllable components.

**Mechanism**:

**Theorem 3.1 (Fixed Backbone)**: For a bounded loss $\ell$, the test error satisfies:

$$
L(P, h_t) \leq L(\boldsymbol{D}, h_t) + \sum_{i=1}^{C_t} \frac{n_i}{n} \bar{\epsilon}_i(h_t) + \ell_{\max} \sqrt{\frac{C_t \ln 4 + 2\ln(1/\delta)}{n}}
$$

where $\bar{\epsilon}_i(h_t)$ is a loss robustness term within the local region of class $i$.

**Theorem 3.2 (Changing Backbone)**: Accounting for feature distribution shift induced by backbone updates:

$$
L(P_t, h_t) \leq 2\ell_{\max} \text{TV}(P_t, \hat{P}_t) + L(\hat{\boldsymbol{D}}, h_t) + \sum_{i=1}^{C_t} \frac{n_i}{n} \bar{\epsilon}_i(h_t) + \ell_{\max} \sqrt{\frac{C_t \ln 4 + 2\ln(1/\delta)}{n}}
$$

**Design Motivation**: Each of the three components is controlled by a corresponding method: (1) $\text{TV}(P_t, \hat{P}_t)$, the feature distribution shift, is controlled by IM's incremental merging; (2) $L(\hat{\boldsymbol{D}}, h_t)$, the training loss, is controlled by the first term of LCA; (3) $\bar{\epsilon}_i$, robustness, is controlled by the second term of LCA. The theory directly validates the design rationale of IM+LCA.

### Key Design 4: Classifier Architecture

**Function**: A separate MLP classification head is added for each new task; at inference, the outputs of all heads are concatenated.

**Mechanism**: Inference is performed as $h(x) = \text{concat}(h(x;\theta_1^{\text{cls}}), \ldots, h(x;\theta_t^{\text{cls}}))$. Each class is represented by a Gaussian distribution $\mathcal{N}_i$ in feature space, and LCA alignment samples synthetic features from each class Gaussian.

**Design Motivation**: Independent classifier heads prevent old heads from being modified during new task training, reducing forgetting. The additional storage cost is only the per-class mean and covariance, $\mathcal{O}(n)$, which is far more efficient than storing raw samples.

## Key Experimental Results

### Table 1: Average Accuracy Comparison on 7 Benchmark Datasets (ViT-B/16-IN1K)

| Method | CIFAR100 | IN-R | IN-A | CUB | OB | VTAB | CARS | Overall |
|--------|----------|------|------|-----|----|------|------|---------|
| CODA-Prompt | 91.0 | 78.2 | 48.1 | 75.6 | 71.0 | 65.6 | 26.3 | 65.1 |
| DualPrompt | 86.7 | 74.6 | 55.3 | 78.9 | 74.4 | 84.0 | 49.4 | 71.9 |
| EASE | 91.7 | 82.4 | 67.8 | 89.5 | 80.8 | 93.3 | 48.1 | 79.1 |
| MOS | 94.3 | 83.3 | 67.6 | 92.3 | 86.1 | 92.4 | 71.4 | 83.9 |
| SLCA | 93.7 | 85.1 | 45.1 | 90.2 | 82.7 | 91.1 | 74.6 | 80.4 |
| IM (merge only) | 92.8 | 84.3 | 66.5 | 86.7 | 81.1 | 84.6 | 70.1 | 80.9 |
| **IM+LCA** | **94.8** | **85.8** | **75.0** | 90.8 | 81.4 | **95.2** | **76.2** | **85.6** |

### Table 2: Robustness Comparison (CIFAR100-C / CIFAR100-P)

| Metric | IM | IM+LCA | Gain |
|--------|----|--------|------|
| CIFAR100-C Average Accuracy | ~88% | ~90% | +2% |
| CIFAR100-P Average Accuracy | ~86% | ~88.5% | +2.5% |
| CIFAR100-C Severity 5 | Lower | Higher | Significant improvement |
| Overall Robustness Score | Baseline | Superior | Consistent improvement |

### LCA as a Plug-in Component for Other Methods

| Method | Original | +LCA | Effect |
|--------|----------|------|--------|
| SLCA (base) | Baseline | SLCA-LCA | Improvements on IN-A, CUB, VTAB, CARS |
| MOS (base) | Baseline | MOS-LCA | Improvements across multiple datasets; CIFAR100 reaches 93.1% |

## Key Findings

1. **Classifier alignment is the critical bottleneck in CIL**: The performance gains from IM to IM+LCA are consistently significant across all 7 datasets, with a particularly notable improvement of 8.5% on ImageNet-A (66.5→75.0), confirming that classifier mismatch after backbone merging is a primary performance bottleneck.

2. **Importance of the robustness regularizer**: The second term of LCA (loss sensitivity regularization) yields robustness improvements of +2% on CIFAR100-C and +2.5% on CIFAR100-P, with consistent gains across all 19 corruption types and multiple perturbation categories.

3. **Composability of LCA**: LCA can be embedded as a plug-in into methods such as SLCA and MOS. Even without hyperparameter tuning (fixing $\lambda=0.1$), it consistently improves performance across multiple datasets.

4. **Effectiveness of merging only PEFT parameters**: Merging only LoRA parameters, without merging the full backbone, achieves efficient knowledge consolidation with minimal storage overhead.

5. **Robustness of $\lambda$ selection**: $\lambda=0.1$ delivers stable performance across all datasets. Excessively large $\lambda$ leads to performance degradation due to over-regularization, consistent with theoretical expectations.

## Highlights & Insights

- **Loss sensitivity as a regularization target**: Unlike conventional weight regularization or feature alignment, LCA directly constrains the rate of change of the loss function over the input space. This "loss surface flattening" concept is conceptually related to Sharpness-Aware Minimization (SAM), but is applied specifically to the classifier alignment problem.

- **Theory-driven method design**: The three-component decomposition in Theorem 3.2 (distribution shift + training loss + robustness) directly motivates the dual-component design of IM+LCA, with each component responsible for controlling one theoretical error term. Such tight coupling between theory and method design is relatively uncommon in the CIL literature.

- **Effective use of synthetic samples**: The method requires no exemplar memory or data replay; storing only the per-class mean and covariance and sampling from Gaussian distributions is sufficient for feature-space classifier alignment, thereby avoiding privacy concerns and storage overhead.

- **Simplicity and efficiency**: The overall method requires neither backbone expansion (as in EASE), complex inference procedures (as in MOS), nor additional memory buffers. A single LCA alignment step after merging suffices, yielding a straightforward and effective solution.

## Limitations & Future Work

1. **LCA operates only at the classifier alignment stage**: The LCA loss is not integrated into the end-to-end backbone training pipeline. The authors acknowledge that incorporating LCA into backbone training could potentially further improve robustness.

2. **Limitations of the Gaussian assumption**: Representing each class with a single Gaussian distribution may fail to capture multimodal or asymmetric structures in the true feature distribution, particularly on complex fine-grained datasets.

3. **Theoretical analysis assumes a fixed backbone**: Theorem 3.1 holds under a fixed backbone; while Theorem 3.2 introduces a distribution shift term, it does not directly analyze the dynamics during backbone training.

4. **Limited evaluation beyond CIL**: Although the LCA loss is general in nature, it is validated only in the CIL setting and has not been tested in other continual learning scenarios (e.g., domain-incremental, task-incremental learning) or general classification tasks.

5. **Limited gains on the OB dataset**: IM+LCA improves OmniBenchmark accuracy by only 0.3% (81.1→81.4%), remaining below MOS's 86.1%, suggesting that the method may not be advantageous under certain distribution settings.

## Related Work & Insights

### vs. EASE (Zhou et al., 2024)
EASE integrates new tasks via an expandable subspace and reweights old classifiers using semantic similarity. In contrast, LCA requires no backbone architecture expansion, incurs lower storage overhead, and directly aligns classifiers through a theoretically grounded loss function. IM+LCA (85.6%) substantially outperforms EASE (79.1%) in overall accuracy, with especially large margins on IN-A (+7.2%), VTAB (+1.9%), and CARS (+28.1%).

### vs. MOS (Sun et al., 2025b)
MOS dynamically selects an appropriate backbone adapter at inference time, emphasizing inference-phase adaptation. IM+LCA, by contrast, completes alignment in a single post-training step, resulting in simpler inference. Although MOS achieves higher accuracy on CUB (92.3 vs. 90.8) and OB (86.1 vs. 81.4), IM+LCA leads substantially on IN-A (+7.4%), VTAB (+2.8%), and CARS (+4.8%), and achieves higher overall accuracy (85.6% vs. 83.9%).

### vs. SLCA (Zhang et al., 2023)
SLCA employs a small learning rate for backbone training to mitigate forgetting, but backbone drift still causes classifier misalignment. IM+LCA directly addresses this issue, improving IN-A accuracy from 45.1% to 75.0% (+29.9%) and achieving 85.6% vs. 80.4% overall. Furthermore, LCA can serve as a complementary component for SLCA (SLCA-LCA) to further improve performance.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The design of the LCA loss is novel — using loss sensitivity as a regularization target with a theoretically grounded error decomposition. The incremental merging strategy is related to prior work, but merging only PEFT parameters without a pruning stage constitutes a new contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluation covers 7 benchmark datasets with 3-seed mean and standard deviation reporting, plug-in combination experiments with other methods, CIFAR100-C/P robustness evaluation, hyperparameter sensitivity analysis, and ablation studies over multiple merging strategies.
- **Writing Quality**: ⭐⭐⭐⭐ The theoretical analysis is clear and complete; method descriptions are concise; algorithmic pseudocode is well-presented; overall paper structure is sound.
- **Value**: ⭐⭐⭐⭐ LCA is simple to implement and can be embedded as a plug-in into existing CIL methods without additional storage or complex inference, making it well-suited for practical deployment.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery](../../AAAI2026/llm_evaluation/goal_geometrically_optimal_alignment_for_continual_generalized_category_discover.md)
- [\[ICLR 2026\] Biologically Plausible Online Hebbian Meta-Learning: Two-Timescale Local Rules for Spiking Neural Brain Interfaces](biologically_plausible_online_hebbian_meta-learning_two-timescale_local_rules_fo.md)
- [\[ICLR 2026\] PlanetAlign: A Comprehensive Python Library for Benchmarking Network Alignment](planetalign_a_comprehensive_python_library_for_benchmarking_network_alignment.md)
- [\[NeurIPS 2025\] Exploiting Task Relationships in Continual Learning via Transferability-Aware Task Embeddings](../../NeurIPS2025/llm_evaluation/exploiting_task_relationships_in_continual_learning_via_transferability-aware_ta.md)
- [\[ICLR 2026\] In-Context Learning for Pure Exploration](in-context_learning_for_pure_exploration.md)

<!-- RELATED:END -->
