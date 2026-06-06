---
title: >-
  [Paper Note] What Linear Probes Miss: Multi-View Probing for Weight-Space Learning
description: >-
  [ICML2026][Interpretability][Weight-space learning] This paper argues that single-view first-order probes miss row-column interactions and second-order correlation structures within weight matrices. It proposes MVProbe…
tags:
  - "ICML2026"
  - "Interpretability"
  - "Weight-space learning"
  - "probing"
  - "model identification"
  - "Gram matrix"
  - "LoRA identification"
date: 2026-05-08
content_hash: 39fdbe5c9558dd8e
---

# What Linear Probes Miss: Multi-View Probing for Weight-Space Learning

**Conference**: ICML2026  
**arXiv**: [2605.23410](https://arxiv.org/abs/2605.23410)  
**Code**: https://github.com/AI-hew-math/MVProbe  
**Area**: Interpretability / Weight-Space Learning  
**Keywords**: Weight-space learning, probing, model identification, Gram matrix, LoRA identification

## TL;DR
This paper argues that single-view first-order probes miss row-column interactions and second-order correlation structures within weight matrices. It proposes MVProbe, which utilizes multi-view representations—including first-order row/column projections and second-order row/column Gram branches—significantly outperforming ProbeX in Model Jungle and Stable Diffusion LoRA identification tasks.

## Background & Motivation
**Background**: The rapid expansion of open-source model repositories often results in checkpoints lacking comprehensive documentation regarding datasets, tasks, or capabilities. Weight-space learning attempts to infer training categories, data distributions, or model attributes directly from model parameters without depending on external metadata. Flattening weights is computationally expensive and destroys structural information; thus, probing methods use learnable probe vectors to pass through weight matrices, generating lightweight and permutation-equivariant representations.

**Limitations of Prior Work**: Single-layer probing, exemplified by ProbeX, scales to large models but relies primarily on single-view first-order projections like $XU$. This approach essentially observes responses of each row along probe directions, ignoring column-side structures and pairwise correlations between rows or columns. Different weight matrices can produce identical first-order responses as long as their differences lie within the probe's nullspace.

**Key Challenge**: The semantics of a weight matrix reside not only in linear projections of individual rows or columns but also in the similarity structures between neurons and input features. To maintain scalability, one cannot directly construct complex graphs or flatten full parameters; however, to improve identification capabilities, the method must capture more geometric information than first-order single-view probes.

**Goal**: The authors aim to retain the efficiency of single-layer probing while addressing the expressivity gap of first-order methods. MVProbe observes row, column, row Gram, and column Gram simultaneously through four complementary branches. Theoretical analysis explains why second-order branches distinguish matrices that first-order probes cannot and why multi-order responses require standardization.

**Key Insight**: Probe vectors are interpreted as learnable landmarks in the geometry of the weight matrix. First-order probes observe projections onto landmark directions, while second-order Gram probes observe responses of sample-to-sample similarities to landmark combinations. This enables the acquisition of kernel-like geometric information without explicitly forming massive graph structures.

**Core Idea**: Multi-view probing is used to simultaneously capture first-order directional responses and second-order similarity structures of weight matrices, with per-sample standardization applied to each branch to enable balanced fusion of signals across different orders.

## Method
MVProbe takes a single representative layer's weight matrix $X \in \mathbb{R}^{m \times n}$ as input to predict attribute labels for the checkpoint, such as CIFAR-100 classes used in fine-tuning or ImageNet classes corresponding to a LoRA. Unlike ProbeX, which only computes $XU$, MVProbe extracts four responses from the same matrix: row-wise first-order, column-wise first-order, row-wise Gram second-order, and column-wise Gram second-order. Each response is standardized independently before being mapped to a common dimension by branch-specific projections. The four concatenated branches are then fed into a shared encoder and classification head.

### Overall Architecture
Given a weight matrix $X$, MVProbe learns four probe matrices $U, V, W, Z$. The row first-order branch computes $XU$, and the column first-order branch computes $X^\top V$. The row kernel branch computes $XX^\top W$, and the column kernel branch computes $X^\top XZ$. to prevent second-order branches from dominating due to naturally larger scales, each branch response matrix $S$ is transformed as $\tilde{S}=(S-\mu(S))/(\sigma(S)+\epsilon)$. After standardization, branches are projected into $f_i$ via MLPs, concatenated as $[f_1;f_2;f_3;f_4]$, and processed by a shared encoder $\psi$ and classifier $\phi$ to output multi-label predictions $\hat{y}$.

### Key Designs
1.  **Symmetric Row-Column First-Order Probing**:
    - **Function**: Simultaneously observes patterns of how output neurons aggregate inputs and how input coordinates connect to outputs.
    - **Mechanism**: $XU$ is a row-centric sketch where each row represents an output neuron's response along probe directions; $X^\top V$ is a column-centric sketch where each row represents the connection pattern of an input dimension to the output-side probes. Theoretically, $X_1 \ne X_2$ can exist such that $X_1U = X_2U$ while $X_1^\top V \ne X_2^\top V$.
    - **Design Motivation**: Neural network weight matrices possess geometry on both input and output sides. Single-sided first-order probes completely ignore variations falling into the probe's nullspace. Adding the transposed perspective reduces these blind spots.

2.  **Gram-based Second-Order Interaction Branches**:
    - **Function**: Captures pairwise similarity between rows and between columns, supplementing correlation structures invisible to first-order projections.
    - **Mechanism**: The row Gram $K_{row}=XX^\top$ encodes similarities between output neurons, while the column Gram $K_{col}=X^\top X$ encodes similarities between input features. MVProbe avoids explicitly forming large Gram matrices by computing $XX^\top W=X(X^\top W)$ and $X^\top XZ=X^\top(XZ)$, maintaining $O(mnr)$ complexity.
    - **Design Motivation**: Theorem 4.1 shows that when $rank(U) < n$, one can construct two matrices with identical first-order responses but different second-order responses. Thus, second-order branches provide non-redundant information that separates collapsed weight geometries.

3.  **Per-sample Standardization and Simple Fusion**:
    - **Function**: Prevents second-order responses from overwhelming first-order branches due to larger scales, allowing all four views to contribute to decision-making.
    - **Mechanism**: Theoretical analysis indicates that for i.i.d. Gaussian weights, the expected ratio of second-order to first-order response norms is approximately $O(n\sigma^2)$. Direct concatenation would cause higher-order branches to dominate. MVProbe independently subtracts the mean and divides by the standard deviation for each sample and branch, ensuring the Frobenius norm of each branch relates to the number of elements rather than the order.
    - **Design Motivation**: Multi-view methods without scale control may only learn from the branch with the largest magnitude. Standardization with simple concatenation was found to be more stable than L2 normalization or learned weighting in experiments.

### Loss & Training
The training objective is the standard multi-label binary cross-entropy loss $\mathcal{L}=\mathcal{L}_{BCE}(\hat{y},y)$. Each branch uses $r=128$ probes with a projection dimension of 128, resulting in a final representation dimension of 512. The model is trained for 500 epochs using the Adam optimizer with a learning rate of $3 \times 10^{-4}$ and a batch size of 128, completing on a single RTX 3090. Optimal representative layers used in Model Jungle: ResNet 67, SupViT 59, MAE 64, DINO 47; for Stable Diffusion LoRA, layer 46 is used.

## Key Experimental Results

### Main Results
| Dataset / Architecture | Metric | MVProbe | Prev. SOTA | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Model Jungle ResNet | Accuracy | 92.24 | ProbeX×4 87.16 | +5.08 |
| Model Jungle SupViT | Accuracy | 92.33 | ProbeX×4 90.33 | +2.00 |
| Model Jungle MAE | Accuracy | 81.62 | ProbeX×4 77.26 | +4.36 |
| Model Jungle DINO | Accuracy | 78.29 | ProbeX×4 73.25 | +5.04 |
| SD200 LoRA In-Dist. | Accuracy | 99.80±0.00 | ProbeX 98.48±0.48 | +1.32 |
| SD1k LoRA Zero-shot | Accuracy | 97.96±0.29 | ProbeX 52.42±2.48 | +45.54 |

### Ablation Study
| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| $XU$ only | ResNet 90.42 / DINO 74.17 | Single row first-order branch is strong but inferior to the full model. |
| $X^\top V$ only | ResNet 88.94 / DINO 72.04 | Column first-order provides complementary info but is weaker alone. |
| second-order only | SupViT 92.04 / MAE 80.57 | Second-order combinations approach the full model on some architectures. |
| MVProbe all four | ResNet 92.24 / etc. | Four branches perform best across all architectures. |
| w/o Std vs w/ Std | Avg 65.9 → 68.8 | Standardization yields +2.8 average gain; 89.2% of layers benefit. |
| all-layer win rate | 95.1% | MVProbe outperforms ProbeX in 311 out of 327 available layers. |

### Key Findings
- Merely increasing the number of probes in ProbeX is insufficient. ProbeX×4 remains inferior to MVProbe, indicating gains stem from perspective design rather than parameter count or probe quantity.
- Second-order Gram branches provide complementary information rather than just a stronger replacement for first-order probes. On DINO, second-order alone is slightly lower than first-order, yet the full combination is best, showing different architectures require different view combinations.
- Standardization is a necessary component. Without it, multi-order scale imbalances occur; with it, gains average +2.8%, notably +4.2 and +4.1 on DINO and ResNet respectively.
- The LoRA experiments demonstrate the most significant gap. In the difficult SD1k setting (1000 classes, 5 models per class), ProbeX in-distribution reaches only 35.75%, while MVProbe achieves 97.88%.

## Highlights & Insights
- The paper clarifies the failure mode of current probing: it is not that probing itself is flawed, but that single-view first-order sketches collapse nullspaces and pairwise interaction structures. Theorem 4.1 provides a clean construction for this intuition.
- MVProbe's design is engineering-friendly. While second-order branches appear to require Gram matrices, using the associative property to write them as $X(X^\top W)$ and $X^\top(XZ)$ maintains $O(mnr)$ complexity, with training time comparable to ProbeX×4.
- Per-sample standardization is a subtle but critical detail. Multi-branch models often use direct concatenation, but this work uses scale theory to explain the bias toward higher-order responses and validates it with cross-layer ablations.
- From an interpretability perspective, MVProbe provides a lightweight tool for checkpoint analysis: even without metadata, training categories or LoRA attributes can be inferred through weight geometry, aiding model repository governance and selection.

## Limitations & Future Work
- The method still relies on selecting a single representative layer. While MVProbe is more robust to layer selection, absolute accuracy on MAE and DINO remains lower than ResNet/SupViT, suggesting insufficient information in single layers of certain architectures.
- Current tasks are limited to training category and LoRA category identification. Whether more complex attributes like model capabilities, bias, safety, or data leakage can be reliably predicted from the same representations requires further experimentation.
- Multi-view branches are manually defined and not adaptive to architecture types. Since optimal views and depths vary for ResNet, ViT, and LoRA, future work may require architecture-aware branch selection.
- Weight-space identification may introduce risks regarding privacy and model provenance. If training data attributes can be recovered from weights, discussions on data governance and release strategies are necessary.

## Related Work & Insights
- **vs ProbeX**: ProbeX demonstrated that single-layer probing scales to large models but primarily uses first-order single-view representations; MVProbe adds column and Gram views within the same single-layer setting, markedly improving accuracy and layer robustness.
- **vs ProbeGen / Neural Graph**: ProbeGen and graph methods are valuable for small models or multi-layer settings but are computationally heavy for large weight matrices; MVProbe maintains the lightweight nature of probing while introducing second-order geometry.
- **vs hand-crafted statistics / StatNN**: Statistical methods look at coarse features like mean and variance, failing to represent inter-neuron relationships; MVProbe's Gram branches directly model these correlation structures.
- **Insights**: For model repository search, automatic LoRA labeling, checkpoint deduplication, and model provenance analysis, MVProbe-style weight geometry representations can serve as foundational features, optionally combined with limited real-world evaluation or metadata.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-view probe and Gram branch concept is clear, and the theoretical motivation is more persuasive than simply stacking branches.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers Model Jungle, all-layer win rates, standardization ablations, high-order branch ablations, and SD LoRA.
- Writing Quality: ⭐⭐⭐⭐ Strong connection between method and theory; tables are informative, though some symbols are dense, requiring familiarity with weight-space learning.
- Value: ⭐⭐⭐⭐ Highly practical for model identification, weight-space analysis, and model repository management, providing a clear direction for probing methods beyond first-order linear responses.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](../../ICLR2026/interpretability/beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)
- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](../../ACL2026/interpretability/rhetorical_questions_in_llm_representations_a_linear_probing_study.md)
- [\[ICLR 2026\] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](../../ICLR2026/interpretability/decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)
- [\[ACL 2026\] Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States](../../ACL2026/interpretability/linear_probes_detect_task_format_not_reasoning_mode_in_language_model_hidden_sta.md)
- [\[AAAI 2026\] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](../../AAAI2026/interpretability/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)

</div>

<!-- RELATED:END -->
