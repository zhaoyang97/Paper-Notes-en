---
title: >-
  [Paper Note] What Linear Probes Miss: Multi-View Probing for Weight-Space Learning
description: >-
  [ICML 2026][Interpretability][probing] Academic paper note for What Linear Probes Miss: Multi-View Probing for Weight-Space Learning.
tags:
  - ICML 2026
  - Interpretability
  - probing
date: 2026-05-08
content_hash: 0af964fbabcb67f6
---
XᵀW)"]
        C2["Col Gram Xᵀ(XZ)"]
    end
    X --> FO
    X --> SO
    FO --> ST["Per-sample Standardization and Simple Fusion<br/>Mean Subtraction and Std Dev Division"]
    SO --> ST
    ST --> P["Branch MLP Projection + Concatenation [f1;f2;f3;f4]"]
    P --> E["Shared Encoder ψ + Classifier φ"]
    E --> Y["Multi-label Prediction ŷ<br/>Training / LoRA Classes"]
```

### Key Designs
1.  **Row-Column Symmetric First-Order Probing**:
    *   **Function**: Simultaneously observes the patterns of output neurons aggregating inputs and input coordinates connecting to outputs.
    *   **Mechanism**: $XU$ is a row-centric sketch, where each row represents the response of an output neuron along probe directions; $X^\top V$ is a column-centric sketch, where each row represents the connection pattern of an input dimension with output-side probes. Theoretically, $X_1 \neq X_2$ exists such that $X_1U = X_2U$ but $X_1^\top V \neq X_2^\top V$.
    *   **Design Motivation**: Neural network weight matrices have geometries on both input and output sides. Single-sided first-order probes completely ignore changes falling into the probe nullspace. Adding a transposed perspective reduces such blind spots.

2.  **Gram-based Second-Order Interaction Branches**:
    *   **Function**: Captures pairwise similarities between rows and between columns, supplementing correlation structures invisible to first-order projections.
    *   **Mechanism**: Row Gram $K_{row} = XX^\top$ encodes similarity between output neurons; column Gram $K_{col} = X^\top X$ encodes similarity between input features. MVProbe does not explicitly form large Gram matrices but instead combines them with probes to compute $XX^\top W = X(X^\top W)$ and $X^\top XZ = X^\top(XZ)$, maintaining $O(mnr)$ complexity.
    *   **Design Motivation**: Theorem 4.1 shows that when $rank(U) < n$, one can construct two matrices with identical first-order responses but different second-order responses. Thus, second-order branches provide complementary information rather than redundant data, separating weight geometries collapsed by first-order probes.

3.  **Per-Sample Standardization and Simple Fusion**:
    *   **Function**: Prevents second-order responses from overwhelming first-order branches due to larger scales, allowing all four perspectives to contribute to the decision.
    *   **Mechanism**: Theoretical analysis shows that for i.i.d. Gaussian weights, the ratio of the expected norm of second-order responses to first-order responses is approximately $O(n\sigma^2)$. Direct concatenation would let higher-order branches dominate. MVProbe independently subtracts the mean and divides by the standard deviation for each sample and each branch, making the branch Frobenius norm dependent on the number of elements rather than the order.
    *   **Design Motivation**: Without scale control in multi-view methods, the model might only learn from the branch with the largest magnitude. Standardization followed by simple concatenation was chosen because it proved more stable than L2 normalization or learned weighting in experiments.

### Loss & Training
The training objective is the standard multi-label binary cross-entropy loss $\mathcal{L} = \mathcal{L}_{BCE}(\hat{y}, y)$. In the implementation, each branch uses $r=128$ probes with a projection dimension of 128, resulting in a final representation dimension of 512. The Adam optimizer is used with a learning rate of $3 \times 10^{-4}$ and a batch size of 128 for 500 epochs. Training can be completed on a single RTX 3090. Optimal single layers are used for Model Jungle: ResNet 67, SupViT 59, MAE 64, DINO 47; for Stable Diffusion LoRA, layer 46 is used.

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
| $XU$ only | ResNet 90.42 / DINO 74.17 | Single row first-order branch is strong but inferior to full version |
| $X^\top V$ only | ResNet 88.94 / DINO 72.04 | Column first-order provides complementary but weaker signal |
| second-order only | SupViT 92.04 / MAE 80.57 | Second-order combination close to full on some architectures, indicating strong Gram structure info |
| MVProbe all four | ResNet 92.24 / etc. | All four branches yield best results across all architectures |
| w/o Std vs w/ Std | Avg 65.9 → 68.8 | Standardization improves avg by +2.8; 89.2% of layers benefit, matching scale analysis |
| all-layer win rate | 95.1% | MVProbe outperforms ProbeX on 311 out of 327 available layers |

### Key Findings
- Simply increasing the number of probes in ProbeX is insufficient. ProbeX×4 still performs worse than MVProbe, indicating that gains stem from the view design rather than parameter count or probe quantity.
- Second-order Gram branches provide complementary information, not just a stronger first-order substitute. On DINO, second-order only was slightly lower than first-order, yet the full version remained best, suggesting different architectures require different view combinations.
- Standardization is a necessary component. Without it, multi-order response scales are unbalanced. Standardization provided an average improvement of +2.8%, particularly +4.2 and +4.1 on DINO and ResNet, respectively.
- LoRA experiments showcase the largest gap. In the difficult SD1k setup (1000 classes, 5 models per class), ProbeX in-distribution achieved only 35.75%, whereas MVProbe reached 97.88%.

## Highlights & Insights
- The paper clearly explains the failure of previous probing: it's not that probing is inherently limited, but that single-view first-order sketches collapse nullspace and pairwise interaction structures. Theorem 4.1 provides a clean construction for this intuition.
- MVProbe is engineering-friendly. While second-order branches appear to require forming Gram matrices, they are computed as $X(X^\top W)$ and $X^\top (XZ)$ via associativity, maintaining $O(mnr)$ complexity with training time nearly equivalent to ProbeX×4.
- Per-sample standardization is a critical but often overlooked detail. Multi-branch models often use simple concatenation; this work uses scale theory to explain why this biases toward high-order responses and validates it with cross-layer ablations.
- From an interpretability perspective, MVProbe provides a lightweight tool for checkpoint analysis: even without metadata, it can infer training categories or LoRA attributes through weight geometry, aiding in model repository governance and selection.

## Limitations & Future Work
- The method still relies on selecting a single representative layer. Although MVProbe is more stable regarding layer selection, absolute accuracy for MAE and DINO is still lower than ResNet/SupViT, suggesting single-layer weights in some architectures contain insufficient information.
- Current tasks are mainly training category and LoRA category identification. Whether more complex attributes like model capability, bias, safety, or data leakage can be reliably predicted from the same representation requires further experimentation.
- Multi-view branches are manually defined and not adaptive to architecture types. Since optimal views and layer depths vary for ResNet, ViT, and LoRA, future work might require architecture-aware branch selection.
- Weight-space identification itself may pose privacy and model provenance risks. If training data attributes can be recovered from weights, corresponding discussions on data governance and release strategies are necessary.

## Related Work & Insights
- **vs ProbeX**: ProbeX proved single-layer probing can scale to large models but mostly used first-order single-view representations; MVProbe supplements this with column and Gram views in the same single-layer setting, significantly improving accuracy and layer robustness.
- **vs ProbeGen / Neural Graph**: ProbeGen and graph methods are valuable for small models or multi-layer settings but are computationally heavy for large weight matrices; MVProbe maintains the lightweight nature of probing while introducing second-order geometry.
- **vs hand-crafted statistics / StatNN**: Statistical methods only look at coarse-grained features like mean, variance, and quantiles, failing to express inter-neuron relationships; MVProbe's Gram branch directly models these correlation structures.
- **Insight**: For model repository search, LoRA auto-labeling, checkpoint deduplication, and model provenance analysis, MVProbe-like weight geometric representations can serve as foundational features, potentially combined with sparse real evaluations or metadata.

## Rating
- Novelty: ⭐⭐⭐⭐ Multi-view probes and the Gram branch idea are clear, and the theoretical motivation is more convincing than simply stacking branches.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers Model Jungle, all-layer win rates, standardization ablation, high-order branch ablation, and SD LoRA.
- Writing Quality: ⭐⭐⭐⭐ Strong link between methodology and theory; tables are informative, though some notation is dense and requires familiarity with weight-space learning.
- Value: ⭐⭐⭐⭐ Highly practical for model identification, weight-space analysis, and model repository management; provides a clear direction for how probing can move beyond first-order linear responses.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning](../../ICLR2026/interpretability/domain_expansion_a_latent_space_construction_framework_for_multi-task_learning.md)
- [\[ICLR 2026\] Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](../../ICLR2026/interpretability/beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)
- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](../../ACL2026/interpretability/rhetorical_questions_in_llm_representations_a_linear_probing_study.md)
- [\[ACL 2026\] Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States](../../ACL2026/interpretability/linear_probes_detect_task_format_not_reasoning_mode_in_language_model_hidden_sta.md)
- [\[AAAI 2026\] Share Your Attention: Transformer Weight Sharing via Matrix-Based Dictionary Learning](../../AAAI2026/interpretability/share_your_attention_transformer_weight_sharing_via_matrix-based_dictionary_lear.md)

</div>

<!-- RELATED:END -->
