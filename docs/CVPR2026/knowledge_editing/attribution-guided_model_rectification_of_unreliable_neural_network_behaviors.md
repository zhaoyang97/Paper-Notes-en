---
title: >-
  [Paper Note] Attribution-Guided Model Rectification of Unreliable Neural Network Behaviors
description: >-
  [CVPR 2026][Knowledge Editing][rank-one model editing] This paper proposes an attribution-guided dynamic model rectification framework that repurposes rank-one model editing from domain adaptation to behavior rectification. By quantifying per-layer editability via Integrated Gradients, the framework automatically localizes suspect layers and repairs three categories of unreliable behaviors—backdoor attacks, spurious correlations, and feature leakage—using as few as a single clean sample.
tags:
  - CVPR 2026
  - Knowledge Editing
  - rank-one model editing
  - attribution analysis
  - backdoor defense
  - spurious correlations
  - model rectification
date: 2026-05-08
content_hash: 5a8f75a16132e3b4
---

# Attribution-Guided Model Rectification of Unreliable Neural Network Behaviors

**Conference**: CVPR 2026
**arXiv**: [2603.15656](https://arxiv.org/abs/2603.15656)
**Code**: None
**Area**: Knowledge Editing
**Keywords**: rank-one model editing, attribution analysis, backdoor defense, spurious correlations, model rectification

## TL;DR
This paper proposes an attribution-guided dynamic model rectification framework that repurposes rank-one model editing from domain adaptation to behavior rectification. By quantifying per-layer editability via Integrated Gradients, the framework automatically localizes suspect layers and repairs three categories of unreliable behaviors—backdoor attacks, spurious correlations, and feature leakage—using as few as a single clean sample.

## Background & Motivation

**Background**: Neural networks exhibit unreliable behaviors under distribution inconsistency, including neural Trojans (backdoor triggers), spurious correlations, and feature leakage, severely limiting deployment in safety-critical scenarios.

**Limitations of Prior Work**: Mainstream repair strategies rely on data cleansing and model retraining, incurring substantial computational and human costs. Rank-one model editing has demonstrated knowledge-editing capabilities in generative and discriminative models, yet applying it for domain adaptation faces two structural bottlenecks: (1) out-of-span residuals—a new key $k^*$ outside the training key span destroys existing associations (Lemma 1); and (2) sample complexity—accurate key estimation under distribution shift requires large numbers of samples (Lemma 2).

**Key Challenge**: Existing model editing methods operate on the final feature layer by default (due to its high-level semantic encoding), yet experiments show that editability varies substantially across layers—per-layer rectification rankings on ResNet-18 differ by several-fold (Fig. 2), and no single layer is universally optimal.

**Goal**: (1) Repair unreliable model behaviors under data-efficient conditions (even with a single clean sample); (2) automatically localize the most critical layer responsible for unreliable behavior, rather than manually designating a fixed layer.

**Key Insight**: Rank-one editing is repositioned from "domain adaptation" to "behavior rectification." The structural property that the model has already observed clean–corrupted pairs during training is exploited to circumvent the theoretical bottlenecks of standard editing; attribution methods then quantify per-layer editability to guide layer selection.

**Core Idea**: Integrated Gradients attributions are computed along the corrupted→clean path for each layer, projected onto the rank-one editing direction to produce a scalar editability score. The layer with the highest score is selected for editing, and the process iterates until the behavior is rectified.

## Method

### Overall Architecture
The framework operates as a three-step loop. **Step 1 (Attribution Computation)**: Given a corrupted sample $\tilde{x}$ and a clean sample $x$, Integrated Gradients attributions $M^l(x, \tilde{x})$ are computed for each layer using $\tilde{x}$ as the reference. **Step 2 (Layer Localization)**: Attributions are projected onto the rank-one editing direction to yield a scalar editability score $\hat{G}_l = \|M^{*,l}\|_F$, and the layer with the maximum score is selected: $l^* = \arg\max_l \hat{G}_l$. **Step 3 (Rank-One Editing)**: A rank-one weight update is applied to layer $l^*$ to establish a new corrupted key → clean value association. The dynamic framework embeds these three steps in a while loop, re-localizing and editing each round until the prediction gap $\delta^*$ falls below a threshold or the performance degradation exceeds a budget $\epsilon$.

### Key Designs

1. **Theoretical Advantages of the Rectification Setting (Rectifiability & Span-Aligned Control)**

   - **Function**: Formally proves two structural advantages of the behavior rectification setting over domain adaptation.
   - **Mechanism**: Because the model has observed both clean and corrupted samples during training, the corrupted key $k^*$ naturally lies within the span of training keys (Proposition 4.1, Rectifiability), eliminating out-of-span residuals and preserving existing key–value associations. Simultaneously, paired supervision drives key estimation error to zero without requiring large numbers of new-domain samples (Proposition 4.2, Span-Aligned Control).
   - **Design Motivation**: These two properties theoretically explain why a single sample suffices for repair—the rectification setting structurally avoids both bottlenecks of domain-adaptation editing.

2. **Attribution-Guided Layer Localization**

   - **Function**: Automatically identifies the layer that contributes most to unreliable behavior and is most amenable to editing.
   - **Mechanism**: IG attributions are computed with corrupted $\tilde{x}$ as the reference and clean $x$ as the input: $M^l_i(x, \tilde{x}) = (f_l(x_i) - f_l(\tilde{x}_i)) \cdot \int_0^1 \frac{\partial f(\hat{x})}{\partial f_l(\hat{x}_i)} d\alpha$. The Completeness axiom (Lemma 3: attributions across all layers sum to the output change $f(x)-f(\tilde{x})$) enables cross-layer comparability. Projecting the attribution onto the editing direction $M^{*,l} = M^l \cdot (C^{-1}k^*)^T$ and taking the Frobenius norm yields the scalar score.
   - **Design Motivation**: Attribution magnitude alone measures only "how important is this layer to the unreliable behavior"; projecting onto the editing direction measures "how much unreliability would editing this layer reduce"—a cross signal of attribution and editing direction that directly correlates with the first-order descent coefficient $G_l$.

3. **Dynamic Model Rectification Framework (Algorithm 1)**

   - **Function**: Iteratively localizes and edits multiple layers for adaptive repair.
   - **Mechanism**: The while loop checks whether the prediction gap $\delta^* = f(x) - f(\tilde{x})$ exceeds target $\delta$ and whether performance degradation $\epsilon^*$ remains within budget. Each round: attribution-based localization of the suspect layer → rank-one editing for $T$ epochs → evaluation of the edited model → if degradation is acceptable, the edit is retained and the loop continues; otherwise the edit is rolled back and the procedure returns.
   - **Design Motivation**: After editing one layer the model state changes, and a previously suboptimal layer may become the new bottleneck. The dynamic framework enables progressive exploration of multi-layer rectification paths.

### Loss & Training
The optimization objective for rank-one editing is a constrained least-squares problem: $\min_\Lambda \|v^* - f_l(k^*; W')\|$, subject to $W' = W + \Lambda(C^{-1}k^*)^T$, where $k^*$ is the feature key of the corrupted sample at the target layer, $v^*$ is the output value of the clean sample at that layer, and $C = KK^T$ is the second-order moment statistics of training-sample keys. The update matrix is an outer product of two vectors (rank-one) and admits a closed-form solution. In the dynamic framework, performance degradation is assessed via a validation metric $\zeta$ after $T$ editing epochs per round.

## Key Experimental Results

### Main Results

**Table 1: Backdoor Attack Repair (CIFAR-10 & ImageNet)**

| Method | #Samples | CIFAR-10 OA↑ | CIFAR-10 ASR↓ | ImageNet OA↑ | ImageNet ASR↓ |
|--------|:---:|:---:|:---:|:---:|:---:|
| Trojaned model | - | 93.67 | 99.94 | 69.05 | 87.24 |
| Fine-tune | 1 | 90.83 | 73.07 | 65.95 | 79.91 |
| Fine-tune | 20 | 91.58 | 13.22 | 68.42 | 21.86 |
| P-ClArC | 20 | 89.97 | 6.21 | 65.42 | 8.09 |
| A-ClArC | 20 | 92.53 | 6.32 | 67.17 | 8.73 |
| Stat. rectifying | 1 | 92.93 | 2.57 | 67.87 | 3.01 |
| **Dyn. rectifying** | **1** | **93.65** | **1.34** | 66.77 | **1.61** |
| **Dyn. rectifying** | **20** | **93.61** | **0.26** | **68.84** | **0.12** |

**Table 4: Spurious Correlation Mitigation (CIFAR-10 & ImageNet; Spurious column denotes deviation relative to Clean↓)**

| Method | #Samples | CIFAR-10 Overall↑ | Clean↑ | Spurious Dev.↓ | ImageNet Overall↑ | Clean↑ | Spurious Dev.↓ |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Benign model | - | 94.00 | 94.42 | +5.58 | 69.04 | 81.25 | +10.41 |
| A-ClArC | 20 | 92.41 | 76.77 | +2.57 | 67.01 | 75.66 | +6.59 |
| P-ClArC | 20 | 88.29 | 16.89 | +0.23 | 66.84 | 8.32 | +2.59 |
| **Dyn. rectifying** | **1** | 92.93 | 94.29 | **+1.86** | 67.50 | 81.66 | **+4.17** |
| **Dyn. rectifying** | **20** | **93.99** | 94.30 | **+0.12** | **68.94** | 81.25 | **+2.08** |

**Table 5: Feature Leakage Mitigation (BlockMNIST)**

| Method | #Samples | Accuracy↑ | Feature Leakage↓ |
|--------|:---:|:---:|:---:|
| Benign model | - | 99.17 | 3.597 |
| IG-SUM regularization | - | 94.14 | 3.417 |
| Fine-tune | 20 | 98.67 | 2.929 |
| Stat. rectifying | 1 | 98.97 | 2.655 |
| **Dyn. rectifying** | **1** | **99.03** | **2.417** |

### Ablation Study

**Static vs. Dynamic Rectification (CIFAR-10 backdoor, n=1)**

| Configuration | OA↑ | ASR↓ | Note |
|---------------|:---:|:---:|------|
| Patched model (n=20) | 89.70 | 12.19 | Neuron pruning, OA loss |
| Static rectifying (n=1) | 92.93 | 2.57 | Only the final layer edited |
| **Dynamic rectifying (n=1)** | **93.65** | **1.34** | Attribution localization + iterative editing |

**Trigger Generalization — Varying Visibility (ResNet-18, CIFAR-10, n=1, trained with φ=0.5)**

| Method | OA↑ | ASR(0.3)↓ | ASR(0.5)↓ | ASR(0.7)↓ | ASR(1.0)↓ |
|--------|:---:|:---:|:---:|:---:|:---:|
| Patched | 89.61 | 30.84 | 26.86 | 32.42 | 37.19 |
| **Dyn. rectifying** | **91.21** | **6.84** | **5.17** | **7.65** | **7.91** |

**Trigger Generalization — Varying Position (rectified with BR sample, n=1)**

| Method | OA↑ | ASR(BR)↓ | ASR(TL)↓ | ASR(C)↓ | ASR(BL)↓ |
|--------|:---:|:---:|:---:|:---:|:---:|
| Patched | 89.22 | 29.31 | 34.42 | 34.58 | 34.88 |
| **Dyn. rectifying** | **90.85** | **6.36** | **9.24** | **9.47** | **8.95** |

**Real-World Scenario: ISIC Skin Lesion (EfficientNet-B4, n=10)**

| Method | Overall↑ | Clean↑ | Spurious Dev.↓ |
|--------|:---:|:---:|:---:|
| Benign model | 79.00 | 61.50 | +26.00 |
| Fine-tune (n=20) | 80.50 | 53.00 | +11.50 |
| A-ClArC (n=20) | 79.50 | 54.50 | +5.00 |
| Stat. rectifying (n=10) | 79.50 | 60.00 | +4.50 |
| **Dyn. rectifying (n=10)** | **80.00** | 61.00 | **+1.50** |

### Key Findings
- **Extreme Data Efficiency**: With a single clean sample, dynamic rectification reduces ASR on CIFAR-10 from 99.94% to 1.34% while leaving OA virtually unchanged (93.67→93.65).
- **Dynamic Consistently Outperforms Static**: Across all scenarios, dynamic rectification uniformly surpasses static rectification (which edits only the final layer), validating the necessity of attribution-based layer localization.
- **Cross-Trigger Generalization**: Rectification using a trigger sample of a specific visibility or position generalizes effectively to other visibilities (0.3–1.0) and positions (TL/C/BL/BR).
- **Unified Effectiveness Across Three Scenarios**: The same framework performs well across backdoor attacks, spurious correlations, and feature leakage, demonstrating the generalization capability of the unified formulation.
- **Real-World Validation**: On the ISIC skin lesion dataset, using 10 manually cleaned samples reduces the spurious deviation from +26.00 to +1.50, substantially outperforming A-ClArC and fine-tuning.
- **The Cost of P-ClArC**: While P-ClArC narrows the spurious deviation, it does so at the cost of a dramatic collapse in clean accuracy (CIFAR-10: 94.42→16.89); the proposed method does not exhibit this problem.

## Highlights & Insights
- **Theory–Empiricism Loop**: Propositions on Rectifiability and Span-Aligned Control rigorously explain why a single sample suffices—the corrupted key lies within the training span (no residuals) and paired supervision eliminates estimation error, making data efficiency a structural guarantee rather than a coincidence.
- **Attribution × Editing Direction Cross-Signal**: Rather than selecting layers by attribution magnitude alone (which only measures "how important is this layer"), the attributions are projected onto the rank-one update direction to measure "how much unreliability would editing this layer reduce"—an intuitive yet theoretically grounded signal directly related to the first-order descent coefficient $G_l$.
- **No Inference Overhead**: Unlike P-ClArC/A-ClArC, which append artifact modules, this method directly edits weights, leaving the model architecture and inference cost entirely unchanged.
- **BlockMNIST Visualization**: IG attribution heatmaps visually demonstrate that the benign model allocates attribution to the null patch (leakage), whereas the rectified model concentrates attribution in the digit region (Fig. 5), providing interpretability evidence.

## Limitations & Future Work
- **Requires Clean–Corrupted Pairs**: Rectification presupposes knowledge of which samples are corrupted and access to their clean counterparts; in practice, an additional backdoor detection step (e.g., SpRAy, SPECTRE) may be required.
- **Linear Association Assumption**: Rank-one editing treats each layer as a linear associative memory, which may be insufficiently precise for highly nonlinear layers.
- **Classification Tasks Only**: Applicability to object detection, segmentation, generation, and other tasks has not been explored.
- **Unknown Scalability to Large Models**: Experiments are based on medium-scale CNNs (ResNet-18/EfficientNet-B4); effectiveness on ViT/LLM-scale models remains to be verified. (Token-level causal analysis can aid layer localization in LLMs, but no analogous mechanism exists for vision models.)
- **Layer Localization Cost**: Computing Integrated Gradients and estimating second-order moment statistics $C$ for all layers introduces additional computational overhead that grows with network depth.

## Related Work & Insights
- **vs. ROME/MEMIT**: ROME edits factual knowledge in LLMs; this work adapts rank-one editing to behavior rectification in visual discriminative models. The core contributions lie in the theoretical analysis of the rectification setting and attribution-based layer selection.
- **vs. P-ClArC/A-ClArC**: Those methods append artifact modules to patch the model; this work directly edits weights—incurring no inference overhead and requiring fewer samples. The clean-accuracy collapse observed with P-ClArC does not occur here.
- **vs. Fine-tuning**: Fine-tuning with n=1 leaves ASR as high as 73%; with n=20, OA degrades noticeably. Rank-one editing is more precise and better preserves OA.
- **vs. Neural Cleanse/SPECTRE**: These methods focus on detecting the presence of backdoors; this work focuses on repair—the two can be combined into a detect-then-rectify pipeline.
- **Inspiration**: The idea of projecting attributions onto the editing direction can be generalized to automatic layer selection in LLM knowledge editing, where ROME/MEMIT also face the layer selection problem.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The attribution-guided layer localization idea is novel, and the theoretical analysis of the rectification setting makes a substantive contribution; however, the rank-one editing framework itself is not new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three categories of unreliable behaviors × multiple datasets × trigger variants × generalization experiments × real-world ISIC scenario; the experimental design is comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ The Lemma/Proposition chain in the theoretical sections is clear and rigorous, though the main text is somewhat lengthy.
- **Value**: ⭐⭐⭐⭐ The practical value of single-sample backdoor repair is high, but the requirement for clean–corrupted pair priors constitutes a deployment barrier.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Fine-tuning Done Right in Model Editing](../../ICLR2026/knowledge_editing/fine-tuning_done_right_in_model_editing.md)
- [\[ICLR 2026\] Energy-Regularized Sequential Model Editing on Hyperspheres](../../ICLR2026/knowledge_editing/energy-regularized_sequential_model_editing_on_hyperspheres.md)
- [\[ICLR 2026\] EAMET: Robust Massive Model Editing via Embedding Alignment Optimization](../../ICLR2026/knowledge_editing/eamet_robust_massive_model_editing_via_embedding_alignment_optimization.md)
- [\[ACL 2026\] FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing](../../ACL2026/knowledge_editing/fable_fine-grained_fact_anchoring_for_unstructured_model_editing.md)
- [\[ICLR 2026\] Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing](../../ICLR2026/knowledge_editing/bilinear_representation_mitigates_reversal_curse_and_enables_consistent_model_ed.md)

</div>

<!-- RELATED:END -->
