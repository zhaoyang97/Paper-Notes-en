---
title: >-
  [Paper Note] Quantized Residuals to Continuous Prompts for Few-Shot Class Incremental Learning in Vision-Language Models
description: >-
  [CVPR 2026][Self-Supervised Learning][FSCIL] QR-Prompt discretely quantizes the "residuals" between CLIP visual and text features—which are typically smoothed out by contrastive learning—into a set of frozen Discriminative Subspace Codebooks (DSQ). These discrete codes are then translated into class-adaptive continuous prompts via a Hierarchical Prompt Encoder (H
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - FSCIL
date: 2026-05-08
content_hash: 2362c43e41349aea
---
# Quantized Residuals to Continuous Prompts for Few-Shot Class Incremental Learning in Vision-Language Models

**Conference**: CVPR 2026  
**Paper**: [CVF OpenAccess](https://openaccess.thecvf.com/content/CVPR2026/html/Sinha_Quantized_Residuals_to_Continuous_Prompts_for_Few-Shot_Class_Incremental_Learning_CVPR_2026_paper.html)  
**Code**: None (Not provided in the paper)  
**Area**: Self-Supervised/Continual Learning · Few-Shot Class Incremental Learning (FSCIL) · Vision-Language Models  
**Keywords**: FSCIL, CLIP Residuals, Product Quantization, Continuous Prompts, Catastrophic Forgetting

## TL;DR
QR-Prompt discretely quantizes the "residuals" between CLIP visual and text features—which are typically smoothed out by contrastive learning—into a set of frozen Discriminative Subspace Codebooks (DSQ). These discrete codes are then translated into class-adaptive continuous prompts via a Hierarchical Prompt Encoder (HPE) and a Prompt Combiner (PC). This mechanism balances stability and plasticity in FSCIL, outperforming existing SOTA methods on CUB200, CIFAR100, and miniImageNet.

## Background & Motivation

**Background**: Few-Shot Class Incremental Learning (FSCIL) requires models to learn new classes from very few samples (e.g., 5-way 5-shot) in each session without forgetting old ones. Early methods mostly relied on pure vision backbones (e.g., ResNet-18) for prototype alignment or dynamic network expansion. Recent shifts utilize Vision-Language Models (VLM) like CLIP for prompt learning, as multi-modal aligned representations offer better transferability.

**Limitations of Prior Work**: The authors identify a dual conflict. First, the "rigidity vs. flexibility" trade-off in prompts: fully optimizable prompts (L2P, DualPrompt, CODA-Prompt) offer plasticity but suffer from semantic drift as sessions progress; static or quantized prompts (VQ-Prompt) are stable but lack fine-grained discriminative power, causing subtle inter-class differences to collapse into the same quantized code. Second, CLIP's contrastive pre-training achieves global alignment through "feature decorrelation and homogenization," which effectively **flattens natural correlations between visual attributes** (structural cues like color, texture, shape). This makes the feature manifold more uniform, suppressing fine-grained details crucial for few-shot generalization.

**Key Challenge**: The plasticity $\leftrightarrow$ stability trade-off is extremely fragile in FSCIL (slight representation drift triggers catastrophic forgetting). Compounded by VLM contrastive objectives that suppress fine-grained discriminative information, achieving both "stability" and "granularity" becomes nearly impossible.

**Key Insight**: The authors observe that the **residual** $r = x^v - x^t$ between CLIP visual embeddings $x^v$ and text embeddings $x^t$ preserves the local manifold structure suppressed by contrastive learning. Two pieces of evidence support this: (a) cross-class mutual correlation of residuals is significantly higher than that of decorrelated visual features (Fig 1a); (b) residual magnitude correlates with the magnitude of the visual feature's second-order Hessian term $H_{f_v}(X)[\delta_i,\delta_i] \propto f_v(X+\delta_i)+f_v(X-\delta_i)-2f_v(X)$ (Fig 1b). This suggests that directions with larger residuals correspond to regions of high manifold curvature and rich fine-grained variation. In short: **residuals are "curvature-aware" complementary signals** containing details underfitted by text embeddings.

**Core Idea**: Convert residuals from "discrete quantization to continuous prompts." By quantizing the residual space into frozen discriminative codebooks (acting as stable anchors), and then re-encoding these discrete codes into class-adaptive continuous prompts (providing plasticity), the model resolves the FSCIL dilemma.

## Method

### Overall Architecture
QR-Prompt takes an image and its class text template as input and outputs a class-adaptive continuous prompt injected into the CLIP text encoder. The pipeline consists of three stages: **1) Calculate residuals $r = x^v - x^t$ using frozen CLIP encoders; 2) Discretize residuals into nearest code indices via Discriminative Subspace Quantization (DSQ); 3) Translate indices back to continuous fine-grained features via Hierarchical Prompt Encoder (HPE) and aggregate them into a single adaptive prompt via Prompt Combiner (PC).**

Stability is ensured through session division: In the **base session**, the DSQ codebook (including rotation matrices) and HPE+PC are trained end-to-end. In **incremental sessions**, the DSQ codebook is **frozen** as an invariant anchor, while only the lightweight HPE and PC modules are fine-tuned. Since prompts rely on frozen codebooks, the model **does not need to store class-level statistics** after each session, reducing memory overhead and resisting forgetting.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Image + Class Template"] --> B["Frozen CLIP Encoders<br/>Residual r = xv − xt"]
    B --> C["Discriminative Subspace Quantization DSQ<br/>M Subspaces + Rotation R + Fisher Regularization<br/>(Frozen after Base Session)"]
    C -->|Indices per Subspace| D["Hierarchical Prompt Encoder HPE<br/>Table Lookup + Cross-Subspace Attention"]
    D --> E["Prompt Combiner PC<br/>Residual-modulated query + Cross-Attention"]
    E -->|Prepend to 'a photo of a {class}'| F["Frozen CLIP Text Encoder<br/>→ Adaptive Text Features"]
```

### Key Designs

**1. Discriminative Subspace Quantization (DSQ): Discretizing Residuals into Frozen "Curvature-Aware" Codebooks**

DSQ quantizes residuals rather than visual embeddings themselves, as residuals preserve curvature details. It divides the residual space $\mathbb{R}^D$ into $M$ subspaces, each with a codebook $C_m = \{c_{m,1},\dots,c_{m,K}\}$. A residual $r_i$ is represented as concatenated nearest codes $\hat r_i = \{c_{1,q_{i1}},\dots,c_{M,q_{iM}}\}$. 

The novelty lies in modeling the manifold geometry: an orthogonal transform $R \in \mathbb{R}^{D\times D}$ aligns subspaces with the **principal curvature directions** of the residual field. Fisher-style regularization is introduced to maintain discriminability using between-class scatter $S_b$ and within-class scatter $S_w$:

$$\mathcal{L} = \sum_i \|R^\top r_i - \hat r_i\|^2 - \lambda\, \mathrm{trace}\big((S_w + \epsilon I)^{-1} S_b\big),$$

The first term minimizes reconstruction error, while the second (Fisher term) maximizes class separation. The codebook is **frozen after the base session**, ensuring cross-session consistency and preventing overfitting on small incremental data.

**2. Hierarchical Prompt Encoder (HPE): Translating Discrete Codes to Continuous Interactive Features**

HPE uses two stages to upsample discrete indices into context-aware prompt vectors. The first stage is **Hierarchical Embedding Lookup**, where $M$ independent tables $\{E_1,\dots,E_M\}$ map indices to $D_p$-dimensional continuous vectors $V_i = \{E_1(c_{1,i}),\dots,E_M(c_{M,i})\} \in \mathbb{R}^{M\times D_p}$. 

The second stage is **Cross-Subspace Attention**, which applies multi-head self-attention across the $M$ subspace embeddings of a batch to model attribute dependencies (e.g., combining "red" and "striped" into a coherent bird feature). This creates a hierarchy: DSQ Codebook (low-level curvature) $\rightarrow$ Embedding Lookup (mid-level continuity) $\rightarrow$ Cross-Subspace Attention (high-level fine-grained semantics).

**3. Prompt Combiner (PC): Aggregating Fragmented Features into Class-Adaptive Prompts**

PC uses a learnable query to aggregate HPE outputs. The query is modulated by the class mean residual $\mu_y$ to embed class-adaptivity directly:

$$Q_p = Q_0 \odot \big(1 + \tanh(W_c \mu_y)\big),$$

where $Q_0$ is a base query and $W_c$ projects the residual onto the query space. Modulated queries perform cross-attention over HPE outputs $A_L$ to produce a single prompt vector $p$, which is mapped to CLIP space as $p^* = W_t p$. This adaptive combination provides plasticity while the frozen DSQ ensures stability.

### Loss & Training
DSQ is optimized for 15 epochs in the base session using the joint "reconstruction + Fisher" objective. HPE and PC are trained end-to-end via an InfoNCE contrastive loss to align generated text features with visual features:

$$\mathcal{L} = -\log \frac{\exp(f(z_t, x^v)/\tau)}{\sum_{i=1}^{N}\exp(f(z_t^i, x^v)/\tau)},$$

In incremental sessions, only HPE/PC are fine-tuned for 20 epochs, while DSQ remains frozen.

### Theoretical Analysis
**Theorem 1 (Generalization Bound)**: In a PAC-Bayesian framework, the generalization gap $\Delta$ grows **logarithmically** with the number of new codes introduced per session. Freezing the codebook ($s_j=0$) yields a tighter bound.
**Theorem 2 (Margin Preservation Bound)**: Quantization error's impact on classification margin is bounded by $\mathbb{E}[\gamma_{y,c} - \hat\gamma_{y,c}] \le \beta\sqrt{2\,\mathrm{tr}(P_U \Sigma_q P_U)}$. Increasing $M$ or $K$ tightens this bound, and DSQ's transformation $R$ helps align quantization error with the manifold's local linear regions.

## Key Experimental Results

### Main Results
Evaluation on CUB200, CIFAR100, and miniImageNet using a CLIP ViT-B/16 backbone.

| Dataset | Metric | QR-Prompt | Prev. SOTA | Note |
|--------|------|-----------|----------|------|
| CUB200 | Avg(%)↑ | **82.12** | 80.49 (BiMC) | Highest performance; 80.68% in final session |
| CUB200 | PD(%)↓ | **6.17** | 7.00 (BiMC) | Smallest drop; most stable |
| CIFAR100 | PD(%)↓ | **7.88** | 8.37 (BiMC) | Lowest drop compared to VQPrompt's 20.21 PD |
| miniImageNet | Avg(%)↑ | **97.43** | 96.61 (VQPrompt)| Significant gain in accuracy |

VQ-Prompt shows high initial accuracy but collapses in later sessions (PD of 26.87 on CUB200), whereas QR-Prompt remains stable due to its frozen discriminative codebooks.

### Ablation Study
Ablation on CUB200 (✓: Used / ✗: Removed):

| Feature | DSQ Rot. | HPE Attn. | PC | Final Session Acc(%) |
|------|-----|-----|----|----|
| Residual | ✓ | ✓ | ✓ | **80.68** |
| Residual | ✗ | ✓ | ✓ | 78.46 |
| Residual | ✓ | ✗ | ✓ | 78.42 |
| Residual | ✓ | ✓ | ✗ | 80.07 |

Removing DSQ rotation causes the largest drop in later sessions, highlighting the importance of aligning the residual curvature. Using visual features instead of residuals also led to performance degradation.

### Key Findings
- **Subspace Count $M$ > Codebook Size $K$**: Increasing $M$ to 32 improves accuracy by better modeling local residual structures, whereas increasing $K$ provides diminishing returns.
- **$\lambda$ Sensitivity**: Optimal $\lambda$ is in the range $[0.1, 0.2]$. Values too high distort the manifold geometry.
- **Interpretability**: Saliency maps show QR-Prompt focuses on discriminative attributes (e.g., beak shape, head color) while Zero-Shot CLIP often attends to background noise.

## Highlights & Insights
- **Residuals as Curvature Signals**: The observation that $r = x^v - x^t$ captures information suppressed by contrastive learning is a powerful conceptual pivot.
- **Discrete-Continuous Division of Labor**: Using frozen discrete codebooks for stability and differentiable continuous decoders for plasticity effectively decouples the FSCIL trade-off.
- **Memory Efficiency**: No need to store class-level statistics (mean/covariance), making it suitable for long session sequences.
- **Theory-Driven Design**: Hyperparameters like $M$ and $K$ are directly guided by the margin preservation bound in Theorem 2.

## Limitations & Future Work
- **Attribute Overlap Assumption**: The method assumes new classes share some attribute distribution with base classes. If a new class contains entirely novel attributes not seen in the base session, the frozen codebook may underperform.
- **Low-Res Data**: Gains on CIFAR100 are smaller than on CUB200, as low-resolution images provide less curvature information in residuals.
- **Future Work**: Exploring "incremental expansion" of codebooks to cover novel attributes without violating stability bounds.

## Related Work & Insights
- **vs. VQ-Prompt**: VQ-Prompt quantizes original visual embeddings, leading to semantic drift and performance collapse in later sessions. QR-Prompt quantizes residuals and uses a hierarchical decoder to maintain granularity.
- **vs. CODA-Prompt/DualPrompt**: These rely on key-query prompt selection but lack an explicit discriminative anchor, making them more prone to interference.
- **vs. BiMC/FDR**: QR-Prompt outperforms calibration-based or keyword-decomposition methods by mining information from the geometric structure of multi-modal residuals.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (Residuals as curvature signals + Discrete-Continuous division)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Standard benchmarks + extensive ablation + theoretical validation)
- **Writing Quality**: ⭐⭐⭐⭐ (Logical flow; high-quality visualizations)
- **Value**: ⭐⭐⭐⭐ (Generalizable paradigm for VLM-based continual learning)

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Semantic-Guided Global-Local Collaborative Prompt Learning for Few-Shot Class Incremental Learning](semantic-guided_global-local_collaborative_prompt_learning_for_few-shot_class_in.md)
- [\[CVPR 2026\] HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning](hycal_training_free_prototype_calibration_for_cross_discipline_fscil.md)
- [\[CVPR 2026\] Exemplar-Free Class Incremental Learning via Preserving Class-Discriminative Structure](exemplar-free_class_incremental_learning_via_preserving_class-discriminative_str.md)
- [\[CVPR 2026\] Few-Shot Hybrid Incremental Learning: Continually Learning under Data Scarcity and Task Uncertainty](few-shot_hybrid_incremental_learningcontinually_learning_under_data_scarcity_and.md)
- [\[ICLR 2026\] PonderLM: Pretraining Language Models to Ponder in Continuous Space](../../ICLR2026/self_supervised/ponderlm_pretraining_language_models_to_ponder_in_continuous_space.md)

</div>

<!-- RELATED:END -->
