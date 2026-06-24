---
title: >-
  [Paper Note] Cutting the Skip: Training Residual-Free Transformers
description: >-
  [ICLR2026][Optimization][Residual Connections] Starting from the condition number of the Transformer Jacobian, this paper reveals that the essence of skip (residual) connections is to improve the network's condition number. Based on this, a scheme is proposed that **modifies only the initialization without changing the architecture**. This allows a Transformer completely devoid of residual connections to be trained as fast as models with residuals for the first time…
tags:
  - "ICLR2026"
  - "Optimization"
  - "Residual Connections"
  - "Network Initialization"
  - "Jacobian Condition Number"
  - "Second-order Optimization"
  - "Vision Transformer"
date: 2026-05-08
content_hash: 41739b67e8347647
---

# Cutting the Skip: Training Residual-Free Transformers

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=iJl3L059s6](https://openreview.net/forum?id=iJl3L059s6)  
**Code**: None (Paper states release after acceptance)  
**Area**: Network Optimization / Transformer Initialization  
**Keywords**: Residual Connections, Network Initialization, Jacobian Condition Number, Second-order Optimization, Vision Transformer

## TL;DR
Starting from the condition number of the Transformer Jacobian, this paper reveals that the essence of skip (residual) connections is to improve the network's condition number. Based on this, a scheme is proposed that **modifies only the initialization without changing the architecture**. This allows a Transformer completely devoid of residual connections to be trained as fast as models with residuals for the first time, while learning more abstract and hierarchically clear representations on dense prediction tasks, outperforming residual-based baselines.

## Background & Motivation
**Background**: Modern Transformers rely almost entirely on skip/residual connections to be trained at great depths. From ResNet to the Transformer by Vaswani et al., residuals are regarded as the cornerstone of deep network trainability. The common explanation is that they improve the condition number of the network Jacobian, allowing gradients to flow smoothly.

**Limitations of Prior Work**: While residual connections stabilize optimization, they simultaneously compromise the **hierarchical abstract structure** that deep networks should possess. The original intent of deep networks is to make representations increasingly abstract layer by layer; however, residuals continuously reinject shallow information into deeper layers, acting as "shortcuts." Consequently, residual-equipped networks behave like an ensemble of shallow sub-networks (Veit et al.). After convergence, many deep layers in Transformers contribute minimally and can even be pruned with negligible performance loss (Gromov et al.). That is, residuals cause a severe decoupling between "nominal depth" and "effective depth," masking the representation gains intended by depth.

**Key Challenge**: Residuals are indispensable for optimization but harmful for representation hierarchy—an inherent conflict between "stability vs. hierarchical abstraction." To obtain truly hierarchical representations, one must remove residuals; yet once removed, skipless Transformers exhibit extremely poor Jacobian condition numbers under random initialization and fail to train.

**Goal**: To enable stable and efficient training of Transformers completely without residual connections **without modifying the standard Transformer architecture**, thereby systematically studying "truly deep" ViTs for the first time.

**Key Insight**: The only prior work training skipless Transformers (He et al. 2023) **modified self-attention blocks** to maintain a well-conditioned forward kernel, but this broke the standard architecture, was incompatible with FlashAttention, and still converged significantly slower. This paper takes a different perspective: instead of modifying the forward kernel, it directly targets the **condition number of the network Jacobian**. Given that the role of residuals is to improve the condition number, can a "theoretically correct initialization" compensate for this benefit?

**Core Idea**: Analysis reveals that residuals in the Jacobian are equivalent to adding an identity matrix $I$, which shifts the singular spectrum of self-attention sub-blocks away from 0 and regularizes the minimum singular value. Therefore, by using a **carefully designed weight initialization**, the self-attention sub-blocks can be made naturally well-conditioned at initialization, reproducing this stability without adding residuals or changing the architecture.

## Method

### Overall Architecture
The objective is pure: after removing all residual connections, the Jacobian condition number of standard Transformer blocks explodes at initialization, preventing training. This work aims to **suppress this condition number to a healthy state solely through initialization**. The logic follows three steps: first, theoretically decompose the Jacobian of a residual-free Transformer block to identify which term the residuals specifically improve (Sections 4.2–4.3); second, derive an initialization scheme for the four self-attention projection matrices $W^Q, W^K, W^V, W^O$ based on this, ensuring the derivative matrix $K_\ell$ of the self-attention sub-block is well-conditioned at the start (Sections 5.1–5.2); finally, pair this with a second-order optimizer, SOAP, to maintain these "good initial conditions" throughout training, matching or exceeding the convergence speed of residual-based baselines.

This is an analysis-driven methodological paper—the core lies in the derivation of matrix condition numbers and initialization construction rather than a multi-module serial pipeline. The method introduces no new modules, does not modify the forward computation graph, and is fully compatible with FlashAttention.

Let the Transformer block be Self-Attention (SAB) + Feed-Forward Network (FFN): with residuals, $X_\ell = \hat X_{\ell-1} + \mathrm{SA}(\hat X_{\ell-1})$ and $\hat X_\ell = X_\ell + \mathrm{MLP}(X_\ell)$; skipless means removing these $+\hat X_{\ell-1}$ and $+X_\ell$ terms. Here $\mathrm{SA}(X) = AVW^O$, and attention $A = \eta(QK^\top)$, where $\eta$ is softmax.

### Key Designs

**1. Jacobian Decomposition: Identifying which term the residuals improve**

Writing the network as $f(x;\theta)$ (omitting token embeddings and output heads), the network Jacobian $J = \partial F/\partial\theta$ is split into block-columns for SAB and FFN sub-blocks. The authors adopt a simplifying assumption: the total network condition number is controlled by the worst sub-block condition number, $\kappa(J) \le \max_\ell\{\kappa(J_\ell), \kappa(\hat J_\ell)\}$, and it is known that the self-attention sub-block $J_\ell$ is much worse than the FFN sub-block $\hat J_\ell$—thus the bottleneck is self-attention.

The key comparison: let $K_\ell$ and $\hat K_\ell$ be the derivatives of SA and MLP outputs with respect to their inputs. With residuals, the derivative for parameter $\ell$ in SA involves products like $(\hat K_i + I_{nd})(K_i + I_{nd})$, which are **augmented by the identity matrix**. Without residuals, this becomes a bare product of $\hat K_i K_i$. This $+I_{nd}$ is the "magic" of residuals: it shifts the spectrum of an otherwise ill-conditioned $K_\ell$ (where the minimum singular value is near 0) and regularizes those near-zero values, improving the condition number. This leads to the core question: **Is there another way to achieve $\kappa(K_\ell) \approx \kappa(K_\ell + I)$ in a skipless setting?**

**2. $W^V W^O$ Scaled Orthogonal Initialization: Making the "Common Factor" condition number exactly 1**

Expanding the derivative of the self-attention sub-block:

$$K_\ell = (\hat X_{\ell-1}W^V_\ell W^O_\ell \otimes I_n)^\top A'_\ell + (W^V_\ell W^O_\ell)^\top \otimes A_\ell$$

The product $W^V_\ell W^O_\ell$ appears in both terms as a "common factor." To make $K_\ell$ well-conditioned, this product itself must be well-conditioned; ideally, it should be a (scaled) orthogonal matrix so all singular values are equal and $\kappa(W^V_\ell W^O_\ell) = 1$. The approach is direct: sample a zero-mean, unit-variance random square matrix $Q \in \mathbb R^{d\times d}$, perform SVD $Q = USV^\top$, and set $W^V_\ell = c \cdot U$ and $W^O_\ell = c \cdot V^\top$ ($c$ is a scaling constant). Thus $W^V_\ell W^O_\ell$ becomes a scaled orthogonal matrix, pinning this factor in the second term of $K_\ell$ to an optimal condition number.

**3. $W^Q W^{K\top}$ Diagonally Dominant Initialization: Rescuing the Attention Map from "Uniform Matrices"**

Orthogonal $W^V W^O$ is insufficient; $K_\ell$ also contains the attention matrix $A_\ell = \mathrm{softmax}(M_\ell)$, where $M_\ell = \hat X_{\ell-1}W^Q_\ell W^{K\top}_\ell\hat X^\top_{\ell-1}$. Its condition number depends on the structure of logits $M_\ell$. Proposition 1 highlights two extremes: if the range of logits per row $\Delta \ll \tau$ ("diffuse rows"), the softmax approaches a rank-1 uniform matrix $\frac1n\mathbf{1}\mathbf{1}^\top$, where $\kappa \gtrsim \tau/\Delta$ and worsens as token count $n$ increases. Conversely, if $M_\ell$ is **diagonally dominant** ($M_{ii} - \max_{j \ne i} M_{ij} \ge \gamma > 0$), the softmax approaches the identity matrix, yielding a good condition number. In random initialization, logits are "diffuse," and the attention map approximates a uniform matrix—the primary cause of ill-conditioning in $K_\ell$.

The solution is to initialize query/key projections as $W^Q_\ell W^{K\top}_\ell = \alpha Z + \beta I$, where $Z_{ij} \sim \mathcal N(0, 1/d)$ and $\alpha, \beta$ are constants. While this "mimetic initialization" empirically improves convergence, this paper provides a **theoretical motivation**: the identity term $\beta I$ encourages diagonal dominance in $W^Q_\ell W^{K\top}_\ell$, making the initial attention operator well-conditioned. The authors acknowledge that diagonal dominance in $W^Q W^{K\top}$ does not automatically ensure it for the projected $X^\top W^Q W^{K\top} X$; discussions on conditions for this transfer are provided in the appendix. Together (Proposition 2), these initializations ensure the entire $K_\ell$ is well-conditioned—the intuition being that the maximum singular value of the perturbation term $E_\ell$ is smaller than the minimum singular value of the dominant term $B_\ell$, such that $\kappa(K_\ell) \approx \kappa(B_\ell)$, removing the obstacle to training skipless Transformers.

**4. Pairing with SOAP Second-order Optimizer: Maintaining healthy conditions throughout**

A good start is only the beginning; condition numbers drift during training. This paper pairs the above initialization with the SOAP second-order optimizer (Vyas et al. 2025). Experiments show that while initialization alone recovers much performance under AdamW, switching to SOAP allows skipless ViT to **converge as fast as residual-equipped ViT** within the standard 300 epochs and eventually outperform it. The initialization provides a well-conditioned start, and the second-order optimizer maintains it across the ill-conditioned self-attention landscape.

### Loss & Training
No new loss functions are introduced. Supervised experiments use ViT-Base (12 layers, 12 heads, head dim 64, token dim 768). The skipless model removes all residuals in SAB and FFN. Self-attention weights use the proposed initialization ($\alpha=2, \beta=0.6, c=3$), and MLP parameters use scale-corrected uniform orthogonal initialization. Drop path is disabled for skipless. Evaluations on ImageNet-1k compare AdamW and SOAP. Self-supervised experiments use the DINO framework with ViT-Small ($\alpha=1.8, \beta=1, c=3$). Initialization hyperparameters $(\alpha, \beta, c)$ are noted to be insensitive.

## Key Experimental Results

### Main Results
Validation accuracy on ImageNet-1k for ViT-Base (Supervised): Removing residuals with AdamW directly collapses performance to 61.4%. The proposed initialization rescues it to 78.1%. When paired with SOAP, skipless reaches **80.8%**, outperforming the residual baseline by 0.5 percentage points.

| Configuration | Optimizer | Accuracy |
|---------------|-----------|----------|
| ViT-Base w/ Residuals | AdamW | 80.3% |
| ViT-Base w/ Residuals | SOAP | 80.1% |
| skipless (no init) | AdamW | 61.4% |
| skipless (no init) | SOAP | 77.0% |
| skipless + Ours | AdamW | 78.1% |
| skipless + Ours | SOAP | **80.8%** |

Self-supervised (DINO ViT-Small, 300 epochs) mIoU for dense linear probe segmentation and object discovery via TokenCut; skipless overall outperforms the residual baseline on dense tasks:

| Task / Dataset | Evaluation | Optimizer | w/ Residuals | skipless |
|----------------|------------|-----------|--------------|----------|
| Seg VOC2012 | Single-layer | AdamW | 56.3 | **62.3** |
| Seg VOC2012 | Multi-scale | AdamW | 61.6 | **65.4** |
| Seg COCOStuff | Single-layer | AdamW | 24.6 | **24.9** |
| Seg ADE20K | Single-layer | AdamW | 23.7 | 22.5 |
| Seg ADE20K | Multi-scale | AdamW | 26.0 | **26.3** |
| Obj Disc VOC2012 | TokenCut | SOAP | 49.4 | **63.2** |
| Obj Disc COCO20k | TokenCut | SOAP | 27.5 | **46.7** |

### Ablation Study

| Configuration | Key Metric (ImageNet acc.) | Description |
|---------------|----------------------------|-------------|
| skipless base (AdamW) | 61.4% | Residual-free without init; nearly fails |
| + Ours (AdamW) | 78.1% | Initialization alone restores significant performance |
| + Ours + SOAP | 80.8% | Init and second-order optimizer together outperform baseline |

### Key Findings
- **Initialization is the prerequisite for skipless training**: Without it, AdamW performance drops to 61.4%. With it, it reaches 78.1%—the single largest contribution.
- **Initialization $\times$ Second-order Optimizer are complementary**: Initialization alone (AdamW) cannot match the residual baseline; SOAP is required to reach equivalent convergence speed and a 0.5% gain, indicating the "good start" must be maintained.
- **Single-layer vs. Multi-scale reveals hierarchical differences**: Skipless slightly underperforms on the complex ADE20K in single-layer evaluation, attributed to the implicit cross-layer mixing in residual models. However, because skipless enforces stricter hierarchies and more abstract features per layer, it outperforms when **explicitly aggregating multi-scale features**.
- **Shallower but stronger, highlighting efficiency**: Depth analysis shows 9-layer skipless outperforms 12-layer residual ViTs in object discovery; 10-layer skipless matches residual ones in segmentation—validating parameter/depth efficiency.
- **Cleaner Representations**: PCA projections of Layer 11 features into RGB show mottled noise in residual models due to re-injected shallow info; skipless shows clear boundaries and consistent colors within objects, with higher semantic coherence.

## Highlights & Insights
- **Translating "Why residuals work" into a replaceable mathematical fact**: The authors prove residuals serve as the $+I_{nd}$ term in the Jacobian to regularize ill-conditioned singular values. Since it is merely about condition numbers, initialization can provide the same benefit—a brilliant "Aha!" moment.
- **No changes to architecture, compatible with FlashAttention**: Unlike He et al. which modified attention blocks, this work only touches weight initialization. The standard Transformer block is unchanged, making it "plug-and-play" and capable of utilizing FlashAttention hardware acceleration.
- **Orthogonal initialization via a "Common Factor" perspective**: Identifying $W^V W^O$ as a common factor in $K_\ell$ terms and using SVD to pin its condition number to 1 is a transferable strategy for controlling the Jacobian in other structures.
- **Providing theory for empirical mimetic initialization**: The previously empirical $\alpha Z + \beta I$ is justified through "diagonal dominance $\rightarrow$ identity-like softmax $\rightarrow$ well-conditioned."

## Limitations & Future Work
- **Scale Constraints**: Experiments were limited to ViT-Base (~100M parameters). Whether skipless training holds for billion-parameter models remains to be verified.
- **Vision-only Verification**: ViT was chosen for its hierarchical nature and ease of visualization; the effectiveness of this initialization in Language Transformers has not yet been tested.
- **Unverified Theoretical Assumptions**: Condition number analysis relies on assumptions like the worst sub-block dominance and block-incoherence, which the authors acknowledge lack explicit validation, relying instead on empirical stability across depths.
- **Dependence on Second-order Optimizers**: Without SOAP, skipless+init still trails residual models; the overhead of second-order optimizers is an implicit cost.

## Related Work & Insights
- **vs. He et al. 2023**: That work modified architecture to prevent kernel rank collapse. This paper targets the **Jacobian condition number**, modifies only initialization, preserves the architecture, and is verified on vision models. The advantage here is FlashAttention compatibility and faster convergence.
- **vs. Residual Paradigms (He et al. 2016; Vaswani et al. 2017)**: While residuals are traditionally seen as necessary for depth, this paper argues they are necessary for optimization but harmful for hierarchy, providing an existence proof for "training without skips."
- **vs. Studies on Residuals "Shallowing" Networks (Veit et al. 2016; Gromov et al. 2025)**: This work turns the observation that residuals make networks behave like shallow ensembles into a motivation to remove them for better abstraction, supported by PCA and multi-scale gains.
- **vs. Conditioned Initialization (Ji et al. 2025a/b; Saratchandran & Lucey 2025)**: While also focusing on condition numbers, this work focuses on the more aggressive **skipless** setting with specific rules for $W^V W^O$ and $W^Q W^K$.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first to train standard skipless Transformers via pure initialization by treating "residuals = condition number improvement" as a replaceable fact.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive analysis across supervised/self-supervised tasks and visualization, though limited to ~100M parameters in vision.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous logic from Jacobian decomposition to initialization construction, with clear propositions and honest assessments of limitations.
- Value: ⭐⭐⭐⭐ Opens the door for systematic study of "truly deep" skipless ViTs and hierarchical representation learning with a plug-and-play engineering approach.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] WSM: Decay-free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training](wsm_decay-free_learning_rate_schedule_via_checkpoint_merging_for_llm_pre-trainin.md)
- [\[NeurIPS 2025\] Covariances for Free: Exploiting Mean Distributions for Training-free Federated Learning](../../NeurIPS2025/optimization/covariances_for_free_exploiting_mean_distributions_for_training-free_federated_l.md)
- [\[ICLR 2026\] Sign-SGD via Parameter-Free Optimization](sign-sgd_via_parameter-free_optimization.md)
- [\[ICLR 2026\] Markovian Transformers for Informative Language Modeling](markovian_transformers_for_informative_language_modeling.md)
- [\[ICLR 2026\] Learning to Recall with Transformers Beyond Orthogonal Embeddings](learning_to_recall_with_transformers_beyond_orthogonal_embeddings.md)

</div>

<!-- RELATED:END -->
