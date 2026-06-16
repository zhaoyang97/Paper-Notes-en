---
title: >-
  [Paper Note] Hyperbolic Gramian Volumes for Multimodal Alignment
description: >-
  [CVPR 2026][Multimodal VLM][Paper Note] To address the "volume collapse" issue (where $\det \approx 1$ and variance is near 0) of Euclidean Gramian volumes under L2 normalization, this paper translates Gramian volume alignment to hyperbolic (Lorentz model) space to preserve variance. By using a learnable scalar $\alpha$ to perform a convex combination of Euc
tags:
  - CVPR 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: 9907721cf3d7eb1c
---
# Hyperbolic Gramian Volumes for Multimodal Alignment

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Na_Hyperbolic_Gramian_Volumes_for_Multimodal_Alignment_CVPR_2026_paper.html)  
**Code**: To be confirmed  
**Area**: Multimodal VLM / Cross-modal Alignment / Hyperbolic Geometry  
**Keywords**: Gramian Volume, Hyperbolic Geometry, Video-Text Retrieval, Contrastive Learning, Mixed Geometry

## TL;DR
To address the "volume collapse" issue (where $\det \approx 1$ and variance is near 0) of Euclidean Gramian volumes under L2 normalization, this paper translates Gramian volume alignment to hyperbolic (Lorentz model) space to preserve variance. By using a learnable scalar $\alpha$ to perform a convex combination of Euclidean and hyperbolic volumes, the proposed HyperGRAM achieves a zero-shot T2V Recall@1 improvement of +1.8% to +2.9% over Euclidean GRAM across four video-text retrieval benchmarks.

## Background & Motivation
**Background**: Mainstream video-text retrieval follows the path of contrastive learning combined with cosine similarity. Recently, GRAM proposed using "volume," defined by the determinant of the Gram matrix $\mathrm{Vol}(G)=\sqrt{\det(G)}$, as an alignment metric capable of capturing high-order correlations among three modalities (text/video/audio) that cosine similarity misses.

**Limitations of Prior Work**: In Euclidean space, all embeddings are L2-normalized to a unit sphere ($\|x_i\|=1$). After alignment, off-diagonal terms tend toward orthogonality ($\langle x_i,x_j\rangle\approx 0$), resulting in the Gram matrix collapsing toward an identity matrix, where $\det(G_{Euc})\approx 1.0$ and the cross-sample standard deviation is only approximately 0.005. This "volume collapse" simultaneously erases two things: discriminative power across samples and sensitivity to semantic richness within matched pairs.

**Key Challenge**: The authors attribute this issue to geometric capacity. The "explanation space" $S(T)$ of a text description—referring to the set of all (video, audio) pairs semantically consistent with text $T$—grows exponentially with semantic richness ($|S(T)|\propto e^{c\cdot H(T)}$, where $H(T)$ is conditional semantic entropy). However, Euclidean volume only grows polynomially ($V(r)\propto r^3$), failing to accommodate the exponentially expanding explanation space, which inevitably leads to variance collapsing toward a constant.

**Goal**: To enable the volume to fulfill two roles simultaneously: ① A discriminative role: distinguishing between matched and unmatched triplets; ② A semantic role: maintaining variance within matched pairs proportional to $|S(T)|$.

**Key Insight**: Hyperbolic volume grows exponentially ($V(r)\propto e^{3r}$), matching the expansion of the explanation space and naturally preserving variance. However, preliminary experiments found that while pure hyperbolic geometry preserves variance, it underperforms compared to Euclidean GRAM in cross-category discrimination.

**Core Idea**: Euclidean (stable cross-category discrimination) and hyperbolic (intra-class semantic variance) geometries are complementary. Rather than choosing one over the other, a data-driven learnable mixture is used to combine them via a convex combination.

## Method

### Overall Architecture
HyperGRAM does not modify the backbone but replaces the "volume" algorithm: trimodal embeddings (text $x_t$, video $x_v$, audio $x_a$) follow two geometric paths. The Euclidean path calculates $V_{Euc}=\sqrt{\det(G_{Euc})}$ as usual. The hyperbolic path first projects embeddings onto the Lorentz hyperboloid, constructs a Gram matrix using the Lorentzian inner product, and then takes $V_{Hyp}=\sqrt{|\det(G_{Hyp})|}$. These two volumes are combined into a hybrid volume $V_\alpha$ using a scalar $\alpha$ (initialized at 0.5 and optimized via gradients), where $-V_\alpha$ is used as the logits for the volume contrastive loss (supplemented by the DAM hard negative loss from GRAM). This pipeline only replaces Euclidean inner products with Lorentzian ones, introducing near-zero additional parameters.

As this is a pure geometric/loss layer improvement involving matrix and determinant operations without multi-stage serial pipelines, no architecture diagram is provided; the mechanism is clarified via equations.

### Key Designs

**1. Explanation Space Theory: Explaining "Why Hyperbolic" via Exponential Capacity**

This serves as the theoretical foundation of the paper, identifying the root cause of "volume collapse." The authors formalize the explanation space of text $T$ as $S(T)=\{(v,a):(v,a)\text{ is semantically valid for }T\}$ and demonstrate that $|S(T)|$ is linked to the conditional semantic entropy $H(T)=-\mathbb{E}_{(v,a)\sim P(V,A|T)}[\log P(V,A|T)]$. Simple descriptions ("a dog") have concentrated conditional distributions with only dozens of valid combinations, whereas rich descriptions ("exquisite artistic performance with complex soundtrack") have diffuse distributions with $|S(T)|\propto e^{c\cdot H(T)}$ growing exponentially. To allow volume to serve a "semantic role" (where variance is proportional to $|S(T)|$), the geometry must provide exponential capacity. Proposition 1 states that hyperbolic volume $V\propto e^{(d-1)r}$ can represent exponential explanation spaces without saturation, whereas Euclidean polynomial growth $V\propto r^d$ cannot. Lemma 1 further proves that on the Lorentz hyperboloid, the variance of the spatial norm $\|x_i\|$ is transmitted through the time component to $\det(G_{Hyp})$, ensuring $\mathrm{Var}(V_{Hyp})\ge C\sigma^2>0$, while L2 normalization forces $\mathrm{Var}(V_{Euc})\to 0$. This elevates "collapse vs. variance preservation" from an empirical observation to a geometric necessity.

**2. Hyperbolic Gramian Volume and Variance Preservation Mechanism: Keeping the Gram Matrix from Collapsing via the Lorentz Model**

To address the collapse of the Euclidean Gram matrix into an identity matrix, the authors reconstruct the volume using the Lorentz model. The Lorentz hyperboloid is defined as $\mathbb{H}^n=\{x\in\mathbb{R}^{n+1}:\langle x,x\rangle_L=-1,\;x_0>0\}$, with the Lorentzian inner product $\langle x,y\rangle_L=-x_0 y_0+\sum_i x_i y_i$, where the time component is $x_0=\sqrt{1+\|x_{spatial}\|^2}$. Euclidean embeddings are projected onto the hyperboloid via $\pi(x)=[\sqrt{1+\|x\|^2},\,x]$, and the hyperbolic Gram matrix $G_{Hyp}=[\langle\pi(x_i),\pi(x_j)\rangle_L]$ is constructed to compute the pseudo-volume $V_{Hyp}=\sqrt{|\det(G_{Hyp})|}$. The key to variance preservation lies in the position-dependent time component:

$$\langle\pi(x_i),\pi(x_j)\rangle_L = -\sqrt{1+\|x_i\|^2}\sqrt{1+\|x_j\|^2}+x_i^\top x_j \neq \text{const}$$

Unlike L2 normalization which crushes all norms to 1, hyperbolic embeddings allow spatial norms to vary freely. This variation is transmitted through $x_0$ into each Gram term, allowing the matrix to retain structural diversity rather than collapsing. Empirical measurements show the hyperbolic volume distribution spread across $[2.01, 2.49]$ (std $\approx 0.12$), while Euclidean volumes are squeezed near 1.0 (std $\approx 0.005$). The authors select the Lorentz model over the Poincaré ball because the latter involves boundary division $(1-c\|p\|^2)^{-1}$ in its gradients, leading to numerical instability in FP16 mixed precision.

**3. Mixed Geometry Learning: Learnable $\alpha$ for Balancing Discrimination and Semantic Variance**

Since pure hyperbolic geometry can lose cross-category discriminative stability despite preserving variance, the authors combine both volumes rather than choosing one:

$$V_\alpha(T,V,A) = (1-\alpha)\cdot V_{Hyp}(T,V,A) + \alpha\cdot V_{Euc}(T,V,A),\quad \alpha\in[0,1]$$

$\alpha$ is initialized to 0.5 and learned end-to-end via projected gradient updates $\alpha^{(t+1)}=\mathrm{clip}(\alpha^{(t)}-\eta\nabla_\alpha L,0,1)$. This avoids the overhead of maintaining independent subspaces like in product manifolds. Interestingly, across four datasets, the learned $\alpha$ converges to approximately 0.5 (range $[0.48, 0.52]$), suggesting that Euclidean global alignment stability and hyperbolic hierarchical variance discrimination are complementarily weighted—an empirical finding that validates the starting premise of geometric complementarity.

### Loss & Training
The training follows the volume contrastive loss of GRAM, using the negative hybrid volume as similarity logits:

$$L_{volume} = \tfrac{1}{2}\Big[\mathbb{E}_{(T,V,A)}\big[-\log\tfrac{\exp(-V_\alpha(T,V,A))}{\sum_j \exp(-V_\alpha(T,V_j,A_j))}\big] + (\text{symmetric terms})\Big]$$

Matched triplets are optimized to have smaller volumes. This is combined with the Data-Anchor Matching (DAM) hard negative binary classification loss $L_{DAM}$ from GRAM (where hard negatives are sampled according to $p_{hard}\propto\exp(-V_\alpha)$). The final objective is $L=L_{volume}+\beta L_{DAM}$ with $\beta=0.1$. The implementation is based on VAST + EVA-CLIP ViT-g/14 (Vision), BEATs (Audio), and BERT-base (Text), pre-trained for 1 epoch on VAST150k before zero-shot evaluation.

## Key Experimental Results

### Main Results
Zero-shot retrieval Recall@1 (%) on four video-text benchmarks, with HyperGRAM compared directly against Euclidean GRAM:

| Method | MSR-VTT T2V | DiDeMo T2V | ActivityNet T2V | VATEX T2V |
|:---:|:---:|:---:|:---:|:---:|
| PMRL | 54.5 | 50.6 | 56.0 | 80.5 |
| GRAM (Euclidean) | 54.8 | 49.8 | 56.2 | 77.0 |
| Pure Hyperbolic (Ours) | 54.8 | 49.1 | 57.0 | 76.7 |
| **HyperGRAM (Ours)** | **56.6** | **51.3** | **58.2** | **79.9** |
| Gain over GRAM | +1.8 | +1.5 | +2.0 | +2.9 |

On average, +2.05% T2V R@1 and +1.38% V2T R@1 were achieved, setting new SOTAs on MSR-VTT, ActivityNet, and VATEX.

### Ablation Study
Volume statistics (cross-sample variance) providing intuitive proof of "collapse vs. variance preservation":

| Dataset | Euclidean Mean / Std | Hyperbolic Mean / Std |
|:---:|:---:|:---:|
| MSR-VTT | 1.000 / 0.005 | 2.15 / 0.12 |
| DiDeMo | 1.001 / 0.006 | 2.08 / 0.13 |
| ActivityNet | 0.999 / 0.005 | 2.12 / 0.11 |
| VATEX | 1.000 / 0.004 | 2.18 / 0.10 |

Hyperbolic volume variance is 20–25 times higher than Euclidean. In semantic role validation, when 300 MSR-VTT matched triplets are categorized into three complexity levels, hyperbolic volume increases monotonically (Simple 2.08 → Multi-object 2.21 → Complex narrative 2.38, +14%), even when controlling for text length.

### Key Findings
- **Hybrid > Pure Euclidean > Pure Hyperbolic (in discrimination)**: Pure hyperbolic underperforms Euclidean GRAM on benchmarks like DiDeMo/VATEX, showing that variance preservation alone is insufficient; hybrid geometry consistently outperforms both pure geometries.
- **$\alpha$ consistently converges to ≈0.5**: Across four datasets, $\alpha \in [0.48, 0.52]$, implying nearly equal complementarity; the authors hypothesize that datasets with stronger hierarchical structures might prefer $\alpha < 0.5$ (more hyperbolic), but current benchmarks converge near 0.5.
- **Volume-text length correlation changes sign with datasets**: $r=+0.335$ for MSR-VTT (coherent narrative) vs. $r=-0.124$ for DiDeMo (fragmented events). The negative correlation is interpreted as volume "penalizing semantic fragmentation"—where longer but incoherent text results in smaller volumes.

## Highlights & Insights
- **Elevating "geometric choice" to a provable principle**: By using explanation space theory and formal propositions, the authors present "why hyperbolic" as a geometric necessity rather than just an empirical phenomenon (std 0.005 vs 0.12).
- **Minimal changes with stable gains**: The core modification involves replacing inner products with Lorentzian inner products. It adds no new parameters and requires no backbone changes, yet provides stable improvements of ~2%, making it easy to migrate to any existing Gramian or contrastive retrieval framework.
- **"Discriminative + Semantic" dual-role volume** provides a reusable perspective: A single scalar metric handles both cross-sample discrimination and intra-class semantic sensitivity. This idea of "preserving variance rather than collapsing" can be transferred to any retrieval/alignment task where L2 normalization results in representation homogenization.
- **Pragmatic engineering reasons for Lorentz over Poincaré**: Avoiding boundary division prevents numerical explosions in FP16, a critical practical consideration for large-scale mixed-precision training.

## Limitations & Future Work
- **Narrow focus on video-text retrieval**: Evaluation is limited to four video-text benchmarks; transferability to image-text, pure text, or more complex modal combinations remains unknown.
- **Post-hoc explanation for $\alpha$ convergence**: The conclusion that hierarchical data would deviate from 0.5 is speculative and lacks constructive experimental verification; the true source of hybrid gains (hyperbolic variance vs. simple ensemble regularization) needs further decomposition.
- **Weak correlation evidence**: The Pearson $r$ between volume and text length is relatively small (max 0.335). The interpretation of negative correlation as "penalizing semantic fragmentation" is somewhat speculative.
- **Pseudo-volume is not true hyperbolic simplex volume**: $\sqrt{|\det(G_{Hyp})|}$ is a proxy proportional to the Cayley-Menger volume. Theoretical guarantees apply to relative ranking rather than absolute volume, which slightly diminishes the theoretical rigor.

## Related Work & Insights
- **vs. GRAM (Euclidean)**: GRAM pioneered the use of Gramian determinants for high-order multimodal correlation but suffered from volume collapse. This work inherits the volume contrastive + DAM framework while replacing the geometry and adding hybridization, consistently outperforming the original.
- **vs. MERU / Hyperbolic Image-Text Embeddings**: Works like MERU use hyperbolic geometry to handle modality gaps but focus on distance-based pairwise similarity (entailment cones). This paper is the first to bring "volume" as a high-order metric into hyperbolic space.
- **vs. Mixed-curvature / Product Manifold**: Traditional mixed-curvature approaches embed data into multiple independent Euclidean/spherical/hyperbolic subspaces. This work performs a simple scalar convex combination $V_\alpha=(1-\alpha)V_{Hyp}+\alpha V_{Euc}$, which is simpler, learnable end-to-end, and does not require independent subspaces.

## Rating
- Novelty: ⭐⭐⭐⭐ First to bring Gramian volume alignment to hyperbolic space with a learnable mixture; theoretically complete narrative.
- Experimental Thoroughness: ⭐⭐⭐ Solid results across four video-text benchmarks and variance/correlation analysis, though task coverage is narrow.
- Writing Quality: ⭐⭐⭐⭐ Explanation space theory clarifies the motivation well; formulas and illustrations are well-coordinated.
- Value: ⭐⭐⭐⭐ Minimal changes, stable gains, and easy migration; offers valuable insights for any alignment task where L2 normalization causes representation collapse.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

## Related Papers

- [\[CVPR 2026\] Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models](uncertainty-guided_compositional_alignment_with_part-to-whole_semantic_represent.md)
- [\[CVPR 2026\] Towards Dynamic Modality Alignment in Multimodal Continual Learning](towards_dynamic_modality_alignment_in_multimodal_continual_learning.md)
- [\[ICML 2026\] Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal ICL](../../ICML2026/multimodal_vlm/hyper-icl_attention_calibration_with_hyperbolic_anchor_distillation_for_multimod.md)
- [\[CVPR 2026\] Anchor-Guided Gradient Alignment for Incomplete Multimodal Learning](anchor-guided_gradient_alignment_for_incomplete_multimodal_learning.md)
- [\[CVPR 2026\] Linguistic Priors for Visual Decoupling: Towards Symmetric Vision-Brain Alignment](linguistic_priors_for_visual_decoupling_towards_symmetric_vision-brain_alignment.md)

</div>

<!-- RELATED:END -->
