---
title: >-
  [Paper Note] AdS-GNN - a Conformally Equivariant Graph Neural Network
description: >-
  [ICLR 2026][Graph Learning][AdS/CFT] This paper "lifts" point clouds from flat Euclidean space to a higher-dimensional Anti-de Sitter (AdS) space. Leveraging the correspondence in physics between AdS isometry transformations and boundary conformal transformations, the authors construct AdS-GNN, the first Graph Neural Network equivariant to the **full conf
tags:
  - ICLR 2026
  - Graph Learning
  - AdS/CFT
date: 2026-05-08
content_hash: 573a8120c575f877
---
# AdS-GNN - a Conformally Equivariant Graph Neural Network

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=EIyvsL5Cue](https://openreview.net/forum?id=EIyvsL5Cue)  
**Code**: To be confirmed  
**Area**: Graph Learning / Equivariant Neural Networks / Geometric Deep Learning  
**Keywords**: Conformal Symmetry, Equivariant GNN, Anti-de Sitter Space, AdS/CFT, Message Passing

## TL;DR
This paper "lifts" point clouds from flat Euclidean space to a higher-dimensional Anti-de Sitter (AdS) space. Leveraging the correspondence in physics between AdS isometry transformations and boundary conformal transformations, the authors construct AdS-GNN, the first Graph Neural Network equivariant to the **full conformal group** (including translations, rotations, scaling, and non-affine special conformal transformations). The model demonstrates stronger scale generalization on tasks such as SuperPixel MNIST, shape segmentation, and Ising model correlation functions, and can directly read out physically meaningful universal quantities like conformal dimensions from the trained network.

## Background & Motivation
**Background**: Equivariant neural networks are a major line of research in geometric deep learning. Since Cohen & Welling's group equivariant convolutions, the community has thoroughly studied equivariance for translations, rotations, reflections (E(n)), general isometries, and various Lie groups (affine groups, isometry groups on semi-Riemannian manifolds). Methods like E(n) equivariant GNNs (EGNN) condition messages on Euclidean distances $\|p_i-p_j\|$, making them naturally invariant to rotations and translations.

**Limitations of Prior Work**: However, these methods remain limited to "distance-preserving" symmetries. Conformal transformations are a larger group of symmetries that are **angle-preserving but not distance-preserving**—they additionally include scaling (dilatation) and **Special Conformal Transformations** (SCT, a non-affine operation of "inversion, then translation, then inversion"). Scale equivariance has been studied sporadically (multi-scale convolutions, Fourier layers, scale-space theory), but no method handles the full conformal group, particularly the difficult non-affine SCT.

**Key Challenge**: The action of the conformal group $\mathrm{Conf}(\mathbb{R}^d)$ elements in Euclidean space is **non-linear and coordinate-dependent** (e.g., the SCT term $\frac{x'}{\|x'\|^2}=\frac{x}{\|x\|^2}-b$). Designing operators equivariant to it directly on $\mathbb{R}^d$ is extremely difficult—this is the barrier that previous methods could not bypass.

**Key Insight**: The authors borrow a key fact from the AdS/CFT correspondence in theoretical physics—the global conformal group $\mathrm{Conf}_g(\mathbb{R}^d)$ of $d$-dimensional flat space is exactly **isomorphic** to the isometry group $PO(d{+}1,1)$ of the $(d{+}1)$-dimensional Anti-de Sitter space $\mathrm{AdS}_{d+1}$. In other words, "difficult" conformal transformations in flat space become "manageable" isometry transformations in a higher-dimensional AdS space, for which equivariant networks already have mature tools in geometric deep learning.

**Core Idea**: Lift data from $\mathbb{R}^d$ to $\mathrm{AdS}_{d+1}$, and condition message passing on the invariant proper distance in AdS to obtain equivariance to the full conformal group "for free."

## Method

### Overall Architecture
The input to AdS-GNN is a point cloud $\{x_i\}_{i=1}^N$ in $\mathbb{R}^d$ (optionally with features $h_i$), and the output is a conformally invariant representation for each node, which can be used for classification or restored to boundary fields with specified conformal dimensions for regression. The pipeline consists of three stages: first, lift each point in flat space to $\mathrm{AdS}_{d+1}$ (calculating an extra scale coordinate $z$), transforming the point cloud into a graph on the AdS manifold; second, perform message passing on this graph, but the messages only depend on the AdS invariant proper distance $D(X_i,X_j)$ (using AdS-GNN for scalar tasks and Clifford algebra-based AdS-CEGNN for vector tasks); finally, read out according to the task—summing nodes for classification, or multiplying by powers of $z$ to restore invariant features to fields with conformal dimension $\Delta$ for regression.

Intuitively, the extra dimension $z$ encodes the "length scale of the system's degrees of freedom": larger $z$ represents coarser scales. Conformal data is envisioned as "residing on the boundary of $\mathrm{AdS}_{d+1}$ ($z=0$)," while the network extends the boundary data into the AdS bulk for calculation.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input point cloud<br/>x_i ∈ R^d (+ features h_i)"] --> B["Lift to AdS<br/>KNN Centroid to determine z_i<br/>Multiply features by z^Δ"]
    B --> C["Build KNN Graph<br/>by proper distance"]
    C --> D["Invariant Message Passing<br/>Scalar: AdS-GNN<br/>Vector: AdS-CEGNN"]
    D -->|Classification: Sum| E["Conformally Invariant Output"]
    D -->|Regression: Multiply z^{-Δ}| F["Conformal Field O(x)<br/>Read out dimension Δ"]
```

### Key Designs

**1. Lifting to AdS Space: Replacing Non-linear Conformal Transformations with Linear Isometries**

This is the foundation of the paper, addressing the core pain point that conformal transformations are non-linear and difficult to design equivariant operators for on $\mathbb{R}^d$. $\mathrm{AdS}_{d+1}$ can be viewed as one branch of a hyperboloid embedded in $\mathbb{R}^{d+1,1}$ satisfying the constraint $\|Y\|=-1$ (with metric $\eta=\mathrm{diag}(-1,1,\dots,1)$). Solving this constraint with coordinates $X=(x,z)\in\mathbb{R}^d\times\mathbb{R}_{>0}$, the Riemannian metric on AdS is $ds^2=\frac{1}{z^2}\big(\sum_a (dx^a)^2+dz^2\big)$. The key fact is: $PO(d{+}1,1)$ is both the isometry group of $\mathrm{AdS}_{d+1}$ and the global conformal group of $\mathbb{R}^d$—this is the kinematic core of the AdS/CFT correspondence. The AdS manifold has a $d$-dimensional boundary at $z=0$, and the way the isometry group acts on boundary points $(x,z=0)$ is exactly the set of conformal transformations (translations/rotations/scaling/SCT). Thus, "local operations within the AdS bulk ⇒ conformally equivariant operations on the boundary."

**2. AdS Embedding Algorithm: Using Neighbor Centroids for Scale Coordinate $z$**

Naively placing points directly on the boundary at $z=0$ fails—the metric is singular at $z=0$, making the proper distance between any two boundary points infinite. Conversely, forcing a fixed small $z$ breaks symmetry. The author's approach (Algorithm 1) is: temporarily place each point at $z=z_0$ (a small regularization), compute an **AdS centroid** (a generalization of Euclidean centroids to hyperbolic space using the Galperin 1993 construction) for its $k_{\text{lift}}$ neighbors; the $z$-coordinate of the centroid has a finite limit as $z_0\to 0$ and depends on the relative spacing of these points. Finally, embed the point as $X_i=(x_i,\hat z_i)$. Intuitively, "the appropriate length scale for a point = its distance to neighbors." This choice **precisely preserves scale invariance** but **mildly breaks SCT**—which is physically expected, as any regularization inevitably breaks conformal invariance (Cardy). This is formalized in Proposition 4.1.1: the lifting process is strictly equivariant to the subgroup generated by translations, rotations, and scaling.

Furthermore, input features are treated as samples of an underlying conformal field $O(x)$. Bulk features are scalars and do not carry the $\lambda^\Delta$ factor, so lifting features requires multiplying by a scale factor: $h_i^{\text{lifted}}=\hat z_i^{\Delta}\, h_i^{\text{input}}$ (for image data $\Delta=0$, this can be skipped). This step corresponds to the bulk-to-boundary propagator in AdS/CFT.

**3. Invariant Message Passing based on Proper Distance: Replacing Euclidean Distance with AdS Geodesic Distance**

Once points $\{X_i\}$ are in AdS, a GNN is executed. Using EGNN as a blueprint—where messages are $m_{ij}=\psi_e(h_i^l,h_j^l,\|p_i-p_j\|^2)$ depending on Euclidean distance—AdS-GNN changes only one part: replacing Euclidean distance with the $PO(d{+}1,1)$-invariant AdS proper distance:

$$\cosh D(X,X')=\frac{z^2+z'^2+\sum_a (x^a-x'^a)^2}{2zz'},$$

namely $m_{ij}=\psi_e(h_i^l,h_j^l,D(X_i,X_j))$. If edges are not provided, a graph is built via $k_{\text{con}}$ nearest neighbors using proper distance. This yields a conformally equivariant GNN with almost no additional computational overhead, and the proper distance introduces locality across both physical space and scale ($z$-direction). Notably, while the lifting step mildly breaks SCT, the GNN itself is **exactly invariant** to the entire $\mathrm{Conf}(\mathbb{R}^d)$.

**4. AdS-CEGNN for Vector Features + Output Layer (Interpretability Source)**

The proper-distance messages above depend only on distance, thus only producing invariant features. For stronger equivariance (e.g., predicting vector fields), the authors utilize the fact that $\mathrm{AdS}_{d+1}$ is a conformally equivariantly embedded submanifold in $\mathbb{R}^{d+1,1}$, applying $O(d{+}1,1)$ equivariant Clifford group networks (Ruhe et al. 2023). Messages operate on multivectors $M_{ij}=\psi_e(H_i^l,H_j^l,X_i,X_j)$, denoted as AdS-CEGNN, which is equivariant to the restricted conformal group $\mathrm{Conf}(\mathbb{R}^d)=PO_0(d{+}1,1)$. The output stage is the inverse of lifting: to ensure the output is a boundary field with conformal dimension $\Delta$, one takes $O(x_i)=\hat z_i^{-\Delta} h_i^{l_{\text{final}}}$, ensuring it transforms as $O'(\lambda x)=\lambda^{-\Delta}O(x)$ under scaling. Here $\Delta$ can be set as a **trainable parameter**; after training, it directly represents the "learnt conformal dimension"—a source of interpretability.

## Key Experimental Results

### Main Results
The authors categorize tasks into computer vision and physics.

| Task | Baseline | Key Results |
|------|----------|-------------|
| SuperPixel MNIST Classification | EGNN / PΘNITA, etc. | 4.09% error on IID, on par with roto-equivariant methods; **maintains 4.09% on rotation+scale augmented test sets, while EGNN/PΘNITA degrade to random guessing.** |
| Shape Segmentation | EGNN / MPNN | Outperforms EGNN on IID, with significant advantages when training points are sparse. |
| 2d Ising Correlation Regress. | EGNN / MPNN | Lowest relative $L2$ across all scales; **2-point functions are over an order of magnitude better than baselines** (form is fixed by conformal invariance). |
| 3d Ising (non-solvable) | EGNN / MPNN | Significantly superior and recovers $\Delta_\sigma=0.518$. |
| N-body Charged Particles | CEGNN (Ruhe et al., SOTA) | AdS-CEGNN is superior and correctly recovers acceleration field dimension $\Delta_a=2$. |

The most compelling result is the augmented test on SuperPixel MNIST: when rotation and scale are superimposed, methods like EGNN (invariant only to rotation/translation) collapse to random levels, while AdS-GNN remains steady due to its exact scale invariance.

### Ablation Study

| Config / Dimension | Phenomenon | Explanation |
|------|------|------|
| IID vs. Augmented Test | AdS-GNN barely loses performance, EGNN collapses | Robustness from scale equivariance |
| Train points 64→32768 | AdS-GNN advantage is greatest in small-sample regime | Higher sample efficiency from equivariance |
| Extrapolation to coords / $N$ | AdS-GNN generalizes better | Evidence of learning underlying physics |
| Reading out $\Delta_\sigma, \Delta_\epsilon$ | Very close to ground truth (2d $\Delta_\sigma=1/8$...) | Interpretability |

### Key Findings
- **Scale generalization is the primary selling point**: On tasks involving scale variations, AdS-GNN's advantage over EGNN widens, and it does not require multi-scale augmentation.
- **Interpretability is a valuable byproduct**: Universal physical quantities like conformal dimensions, which are independent of microscopic details, can be read directly from the network. In N-body tasks, the correct recovery of $\Delta$ serves as a verification signal for model correctness.
- **Orientation information is a bottleneck**: It slightly underperforms orientation-aware methods like PΘNITA on MNIST and MPNN (using $x_i-x_j$) on Shapes, as it only uses invariant descriptors.

## Highlights & Insights
- **Physics duality as an engineering tool**: While AdS/CFT is usually a topic in quantum gravity, the authors use only the "kinematic" isomorphism (boundary conformal ⇔ bulk isometry) to simplify a difficult equivariance into one manageable by existing tools.
- **Almost zero extra overhead**: Compared to Euclidean EGNN, AdS-GNN simply swaps the distance formula for proper distance, maintaining the same complexity while gaining full conformal equivariance.
- **Physical intuition of $z$**: Explaining the extra dimension as "length scale" and determining it adaptively via neighbors is a pragmatic engineering trade-off that preserves scale invariance while mildly breaking SCT.
- **Trainable Conformal Dimension $\Delta$**: Embedding physical constants into learnable parameters gives the model both predictive power and interpretability, a paradigm applicable to other scientific machine learning tasks.

## Limitations & Future Work
- **SCT is not perfectly equivariant**: The lifting step mildly breaks it (an inevitable cost of regularization), though experiments show the impact is small. Theoretical error bounds are missing.
- **Lack of orientation information**: Relying solely on invariant descriptors makes it less effective than orientation/displacement-based methods on certain shape discrimination tasks.
- **Heavy reliance on physics tasks**: The most impressive results are in Ising and N-body tasks; CV benefits, beyond robustness, are comparable to EGNN on standard benchmarks.
- **Limited extractable information**: Conformal Field Theory is also characterized by 3-point coefficients $c_{abc}$; whether these can be extracted remains an open question.
- **Prospects**: The authors believe this consistency under local scale changes is promising for CV and robotics, where objects appear at different scales but require consistent predictions without massive augmentation.

## Related Work & Insights
- **vs. EGNN / E(n) Equivariant Networks**: EGNN uses Euclidean distance, making it equivariant only to rotations/translations. AdS-GNN expands this to the full conformal group via proper distance, preventing performance collapse under scale shifts.
- **vs. Scale-Equivariant Methods (Multi-scale Conv / Fourier / Scale-space)**: These usually only handle the scaling subgroup. AdS-GNN's geometry **by construction** enforces equivariance to the full conformal group, including non-affine transformations.
- **vs. CEGNN (Ruhe et al. 2023)**: CEGNN uses Clifford networks for $O(n)$ equivariance. AdS-CEGNN elevates this to $\mathbb{R}^{d+1,1}$ using the AdS submanifold property to achieve conformal equivariance.
- **vs. Semi-Riemannian Equivariant Networks**: This work can be seen as a specific implementation of general frameworks (e.g., Weiler/Zhdanov) on AdS, with the addition of the critical "lifting" step.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First GNN equivariant to the full conformal group (including SCT); ingenious use of AdS/CFT.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers various CV and physics tasks; lacks theoretical bounds for symmetry breaking.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation; high entry barrier due to AdS/CFT context.
- **Value**: ⭐⭐⭐⭐⭐ Opens a new path for conformal learning and provides an interpretable paradigm for physical quantity extraction.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On The Expressive Power of GNN Derivatives](on_the_expressive_power_of_gnn_derivatives.md)
- [\[AAAI 2026\] On Stealing Graph Neural Network Models](../../AAAI2026/graph_learning/on_stealing_graph_neural_network_models.md)
- [\[AAAI 2026\] Magnitude-Modulated Equivariant Adapter for Parameter-Efficient Fine-Tuning of Equivariant Graph Neural Networks](../../AAAI2026/graph_learning/magnitude-modulated_equivariant_adapter_for_parameter-efficient_fine-tuning_of_e.md)
- [\[ICLR 2026\] GNN-as-Judge: Unleashing the Power of LLMs for Graph Learning with GNN Feedback](gnn-as-judge_unleashing_the_power_of_llms_for_graph_learning_with_gnn_feedback.md)
- [\[ICML 2026\] Full-Spectrum Graph Neural Network: Expressive and Scalable](../../ICML2026/graph_learning/full-spectrum_graph_neural_network_expressive_and_scalable.md)

</div>

<!-- RELATED:END -->
