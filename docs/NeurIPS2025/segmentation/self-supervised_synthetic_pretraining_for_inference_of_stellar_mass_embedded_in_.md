---
title: >-
  [Paper Note] Self-supervised Synthetic Pretraining for Inference of Stellar Mass Embedded in Dense Gas
description: >-
  [NeurIPS 2025 (Workshop: ML and the Physical Sciences)][Segmentation][self-supervised pretraining] This paper proposes a "synthetic data-driven self-supervised pretraining" paradigm: one million synthetic fractal images are first generated via the Flame algorithm to pretrain a ViT-L/16 encoder using the DINOv2 framework; the frozen encoder is then transferred directly to an extremely limited set of magnetohydrodynamic (MHD) star-formation simulation data, achieving stellar mass prediction via kNN regression ($R^2 = 0.81$) and zero-shot unsupervised semantic segmentation via PCA projection—slightly outperforming a fully supervised ResNet-18 baseline trained on the same data.
tags:
  - "NeurIPS 2025 (Workshop: ML and the Physical Sciences)"
  - Segmentation
  - self-supervised pretraining
  - synthetic fractal images
  - DINOv2
  - ViT
  - stellar mass inference
  - MHD simulation
  - zero-shot semantic segmentation
date: 2026-05-08
content_hash: e86ef25c4c8c9c6b
---

# Self-supervised Synthetic Pretraining for Inference of Stellar Mass Embedded in Dense Gas

**Conference**: NeurIPS 2025 (Workshop: ML and the Physical Sciences)
**arXiv**: [2510.24159](https://arxiv.org/abs/2510.24159)
**Code**: None
**Area**: Self-supervised Learning / Astrophysics / Image Segmentation
**Keywords**: self-supervised pretraining, synthetic fractal images, DINOv2, ViT, stellar mass inference, MHD simulation, zero-shot semantic segmentation

## TL;DR

This paper proposes a "synthetic data-driven self-supervised pretraining" paradigm: one million synthetic fractal images are first generated via the Flame algorithm to pretrain a ViT-L/16 encoder using the DINOv2 framework; the frozen encoder is then transferred directly to an extremely limited set of magnetohydrodynamic (MHD) star-formation simulation data, achieving stellar mass prediction via kNN regression ($R^2 = 0.81$) and zero-shot unsupervised semantic segmentation via PCA projection—slightly outperforming a fully supervised ResNet-18 baseline trained on the same data.

## Background & Motivation

**Background**: Stellar mass is one of the most fundamental physical quantities in astrophysics, governing a star's luminosity, lifetime, evolutionary trajectory, and nucleosynthetic processes—the latter producing the chemical elements that constitute the solar system and life itself. In astronomy, the Initial Mass Function (IMF) describes the distribution of stellar masses and is observed to exhibit a remarkably consistent shape across diverse environments, yet its physical origin remains an open question. Accurately determining the masses of young forming stars (protostars and pre-main-sequence stars) from observational data is central to uncovering the origin of the IMF, but this task faces severe observational challenges.

**Limitations of Prior Work**: Young stars are deeply embedded within their natal molecular clouds, heavily obscured by dense gas and nearly invisible at optical wavelengths. Compounding this, the luminosity of these young stars is dominated by gas accretion rather than stellar radiation, rendering mass-estimation methods commonly applied to main-sequence stars entirely inapplicable. Conventional dynamical estimation methods assume spherical symmetry, but gas distributions in star-forming regions are highly inhomogeneous—intricately structured with filaments and dense cores—making the spherical-symmetry assumption highly unreliable. High-resolution three-dimensional MHD simulations can faithfully capture the physics of star formation, but their computational cost is prohibitive: a single simulation in this work requires 81.2 EFLOPS, resulting in an extremely limited pool of labeled simulation data (approximately 32,000 samples).

**Key Challenge**: Deep learning has demonstrated tremendous potential for image analysis and could, in principle, map complex two-dimensional gas density and velocity field information onto stellar mass. However, the acute scarcity of high-quality labeled data severely constrains the applicability of conventional supervised learning. This constitutes a canonical "data efficiency" challenge: how to train a reliable model under extremely limited labeled data.

**Goal**: Specifically, the paper addresses two key questions: (1) Can cheap synthetic data replace expensive simulation data for model pretraining, enabling the model to acquire general visual representations? (2) Can the pretrained frozen model achieve effective stellar mass prediction and region segmentation on a small amount of physical simulation data without any fine-tuning?

**Key Insight**: The authors' key observation draws from a series of seminal computer vision works—Kataoka et al. (2020) demonstrated that supervised pretraining on fractal images alone can approach ImageNet-22k pretraining performance on natural images. This finding was subsequently extended to ViT architectures and self-supervised learning frameworks. Fractal images offer several distinctive advantages: they can be generated in unlimited quantities via mathematical formulae, they raise no privacy or ethical concerns, and their computational cost is minimal. Crucially, the self-similar structure of fractals shares visual resemblance—at some level—with the gas distributions characteristic of star-forming regions (filamentary structures, nested density gradients, etc.), providing intuitive motivation for cross-domain transfer.

**Core Idea**: One million synthetic fractal images generated by mathematical formulae are used in place of expensive physical simulation data for DINOv2 self-supervised pretraining, enabling the frozen ViT encoder to perform stellar mass prediction and zero-shot semantic segmentation on limited MHD simulation data without any fine-tuning.

## Method

### Overall Architecture

The proposed pipeline divides cleanly into two stages and three evaluation tasks. The first stage is **synthetic data generation and self-supervised pretraining**: one million synthetic fractal images (resolution $336 \times 336$) are generated via an extended Flame algorithm, and a ViT-L/16 encoder is then self-supervisedly pretrained for 100 epochs using the DINOv2 framework. This stage operates entirely on mathematically generated images and involves no astrophysical data whatsoever. The second stage is **frozen feature evaluation**: the pretrained encoder weights are fully frozen and applied to two-dimensional projection maps produced by MHD star-formation simulations. Each simulation snapshot is encoded into a 1024-dimensional feature vector, which is then used for stellar mass prediction via kNN regression and unsupervised semantic segmentation via PCA projection. No backpropagation or parameter updates occur in the second stage—the evaluation is entirely "plug-and-play" zero-shot/frozen-feature inference.

### Key Designs

1. **Synthetic Fractal Image Generation System**:

    - **Function**: An extended version of the Flame algorithm that generates large-scale fractal image datasets via iterated function systems (IFS), serving as a substitute for expensive real data in self-supervised pretraining.
    - **Mechanism**: Each fractal image is generated as follows. A set of parameters $\theta_i = (a_i, b_i, c_i, d_i, e_i, f_i)$ is first randomly sampled; these define an affine transformation $w(\bm{x}; \theta_i) = \begin{pmatrix} a_i & b_i \\ c_i & d_i \end{pmatrix} \bm{x} + \begin{pmatrix} e_i \\ f_i \end{pmatrix}$. At each iteration, nonlinear transformations from the original Flame algorithm (e.g., spherical, bubble) are probabilistically applied to produce the next sample point $\bm{x}_{i+1} = w(\bm{x}_i; \theta_i)$. One million points are sampled per image and interpolated to $336 \times 336$ resolution. To ensure informational richness, only candidate frames covering $\geq 80\%$ of the image plane are accepted, yielding a training dataset of one million images. The total compute for the entire dataset is approximately 2.67 EFLOPS—an average of 2.67 TFLOPS per image—several orders of magnitude lower than the 81.2 EFLOPS required for a single MHD simulation.
    - **Design Motivation**: Fractal images offer three core advantages. First, they can be generated in unlimited quantities at minimal cost via mathematical formulae, entirely bypassing the bottleneck of physical simulation or observational data acquisition. Second, the self-similar structure of fractals (multi-scale repetition, complex geometric morphology) provides rich texture and structural learning signals for visual encoders. Third, prior work has demonstrated that fractal pretraining yields visual representations transferable to natural images, providing a theoretical basis for transfer from fractal images to astrophysical images. The 80% coverage threshold ensures each image contains sufficiently complex structural information, avoiding overly sparse or trivial patterns.

2. **MHD Simulation Data Pipeline and Feature Encoding**:

    - **Function**: Two-dimensional projection maps are extracted from high-resolution three-dimensional MHD simulations and used as input data for downstream tasks.
    - **Mechanism**: The simulation is conducted in a cube of side length 4 parsecs (approximately $3.08 \times 10^{13}$ km), containing 3000 solar masses of gas at an initial uniform proton number density of $1365 \, \text{cm}^{-3}$, with a $10 \, \mu\text{G}$ magnetic field applied along the $z$-axis and an initial velocity field with Mach number 10. The SFUMATO adaptive mesh refinement (AMR) code is used, achieving a finest spatial resolution of $\Delta x \sim 3 \times 10^{-3}$ parsecs—sufficient to resolve the Jeans length with at least 5 grid cells. When gas density exceeds a threshold, unstable dense clumps are replaced by sink particles (accretion radius $5.0 \times 10^{-4}$ pc); sink particles accrete surrounding gas within a fixed radius, and their accreted mass is taken as the protostellar mass. A total of 32,000 snapshots are constructed, each centered on a protostar and spanning a 0.5 pc region. Each snapshot is projected along the $x$, $y$, and $z$ directions to produce $64 \times 64$ two-dimensional maps with three physical channels: column density $N_\text{HI}$, mean line-of-sight velocity $v_\text{los}$, and velocity dispersion $\sigma_v$. Preprocessing applies a logarithmic transform to stellar mass and column density, and min-max normalization to velocity and velocity dispersion.
    - **Design Motivation**: The three-channel design (density, velocity, velocity dispersion) is intended to provide the model with complementary physical information. Column density reflects the total gas accumulation along the line of sight and serves as a fundamental measure of the star-forming environment; mean line-of-sight velocity reveals macroscopic gas motion patterns (e.g., accretion flows, bipolar outflows); velocity dispersion indicates turbulently active regions and potential star-formation hotspots. Using three projection directions ($x$, $y$, $z$) further enhances data diversity and avoids single-viewpoint bias. Cropping to a 0.5 pc region centered on the protostar ensures the target object always appears at the image center, allowing the model to focus on the local environment surrounding the star.

3. **DINOv2 Self-supervised Pretraining and Frozen Feature Transfer**:

    - **Function**: The DINOv2 framework is used to train a ViT-L/16 encoder on synthetic fractal data; the encoder is then fully frozen, and downstream tasks are evaluated in a zero-shot/frozen-feature manner via PCA whitening and kNN regression.
    - **Mechanism**: DINOv2 is an advanced self-supervised learning framework whose core principle is to learn visual representations by enforcing consistency constraints across multiple augmented views (different crops, flips, color jitter, etc.) of the same image. Specifically, it employs a teacher–student architecture in which the teacher network is updated via an exponential moving average (EMA) of the student network parameters. The pretraining configuration uses a ViT-L/16 encoder with patch size 16, input resolution 336, batch size 1024, trained for 100 epochs. A cosine annealing learning rate schedule is adopted with a maximum of 0.04 and 10 warm-up epochs. After pretraining, the encoder parameters are fully frozen. For the downstream stellar mass prediction task, each $64 \times 64$ simulation snapshot is encoded into a 1024-dimensional feature vector. PCA is then fitted on the training set features (retaining all 1024 dimensions, i.e., PCA whitening), and the fitted PCA transform is applied to all data features. Finally, a distance-weighted kNN regressor ($k = 5$) predicts the logarithm of stellar mass in the PCA-whitened feature space. The dataset is split into 24,000 training samples and 8,000 test samples.
    - **Design Motivation**: DINOv2 is preferred over other self-supervised frameworks (e.g., MoCoV3, MAE) for two key reasons. First, DINOv2 has been shown to capture rich semantic structure, achieving strong performance on multiple downstream tasks without fine-tuning—precisely matching the paper's "frozen feature evaluation" requirement. Second, the PCA components of DINOv2 features have been demonstrated to reveal semantically meaningful structure in natural images (e.g., foreground/background segmentation), opening the possibility of unsupervised segmentation for astrophysical images. PCA whitening removes inter-dimensional correlations and standardizes the variance of each dimension, making kNN regression more effective under Euclidean distance metrics. kNN is preferred over linear probing because it involves no parameter learning whatsoever, constituting the "purest" frozen feature evaluation and most faithfully reflecting the quality of the pretrained representations.

### Loss & Training

The paper involves two entirely independent training procedures. The first is the DINOv2 self-supervised pretraining stage, which employs DINOv2's native self-distillation loss: the teacher and student networks produce probability distributions over different augmented views of the same image, and a cross-entropy loss constrains the student's output to be consistent with the teacher's. The teacher network is updated via EMA of the student's parameters and does not participate in gradient backpropagation directly. This stage is conducted entirely on synthetic fractal images without any astrophysical labels.

The second is the supervised ResNet-18 baseline used for comparison, which employs a standard L2 regression loss (mean squared error, MSE) to directly predict the logarithm of stellar mass. This baseline is trained end-to-end on MHD simulation data using the same learning rate schedule as DINOv2 (cosine annealing, maximum learning rate 0.04, 10 warm-up epochs followed by 90 cosine decay epochs).

Notably, in the final downstream evaluation, the DINOv2 encoder undergoes no training whatsoever—no parameter updates, no loss function, no backpropagation. The kNN regressor is also non-parametric, producing predictions solely via distance-weighted averaging over nearest neighbors in the training set. This "zero-learning" evaluation paradigm is the methodological centerpiece of the paper.

## Key Experimental Results

### Main Results

The core experiment is frozen-feature stellar mass regression, comparing different models and initializations on the test set (8,000 samples). Evaluation metrics are the coefficient of determination $R^2$ (higher is better) and root mean squared error RMSE (lower is better).

| Method | Initialization | $R^2$ (↑) | RMSE (↓) |
|--------|---------------|-----------|----------|
| ResNet-18 (supervised) | Random | -1.9 | 0.34 |
| ResNet-18 (supervised) | Pretrained | 0.80 | 0.089 |
| DINOv2 + kNN ($k=5$) | Random | -0.58 | 0.52 |
| DINOv2 + kNN ($k=5$) | Synthetic fractal pretraining | 0.80 | 0.089 |
| DINOv2 + kNN ($k=5$) + PCA whitening | Synthetic fractal pretraining | **0.81** | **0.088** |

These results convey several important signals. First, synthetic fractal pretraining substantially improves model performance—DINOv2 improves from $R^2 = -0.58$ (random initialization, completely unusable) to $R^2 = 0.81$ (after pretraining), an enormous gain. Second, the entirely "zero-parameter-learning" pipeline of self-supervised pretrained DINOv2 with PCA whitening and kNN regression marginally outperforms the fully supervised ResNet-18 baseline trained end-to-end on the same data ($R^2$: 0.81 vs. 0.80; RMSE: 0.088 vs. 0.089). Third, PCA whitening yields a small but consistent gain ($R^2$ from 0.80 to 0.81, RMSE from 0.089 to 0.088), confirming that decorrelation benefits kNN regression.

### Ablation Study

| Configuration | $R^2$ | RMSE | Notes |
|--------------|-------|------|-------|
| DINOv2 (pretrained) + PCA + kNN | 0.81 | 0.088 | Full model: best performance |
| DINOv2 (pretrained) + kNN (no PCA) | 0.80 | 0.089 | Slight drop without PCA whitening |
| DINOv2 (random init) + kNN | -0.58 | 0.52 | Complete failure without pretraining |
| ResNet-18 (random init, supervised) | -1.9 | 0.34 | Random-init ResNet-18 also fails |

This comparison constitutes an effective ablation analysis:

- **Criticality of pretraining**: DINOv2 with random initialization achieves $R^2 = -0.58$ (a negative value indicating predictions worse than simply using the training set mean), whereas synthetic fractal pretraining yields $R^2 = 0.81$. This is the paper's most central finding—synthetic pretraining is the soul of the entire method; without it, everything collapses.
- **Marginal gain from PCA whitening**: PCA whitening yields only a small improvement in $R^2$ (from 0.80 to 0.81), indicating that the raw features produced by pretraining are already quite effective. PCA whitening primarily serves a "finishing touch" role, improving kNN distance computation quality in high-dimensional space through decorrelation.
- **Self-supervised vs. supervised**: DINOv2 self-supervised pretraining (on fractal images, without labels) matches or slightly exceeds fully supervised ResNet-18 training (on simulation data, with labels). This fundamentally demonstrates that, in data-scarce settings, the value of self-supervised representation learning may exceed that of labeled supervisory signals.

### Key Findings

- **Mass-range dependence**: From the scatter plots in Figures 2(b) and 2(c), both methods track true values reasonably well up to approximately $\leq 6 \, M_\odot$, a regime supported by more than $10^2$ training samples. In the transitional range $6\text{–}15 \, M_\odot$, DINOv2 outperforms ResNet-18—DINOv2 captures many true values while ResNet-18 tends to systematically underestimate. At higher masses ($> 15 \, M_\odot$), both models become unreliable due to fewer than 10 training samples. This finding suggests that the representations learned by DINOv2 generalize better in data-sparse regimes.

- **Semantic significance of PCA components**: The zero-shot semantic segmentation experiment reveals rich semantic structure in the pretrained features. Mapping the first three PCA components to RGB color space produces clear color-based partitioning of different regions: black regions correspond to diffuse low-density areas or regions of very high velocity dispersion (the latter potentially indicating ongoing star-formation activity); yellow to yellow-green regions indicate low velocity dispersion; magenta and dodger blue indicate negative and positive line-of-sight velocities, respectively, in high-velocity-dispersion regions where gas may be accreting onto dense cores and contributing to stellar mass growth. Crucially, this semantic segmentation emerges entirely spontaneously, without any labeled data or supervised fine-tuning.

- **Efficacy of cross-domain transfer**: There exists a substantial domain gap between mathematically generated fractal images and physically simulated star-forming regions—the former are abstract geometric patterns, the latter are physically meaningful density and velocity fields. Nevertheless, pretraining remains effective, suggesting that DINOv2 learns not some fractal-specific "shape recognition" capability on fractal images, but rather a more general "visual structure understanding" ability—encoding general visual features such as multi-scale textures, hierarchical structures, and spatial gradients. This finding echoes the philosophy of the original DINOv2 paper: large-scale self-supervised pretraining can yield "universal visual features" whose semantic structure is rich enough to bridge even the enormous domain gap from everyday natural images to physical simulation data.

- **Catastrophic performance of random initialization**: ResNet-18 and DINOv2 with random initialization achieve $R^2$ values of -1.9 and -0.58, respectively, meaning that randomly initialized models not only fail to learn useful information but produce predictions even worse than simply using the training set mean. For ResNet-18, this demonstrates that 32k samples of $64 \times 64$ simulation data are severely insufficient to train even a shallow CNN from scratch. For DINOv2, the 1024-dimensional features of a randomly initialized ViT-L (~300M parameters) are essentially random projections, and kNN queries in a high-dimensional random space naturally yield no meaningful results. This catastrophic "zero-baseline" performance strongly reinforces the value of synthetic pretraining.

## Highlights & Insights

- **A new paradigm for synthetic data pretraining**: The paper's most central contribution lies not in stellar mass prediction per se, but in validating the feasibility of a general paradigm: "self-supervised pretraining on mathematically generated synthetic images → zero-shot transfer to domain-specific tasks." This idea generalizes to any scientific computing domain with scarce labeled data (e.g., microstructural analysis in materials science, remote sensing interpretation in earth science, rare lesion detection in medical imaging). The generation cost per fractal image is only 2.67 TFLOPS, whereas a single MHD simulation snapshot requires 2540 TFLOPS—a cost differential approaching three orders of magnitude—making this approach highly attractive under computational resource constraints.

- **Extreme simplicity of frozen feature evaluation**: The entire downstream evaluation pipeline is remarkably simple: the pretrained encoder is fully frozen with no fine-tuning; PCA whitening is a linear transformation completable in one step; kNN regression is non-parametric with no learnable parameters. No gradient computation or parameter optimization occurs anywhere between "input simulation image" and "output mass prediction." This extreme simplicity implies: (1) no risk of overfitting, since there are literally no parameters to overfit; (2) minimal computational cost, requiring only a single forward pass and a kNN query; (3) strong interpretability, since PCA components can be directly visualized as color maps to reveal the structure of the feature space.

- **Physical meaning revealed by PCA semantic segmentation**: The PCA components of DINOv2 features spontaneously partition star-forming regions into semantically meaningful segments, distinguishing dense cores, accretion flows, turbulent regions, and other physical structures. The "aha moment" of this finding is that a model that has never seen any astrophysical data—having learned only general visual representations from fractal images—can automatically identify physically meaningful regional structures in star-forming environments. This hints that visual self-supervised learning may capture a universal "texture–structure" hierarchical representation that aligns with physical structure.

- **Generalization advantage in data-sparse regimes**: In the data-sparse range $6\text{–}15 \, M_\odot$, DINOv2 frozen features outperform fully supervised ResNet-18. This counterintuitive but deeply significant result indicates that general representations learned via self-supervised pretraining are more robust than task-specific representations from supervised training in low-data regimes. A plausible explanation is that the representation space learned by DINOv2 on synthetic data is smoother and more continuous, enabling kNN regression to produce reasonable predictions via spatial interpolation even in data-sparse regions, whereas ResNet-18's supervised training may overfit to data-dense regions and generalize poorly where data are sparse.

## Limitations & Future Work

- **Depth limitations inherent to a workshop paper**: As a short paper (only 6 pages of main text) at the NeurIPS ML4PS workshop, the experimental validation has limited depth. Systematic ablations over several key hyperparameters are absent (e.g., the value of $k$ in kNN, the number of PCA dimensions retained, the effect of fractal image count, and comparisons across different ViT scales). More importantly, comparisons with other self-supervised methods (MAE, MoCoV3, SimCLR) and alternative synthetic data generation strategies are lacking, making it impossible to confirm whether the specific combination of DINOv2 and Flame fractals is optimal.

- **Gap between simulation and observation**: All experiments are conducted on MHD simulation data rather than real astronomical observations. Simulation data are "ideal"—free of noise, instrumental effects, foreground/background contamination, and with uniform spatial resolution. Real observational data face far more severe challenges, including convolution by the telescope point spread function (PSF), detector noise, incomplete spatial sampling, and distance-dependent spatial resolution variations. The authors acknowledge this limitation in the discussion and propose "constructing datasets from the noise itself" as a potential remedy, but provide no experimental validation.

- **Lack of theoretical grounding for the choice of fractal images**: Although fractal pretraining proves effective, the paper does not explain why fractals are chosen. Is it because the self-similar structure of fractals matches the fractal properties of the interstellar medium? Or simply because fractals provide sufficient visual diversity to train a general feature extractor? If the latter, would other synthetic images (e.g., Perlin noise, random textures from StyleGAN, simple geometric compositions) yield similar results? This critical question is not explored.

- **Absolute level of prediction accuracy**: While $R^2 = 0.81$ substantially exceeds chance, it may still be insufficient for practical astronomical applications. The model accounts for only ~81% of mass variance, and the remaining ~19% of unexplained variance may be physically important (e.g., for distinguishing the contributions of different mass ranges to the IMF). Prediction becomes unreliable at the high-mass end ($> 6 \, M_\odot$)—precisely the regime most critical for IMF research, since massive stars, though rare, have an outsized influence on galactic evolution.

- **Information loss from two-dimensional projection**: Projecting three-dimensional MHD simulations onto two-dimensional maps inevitably discards substantial information. Distinct structures separated in three-dimensional space but overlapping along the line of sight may be conflated—for example, two dense cores far apart in three dimensions but overlapping in projection will have their column densities superimposed, potentially biasing mass predictions. While velocity channels provide a partial proxy for three-dimensional depth information (different line-of-sight velocities corresponding to gas at different distances), this discriminating power is very limited, especially in highly turbulent regions where line broadening confounds velocity signals from different locations. Future work could explore three-dimensional voxel data (e.g., PointNet++, 3D CNNs, or Transformer-based voxel processing architectures) to retain more spatial information, or incorporate position–position–velocity (PPV) data cubes to access richer kinematic information.

- **Potential directions for improvement**: (1) Adding more projection channels (e.g., magnetic field strength, temperature projection maps) to provide richer input information; (2) attempting lightweight fine-tuning of the DINOv2 encoder (e.g., LoRA adapters) to potentially further improve performance without overfitting; (3) replacing purely mathematical fractals with more physically motivated synthetic data (e.g., simplified hydrodynamic simulations, turbulence field generators) to reduce the domain gap between pretraining data and downstream data; (4) cross-validating across multiple MHD simulations with different physical parameter settings to test the generalizability of the approach.

## Related Work & Insights

- **vs. DINOv2 applied to natural images / galaxy images [Oquab et al., 2023; Lanusse et al.]**: The original DINOv2 paper demonstrated powerful zero-shot capabilities on natural images, and subsequent work applied it to galaxy image classification. The distinctive contribution of this paper is that no in-domain data whatsoever is used for pretraining—not even galaxy images, only purely mathematically generated fractal images. This pushes further the recognition that "pretraining data need not come from the same domain as downstream data." The paper's strength lies in validating the possibility of extreme cross-domain transfer; a limitation is that pretraining on galaxy images or interstellar medium observations might yield better performance, but no such comparison is made.

- **vs. Kataoka et al. (FractalDB) [2020] and subsequent fractal pretraining work**: Kataoka et al. pioneered the "fractal image supervised pretraining" paradigm, demonstrating that formula-generated fractals can replace ImageNet for classification pretraining. Subsequent work extended this to ViTs and self-supervised learning. This paper's contribution is to introduce this paradigm into the astrophysics domain for the first time, combining it with DINOv2 to achieve a self-supervised version. The distinction is that the paper addresses regression (mass prediction) and segmentation (PCA visualization) rather than classification, validating the generality of fractal pretraining across a broader range of task types.

- **vs. traditional astrophysical machine learning methods**: Conventional approaches typically rely on hand-crafted features (e.g., radial profiles of column density, statistical measures such as velocity structure functions) or end-to-end CNN training on limited labeled data. The former requires extensive domain expertise to design effective features and may miss important spatial patterns; the latter is prone to overfitting when data are insufficient. The proposed method entirely eliminates the need for hand-crafted feature design and labeled fine-tuning, demonstrating the unique potential of self-supervised representation learning in data-constrained scientific computing settings. This carries important demonstrative value for the broader scientific machine learning community: when labeled data are expensive, investing computational resources in self-supervised pretraining on cheap synthetic data may be more effective than investing in acquiring more labeled data. This paradigm has particularly practical prospects in simulation-driven disciplines such as astrophysics, particle physics, and climate science, where simulation costs are prohibitive but synthetic data can be generated cheaply.

- **Implications for other domains**: The methodological approach of this paper (synthetic data self-supervised pretraining → zero-shot frozen feature transfer) is highly generalizable. Its core insight is that when domain data are scarce, the quality of general visual representations matters more than the quantity of task-specific labels. In medical imaging, high-quality annotated data are equally scarce and expensive (requiring pixel-level annotation by specialists)—could pretraining on synthetic organ shapes, pathological textures, or fractal vascular patterns be effective? In materials science, could models for microstructural analysis be pretrained on mathematically generated crystal structure patterns or Voronoi polyhedra? In remote sensing, could land cover classifiers be pretrained on procedurally generated terrain textures (e.g., Perlin noise terrain)? These are highly promising directions with substantial practical value deserving deeper exploration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The paper is the first to introduce synthetic fractal self-supervised pretraining for astrophysical stellar mass inference; the cross-domain transfer idea is novel, though the core technical components (DINOv2, fractal pretraining, kNN) are all drawn from prior work.
- **Experimental Thoroughness**: ⭐⭐⭐ As a workshop short paper, the experimental setup is reasonable but lacks depth; key ablations (e.g., $k$ value, fractal count, comparisons across encoder architectures) and comparisons with other self-supervised methods are absent.
- **Writing Quality**: ⭐⭐⭐⭐ The paper is clearly structured with well-motivated background, concise and accurate method description, and professional figures and tables, though some details are insufficiently elaborated due to page constraints.
- **Value**: ⭐⭐⭐⭐ The paper provides a clean and effective methodological template for data-constrained scientific computing domains; the synthetic data pretraining idea has broad cross-domain applicability, though current validation is limited to a single task and simulation data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Exploring Structural Degradation in Dense Representations for Self-supervised Learning](exploring_structural_degradation_in_dense_representations_for_self-supervised_le.md)
- [\[ICCV 2025\] Joint Self-Supervised Video Alignment and Action Segmentation](../../ICCV2025/segmentation/joint_self-supervised_video_alignment_and_action_segmentation.md)
- [\[NeurIPS 2025\] Vision Transformers with Self-Distilled Registers](vision_transformers_with_self-distilled_registers.md)
- [\[CVPR 2026\] Follow the Saliency: Supervised Saliency for Retrieval-augmented Dense Video Captioning](../../CVPR2026/segmentation/follow_the_saliency_supervised_saliency_for_retrieval-augmented_dense_video_capt.md)
- [\[ICCV 2025\] Enhancing Transformers Through Conditioned Embedded Tokens](../../ICCV2025/segmentation/enhancing_transformers_through_conditioned_embedded_tokens.md)

</div>

<!-- RELATED:END -->
