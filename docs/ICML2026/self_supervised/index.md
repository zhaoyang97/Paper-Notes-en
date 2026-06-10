---
title: >-
  ICML2026 Self-Supervised Learning Papers · 26 Notes
description: >-
  26 ICML2026 papers in the Self-Supervised Learning area, covering Self-Supervised Learning, Alignment/RLHF, Few-/Zero-Shot Learning and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ICML2026"
  - "Self-Supervised Learning"
  - "AI paper notes"
  - "paper summaries"
  - "Alignment/RLHF"
  - "Few-/Zero-Shot Learning"
item_list:
  - u: "a_refined_generalization_analysis_for_extreme_multi-class_supervised_contrastive/"
    t: "A Refined Generalization Analysis for Extreme Multi-class Supervised Contrastive Representation Learning"
  - u: "beyond_distribution_estimation_simplex_anchored_structural_inference_towards_uni/"
    t: "Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning"
  - u: "can_local_learning_match_self-supervised_backpropagation/"
    t: "Can Local Learning Match Self-Supervised Backpropagation?"
  - u: "data_augmentation_of_contrastive_learning_is_estimating_positive-incentive_noise/"
    t: "Data Augmentation of Contrastive Learning is Estimating Positive-incentive Noise"
  - u: "flag_foundation_model_representation_with_latent_diffusion_alignment_via_graph_f/"
    t: "FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction"
  - u: "from_zero_to_hero_advancing_zero-shot_foundation_models_for_tabular_outlier_dete/"
    t: "From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection"
  - u: "how_neural_is_a_neural_foundation_model/"
    t: "How 'Neural' is a Neural Foundation Model?"
  - u: "inconsistency-aware_minimization_improving_generalization_with_unlabeled_data/"
    t: "Inconsistency-Aware Minimization: Improving Generalization with Unlabeled Data"
  - u: "infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate/"
    t: "InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation"
  - u: "learning_graph_foundation_models_on_riemannian_graph-of-graphs/"
    t: "Learning Graph Foundation Models on Riemannian Graph-of-Graphs"
  - u: "learning_to_extrapolate_to_new_tasks_a_relational_approach_to_task_extrapolation/"
    t: "Learning to Extrapolate to New Tasks: A Relational Approach to Task Extrapolation"
  - u: "lec_linear_expectation_constraints_for_selection-conditioned_risk_control_in_sel/"
    t: "LEC: Linear Expectation Constraints for Selection-Conditioned Risk Control in Selective Prediction and Routing Systems"
  - u: "limix-2m_mitigating_low-rank_collapse_and_attention_bottlenecks_in_tabular_found/"
    t: "LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models"
  - u: "mitigating_label_shift_in_tabular_in-context_learning_via_test-time_posterior_ad/"
    t: "Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment"
  - u: "nitp_next_implicit_token_prediction_for_llm_pre-training/"
    t: "NITP: Next Implicit Token Prediction for LLM Pre-training"
  - u: "numleak_public_numeric_benchmarks_as_latent_labels_in_foundation_models/"
    t: "NumLeak: Public Numeric Benchmarks as Latent Labels in Foundation Models"
  - u: "partco_part-level_correspondence_priors_enhance_category_discovery/"
    t: "PartCo: Part-Level Correspondence Priors Enhance Category Discovery"
  - u: "provable_accuracy_collapse_in_embedding-based_representations_under_dimensionali/"
    t: "Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch"
  - u: "scaling_continual_learning_to_300_tasks_with_bi-level_routing_mixture-of-experts/"
    t: "Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts"
  - u: "statistical_consistency_and_generalization_of_contrastive_representation_learnin/"
    t: "Statistical Consistency and Generalization of Contrastive Representation Learning"
  - u: "text-conditional_jepa_for_learning_semantically_rich_visual_representations/"
    t: "Text-Conditional JEPA for Learning Semantically Rich Visual Representations"
  - u: "the_geometric_mechanics_of_contrastive_representation_learning_alignment_potenti/"
    t: "The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-modal Divergence"
  - u: "the_geometry_of_projection_heads_conditioning_invariance_and_collapse/"
    t: "The Geometry of Projection Heads: Conditioning, Invariance and Collapse"
  - u: "tracer_persistent_regularization_for_robust_multimodal_finetuning/"
    t: "TRACER: Robust Multimodal Fine-Tuning Proven by WMA Teacher + Geometric Decomposition"
  - u: "understanding_self-supervised_learning_via_latent_distribution_matching/"
    t: "Understanding Self-Supervised Learning via Latent Distribution Matching"
  - u: "when_softmax_fails_at_the_top_extreme_value_corrections_for_infonce/"
    t: "When Softmax Fails at the Top: Extreme Value Corrections for InfoNCE"
item_total: 26
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# 🔄 Self-Supervised Learning

**🧪 ICML2026** · **26** paper notes

📌 **Same area in other venues:** [💬 ACL2026 (1)](../../ACL2026/self_supervised/index.md) · [📷 CVPR2026 (30)](../../CVPR2026/self_supervised/index.md) · [🔬 ICLR2026 (14)](../../ICLR2026/self_supervised/index.md) · [🤖 AAAI2026 (13)](../../AAAI2026/self_supervised/index.md) · [🧠 NeurIPS2025 (33)](../../NeurIPS2025/self_supervised/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/self_supervised/index.md)

🔥 **Top topics:** Self-Supervised Learning ×3 · Alignment/RLHF ×2 · Few-/Zero-Shot Learning ×2

**[A Refined Generalization Analysis for Extreme Multi-class Supervised Contrastive Representation Learning](a_refined_generalization_analysis_for_extreme_multi-class_supervised_contrastive.md)**

:   This paper improves the sample complexity upper bound for supervised contrastive learning (where tuples are constructed from a finite labeled data pool). By utilizing two different U-statistic estimators, it achieves a breakthrough in extreme multi-class scenarios, moving from bounds dependent on the minimum class probability to bounds dependent only on the number of classes or the sample scale.

**[Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning](beyond_distribution_estimation_simplex_anchored_structural_inference_towards_uni.md)**

:   This paper proposes SAGE, which replaces "estimating unlabeled data distribution" with "structural inference in the representation space." By utilizing a trio of simplex ETF geometric anchors, high-order graph propagation, and distribution-agnostic reliability weighting, it achieves an average accuracy improvement of 8.52% under the UniSSL setting characterized by extreme label scarcity and arbitrary unlabeled distributions.

**[Can Local Learning Match Self-Supervised Backpropagation?](can_local_learning_match_self-supervised_backpropagation.md)**

:   Ours theoretically proves that local self-supervised learning (local-SSL) can precisely achieve the gradient updates of global backpropagation self-supervised learning (BP-SSL) in deep linear networks. Based on this, the CLAPP++ algorithm is proposed (incorporating 2D spatial dependency and direct feedback), reaching performance parity with global BP-SSL on CIFAR-10/STL-10/Tiny ImageNet and setting a new Prev. SOTA for local-SSL.

**[Data Augmentation of Contrastive Learning is Estimating Positive-incentive Noise](data_augmentation_of_contrastive_learning_is_estimating_positive-incentive_noise.md)**

:   The authors demonstrate that "predefined data augmentation" (e.g., rotation, cropping, flipping) in contrastive learning is equivalent to the point estimation of Positive-incentive Noise ($\pi$-noise). They upgrade $\pi$-noise from a "point estimation" to a learnable distribution by training a $\pi$-noise generator (PiNDA) that applies learnable noise to original images. PiNDA consistently improves performance for SimCLR, BYOL, SimSiam, MoCo, and DINO on vision tasks and is naturally adaptable to non-visual data lacking manual augmentation, such as HAR, Reuters, and Epsilon.

**[FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction](flag_foundation_model_representation_with_latent_diffusion_alignment_via_graph_f.md)**

:   FLAG reformulates the prediction of spatial gene expression from H&E pathology images as a structured distribution generation problem. It utilizes a fixed spatial graph encoder to compress tissue topology into condition vectors and employs a DiT (Diffusion Transformer) for denoising in the gene dimension. By injecting gene-gene regulatory priors through representation alignment with intermediate layers of a Gene Foundation Model (GFM), FLAG elevates Gene Structural Correlation (GSC) and Spatial Structural Correlation (SSC) to new heights while maintaining competitive PCC/MSE performance.

**[From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection](from_zero_to_hero_advancing_zero-shot_foundation_models_for_tabular_outlier_dete.md)**

:   This paper proposes OutFormer, a tabular Prior-Fitted Network (PFN) pretrained on a mixture of three synthetic priors (GMM/SCM/Copula) and stabilized via a multi-armed bandit self-evolving curriculum. It achieves zero-shot tabular outlier detection by consuming training data as in-context information and providing labels in a single forward pass, reaching SOTA rankings on ADBench and two new 1500+ dataset benchmarks with inference latency close to shallow models.

**[How 'Neural' is a Neural Foundation Model?](how_neural_is_a_neural_foundation_model.md)**

:   The authors treat a "SOTA foundation model for mouse visual cortex (FNN)" as a physiological experimental subject. By analyzing its encoder, recurrent, and readout modules using a toolkit consisting of decoding manifolds, encoding manifolds, and decoding trajectories, they find that FNN's fitting accuracy is primarily sustained by a set of homogeneous feature maps in the readout, while only the recurrent module is truly "brain-like." Using a newly proposed "tubularity" metric, they quantitatively show that "early encoding layers lack biological-grade temporal structures," providing explicit recommendations for future neural foundation models to "add recurrence early and reduce feature dimensions in the readout."

**[Inconsistency-Aware Minimization: Improving Generalization with Unlabeled Data](inconsistency-aware_minimization_improving_generalization_with_unlabeled_data.md)**

:   This paper proposes "Local Inconsistency" $S_\rho(\theta)$—the worst-case KL divergence within a parameter ball—which can be computed using only unlabeled data. By employing it as a training regularizer, the resulting IAM optimizer matches or exceeds SAM/ASAM in supervised tasks and provides additional gains in semi-supervised (FixMatch) and self-supervised (SimCLR) scenarios by leveraging unlabeled batches.

**[InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)**

:   InfoAtlas transforms mutual information (MI) estimation from an optimization problem requiring a per-dataset critic network into a "single forward pass" problem using a hypernetwork pre-trained on large-scale synthetic data. It achieves accuracy comparable to neural estimators like MINE/MINDE while providing a 100× speedup.

**[Learning Graph Foundation Models on Riemannian Graph-of-Graphs](learning_graph_foundation_models_on_riemannian_graph-of-graphs.md)**

:   R-GFM treats subgraphs of "different hop counts" as nodes in a higher-level Graph-of-Graphs (GoG). A dynamic MoE router distributes each GoG to Riemannian manifolds (Hyperbolic / Euclidean / Spherical) that best match its curvature. This simultaneously addresses two inherent limitations of existing graph foundation models—fixed receptive fields and single Euclidean embeddings—yielding relative improvements of up to 49% in downstream tasks.

**[Learning to Extrapolate to New Tasks: A Relational Approach to Task Extrapolation](learning_to_extrapolate_to_new_tasks_a_relational_approach_to_task_extrapolation.md)**

:   This paper proposes the Relational Task Extrapolator (RTE), which reinterprets "new tasks outside the training support" as a composition problem of "known anchor tasks + seen inter-task transformations." It trains a relational operator $\Psi$ to assemble anchor-transformation pairs at test time to predict outputs for unknown tasks.

**[LEC: Linear Expectation Constraints for Selection-Conditioned Risk Control in Selective Prediction and Routing Systems](lec_linear_expectation_constraints_for_selection-conditioned_risk_control_in_sel.md)**

:   Addressing the long-standing problem in LLM selective prediction where "UCB risk bounds are too conservative and yield few usable thresholds," the authors reformulate the objective "post-selection error rate $\le \alpha$" into a **linear expectation constraint** involving two 0-1 indicator functions for selection and error. This derivation leads to a finite-sample sufficient condition (Eq. 5) that depends only on the calibration set. It maintains rigorous finite-sample guarantees while being significantly tighter than UCB. The framework naturally extends to two-model routing systems by jointly calibrating two thresholds, showing universal gains in power on CommonsenseQA / TriviaQA / ScienceQA / MM-Vet v2, and accepting 9.5% more samples than Clopper-Pearson UCB on TriviaQA.

**[LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models](limix-2m_mitigating_low-rank_collapse_and_attention_bottlenecks_in_tabular_found.md)**

:   Addressing two critical pathologies in tabular foundation models like TabPFN-v2—severe low-rank collapse in shallow layers and negligible contribution of sample-attention in the final layer to prediction signals—the authors propose using Radial Basis Functions (RaBEL) to expand each scalar into a set of local responses, unlocking "value-direction" degrees of freedom. Furthermore, they reorder bidirectional attention blocks from F→S→N to S→N→F to ensure all attention paths flow into the readout. With only 2M parameters, this approach consistently outperforms the 7M TabPFN-v2 and 27M TabICL on major tabular benchmarks.

**[Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment](mitigating_label_shift_in_tabular_in-context_learning_via_test-time_posterior_ad.md)**

:   The authors propose posterior correction for "tabular foundation models" like TabPFN, which feed training sets directly into attention mechanisms as in-context data. Identifying a severe over-fitting to the training set's majority class, they introduce DistPFN: a posterior reweighting via $\tilde{p}(y) \propto \hat{p}(y)^2 / p_{train}(y)$. Across 253 OpenML datasets, this method improves TabPFN-v2 accuracy from 72.7% to 76.9% under strong label shift ($\beta=5$) without retraining, test prior estimation, or architectural modifications.

**[NITP: Next Implicit Token Prediction for LLM Pre-training](nitp_next_implicit_token_prediction_for_llm_pre-training.md)**

:   NITP provides continuous representation space supervision for the final hidden state by using **shallow representations as implicit targets**. This supplements standard NTP to prevent hidden representation degeneration into low-dimensional anisotropic configurations. It achieves a 5.7% MMLU-Pro improvement on a 9B MoE model and a general 4-6% increase in reasoning tasks, with only ~2% extra computational overhead.

**[NumLeak: Public Numeric Benchmarks as Latent Labels in Foundation Models](numleak_public_numeric_benchmarks_as_latent_labels_in_foundation_models.md)**

:   NumLeak detects and quantifies the degree of memorization of public numerical benchmarks (financial factors, macroeconomic data, climate data) in foundation models through a **four-layer diagnostic protocol**. It reveals how such contamination leaks into downstream financial signals and evaluates risk mitigation via system prompt defenses; Opus 4.7 achieves a within-25 bps accuracy of 0.60 and Pearson $r = 0.99$ on the Mkt-RF factor.

**[PartCo: Part-Level Correspondence Priors Enhance Category Discovery](partco_part-level_correspondence_priors_enhance_category_discovery.md)**

:   PartCo introduces a **plug-and-play** framework to enhance Generalized Category Discovery by explicitly leveraging **part-level feature correspondences** inherent in Vision Transformer patch tokens—improving baselines such as SimGCD, SPTNet, and FlipClass by 2-10% across multiple benchmarks including CUB, Stanford-Cars, and ImageNet-100.

**[Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch](provable_accuracy_collapse_in_embedding-based_representations_under_dimensionali.md)**

:   The authors prove that in typical triplet tasks within contrastive learning, as long as the embedding dimension $d$ is less than a certain constant multiple of the true dimension $D$, the accuracy "collapses" to the 50% baseline of a 1D random embedding regardless of the optimizer used. Furthermore, from an algorithmic perspective, this phenomenon is inapproximable in polynomial time under the Unique Games Conjecture.

**[Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts](scaling_continual_learning_to_300_tasks_with_bi-level_routing_mixture-of-experts.md)**

:   The authors propose CaRE: a **Bi-Level Routing Mixture-of-Experts (BR-MoE)** module inserted into each block of a ViT. It first selects Top-M relevant task routers using entropy via "class perceptors," then activates Top-K task experts through these routers, overlaid with a shared EMA expert. This allows the model to retain old knowledge while absorbing new classes even as the task sequence extends to 300+, filling the gap in "long-sequence CIL" research (and introducing the OmniBenchmark-1K dataset with 1000 classes).

**[Statistical Consistency and Generalization of Contrastive Representation Learning](statistical_consistency_and_generalization_of_contrastive_representation_learnin.md)**

:   This paper establishes the Fisher/statistical consistency for Contrastive Representation Learning (CRL), demonstrating that minimizing upstream contrastive loss is equivalent to optimizing downstream AUC-type retrieval performance. It provides refined generalization bounds of $O(1/m+1/\sqrt n)$ for supervised and $O(1/\sqrt m+1/\sqrt n)$ for self-supervised settings, providing the first theoretical explanation for the phenomenon where CLIP and SimCLR consistently benefit from utilizing tens of thousands of negative samples.

**[Text-Conditional JEPA for Learning Semantically Rich Visual Representations](text-conditional_jepa_for_learning_semantically_rich_visual_representations.md)**

:   This paper proposes TC-JEPA, which conditions the I-JEPA masked feature predictor additionally on image captions. By applying multi-layer sparse cross-attention, patch representations become predictable under textual "prompts," enabling the learning of semantically richer and dense prediction-friendly visual representations without contrastive loss.

**[The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-modal Divergence](the_geometric_mechanics_of_contrastive_representation_learning_alignment_potenti.md)**

:   This paper elevates the InfoNCE loss to a deterministic "population energy" over representation distributions using a measure-theoretic framework, proving that the unimodal case is convex and converges to a unique Gibbs equilibrium, while the symmetric multimodal case exhibits persistent negative symmetric KL coupling, which geometrically and inevitably induces a modality gap.

**[The Geometry of Projection Heads: Conditioning, Invariance and Collapse](the_geometry_of_projection_heads_conditioning_invariance_and_collapse.md)**

:   This paper analyzes projection heads in self-supervised learning (SSL) as trainable metric tensors from a Riemannian geometry perspective. It demonstrates that their role is to dynamically whiten the optimization landscape, escape collapse saddles using negative curvature from smooth activations, and induce metric singularities along data augmentation directions—three mechanisms that together explain the long-standing mystery of "required during training, discarded during inference."

**[TRACER: Robust Multimodal Fine-Tuning Proven by WMA Teacher + Geometric Decomposition](tracer_persistent_regularization_for_robust_multimodal_finetuning.md)**

:   TRACER utilizes closed-form theoretical analysis to geometrically decompose contrastive fine-tuning into a "task subspace" and "orthogonal preservation." It proves that EMA teachers collapse and lose regularization power, and proposes the Weighted Moving Average (WMA) teacher to maintain finite-horizon constraint power with bias-free convergence to the task subspace. On CLIP ViT-B/16, the average ImageNet distribution shift performance improves to 64.07% vs. CaRot's 62.54%.

**[Understanding Self-Supervised Learning via Latent Distribution Matching](understanding_self-supervised_learning_via_latent_distribution_matching.md)**

:   The authors unify contrastive / non-contrastive / predictive SSL as "Latent Distribution Matching (LDM)": maximizing the log-probability of samples under a hypothesized latent model (alignment) + maximizing latent entropy (uniformity). Based on this, they derive nonlinear identifiable predictive SSL equipped with a Kalman predictor.

**[When Softmax Fails at the Top: Extreme Value Corrections for InfoNCE](when_softmax_fails_at_the_top_extreme_value_corrections_for_infonce.md)**

:   This paper interprets InfoNCE as a top-1 selection likelihood, pointing out that standard softmax implicitly assumes a Gumbel tail distribution, whereas high-similarity hard negatives of normalized embeddings more frequently exhibit Weibull behavior with finite endpoints. Consequently, the authors propose WEINCE, a parameter-free method that uses in-batch tail statistics to adaptively mix softmax logits and endpoint shortfall logits, stably improving the quality of self-supervised representations.
