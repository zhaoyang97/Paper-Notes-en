---
title: >-
  ICML2026 Medical Imaging Papers · 24 Notes
description: >-
  24 ICML2026 papers in the Medical Imaging area, covering Medical Imaging, Reasoning, Segmentation, Multimodal/VLM, Alignment/RLHF, Adversarial Robustness and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ICML2026"
  - "Medical Imaging"
  - "AI paper notes"
  - "paper summaries"
  - "Reasoning"
  - "Segmentation"
  - "Multimodal/VLM"
  - "Alignment/RLHF"
  - "Adversarial Robustness"
item_list:
  - u: "are_we_overconfident_in_models_and_results_for_semi-supervised_3d_medical_image_/"
    t: "Are We Overconfident in Models and Results for Semi-Supervised 3D Medical Image Segmentation?"
  - u: "auditing_sybil_explaining_deep_lung_cancer_risk_prediction_through_generative_in/"
    t: "Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions"
  - u: "cascade_conformal_prediction_uncertainty-adaptive_prediction_intervals_for_two-s/"
    t: "CASCADE Conformal Prediction: Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support"
  - u: "discontinuous_galerkin_neural_operator_for_pathology_defocus_deblurring/"
    t: "DGNO: Discontinuous Galerkin Neural Operator for Pathology Defocus Deblurring"
  - u: "dp-kfc_data-free_preconditioning_for_privacy-preserving_deep_learning/"
    t: "DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning"
  - u: "eeg-based_multimodal_learning_via_hyperbolic_mixture-of-curvature_experts/"
    t: "EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts"
  - u: "evidential_reasoning_advances_interpretable_real-world_disease_screening/"
    t: "Evidential Reasoning Advances Interpretable Real-World Disease Screening"
  - u: "factored_classifier-free_guidance/"
    t: "Factored Classifier-Free Guidance"
  - u: "federated_distillation_for_whole_slide_image_via_gaussian-mixture_feature_alignm/"
    t: "Federated Distillation for Whole Slide Image via Gaussian-Mixture Feature Alignment and Curriculum Integration"
  - u: "foundation_vaes_for_3d_ct_reconstruction_augmentation_and_generation/"
    t: "Foundation VAEs for 3D CT Reconstruction, Augmentation, and Generation"
  - u: "geometrically_constrained_stenosis_editing_in_coronary_angiography_via_entropic_/"
    t: "OT-Bridge Editor: Geometrically Constrained Stenosis Editing in Coronary Angiography via Entropic Optimal Transport"
  - u: "learning_multi-scale_hypergraph_for_high-order_brain_connectivity_analysis/"
    t: "Learning Multi-Scale Hypergraph for High-Order Brain Connectivity Analysis"
  - u: "marrying_generative_model_of_healthcare_events_with_digital_twin_of_social_deter/"
    t: "Marrying Generative Model of Healthcare Events with Digital Twin of Social Determinants of Health for Disease Reasoning"
  - u: "medcrp-cl_continual_medical_image_segmentation_via_bayesian_nonparametric_semant/"
    t: "MedCRP-CL: Continual Medical Image Segmentation via Bayesian Nonparametric Semantic Modality Discovery"
  - u: "meg-xl_data-efficient_brain-to-text_via_long-context_pre-training/"
    t: "MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training"
  - u: "pacx-mae_physiology-augmented_chest_x-ray_masked_autoencoder/"
    t: "PaCX-MAE: Physiology-Augmented Chest X-Ray Masked Autoencoder"
  - u: "plug-and-play_diffusion_meets_admm_dual-variable_coupling_for_robust_medical_ima/"
    t: "Plug-and-Play Diffusion Meets ADMM: Dual-Variable Coupling for Robust Medical Image Reconstruction"
  - u: "scaling_vision_transformers_for_functional_mri_with_flat_maps/"
    t: "Scaling Vision Transformers for Functional MRI with Flat Maps"
  - u: "seizure-semiology-suite_s3_a_clinically_multimodal_dataset_benchmark_and_models_/"
    t: "Seizure-Semiology-Suite (S³): A Clinically Multimodal Dataset, Benchmark, and Models for Seizure Semiology Understanding"
  - u: "semir_semantic_minor-induced_representation_learning_on_graphs_for_visual_segmen/"
    t: "SEMIR: Semantic Minor-Induced Representation Learning on Graphs for Visual Segmentation"
  - u: "synermedgen_synergizing_medical_multimodal_understanding_with_generation_via_tas/"
    t: "SynerMedGen: Synergizing Medical Multimodal Understanding with Generation via Task Alignment"
  - u: "the_double_dilemma_in_multi-task_radiology_report_generation_a_gradient_dynamics/"
    t: "CAME-Grad: The Double Dilemma in Multi-Task Radiology Report Generation — A Gradient Dynamics Analysis and Solution"
  - u: "thinking_in_scales_accelerating_gigapixel_pathology_image_analysis_via_adaptive_/"
    t: "PathCTM: Thinking in Scales — Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning"
  - u: "turning_drift_into_constraint_robust_reasoning_alignment_in_non-stationary_envir/"
    t: "Turning Drift into Constraint: Robust Reasoning Alignment in Non-Stationary Multi-Stream Environments"
item_total: 24
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# 🏥 Medical Imaging

**🧪 ICML2026** · **24** paper notes

📌 **Same area in other venues:** [📷 CVPR2026 (56)](../../CVPR2026/medical_imaging/index.md) · [🔬 ICLR2026 (22)](../../ICLR2026/medical_imaging/index.md) · [🤖 AAAI2026 (75)](../../AAAI2026/medical_imaging/index.md) · [🧠 NeurIPS2025 (74)](../../NeurIPS2025/medical_imaging/index.md) · [📹 ICCV2025 (31)](../../ICCV2025/medical_imaging/index.md)

🔥 **Top topics:** Medical Imaging ×12 · Reasoning ×4 · Segmentation ×3 · Multimodal/VLM ×3 · Alignment/RLHF ×3

**[Are We Overconfident in Models and Results for Semi-Supervised 3D Medical Image Segmentation?](are_we_overconfident_in_models_and_results_for_semi-supervised_3d_medical_image_.md)**

:   This paper points out that semi-supervised 3D medical image segmentation suffers from two types of problems: overconfidence in model pseudo-labels and over-optimism in evaluation protocols. It proposes TCSeg, which utilizes confidence-uncertainty dual-axis reliability and tri-space calibration (probability, feature, and image spaces) to suppress confirmation bias. It also advocates for evaluation protocols using multiple random seeds and reporting both best and last checkpoints to provide a more honest assessment of performance.

**[Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions](auditing_sybil_explaining_deep_lung_cancer_risk_prediction_through_generative_in.md)**

:   This paper proposes S(H)NAP—a generative interventional framework based on 3D diffusion bridges using "removal + insertion." It decomposes the decisions of Sybil, a state-of-the-art lung cancer risk prediction model, into an LMPI (Linear + Second-order Interaction Model) consisting of "nodule main effects + pairwise interactions + background." For the first time, it audits the model's reliance on in-hospital artifacts (e.g., ECG electrodes, metallic clothing buttons) and identifies a severe "radial insensitivity" failure mode regarding peripheral lung nodules through causal rather than correlative methods.

**[CASCADE Conformal Prediction: Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support](cascade_conformal_prediction_uncertainty-adaptive_prediction_intervals_for_two-s.md)**

:   The CASCADE framework is proposed to propagate epistemic uncertainty from a first-stage classifier (quantified via Venn-Abers predictors) into second-stage regression prediction intervals. This narrows prediction intervals for high-confidence patients by 38.9% while automatically expanding safety buffers for uncertain cases, achieving adaptive coverage guarantees.

**[DGNO: Discontinuous Galerkin Neural Operator for Pathology Defocus Deblurring](discontinuous_galerkin_neural_operator_for_pathology_defocus_deblurring.md)**

:   DGNO reformulates defocus deblurring in pathological microscopy as an inverse problem of "spatially-varying integral operators." By adopting the Discontinuous Galerkin (DG) style, it decomposes the global kernel into element-local integral operators and interface numerical fluxes. This preserves the physical interpretability of neural operators while handling inherently local discontinuous blur in pathology images. It outperforms SOTAs such as NAFNet, Restormer, and MambaIRv2 on datasets like BBBC006w1.

**[DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning](dp-kfc_data-free_preconditioning_for_privacy-preserving_deep_learning.md)**

:   This paper proposes DP-KFC: based on the observations that "the scale of the Fisher matrix is determined by the architecture and the correlation structure can be approximated by modality-level spectral statistics," structured synthetic noise ($1/f^\alpha$ pink noise for images, Zipf sampling for text) is used to probe the network and reconstruct KFAC preconditioners. This approach consumes no privacy budget and introduces no distribution shift, consistently outperforming DP-SGD and public-data preconditioning methods under strong privacy constraints ($\varepsilon\le 3$).

**[EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts](eeg-based_multimodal_learning_via_hyperbolic_mixture-of-curvature_experts.md)**

:   EEG-MoCE assigns a Lorentz manifold expert with a **learnable curvature** to each modality in EEG-based multimodal learning (emotion/sleep/cognition). It then utilizes "curvature-aware attention"—where "larger curvature $\rightarrow$ richer hierarchical structure $\rightarrow$ higher weight in fusion"—to perform cross-modal integration. The approach achieves cross-subject accuracy gains of +14.14%, +3.34%, and +7.98% on EAV, ISRUC, and Cognitive datasets, respectively.

**[Evidential Reasoning Advances Interpretable Real-World Disease Screening](evidential_reasoning_advances_interpretable_real-world_disease_screening.md)**

:   EviScreen utilizes "normal + pathological" dual knowledge banks for region-level evidence retrieval, followed by evidential reasoning between the current case and evidence using cross-attention and self-attention. It provides both **retrospective interpretability** (identifying which historical cases support the current judgment) and **localization interpretability** (anomaly maps obtained through contrastive retrieval), improving specificity at high recall across four real-world external test sets to SOTA levels.

**[Factored Classifier-Free Guidance](factored_classifier-free_guidance.md)**

:   This paper identifies an "attribute amplification" failure mode of CFG in counterfactual generation with diffusion models—using a single global $\omega$ amplifies not only the target attribute but also unintended ones. The authors propose FCFG: grouping attributes by causal graph and assigning independent guidance weights to each group, which significantly reduces non-target attribute drift and improves counterfactual reversibility on CelebA-HQ / EMBED / MIMIC-CXR.

**[Federated Distillation for Whole Slide Image via Gaussian-Mixture Feature Alignment and Curriculum Integration](federated_distillation_for_whole_slide_image_via_gaussian-mixture_feature_alignm.md)**

:   This paper proposes FedHD: In heterogeneous federated pathology scenarios, it employs Gaussian-mixture feature alignment for "one-to-one" WSI feature-level distillation, then progressively injects cross-institutional synthetic features into local training via curriculum learning. This enables collaboration without sharing raw data or exchanging model parameters and maintains compatibility with heterogeneous MIL architectures and feature extractors. FedHD comprehensively outperforms existing federated and distillation baselines on TCGA-IDH / CAMELYON16 / CAMELYON17.

**[Foundation VAEs for 3D CT Reconstruction, Augmentation, and Generation](foundation_vaes_for_3d_ct_reconstruction_augmentation_and_generation.md)**

:   This paper demonstrates a counter-intuitive yet practical finding—Foundation VAEs pretrained on natural images/videos can serve as a unified interface to simultaneously support CT reconstruction, augmentation, and generation without any medical fine-tuning. The reconstruction behaves as denoising without shifting boundaries; thus, reconstructed maps can serve as denoising augmentations (pancreatic / lung tumor NSD +3.9%), while the latent space can host CT conditional diffusion generation (FVD −3.9%, CT-CLIP +36.2%, multi-disease fidelity AUC +2.76%).

**[OT-Bridge Editor: Geometrically Constrained Stenosis Editing in Coronary Angiography via Entropic Optimal Transport](geometrically_constrained_stenosis_editing_in_coronary_angiography_via_entropic_.md)**

:   OT-Bridge Editor reformulates "editing a vessel stenosis in coronary angiography" as a constrained entropic OT problem in a vessel-structure composite domain. By integrating path-level geometric projection supervision into the Schrödinger Bridge, it achieves pixel-level shape/position controllable synthetic angiography, yielding a relative improvement of 27.8% in downstream stenosis detection mAP@0.5 on the ARCADE public dataset.

**[Learning Multi-Scale Hypergraph for High-Order Brain Connectivity Analysis](learning_multi-scale_hypergraph_for_high-order_brain_connectivity_analysis.md)**

:   MuHL decomposes brain ROI features into multi-resolution representations using graph wavelets with learnable scales and dynamically generates soft hyperedges via a "node embedding × shared projection matrix" mechanism. It achieves 93.2% accuracy on ADNI for 5nd-stage AD classification and 76.8% on PPMI for PD classification, while providing interpretable key ROIs and hyperedges.

**[Marrying Generative Model of Healthcare Events with Digital Twin of Social Determinants of Health for Disease Reasoning](marrying_generative_model_of_healthcare_events_with_digital_twin_of_social_deter.md)**

:   This paper proposes DiffDT: a conditional Latent Diffusion framework that bridges Electronic Health Records (ICD-coded event sequences) with multi-organ biomarker digital twins (imaging-derived tabular features for brain/heart/liver/kidney and brain functional connectivity SPD matrices). The key innovation is an SPD-VQVAE based on Cholesky decomposition that reduces $\mathcal{O}(N^3)$ SPD manifold diffusion to a manifold-preserving and efficient latent space. This allows an AR model to perform multi-pathway disease reasoning via the intermediary path of "generating digital twins $\rightarrow$ predicting next ICD." On the UKB dataset, the AUC for predicting the next occurrence of 1,944 disease categories reached 0.91, setting a new SOTA.

**[MedCRP-CL: Continual Medical Image Segmentation via Bayesian Nonparametric Semantic Modality Discovery](medcrp-cl_continual_medical_image_segmentation_via_bayesian_nonparametric_semant.md)**

:   The Chinese Restaurant Process (CRP) is utilized for online Bayesian nonparametric clustering of clinical text prompts to automatically discover "semantic modalities." Individual LoRA adapters are assigned to each semantic modality and combined with intra-modality EWC. Across 16 medical segmentation tasks, this approach achieves a 73.3% Dice score and a 4.1% forgetting rate, using only 1/6 of the parameters required by MoE baselines.

**[MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training](meg-xl_data-efficient_brain-to-text_via_long-context_pre-training.md)**

:   MEG-XL utilizes 2.5 minutes (191k tokens) of MEG context for mask token pre-training (5–300$\times$ longer than previous methods). When fine-tuned on a 50-word brain-to-text task, it achieves the decoding accuracy of SOTA supervised methods using only 1 hour of data compared to the typical 50 hours, significantly outperforming all existing brain foundation models.

**[PaCX-MAE: Physiology-Augmented Chest X-Ray Masked Autoencoder](pacx-mae_physiology-augmented_chest_x-ray_masked_autoencoder.md)**

:   PaCX-MAE performs LoRA fine-tuning on a Chest X-Ray (CXR) ViT pretrained with MAE, utilizing frozen encoders for ECG and laboratory tests as teachers. Through a dual distillation objective of InfoNCE contrastive learning and cosine regression, "invisible physiological context" is injected into the pure image encoder. At inference time, using only the CXR, the model outperforms the same-architecture MAE baseline across 9 downstream benchmarks, with particularly significant gains in physiology-dependent tasks (MedMod +2.7 AUROC, VinDr +6.5 F1).

**[Plug-and-Play Diffusion Meets ADMM: Dual-Variable Coupling for Robust Medical Image Reconstruction](plug-and-play_diffusion_meets_admm_dual-variable_coupling_for_robust_medical_ima.md)**

:   This paper reintroduces the dual variable of ADMM into the PnP diffusion prior loop, utilizing "duality" to provide integral feedback that eliminates steady-state bias. A frequency-domain Spectral Homogenization module is proposed to whiten structured dual residuals into pseudo-AWGN, preventing the triggering of OOD hallucinations in the diffusion denoiser. It achieves SOTA fidelity and approximately 3× inference acceleration on sparse-view/limited-angle CT and accelerated MRI.

**[Scaling Vision Transformers for Functional MRI with Flat Maps](scaling_vision_transformers_for_functional_mri_with_flat_maps.md)**

:   By projecting 3D fMRI volumes into 2D videos via "cortical flat maps" and feeding them into a standard spacetime MAE-ViT, the authors develop CortexMAE trained on 2.1K hours of HCP data. It significantly outperforms SOTA in cognitive state decoding, validating that flat maps represent the "Goldilocks zone" between voxels (volume) and region-averaging (parcellation). The study also releases Brainmarks, the first open-source fMRI foundation model benchmark, providing the first systematic scaling law for fMRI models and an honest null result showing that individual trait prediction still fails to outperform simple functional connectivity baselines.

**[Seizure-Semiology-Suite (S³): A Clinically Multimodal Dataset, Benchmark, and Models for Seizure Semiology Understanding](seizure-semiology-suite_s3_a_clinically_multimodal_dataset_benchmark_and_models_.md)**

:   This work constructs the first large-scale expert-annotated seizure video dataset, S³ (438 videos, 35,000+ dense labels, 20 ILAE semiology features). It introduces a seven-level hierarchical task benchmark and the clinically-aligned Seizure-RQI report quality metric. The study systematically exposes the failure modes of 11 open-source MLLMs in temporal localization, spatial lateralization, and clinical faithfulness, while improving the Seizure vs. Non-Epileptic Seizure (ES vs. NES) classification F1 to 0.96 through domain-specific fine-tuning and a two-stage neuro-symbolic framework.

**[SEMIR: Semantic Minor-Induced Representation Learning on Graphs for Visual Segmentation](semir_semantic_minor-induced_representation_learning_on_graphs_for_visual_segmen.md)**

:   SEMIR treats the voxel grid as a parent graph $G$, compresses it into a "boundary-aligned" graph minor $H$ (reducing node count from $\sim10^7$ to $\sim10^3$) via parameterized edge contraction, node deletion, and edge deletion. Using only 5–20 few-shot samples, it black-box optimizes $\Theta$ to maximize boundary Dice, then applies a GNN for supernode classification on the minor, and finally performs exact lifting via a bijection between the minor and the voxel grid. On the BraTS / KiTS / LiTS tumor segmentation tasks, SEMIR consistently outperforms nnU-Net on minority class Dice, requiring only a 16GB T4 GPU.

**[SynerMedGen: Synergizing Medical Multimodal Understanding with Generation via Task Alignment](synermedgen_synergizing_medical_multimodal_understanding_with_generation_via_tas.md)**

:   SynerMedGen proposes the "generation-aligned understanding" principle—deriving understanding tasks directly from the same paired synthetic data (via CTS, MI, and TIA tasks). By using a two-stage training process where the understanding branch first learns representations beneficial for synthesis, the model subsequently transfers these to a latent flow matching generation branch, outperforming both specialized synthesis models and existing unified MLLMs across 22 medical synthesis tasks.

**[CAME-Grad: The Double Dilemma in Multi-Task Radiology Report Generation — A Gradient Dynamics Analysis and Solution](the_double_dilemma_in_multi-task_radiology_report_generation_a_gradient_dynamics.md)**

:   This paper uses an SDE framework to analyze the "double dilemma" of gradient conflict between "report generation vs. clinical constraints" in Radiology Report Generation (RRG) multi-task learning—specifically, drift term deviation from Pareto optimality and diffusion term decay preventing escape from local optima. The authors propose the CAME-Grad optimizer (direction rectification + energy injection + adaptive fusion) as a linear scaling plug-and-play alternative, achieving an average clinical efficacy gain of +2.3% / +1.9% across 8 RRG methods on MIMIC-CXR / IU X-Ray datasets.

**[PathCTM: Thinking in Scales — Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning](thinking_in_scales_accelerating_gigapixel_pathology_image_analysis_via_adaptive_.md)**

:   PathCTM reframes Whole Slide Image (WSI) analysis from "exhaustive high-magnification patching" to "continuous multi-scale reasoning from low-resolution global and high-resolution local" views. Based on the Continuous Thought Machine, it introduces the thinking-in-scales paradigm, attention-guided regional pruning, and confidence-aware early stopping. This approach reduces the number of patches by 95.95% and inference time by 95.62% while simultaneously improving the AUC.

**[Turning Drift into Constraint: Robust Reasoning Alignment in Non-Stationary Multi-Stream Environments](turning_drift_into_constraint_robust_reasoning_alignment_in_non-stationary_envir.md)**

:   This work reinterprets the reasoning "drift" among multiple MLLMs as negative sample constraints in DPO, using Plackett-Luce preference loss to simultaneously suppress the divergent trajectories of N source models. As a result, a 7B student model, without ground-truth reports and using only 10% of MIMIC-CXR, surpasses all source teachers in chest X-ray classification and report generation tasks.
