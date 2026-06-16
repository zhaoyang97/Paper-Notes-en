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

📌 **Same area in other venues:** [📷 CVPR2026 (174)](../../CVPR2026/medical_imaging/index.md) · [🔬 ICLR2026 (22)](../../ICLR2026/medical_imaging/index.md) · [🤖 AAAI2026 (75)](../../AAAI2026/medical_imaging/index.md) · [🧠 NeurIPS2025 (74)](../../NeurIPS2025/medical_imaging/index.md) · [📹 ICCV2025 (31)](../../ICCV2025/medical_imaging/index.md)

🔥 **Top topics:** Medical Imaging ×12 · Reasoning ×4 · Segmentation ×3 · Multimodal/VLM ×3 · Alignment/RLHF ×3

**[Are We Overconfident in Models and Results for Semi-Supervised 3D Medical Image Segmentation?](are_we_overconfident_in_models_and_results_for_semi-supervised_3d_medical_image_.md)**

:   This paper highlights that semi-supervised 3D medical image segmentation suffers from overconfidence in both model pseudo-labels and evaluation protocols. It proposes TCSeg, which utilizes dual-axis reliability (confidence-uncertainty) and tri-space calibration (probability, feature, and image) to suppress confirmation bias, while advocating for a rigorous evaluation protocol reporting both best and last checkpoints across multiple random seeds.

**[Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions](auditing_sybil_explaining_deep_lung_cancer_risk_prediction_through_generative_in.md)**

:   This paper proposes S(H)NAP—a generative interventional framework based on 3D diffusion bridges for "removal + insertion." It decomposes the decisions of Sybil, a leading lung cancer risk prediction model, into a Linear + Second-order Interaction Model (LMPI) consisting of "nodule main effects + pairwise interactions + background." For the first time, it audits the model's dependence on in-hospital artifacts (e.g., ECG electrodes, metal buttons) and identifies a severe "radial insensitivity" failure mode for peripheral nodules through causal rather than correlative methods.

**[CASCADE Conformal Prediction: Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support](cascade_conformal_prediction_uncertainty-adaptive_prediction_intervals_for_two-s.md)**

:   The CASCADE framework is proposed to propagate epistemic uncertainty from a first-stage classifier (quantified via Venn-Abers predictors) into second-stage regression prediction intervals. This narrows intervals for high-confidence patients by 38.9% while automatically expanding safety buffers for uncertain cases, achieving adaptive coverage guarantees.

**[DGNO: Discontinuous Galerkin Neural Operator for Pathology Defocus Deblurring](discontinuous_galerkin_neural_operator_for_pathology_defocus_deblurring.md)**

:   DGNO reformulates defocus deblurring of pathological microscopy images as an inverse problem of "spatially varying integral operators." Using a Discontinuous Galerkin (DG) style, it decomposes the global kernel into element-local integral operators and interface numerical fluxes. This preserves the physical interpretability of neural operators while effectively handling the inherently local discontinuous blur in pathological images, surpassing SOTAs such as NAFNet, Restormer, and MambaIRv2 on datasets like BBBC006w1.

**[DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning](dp-kfc_data-free_preconditioning_for_privacy-preserving_deep_learning.md)**

:   This paper proposes DP-KFC: based on the observation that "the scaling of the Fisher matrix is determined by the architecture, and the correlation structure can be approximated by modality-level spectral statistics," it reconstructs KFAC preconditioners by probing the network with structured synthetic noise (1/f^\alpha pink noise for images, Zipf sampling for text). This approach neither consumes the privacy budget nor introduces distribution shifts, consistently outperforming DP-SGD and public data preconditioning methods under strong privacy ($\varepsilon \le 3$).

**[EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts](eeg-based_multimodal_learning_via_hyperbolic_mixture-of-curvature_experts.md)**

:   EEG-MoCE assigns a Lorentz manifold expert with **learnable curvature** to each modality in EEG-based multimodal learning (emotion/sleep/cognition). It utilizes curvature-aware attention, where "higher curvature signifies richer hierarchical structure and thus higher weight in fusion," to perform cross-modal integration. This approach achieves cross-subject accuracy gains of +14.14%, +3.34%, and +7.98% on the EAV, ISRUC, and Cognitive datasets, respectively.

**[Evidential Reasoning Advances Interpretable Real-World Disease Screening](evidential_reasoning_advances_interpretable_real-world_disease_screening.md)**

:   EviScreen utilizes "Normal + Pathological" dual knowledge banks for region-level evidence retrieval, followed by cross-attention and self-attention to perform evidential reasoning between the current case and retrieved evidence. This approach provides both **retrospective interpretability** (identifying which historical cases support the current judgment) and **localization interpretability** (abnormality maps from contrastive retrieval), achieving SOTA specificity at high recall levels across four real-world external test sets.

**[Factored Classifier-Free Guidance](factored_classifier-free_guidance.md)**

:   This paper identifies the "attribute amplification" failure mode of CFG in diffusion model counterfactual generation—where a single global $\omega$ amplifies attributes that should remain unchanged. The authors propose FCFG: grouping attributes according to a causal graph and assigning independent guidance weights to each group. This significantly reduces off-target attribute drift and improves counterfactual reversibility on CelebA-HQ, EMBED, and MIMIC-CXR.

**[Federated Distillation for Whole Slide Image via Gaussian-Mixture Feature Alignment and Curriculum Integration](federated_distillation_for_whole_slide_image_via_gaussian-mixture_feature_alignm.md)**

:   This paper proposes FedHD: In heterogeneous federated pathology scenarios, it employs Gaussian-mixture feature alignment for "one-to-one" WSI feature-level distillation. It then progressively injects cross-institutional synthetic features into local training via curriculum learning. This allows institutions to collaborate without sharing raw data or exchanging model parameters. Compatible with heterogeneous MIL architectures and feature extractors, it comprehensively outperforms existing federated and distillation baselines on TCGA-IDH, CAMELYON16, and CAMELYON17.

**[Foundation VAEs for 3D CT Reconstruction, Augmentation, and Generation](foundation_vaes_for_3d_ct_reconstruction_augmentation_and_generation.md)**

:   This paper demonstrates a counter-intuitive yet practical discovery: Foundation VAEs pretrained on natural images/videos serve as a unified interface for CT reconstruction, augmentation, and generation without any medical fine-tuning. Reconstruction acts as denoising without shifting boundaries; thus, reconstructed images can serve as denoising augmentation (pancreatic / lung tumor NSD +3.9%), while the latent space supports conditional CT diffusion generation (FVD −3.9%, CT-CLIP +36.2%, multi-disease fidelity AUC +2.76%).

**[OT-Bridge Editor: Geometrically Constrained Stenosis Editing in Coronary Angiography via Entropic Optimal Transport](geometrically_constrained_stenosis_editing_in_coronary_angiography_via_entropic_.md)**

:   OT-Bridge Editor reformulates "editing a vascular stenosis on coronary angiography" as a constrained entropic OT problem in a vessel-structure composite domain. By employing Schrödinger Bridge with path-level geometric projection supervision, it achieves pixel-level shape/position controllable synthetic angiography, resulting in a 27.8% relative gain in downstream stenosis detection mAP@0.5 on the ARCADE public dataset.

**[Learning Multi-Scale Hypergraph for High-Order Brain Connectivity Analysis](learning_multi-scale_hypergraph_for_high-order_brain_connectivity_analysis.md)**

:   MuHL decomposes brain ROI features into multi-resolution representations using graph wavelets with learnable scales, then dynamically generates soft hyperedges via a "node embedding × shared projection matrix" mechanism. This approach achieves 93.2% Acc on ADNI and 76.8% Acc on PPMI for multi-stage AD/PD classification, while identifying interpretable key ROIs and hyperedges.

**[Marrying Generative Model of Healthcare Events with Digital Twin of Social Determinants of Health for Disease Reasoning](marrying_generative_model_of_healthcare_events_with_digital_twin_of_social_deter.md)**

:   This paper proposes DiffDT: a conditional Latent Diffusion framework connecting electronic health records (ICD-coded event sequences) with multi-organ biomarker digital twins (imaging-derived tabular features of brain/heart/liver/kidney and brain functional connectivity SPD matrices). The key innovation is an SPD-VQVAE based on Cholesky decomposition that reduces $\mathcal{O}(N^3)$ SPD manifold diffusion to a manifold-preserving and efficient latent space. An AR model then utilizes the mediating path of "generating digital twins → predicting the next ICD" to complete multi-pathway disease reasoning. On UKB, the next-event prediction AUC for 1944 disease classes reached 0.91, setting a new SOTA.

**[MedCRP-CL: Continual Medical Image Segmentation via Bayesian Nonparametric Semantic Modality Discovery](medcrp-cl_continual_medical_image_segmentation_via_bayesian_nonparametric_semant.md)**

:   The authors utilize the Chinese Restaurant Process (CRP) for online Bayesian nonparametric clustering of clinical text prompts to automatically discover "semantic modalities." Each discovered modality is assigned an independent LoRA adapter combined with intra-modality EWC. This approach pushes the Dice coefficient to 73.3% while reducing the forgetting rate to 4.1% across 16 medical segmentation tasks, utilizing only 1/6 of the parameters required by MoE baselines.

**[MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training](meg-xl_data-efficient_brain-to-text_via_long-context_pre-training.md)**

:   MEG-XL utilizes a 2.5-minute (191k tokens) MEG context for masked token pre-training (5–300$\times$ longer than previous methods), then fine-tunes on a 50-word Brain-to-Text task. With only 1 hour of data, it achieves the decoding accuracy of SOTA supervised methods using 50 hours of data and significantly outperforms all existing brain foundation models.

**[PaCX-MAE: Physiology-Augmented Chest X-Ray Masked Autoencoder](pacx-mae_physiology-augmented_chest_x-ray_masked_autoencoder.md)**

:   PaCX-MAE builds upon an MAE pre-trained chest X-ray ViT, using LoRA fine-tuning to treat ECG and laboratory test encoders as frozen teachers. Through dual distillation involving InfoNCE contrastive loss and cosine regression, "invisible physiological context" is injected into the image-only encoder. During inference, the model requires only chest X-rays to outperform the same-architecture MAE baseline across 9 downstream benchmarks, with significant gains on physiology-dependent tasks (MedMod +2.7 AUROC, VinDr +6.5 F1).

**[Plug-and-Play Diffusion Meets ADMM: Dual-Variable Coupling for Robust Medical Image Reconstruction](plug-and-play_diffusion_meets_admm_dual-variable_coupling_for_robust_medical_ima.md)**

:   This paper reintegrates the dual variables of ADMM into the PnP diffusion prior loop, utilizing "duality" to provide integral feedback that eliminates steady-state bias. A frequency-domain Spectral Homogenization module is employed to whiten structured dual residuals into pseudo-AWGN, preventing Out-of-Distribution (OOD) hallucinations in the diffusion denoiser. It achieves SOTA fidelity on sparse-view/limited-angle CT and accelerated MRI with approximately a $3\times$ inference speedup.

**[Scaling Vision Transformers for Functional MRI with Flat Maps](scaling_vision_transformers_for_functional_mri_with_flat_maps.md)**

:   By projecting 3D fMRI volumes into 2D videos via "cortical flat maps" and feeding them into a standard spacetime MAE-ViT, the authors develop CortexMAE, trained on 2.1K hours of HCP data. It significantly outperforms SOTA in cognitive state decoding and validates flat maps as the "goldilocks zone" between voxel-wise (volume) and region-average (parcellation) representations. Simultaneously, the first open-source fMRI foundation model benchmark, Brainmarks, is released, providing the first systemic scaling laws for fMRI and an honest null result showing that foundation models still struggle to beat simple functional connectivity baselines in individual trait prediction.

**[Seizure-Semiology-Suite (S³): A Clinically Multimodal Dataset, Benchmark, and Models for Seizure Semiology Understanding](seizure-semiology-suite_s3_a_clinically_multimodal_dataset_benchmark_and_models_.md)**

:   This paper constructs the first large-scale expert-annotated seizure video dataset, S³ (438 clips, 35,000+ dense labels, 20 ILAE semiology features). It introduces a seven-level hierarchical task benchmark and a clinically aligned Seizure-RQI report quality metric. The study systematically exposes 11 open-source MLLMs' failure modes in temporal localization, spatial lateralization, and clinical faithfulness, and achieves an ES vs. NES classification F1 of 0.96 through domain fine-tuning and a two-stage neuro-symbolic framework.

**[SEMIR: Semantic Minor-Induced Representation Learning on Graphs for Visual Segmentation](semir_semantic_minor-induced_representation_learning_on_graphs_for_visual_segmen.md)**

:   SEMIR treats the voxel grid as a base graph $G$ and compresses it into a "boundary-aligned" graph minor $H$ via parameterized edge contraction, node deletion, and edge deletion (reducing node count from $\sim10^7$ to $\sim10^3$). It utilizes 5–20 few-shot samples for black-box optimization of $\Theta$ to maximize boundary Dice, performs supernode classification with a GNN on the minor, and finally returns to the original grid via a bijective exact lifting. It consistently outperforms nnU-Net on minority class Dice across BraTS, KiTS, and LiTS tumor segmentation tasks, requiring only a 16GB T4 GPU.

**[SynerMedGen: Synergizing Medical Multimodal Understanding with Generation via Task Alignment](synermedgen_synergizing_medical_multimodal_understanding_with_generation_via_tas.md)**

:   SynerMedGen proposes the "generation-aligned understanding" principle—deriving understanding tasks directly from the same paired synthetic data (via CTS, MI, and TIA tasks). By employing a two-stage training process, the understanding branch first learns representations beneficial for synthesis before transitioning to the latent flow matching generation branch. This approach outperforms both specialized synthesis models and existing unified MLLMs across 22 medical synthesis tasks.

**[CAME-Grad: The Double Dilemma in Multi-Task Radiology Report Generation — A Gradient Dynamics Analysis and Solution](the_double_dilemma_in_multi-task_radiology_report_generation_a_gradient_dynamics.md)**

:   This paper utilizes an SDE framework to analyze the dual nature of gradient conflicts between "report generation vs. clinical constraints" in Radiology Report Generation (RRG) — drift term deviation from Pareto optimality and diffusion term decay failing to escape local optima. The authors propose the CAME-Grad optimizer (Direction Rectification + Energy Injection + Adaptive Fusion) as a plug-and-play alternative to linear scaling, achieving average gains of +2.3% and +1.9% in clinical efficacy across 8 RRG methods on MIMIC-CXR and IU X-Ray.

**[PathCTM: Thinking in Scales — Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning](thinking_in_scales_accelerating_gigapixel_pathology_image_analysis_via_adaptive_.md)**

:   PathCTM reformulates Whole Slide Image (WSI) analysis from "exhaustive high-magnification patching" to "continuous multi-scale reasoning from low-magnification global to high-magnification local". Based on Continuous Thought Machines, it introduces the thinking-in-scales paradigm + attention-guided region pruning + confidence-aware early stopping, reducing patch counts by 95.95% and inference time by 95.62% while maintaining or even improving AUC.

**[Turning Drift into Constraint: Robust Reasoning Alignment in Non-Stationary Multi-Stream Environments](turning_drift_into_constraint_robust_reasoning_alignment_in_non-stationary_envir.md)**

:   This paper reinterprets reasoning "drift" among multiple MLLMs as negative constraints in DPO. By employing a Plackett-Luce preference loss to simultaneously suppress divergent trajectories from $N$ source models, a 7B student model outperforms all source teachers in chest X-ray classification and report generation tasks using only 10% of MIMIC-CXR data without requiring ground-truth reports.
