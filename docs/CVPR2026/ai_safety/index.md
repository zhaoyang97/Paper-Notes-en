---
title: >-
  CVPR2026 AI Safety Papers · 24 Notes
description: >-
  24 CVPR2026 papers in the AI Safety area, covering Adversarial Robustness, Federated Learning, Multimodal/VLM, Watermarking, Robotics and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "CVPR2026"
  - "AI Safety"
  - "AI paper notes"
  - "paper summaries"
  - "Adversarial Robustness"
  - "Federated Learning"
  - "Multimodal/VLM"
  - "Watermarking"
  - "Robotics"
item_list:
  - u: "a_unified_perspective_on_adversarial_membership_manipulation_in_vision_models/"
    t: "A Unified Perspective on Adversarial Membership Manipulation in Vision Models"
  - u: "all_vehicles_can_lie_efficient_adversarial_defense_in_fully_untrusted-vehicle_co/"
    t: "All Vehicles Can Lie: Efficient Adversarial Defense in Fully Untrusted-Vehicle Collaborative Perception via Pseudo-Random Bayesian Inference"
  - u: "clustermark_towards_robust_watermarking_for_autoregressive_image_generators_with/"
    t: "ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering"
  - u: "computation_and_communication_efficient_federated_unlearning_via_on-server_gradi/"
    t: "Computation and Communication Efficient Federated Unlearning via On-server Gradient Conflict Mitigation and Expression"
  - u: "decoupling_defense_strategies_for_robust_image_watermarking/"
    t: "AdvMark: Decoupling Defense Strategies for Robust Image Watermarking"
  - u: "domain-skewed_federated_learning_with_feature_decoupling_and_calibration/"
    t: "Domain-Skewed Federated Learning with Feature Decoupling and Calibration"
  - u: "enhancing_out-of-distribution_detection_with_extended_logit_normalization/"
    t: "Enhancing Out-of-Distribution Detection with Extended Logit Normalization"
  - u: "fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation/"
    t: "FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation"
  - u: "feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift/"
    t: "FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift"
  - u: "federated_active_learning_extreme_noniid/"
    t: "Federated Active Learning Under Extreme Non-IID and Global Class Imbalance"
  - u: "fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_/"
    t: "FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning"
  - u: "generative_adversarial_perturbations_with_cross-paradigm_transferability_on_loca/"
    t: "Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting"
  - u: "irisfp_adversarial-example-based_model_fingerprinting_with_enhanced_uniqueness_a/"
    t: "IrisFP: Adversarial-Example-based Model Fingerprinting with Enhanced Uniqueness and Robustness"
  - u: "one-to-more_high-fidelity_training-free_anomaly_generation_with_attention_control/"
    t: "One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control"
  - u: "proxyfl_a_proxy-guided_framework_for_federated_semi-supervised_learning/"
    t: "ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning"
  - u: "recovermark_robust_watermarking_for_localization_and_recovery_of_manipulated_fac/"
    t: "RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces"
  - u: "subflot_submodel_extraction_for_efficient_and_personalized_federated_learning_vi/"
    t: "SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport"
  - u: "tiacam_text-anchored_invariant_feature_learning_with_auto-augmentation_for_camer/"
    t: "TIACam: Text-Anchored Invariant Feature Learning with Auto-Augmentation for Camera-Robust Zero-Watermarking"
  - u: "towards_highly_transferable_vision-language_attack_via_semantic-augmented_dynami/"
    t: "Towards Highly Transferable Vision-Language Attack via Semantic-Augmented Dynamic Contrastive Interaction"
  - u: "tutor-student_reinforcement_learning_a_dynamic_curriculum_for_robust_deepfake_de/"
    t: "Tutor-Student Reinforcement Learning: A Dynamic Curriculum for Robust Deepfake Detection"
  - u: "unigame_turning_a_unified_multimodal_model_into_its_own_adversary/"
    t: "UniGame: Turning a Unified Multimodal Model Into Its Own Adversary"
  - u: "when_robots_obey_the_patch_universal_transferable_patch_attacks_on_vision-langua/"
    t: "When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models"
  - u: "your_classifier_can_do_more_towards_balancing_the/"
    t: "Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation"
  - u: "φ-dpo_fairness_direct_preference_optimization_approach_to_continual_learning_in_/"
    t: "$\\varphi$-DPO: Fairness Direct Preference Optimization Approach to Continual Learning in Large Multimodal Models"
item_total: 24
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# 🛡️ AI Safety

**📷 CVPR2026** · **24** paper notes

📌 **Same area in other venues:** [🧪 ICML2026 (41)](../../ICML2026/ai_safety/index.md) · [💬 ACL2026 (4)](../../ACL2026/ai_safety/index.md) · [🔬 ICLR2026 (27)](../../ICLR2026/ai_safety/index.md) · [🤖 AAAI2026 (45)](../../AAAI2026/ai_safety/index.md) · [🧠 NeurIPS2025 (73)](../../NeurIPS2025/ai_safety/index.md) · [📹 ICCV2025 (24)](../../ICCV2025/ai_safety/index.md)

🔥 **Top topics:** Adversarial Robustness ×13 · Federated Learning ×8 · Multimodal/VLM ×5 · Watermarking ×4 · Robotics ×3

**[A Unified Perspective on Adversarial Membership Manipulation in Vision Models](a_unified_perspective_on_adversarial_membership_manipulation_in_vision_models.md)**

:   This work is the first to reveal the adversarial membership manipulation vulnerability in membership inference attacks (MIA) against vision models — imperceptible perturbations can forge non-members as members to deceive auditing. It identifies a gradient norm collapse signature in forged members, and proposes a gradient-geometry-based detection strategy (MFD) and an adversarially robust inference framework (AR-MIA).

**[All Vehicles Can Lie: Efficient Adversarial Defense in Fully Untrusted-Vehicle Collaborative Perception via Pseudo-Random Bayesian Inference](all_vehicles_can_lie_efficient_adversarial_defense_in_fully_untrusted-vehicle_co.md)**

:   This paper proposes the Pseudo-Random Bayesian Inference (PRBI) framework for collaborative perception scenarios where **all vehicles are untrusted**. By leveraging inter-frame temporal consistency as a self-referential signal, PRBI employs pseudo-random grouping combined with Bayesian inference to efficiently identify and exclude malicious vehicles at an average cost of only 2.5 validations per frame, recovering detection accuracy to 79.4%–86.9% of the pre-attack baseline.

**[ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering](clustermark_towards_robust_watermarking_for_autoregressive_image_generators_with.md)**

:   This paper proposes ClusterMark, a watermarking scheme based on visual token clustering that adapts KGW-style LLM watermarking to autoregressive image generators. By assigning visually similar tokens to the same green/red partition, it significantly improves watermark robustness under image perturbations while preserving image quality.

**[Computation and Communication Efficient Federated Unlearning via On-server Gradient Conflict Mitigation and Expression](computation_and_communication_efficient_federated_unlearning_via_on-server_gradi.md)**

:   This paper proposes FOUL, a two-stage framework that decouples causal and non-causal features during training and performs on-server gradient conflict matching during unlearning, achieving efficient federated unlearning with low communication overhead without accessing client data.

**[AdvMark: Decoupling Defense Strategies for Robust Image Watermarking](decoupling_defense_strategies_for_robust_image_watermarking.md)**

:   AdvMark proposes a two-stage decoupled defense framework: Stage 1 Encoder Adversarial Training (EAT) pushes watermarked images into non-attackable regions to resist adversarial attacks; Stage 2 performs direct image optimization to defend against distortion and regeneration attacks while preserving adversarial robustness. Evaluated across 9 watermarking methods × 10 attack types, AdvMark improves distortion/regeneration/adversarial accuracy by 29%/33%/46% respectively, while achieving the best image quality.

**[Domain-Skewed Federated Learning with Feature Decoupling and Calibration](domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)**

:   This paper proposes F²DC, a framework that employs a Domain Feature Decoupler (DFD) and a Domain Feature Corrector (DFC) to decompose local client features in federated learning into domain-robust features and domain-related features. Rather than discarding the latter, F²DC calibrates them to recover entangled class-discriminative information, and combines this with a domain-aware aggregation strategy. The method consistently outperforms state-of-the-art approaches across three multi-domain datasets.

**[Enhancing Out-of-Distribution Detection with Extended Logit Normalization](enhancing_out-of-distribution_detection_with_extended_logit_normalization.md)**

:   This paper identifies two forms of feature collapse induced by LogitNorm during training—dimensional collapse and origin collapse—and proposes a hyperparameter-free Extended Logit Normalization (ELogitNorm) that replaces the distance-to-origin scaling factor with the distance from features to the decision boundary. ELogitNorm significantly improves both post-hoc OOD detection performance and confidence calibration without sacrificing classification accuracy.

**[FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)**

:   This paper proposes FedAFD, a framework that simultaneously improves model performance for both heterogeneous clients and the server in multimodal federated learning through a three-stage design comprising bi-level adversarial alignment, granularity-aware feature fusion, and similarity-guided ensemble distillation.

**[FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)**

:   This paper proposes FedDAP, a domain-aware prototype federated learning framework that addresses global model performance degradation caused by client-side domain shift in federated learning. FedDAP constructs domain-specific global prototypes and employs a dual prototype alignment strategy comprising intra-domain alignment and cross-domain contrastive learning.

**[Federated Active Learning Under Extreme Non-IID and Global Class Imbalance](federated_active_learning_extreme_noniid.md)**

:   This paper systematically analyzes the impact of global class imbalance and client heterogeneity on query model selection in federated active learning (FAL), derives three core Observations, and proposes FairFAL—a class-fair FAL framework featuring adaptive query model selection, prototype-guided pseudo-labeling, and two-stage uncertainty-diversity balanced sampling—consistently outperforming all baselines across five benchmark datasets.

**[FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)**

:   This paper proposes FedRE, a framework that achieves a three-way balance among performance, privacy protection, and communication overhead in model-heterogeneous federated learning via "entangled representations"—aggregating all local representations of each client into a single cross-class representation using normalized random weights.

**[Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting](generative_adversarial_perturbations_with_cross-paradigm_transferability_on_loca.md)**

:   This paper proposes CrowdGen, the first cross-paradigm adversarial attack framework targeting both density-map and point-regression crowd counting models. A lightweight UNet generator combined with a multi-task loss (logit suppression, density suppression, GradCAM guidance, and frequency-domain constraint) achieves high transferability (TR up to 1.69) across seven SOTA crowd counting models while maintaining visual imperceptibility (~19 dB PSNR), increasing attack MAE by an average factor of 7×.

**[IrisFP: Adversarial-Example-based Model Fingerprinting with Enhanced Uniqueness and Robustness](irisfp_adversarial-example-based_model_fingerprinting_with_enhanced_uniqueness_a.md)**

:   This paper proposes IrisFP, a model fingerprinting framework that simultaneously enhances fingerprint uniqueness and robustness through three innovations: placing fingerprints at the intersection of multi-class decision boundaries, constructing composite sample fingerprints, and performing statistically-guided fingerprint selection. IrisFP consistently achieves higher AUC than state-of-the-art methods across 5 datasets.

**[One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control](one-to-more_high-fidelity_training-free_anomaly_generation_with_attention_control.md)**

:   O2MAG proposes a training-free few-shot anomaly generation method that synthesizes diverse and realistic anomalies from a single reference anomaly image via a tri-branch diffusion process with self-attention grafting (TriAG). It incorporates Anomaly Guidance Optimization (AGO) to align textual semantics and Dual Attention Enhancement (DAE) to ensure complete mask-region filling. The method significantly outperforms existing approaches on downstream anomaly detection benchmarks using MVTec-AD.

**[ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning](proxyfl_a_proxy-guided_framework_for_federated_semi-supervised_learning.md)**

:   ProxyFL is proposed as a framework that leverages classifier weights as unified proxies to simultaneously mitigate external heterogeneity (cross-client distribution discrepancy) and internal heterogeneity (distribution mismatch between labeled and unlabeled data) in federated semi-supervised learning, achieving substantial improvements over existing FSSL methods across multiple benchmarks.

**[RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces](recovermark_robust_watermarking_for_localization_and_recovery_of_manipulated_fac.md)**

:   This paper proposes RecoverMark, a robust watermarking framework that embeds facial content itself as a watermark into the background region, simultaneously achieving tampering localization, original content recovery, and copyright verification while remaining effective under watermark removal attacks.

**[SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport](subflot_submodel_extraction_for_efficient_and_personalized_federated_learning_vi.md)**

:   This paper proposes SubFLOT, a framework that leverages Optimal Transport (OT) on the server side to align the parameter distributions of a global model with clients' historical models, enabling personalized pruning without access to raw data. Combined with an adaptive regularization mechanism to suppress pruning-induced parameter drift, SubFLOT substantially outperforms existing federated pruning methods across multiple datasets.

**[TIACam: Text-Anchored Invariant Feature Learning with Auto-Augmentation for Camera-Robust Zero-Watermarking](tiacam_text-anchored_invariant_feature_learning_with_auto-augmentation_for_camer.md)**

:   This paper proposes TIACam, a framework that simulates camera distortions via a learnable auto-augmentor, learns invariant features through text-anchored cross-modal adversarial training, and binds binary messages to features via a zero-watermarking head—achieving camera-robust zero-watermarking without modifying any image pixels. TIACam attains state-of-the-art bit accuracy across three real-world scenarios: screen recapture, print-and-scan, and screenshot.

**[Towards Highly Transferable Vision-Language Attack via Semantic-Augmented Dynamic Contrastive Interaction](towards_highly_transferable_vision-language_attack_via_semantic-augmented_dynami.md)**

:   This paper proposes SADCA (Semantic-Augmented Dynamic Contrastive Attack), which iteratively disrupts cross-modal semantic consistency between adversarial images and texts via a dynamic contrastive interaction mechanism and a semantic augmentation module. SADCA significantly improves adversarial transferability against vision-language pre-training (VLP) models, surpassing existing SOTA methods in both cross-model and cross-task attack settings.

**[Tutor-Student Reinforcement Learning: A Dynamic Curriculum for Robust Deepfake Detection](tutor-student_reinforcement_learning_a_dynamic_curriculum_for_robust_deepfake_de.md)**

:   This paper proposes the Tutor-Student Reinforcement Learning (TSRL) framework, which formulates the training process of a deepfake detector as a Markov Decision Process. A "tutor" (PPO agent) dynamically assigns loss weights to individual samples based on their visual features and historical learning dynamics (EMA loss, forgetting count). A "state-change" reward signal guides the "student" (detector) to prioritize high-value samples, substantially improving generalization in cross-dataset and cross-method evaluations.

**[UniGame: Turning a Unified Multimodal Model Into Its Own Adversary](unigame_turning_a_unified_multimodal_model_into_its_own_adversary.md)**

:   UniGame proposes the first self-adversarial post-training framework for unified multimodal models (UMMs). By attaching a lightweight perturber at the shared visual token interface, the generation branch actively constructs semantically consistent adversarial samples to challenge the understanding branch, forming a minimax self-play game that substantially improves consistency (+4.6%), understanding (+3.6%), generation, and robustness.

**[When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models](when_robots_obey_the_patch_universal_transferable_patch_attacks_on_vision-langua.md)**

:   This paper proposes the UPA-RFAS framework, which learns a single physical adversarial patch to achieve universal, transferable black-box attacks against VLA robot policies through a combination of feature-space displacement, attention hijacking, and semantic misalignment.

**[Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation](your_classifier_can_do_more_towards_balancing_the.md)**

:   This paper analyzes the energy landscape to reveal the complementarity between adversarial training (AT) and JEM—AT aligns the clean-adversarial energy distribution (→ robustness); JEM aligns the clean-generated energy distribution (→ accuracy + generation). The proposed EB-JDAT models the joint distribution $p(\mathbf{x}, \tilde{\mathbf{x}}, y)$ and employs min-max energy optimization to align the energy distributions of all three data types. On CIFAR-10, AutoAttack robustness reaches 68.76% (surpassing SOTA AT by +10.78%), while maintaining 90.39% clean accuracy and competitive generation quality with FID=27.42.

**[$\varphi$-DPO: Fairness Direct Preference Optimization Approach to Continual Learning in Large Multimodal Models](φ-dpo_fairness_direct_preference_optimization_approach_to_continual_learning_in_.md)**

:   This paper proposes $\varphi$-DPO, which adopts DPO as a continual learning paradigm (using the previous-step model as the reference policy) and introduces a fairness modulation factor $(1-p)^\gamma$ inspired by focal loss to balance gradient contributions across data groups. The authors theoretically prove that the gradient bias approaches zero as $\gamma \to \infty$, and achieve state-of-the-art performance on the CoIN and MLLM-CL benchmarks.
