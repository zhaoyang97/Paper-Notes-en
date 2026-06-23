---
title: >-
  NeurIPS2025 Image Restoration Papers · 26 Notes
description: >-
  26 NeurIPS2025 papers in the Image Restoration area, covering Image Restoration, Super-Resolution, Diffusion Models, Layout & Composition and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "NeurIPS2025"
  - "Image Restoration"
  - "AI paper notes"
  - "paper summaries"
  - "Super-Resolution"
  - "Diffusion Models"
  - "Layout & Composition"
item_list:
  - u: "adaptive_discretization_for_consistency_models/"
    t: "Adaptive Discretization for Consistency Models"
  - u: "audio_super-resolution_with_latent_bridge_models/"
    t: "Audio Super-Resolution with Latent Bridge Models"
  - u: "dp2o-sr_direct_perceptual_preference_optimization_for_real-world_image_super-res/"
    t: "DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution"
  - u: "dynaguide_steering_diffusion_polices_with_active_dynamic_guidance/"
    t: "DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance"
  - u: "elucidated_rolling_diffusion_models_for_probabilistic_forecasting_of_complex_dyn/"
    t: "Elucidated Rolling Diffusion Models for Probabilistic Forecasting of Complex Dynamics"
  - u: "encoder-decoder_diffusion_language_models_for_efficient_training_and_inference/"
    t: "Encoder-Decoder Diffusion Language Models for Efficient Training and Inference"
  - u: "enhancing_infrared_vision_progressive_prompt_fusion_network_and_benchmark/"
    t: "Enhancing Infrared Vision: Progressive Prompt Fusion Network and Benchmark"
  - u: "fiper_factorized_features_for_robust_image_super-resolution_and_compression/"
    t: "FIPER: Factorized Features for Robust Image Super-Resolution and Compression"
  - u: "gc4nc_a_benchmark_framework_for_graph_condensation_on_node_classification_with_n/"
    t: "GC4NC: A Benchmark Framework for Graph Condensation on Node Classification with New Insights"
  - u: "implicit_augmentation_from_distributional_symmetry_in_turbulence_super-resolutio/"
    t: "Implicit Augmentation from Distributional Symmetry in Turbulence Super-Resolution"
  - u: "improving_diffusion-based_inverse_algorithms_under_few-step_constraint_via_learn/"
    t: "Improving Diffusion-based Inverse Algorithms under Few-Step Constraint via Learnable Linear Extrapolation"
  - u: "latent_harmony_synergistic_unified_uhd_image_restoration_via_latent_space_regula/"
    t: "Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement"
  - u: "latent_harmony_synergistic_unified_uhd_image_restoration_with_pre-trained_diffus/"
    t: "Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement"
  - u: "learning_cocoercive_conservative_denoisers_via_helmholtz_decomposition_for_poiss/"
    t: "Learning Cocoercive Conservative Denoisers via Helmholtz Decomposition for Poisson Inverse Problems"
  - u: "luminance-aware_statistical_quantization_unsupervised_hierarchical_learning_for_/"
    t: "Luminance-Aware Statistical Quantization: Unsupervised Hierarchical Learning for Illumination Enhancement"
  - u: "map_estimation_with_denoisers_convergence_rates_and_guarantees/"
    t: "MAP Estimation with Denoisers: Convergence Rates and Guarantees"
  - u: "modem_a_morton-order_degradation_estimation_mechanism_for_adverse_weather_image_/"
    t: "MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration"
  - u: "moe-gyro_self-supervised_over-range_reconstruction_and_denoising_for_mems_gyrosc/"
    t: "MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes"
  - u: "ms-bart_unified_modeling_of_mass_spectra_and_molecules_for_structure_elucidation/"
    t: "MS-BART: Unified Modeling of Mass Spectra and Molecules for Structure Elucidation"
  - u: "real-world_adverse_weather_image_restoration_via_dual-level_reinforcement_learni/"
    t: "Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start"
  - u: "rethinking_circuit_completeness_in_language_models_and_or_and_adder_gates/"
    t: "Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates"
  - u: "rethinking_nighttime_image_deraining_via_learnable_color_space_transformation/"
    t: "Rethinking Nighttime Image Deraining via Learnable Color Space Transformation"
  - u: "rgb-to-polarization_estimation_a_new_task_and_benchmark_study/"
    t: "RGB-to-Polarization Estimation: A New Task and Benchmark Study"
  - u: "scsplit_bringing_severity_cognizance_to_image_decomposition_in_fluorescence_micr/"
    t: "scSplit: Bringing Severity Cognizance to Image Decomposition in Fluorescence Microscopy"
  - u: "spiking_meets_attention_efficient_remote_sensing_image_super-resolution_with_att/"
    t: "Spiking Meets Attention: Efficient Remote Sensing Image Super-Resolution with Attention Spiking Neural Networks"
  - u: "the_effect_of_optimal_self-distillation_in_noisy_gaussian_mixture_model/"
    t: "The Effect of Optimal Self-Distillation in Noisy Gaussian Mixture Model"
item_total: 26
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# 🖼️ Image Restoration

**🧠 NeurIPS2025** · **26** paper notes

📌 **Same area in other venues:** [📷 CVPR2026 (107)](../../CVPR2026/image_restoration/index.md) · [🔬 ICLR2026 (61)](../../ICLR2026/image_restoration/index.md) · [🧪 ICML2026 (21)](../../ICML2026/image_restoration/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/image_restoration/index.md) · [📹 ICCV2025 (31)](../../ICCV2025/image_restoration/index.md)

🔥 **Top topics:** Image Restoration ×8 · Super-Resolution ×5 · Diffusion Models ×4 · Layout & Composition ×2

**[Adaptive Discretization for Consistency Models](adaptive_discretization_for_consistency_models.md)**

:   This paper proposes ADCM, which formalizes the discretization step size of consistency models as a constrained optimization problem balancing local consistency (trainability) and global consistency (stability), derives a closed-form solution via the Gauss-Newton method, and achieves adaptive discretization that surpasses all prior CMs on CIFAR-10 using less than 25% of the training budget.

**[Audio Super-Resolution with Latent Bridge Models](audio_super-resolution_with_latent_bridge_models.md)**

:   This paper proposes AudioLBM, which compresses audio waveforms into a continuous latent space and employs a bridge model to realize a latent-to-latent generation process from low-resolution to high-resolution. Combined with frequency-aware training for broader data utilization and a cascaded design to surpass the 48kHz ceiling, AudioLBM comprehensively outperforms methods such as AudioSR across speech, sound effects, and music, while achieving any-to-192kHz audio super-resolution for the first time.

**[DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution](dp2o-sr_direct_perceptual_preference_optimization_for_real-world_image_super-res.md)**

:   This paper proposes DP²O-SR, a framework that exploits the inherent stochasticity of diffusion models to generate diverse super-resolution outputs, constructs preference pairs via a hybrid perceptual reward, and introduces a Hierarchical Preference Optimization (HPO) strategy to adaptively weight training pairs — significantly improving perceptual quality in real-world image super-resolution without any human annotations.

**[DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance](dynaguide_steering_diffusion_polices_with_active_dynamic_guidance.md)**

:   This paper proposes DynaGuide, which applies classifier guidance to a frozen pretrained diffusion policy at inference time via an external latent dynamics model, steering the robot toward arbitrary positive/negative goals without modifying policy weights. It achieves an average success rate of 70% on CALVIN simulation and 80% on a real robot.

**[Elucidated Rolling Diffusion Models for Probabilistic Forecasting of Complex Dynamics](elucidated_rolling_diffusion_models_for_probabilistic_forecasting_of_complex_dyn.md)**

:   This paper proposes ERDM, the first framework to successfully unify the Rolling Diffusion paradigm with the principled design choices of EDM (noise schedule, preconditioning, Heun sampler). By employing a progressive noise schedule that explicitly models growing uncertainty, ERDM significantly outperforms autoregressive EDM baselines on Navier-Stokes and ERA5 weather forecasting benchmarks.

**[Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)**

:   This paper proposes E2D2, an encoder-decoder architecture for discrete diffusion language models that performs iterative denoising via a lightweight decoder while periodically updating representations through a large encoder, achieving faster inference (~3× vs. MDLM) and more efficient block diffusion training (halving FLOPs).

**[Enhancing Infrared Vision: Progressive Prompt Fusion Network and Benchmark](enhancing_infrared_vision_progressive_prompt_fusion_network_and_benchmark.md)**

:   To address the challenge of coupled degradations (low contrast, blur, and noise) in thermal infrared (TIR) images, this paper proposes PPFN, a progressive prompt fusion network with a dual-prompt design, along with the Selective Progressive Training (SPT) strategy. The authors also construct HM-TIR, the first large-scale multi-scene TIR benchmark dataset. The proposed method achieves an 8.76% PSNR improvement in composite degradation scenarios.

**[FIPER: Factorized Features for Robust Image Super-Resolution and Compression](fiper_factorized_features_for_robust_image_super-resolution_and_compression.md)**

:   This paper proposes a Factorized Features representation that decomposes images into learnable non-uniform bases and spatially variant coefficients, augmented with sawtooth coordinate transformation and multi-frequency modulation. The approach achieves a 204.4% relative PSNR gain at 4× super-resolution (HAT-L-F vs. SwinIR) and a 21.09% BD-rate reduction over VTM in image compression.

**[GC4NC: A Benchmark Framework for Graph Condensation on Node Classification with New Insights](gc4nc_a_benchmark_framework_for_graph_condensation_on_node_classification_with_n.md)**

:   This paper proposes GC4NC—the first systematic benchmark framework for graph condensation (GC)—which evaluates multiple GC methods across 8 dimensions (performance / efficiency / privacy protection / denoising / NAS effectiveness / transferability, etc.), finding that trajectory matching methods achieve the best performance, structure-free methods are most efficient, and graph condensation significantly outperforms image condensation under 1000× compression.

**[Implicit Augmentation from Distributional Symmetry in Turbulence Super-Resolution](implicit_augmentation_from_distributional_symmetry_in_turbulence_super-resolutio.md)**

:   This paper demonstrates that the statistical isotropy of turbulence itself constitutes a form of implicit data augmentation, enabling standard CNNs to partially learn rotational equivariance in super-resolution tasks without explicit rotation augmentation or equivariant architectures. The authors further show that the scale dependence of equivariance error is consistent with Kolmogorov's local isotropy hypothesis.

**[Improving Diffusion-based Inverse Algorithms under Few-Step Constraint via Learnable Linear Extrapolation](improving_diffusion-based_inverse_algorithms_under_few-step_constraint_via_learn.md)**

:   This paper proposes Learnable Linear Extrapolation (LLE), which combines current and historical clean data estimates via learnable linear coefficients to enhance any diffusion inverse problem algorithm conforming to the Sampler-Corrector-Noiser paradigm under few-step (3–5 steps) constraints. The method requires only 50 training samples and a few minutes of training, yielding consistent improvements across 9+ algorithms × 5 tasks.

**[Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement](latent_harmony_synergistic_unified_uhd_image_restoration_via_latent_space_regula.md)**

:   This paper proposes Latent Harmony, a two-stage framework that constructs a generalizable VAE (LH-VAE) via latent space regularization, and introduces a high-frequency-guided controllable LoRA fine-tuning mechanism, achieving flexible fidelity-perceptual quality trade-offs in unified multi-degradation UHD image restoration while preserving structural integrity.

**[Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement](latent_harmony_synergistic_unified_uhd_image_restoration_with_pre-trained_diffus.md)**

:   This paper proposes Latent Harmony, a two-stage framework that constructs a degradation-robust LH-VAE via latent space regularization, and subsequently applies high-frequency-guided LoRA fine-tuning to independently optimize the encoder (fidelity) and decoder (perceptual quality), achieving a unified solution to the generalization–reconstruction–perception trilemma in all-in-one UHD image restoration.

**[Learning Cocoercive Conservative Denoisers via Helmholtz Decomposition for Poisson Inverse Problems](learning_cocoercive_conservative_denoisers_via_helmholtz_decomposition_for_poiss.md)**

:   This paper introduces the concept of Cocoercive Conservative (CoCo) denoisers and proposes a novel training strategy via generalized Helmholtz decomposition — Hamiltonian regularization to promote conservativeness and spectral regularization to promote cocoerciveness — enabling denoisers to serve as proximal operators of implicit weakly convex priors, thereby achieving convergence-guaranteed and high-performance PnP methods for Poisson inverse problems (photon-limited deconvolution, low-dose CT, etc.).

**[Luminance-Aware Statistical Quantization: Unsupervised Hierarchical Learning for Illumination Enhancement](luminance-aware_statistical_quantization_unsupervised_hierarchical_learning_for_.md)**

:   This paper proposes the LASQ framework, which reformulates low-light image enhancement (LLIE) as a statistical sampling process over hierarchical luminance distributions. By exploiting the power-law distribution inherent in natural luminance transitions, LASQ employs MCMC sampling to generate hierarchical luminance adaptation operators (LAOs) that are embedded into the forward process of a diffusion model, enabling fully unsupervised enhancement without requiring any normal-light reference images.

**[MAP Estimation with Denoisers: Convergence Rates and Guarantees](map_estimation_with_denoisers_convergence_rates_and_guarantees.md)**

:   This paper proves that a simple iterative averaging algorithm based on MMSE denoisers—closely related to practical methods such as Cold Diffusion—provably converges to the proximal operator of the negative log-prior under log-concave prior assumptions, achieving a convergence rate of $\tilde{O}(1/k)$. The work provides rigorous theoretical foundations for a class of denoising methods that have demonstrated empirical success but lacked theoretical guarantees, and embeds the approach within a proximal gradient descent framework for MAP estimation.

**[MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration](modem_a_morton-order_degradation_estimation_mechanism_for_adverse_weather_image_.md)**

:   This paper proposes the MODEM framework, which combines Morton-encoded spatial scanning with selective state space models (SSMs) to capture spatially heterogeneous weather degradation patterns. Equipped with a dual degradation estimation module that provides both global and local priors, MODEM achieves state-of-the-art unified adaptive restoration across multiple adverse weather degradation types.

**[MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes](moe-gyro_self-supervised_over-range_reconstruction_and_denoising_for_mems_gyrosc.md)**

:   This paper proposes MoE-Gyro, a self-supervised Mixture-of-Experts framework that simultaneously addresses the fundamental range–noise trade-off in MEMS gyroscopes via an Over-Range Reconstruction Expert (ORE, incorporating Gaussian-Decay Attention and physics-informed constraints) and a Denoising Expert (DE, incorporating dual-branch complementary masking and FFT-guided augmentation). The measurable range is extended from ±450°/s to ±1500°/s, and bias instability is reduced by 98.4%.

**[MS-BART: Unified Modeling of Mass Spectra and Molecules for Structure Elucidation](ms-bart_unified_modeling_of_mass_spectra_and_molecules_for_structure_elucidation.md)**

:   This paper proposes MS-BART, which maps molecular fingerprints and molecular structures (SELFIES) into a shared token space via a unified vocabulary, performs multi-task pretraining on 4 million fingerprint–molecule pairs, and subsequently applies experimental spectra fine-tuning and chemical feedback alignment to enable efficient generation of molecular structures from mass spectra.

**[Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start](real-world_adverse_weather_image_restoration_via_dual-level_reinforcement_learni.md)**

:   This paper proposes a Dual-level Reinforcement Learning (DRL) framework that combines a physics-driven million-scale synthetic weather dataset, HFLS-Weather, for high-quality cold-start training, and achieves adaptive real-world adverse weather image restoration through Perturbation-driven Image Quality Optimization (PIQO) at the local level and global meta-controller multi-agent collaboration.

**[Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates](rethinking_circuit_completeness_in_language_models_and_or_and_adder_gates.md)**

:   This paper systematically introduces AND, OR, and ADDER gates to decompose language model circuits, reveals that circuit incompleteness primarily stems from the omission of OR gates, and proposes a framework combining noising and denoising interventions to fully recover all three gate types while guaranteeing both faithfulness and completeness.

**[Rethinking Nighttime Image Deraining via Learnable Color Space Transformation](rethinking_nighttime_image_deraining_via_learnable_color_space_transformation.md)**

:   Motivated by the statistical finding that nighttime rain exhibits far greater contrast in the Y channel (luminance) of YCbCr than in RGB, this work proposes a learnable Color Space Converter (CSC) that performs deraining in the Y channel, an Implicit Illumination Guidance (IIG) module that encodes non-uniform nighttime illumination, and a photorealistic dataset HQ-NightRain constructed via illumination-aware synthesis. The three components jointly yield substantial improvements in nighttime deraining performance.

**[RGB-to-Polarization Estimation: A New Task and Benchmark Study](rgb-to-polarization_estimation_a_new_task_and_benchmark_study.md)**

:   This paper formally defines the novel task of estimating polarization components (S₁/S₂/S₃) from standard RGB images, establishes the first systematic benchmark encompassing both restoration-based and generative methods, and finds that pretrained MAE achieves the best overall pixel-level accuracy (PSNR 24.74). Restoration-based methods consistently outperform diffusion-based generative methods, with pretrained weight transfer identified as a critical advantage.

**[scSplit: Bringing Severity Cognizance to Image Decomposition in Fluorescence Microscopy](scsplit_bringing_severity_cognizance_to_image_decomposition_in_fluorescence_micr.md)**

:   This paper proposes scSplit, which introduces a severity-cognizant input normalization module (SCIN) and a regression network (Reg) to endow an InDI-based iterative image decomposition framework with awareness of the mixing severity of two overlapping structures in fluorescence microscopy images. The method unifies image splitting and bleedthrough removal across five public datasets under a single framework.

**[Spiking Meets Attention: Efficient Remote Sensing Image Super-Resolution with Attention Spiking Neural Networks](spiking_meets_attention_efficient_remote_sensing_image_super-resolution_with_att.md)**

:   This paper proposes SpikeSR, the first attention-based spiking neural network (SNN) framework for remote sensing image super-resolution. By incorporating Spiking Attention Blocks (SAB) that combine Hybrid Dimensional Attention (HDA) and Deformable Similarity Attention (DSA), SpikeSR achieves state-of-the-art performance on AID/DOTA/DIOR while maintaining high computational efficiency.

**[The Effect of Optimal Self-Distillation in Noisy Gaussian Mixture Model](the_effect_of_optimal_self-distillation_in_noisy_gaussian_mixture_model.md)**

:   This paper presents a rigorous theoretical analysis of hyperparameter-optimized multi-stage self-distillation on noisy Gaussian mixture data using the replica method from statistical physics. It reveals that the denoising effect of hard pseudo-labels is the primary driver of performance gains in self-distillation, that moderate-sized datasets benefit the most, and proposes two practical improvement strategies—early stopping (limiting the number of distillation stages) and bias parameter fixing. Theoretical predictions are validated through experiments on CIFAR-10 with ResNet.
