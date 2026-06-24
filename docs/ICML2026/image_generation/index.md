---
title: >-
  ICML2026 Image Generation Papers · 141 Notes
description: >-
  141 ICML2026 papers in the Image Generation area, covering Diffusion Models, Text-to-Image, Alignment/RLHF, Layout & Composition, Adversarial Robustness, Image Editing and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ICML2026"
  - "Image Generation"
  - "AI paper notes"
  - "paper summaries"
  - "Diffusion Models"
  - "Text-to-Image"
  - "Alignment/RLHF"
  - "Layout & Composition"
  - "Adversarial Robustness"
  - "Image Editing"
item_list:
  - u: "a_diffusive_classification_loss_for_learning_energy-based_generative_models/"
    t: "A Diffusive Classification Loss for Learning Energy-based Generative Models"
  - u: "a_kinetic_energy_perspective_of_flow_matching/"
    t: "A Kinetic Energy Perspective of Flow Matching"
  - u: "a_systematic_investigation_of_rl-jailbreaking_in_llms/"
    t: "A Systematic Investigation of RL-Jailbreaking in LLMs"
  - u: "a_unified_framework_for_diffusion_model_unlearning_with_f-divergence/"
    t: "A Unified Framework for Diffusion Model Unlearning with f-Divergence"
  - u: "adaeraser_training-free_object_removal_via_adaptive_attention_suppression/"
    t: "AdaEraser: Training-Free Object Removal via Adaptive Attention Suppression"
  - u: "adapting_noise_to_data_generative_flows_from_1d_processes/"
    t: "Adapting Noise to Data: Generative Flows from Learned 1D Processes"
  - u: "adversarial_flow_models/"
    t: "Adversarial Flow Models"
  - u: "aesformer_transform_everyday_photos_into_beautiful_memories/"
    t: "AesFormer: Transform Everyday Photos into Beautiful Memories"
  - u: "ag-repa_causal_layer_selection_for_representation_alignment_in_audio_flow_matchi/"
    t: "AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching"
  - u: "alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models/"
    t: "Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models"
  - u: "anomaly-preference_image_generation/"
    t: "Anomaly-Preference Image Generation (APO)"
  - u: "ateliereval_agentic_evaluation_of_humans_llms_as_text-to-image_prompters/"
    t: "AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters"
  - u: "balancing_fidelity_and_diversity_in_diffusion_models_via_symmetric_attention_dec/"
    t: "Balancing Fidelity and Diversity in Diffusion Models via Symmetric Attention Decomposition: Hopfield Perspective"
  - u: "barriers_to_counterfactual_credit_attribution_for_autoregressive_models/"
    t: "Barriers to Counterfactual Credit Attribution for Autoregressive Models"
  - u: "bayesian_tensor_decomposition_with_diffusion_model_prior/"
    t: "Bayesian Tensor Decomposition with Diffusion Model Prior"
  - u: "beyond_generative_priors_minority_sampling_with_jepa-guided_diffusion/"
    t: "Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion"
  - u: "bootstrap_your_generator_unpaired_visual_editing_with_flow_matching/"
    t: "Bootstrap Your Generator: Unpaired Visual Editing with Flow Matching"
  - u: "breaking_the_lock-in_diversifying_text-to-image_generation_via_representation_mo/"
    t: "Breaking the Lock-in: Diversifying Text-to-Image Generation via Representation Modulation"
  - u: "budget-constrained_step-level_diffusion_caching/"
    t: "Budget-Constrained Step-Level Diffusion Caching"
  - u: "caracal_causal_architecture_via_spectral_mixing/"
    t: "Caracal: Causal Architecture via Spectral Mixing"
  - u: "clear_context-aware_learning_with_end-to-end_mask-free_inference_for_adaptive_vi/"
    t: "CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal"
  - u: "coarse-grained_boltzmann_generators/"
    t: "Coarse-Grained Boltzmann Generators"
  - u: "cocoedit_content-consistent_image_editing_via_region_regularized_reinforcement_l/"
    t: "CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning"
  - u: "cofrgenet_continued_fraction_architectures_for_language_generation/"
    t: "CoFrGeNet: Continued Fraction Architectures for Language Generation"
  - u: "compositional_generative_modeling_from_decentralized_data/"
    t: "Compositional Generative Modeling from Decentralized Data"
  - u: "compression_as_adaptation_implicit_visual_representation_with_diffusion_foundati/"
    t: "Compression as Adaptation: Implicit Visual Representation with Diffusion Foundation Models"
  - u: "conf-gen_conformal_uncertainty_quantification_for_generative_models/"
    t: "Conf-Gen: Conformal Uncertainty Quantification for Generative Models"
  - u: "conflict-aware_additive_guidance_for_flow_models_under_compositional_rewards/"
    t: "Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards"
  - u: "conformal_reliability_a_new_evaluation_metric_for_conditional_generation/"
    t: "Conformal Reliability: A New Evaluation Metric for Conditional Generation"
  - u: "content-style_identification_via_differential_independence/"
    t: "Content-Style Identification via Differential Independence"
item_total: 141
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# 🎨 Image Generation

**🧪 ICML2026** · **141** paper notes

📌 **Same area in other venues:** [📷 CVPR2026 (492)](../../CVPR2026/image_generation/index.md) · [🔬 ICLR2026 (353)](../../ICLR2026/image_generation/index.md) · [💬 ACL2026 (5)](../../ACL2026/image_generation/index.md) · [🤖 AAAI2026 (79)](../../AAAI2026/image_generation/index.md) · [🧠 NeurIPS2025 (221)](../../NeurIPS2025/image_generation/index.md) · [📹 ICCV2025 (213)](../../ICCV2025/image_generation/index.md)

🔥 **Top topics:** Diffusion Models ×49 · Text-to-Image ×10 · Alignment/RLHF ×8 · Layout & Composition ×7 · Adversarial Robustness ×4

**[A Diffusive Classification Loss for Learning Energy-based Generative Models](a_diffusive_classification_loss_for_learning_energy-based_generative_models.md)**

:   This paper proposes DiffCLF, which reformulates energy estimation across temporal noise levels as a classification problem. By training jointly with DSM, it learns more reliable energy functions without requiring expensive maximum likelihood sampling, specifically alleviating the "mode blindness" of score matching regarding multi-modal weights.

**[A Kinetic Energy Perspective of Flow Matching](a_kinetic_energy_perspective_of_flow_matching.md)**

:   This paper treats flow matching sampling trajectories as particle motions and defines Kinetic Path Energy (KPE) to measure the cumulative kinetic energy of the generation process for each sample. Based on this, a training-free strategy called Kinetic Trajectory Shaping (KTS) is proposed to enhance generation quality while suppressing memorization caused by late-stage energy spikes.

**[A Systematic Investigation of RL-Jailbreaking in LLMs](a_systematic_investigation_of_rl-jailbreaking_in_llms.md)**

:   This paper investigates RL-based LLM jailbreaking as a decomposable POMDP system, finding that environment definition factors—such as reward functions, episode length, and the number of training questions—determine automated red teaming success rates more significantly than the choice of RL algorithm.

**[A Unified Framework for Diffusion Model Unlearning with f-Divergence](a_unified_framework_for_diffusion_model_unlearning_with_f-divergence.md)**

:   This paper generalizes MSE/KL alignment in diffusion model concept unlearning to arbitrary $f$-divergence, proposing the f-DMU framework. It identifies that closed-form Hellinger loss is often more stable and better at preserving non-target concepts than MSE.

**[AdaEraser: Training-Free Object Removal via Adaptive Attention Suppression](adaeraser_training-free_object_removal_via_adaptive_attention_suppression.md)**

:   AdaEraser adaptively modulates the self-attention suppression intensity of diffusion models based on the "object presence degree." It simultaneously improves object removal completeness and background reconstruction quality without training, outperforming both training-based and training-free object removal methods on Mulan and OABench.

**[Adapting Noise to Data: Generative Flows from Learned 1D Processes](adapting_noise_to_data_generative_flows_from_1d_processes.md)**

:   This paper argues that the default Gaussian latent in flow/diffusion models is not always suitable for the data distribution. It proposes constructing a data-adaptive product prior using learnable 1D quantile functions to jointly learn the noise and velocity field in flow matching, thereby shortening the transport path and improving performance on heavy-tailed weather data and low-capacity image generation.

**[Adversarial Flow Models](adversarial_flow_models.md)**

:   The authors add an optimal transport regularization term $\|G(z)-z\|^2$ to the GAN training objective, constraining the GAN's "arbitrary transport map" to a unique Wasserstein-2 optimal transport map. This allows adversarial training on pure Transformers to stabilize for the first time and perform end-to-end single-step generation. On ImageNet-256, the 1NFE FID reaches **2.38** (XL/2) and **1.94** (112-layer recursive model).

**[AesFormer: Transform Everyday Photos into Beautiful Memories](aesformer_transform_everyday_photos_into_beautiful_memories.md)**

:   AesFormer defines aesthetic photo enhancement as Aesthetic Photo Reconstruction (APR). It introduces a two-stage framework that first generates a photography action plan and then executes structural editing, transforming errors in composition, perspective, and pose into executable edits. It significantly outperforms open-source editors on AesRecon and approaches the performance of Nano Banana Pro.

**[AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching](ag-repa_causal_layer_selection_for_representation_alignment_in_audio_flow_matchi.md)**

:   AG-REPA discovers that the "layers storing semantic information" and the "layers actually driving the velocity field" in audio Flow Matching do not coincide. It proposes using forward-only gate ablation to select layers with the highest causal contribution for representation alignment, achieving faster convergence and lower FAD than fixed-layer REPA in speech and general audio generation.

**[Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)**

:   This paper proposes Alignment-Guided Score Matching, which utilizes a reward-free Plackett-Luce alignment reward to directly incorporate positive and negative text-image matching signals into the diffusion score matching objective. By training lightweight soft tokens, it improves T2I semantic alignment while mitigating common repetition and counting errors found in SoftREPA.

**[Anomaly-Preference Image Generation (APO)](anomaly-preference_image_generation.md)**

:   The authors reformulate "few-shot anomaly image generation" as a "preference optimization problem without manual annotation." Using real anomalies as positive samples and the denoising bias of a reference model at the same timestep as implicit negative samples, they align the diffusion model with the anomaly distribution via a DPO-style loss. Combined with Time-Aware Capacity Allocation (TACA) to adjust LoRA rank by timestep for structural diversity and hierarchical CFG for text-anomaly alignment, APO achieves state-of-the-art results in both realism and diversity on benchmarks like MVTec.

**[AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters](ateliereval_agentic_evaluation_of_humans_llms_as_text-to-image_prompters.md)**

:   AtelierEval is the first to evaluate the "prompter" in the text-to-image workflow. Using 360 expert tasks, three cognitive task categories, and the AtelierJudge agentic evaluator system, it quantitatively measures the prompting proficiency of humans and MLLMs, discovering that image-mimicry prompting is often more reliable than pure text-planning prompting.

**[Balancing Fidelity and Diversity in Diffusion Models via Symmetric Attention Decomposition: Hopfield Perspective](balancing_fidelity_and_diversity_in_diffusion_models_via_symmetric_attention_dec.md)**

:   Decomposes the $\mathbf{QK}^\top$ attention matrix in diffusion models into symmetric components (energy landscape) and anti-symmetric components (circulatory dynamics), derives Hopfield-style stability measures to diagnose metastable mixing, and achieves a training-free controllable fidelity-diversity trade-off by regulating the anti-symmetric component.

**[Barriers to Counterfactual Credit Attribution for Autoregressive Models](barriers_to_counterfactual_credit_attribution_for_autoregressive_models.md)**

:   This paper formally investigates the problem of "Counterfactual Credit Attribution (CCA)" for generative models in RAG/in-context deployment. It proves two surprising negative results: (1) An autoregressive rollout is not necessarily CCA even if the underlying next-token predictor is $(0,0)$-CCA—CCA does not compose naturally under autoregression like DP does; (2) Black-box "CCA retrofitting" for a deployed non-attributing model requires an exponential number of queries relative to the output length $\ell$.

**[Bayesian Tensor Decomposition with Diffusion Model Prior](bayesian_tensor_decomposition_with_diffusion_model_prior.md)**

:   DiffBCP injects pre-trained diffusion models as implicit data priors into Bayesian CP tensor decomposition. By employing a split Gibbs sampler to achieve tractable posterior inference, it substantially outperforms traditional and deep tensor decomposition baselines in image inpainting and denoising tasks (with a PSNR gain of up to +2.33 dB on FFHQ).

**[Beyond Generative Priors: Minority Sampling with JEPA-Guided Diffusion](beyond_generative_priors_minority_sampling_with_jepa-guided_diffusion.md)**

:   Ours proposes JEPA Guidance, utilizing implicit density signals from JEPA (e.g., DINOv2) encoders to guide the sampling of diffusion models. This shifts the definition of minority samples from "low density under the generative model's prior" to "low density under a world prior," achieving more semantically meaningful rare sample generation in unconditional, class-conditional, and text-to-image scenarios.

**[Bootstrap Your Generator: Unpaired Visual Editing with Flow Matching](bootstrap_your_generator_unpaired_visual_editing_with_flow_matching.md)**

:   The authors propose Bootstrap Your Generator (ByG), an editing training framework for flow matching that does not require paired data. It extracts editing direction priors from a frozen base model, maintains source structure via cycle consistency, and bridges the training-inference gap using gradient routing. It outperforms supervised baselines trained on millions of paired samples in both image and video editing.

**[Breaking the Lock-in: Diversifying Text-to-Image Generation via Representation Modulation](breaking_the_lock-in_diversifying_text-to-image_generation_via_representation_mo.md)**

:   The authors discover that Transformer-based text-to-image models cause the "zero-frequency spatial mean (DC component)" to rapidly align across different random seeds during the early stages of denoising, prematurely locking the global layout. Consequently, they propose DAVE—a lightweight attenuation of the DC component in intermediate representations during early generation stages. This approach unlocks sample diversity for the same prompt with almost zero overhead while maintaining image quality and text-image alignment.

**[Budget-Constrained Step-Level Diffusion Caching](budget-constrained_step-level_diffusion_caching.md)**

:   BudCache transforms step-level caching for diffusion models from "passively triggered by thresholds with input-dependent latency" into "fixing a compute budget $B$ first, then searching for the optimal caching strategy offline." Using a hybrid of Simulated Annealing and Hill Climbing, it produces a static caching mask in minutes. For tight budgets, it employs teacher-student distillation to realign the schedule, outperforming heuristic caching methods like TeaCache and MagCache in quality across FLUX.1-dev and Wan2.1 under identical latency.

**[Caracal: Causal Architecture via Spectral Mixing](caracal_causal_architecture_via_spectral_mixing.md)**

:   Caracal replaces the $\mathcal{O}(L^2)$ attention in Transformers with an $\mathcal{O}(L \log L)$ Multi-Head Fourier (MHF) module. It achieves strict causal masking in the frequency domain via a "pad-FFT-multiply-iFFT-truncate" mechanism and completely removes positional embeddings. Using only standard FFT operators (without relying on custom CUDA kernels like Mamba), it matches the performance of Llama, Mamba, Mamba-2, and Jamba across scales from Tiny to Large.

**[CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal](clear_context-aware_learning_with_end-to-end_mask-free_inference_for_adaptive_vi.md)**

:   This paper proposes CLEAR for video subtitle removal: a two-stage training approach (Stage I learns self-supervised subtitle prior masks using a dual encoder with orthogonal decoupling; Stage II adds LoRA and an occlusion head to the Wan2.1 video diffusion model for adaptive weighting). The inference requires no masks or text detectors. By training only 0.77% of parameters, it achieves a PSNR of 26.80 dB on a Chinese test set (+6.77 dB over the strongest baseline) and demonstrates zero-shot generalization to six languages.

**[Coarse-Grained Boltzmann Generators](coarse-grained_boltzmann_generators.md)**

:   The authors propose Coarse-Grained Boltzmann Generators (CG-BGs), which combine normalizing flow generative models with learned Potential of Mean Force (PMF) in coarse-grained coordinate space for importance sampling. This achieves asymptotically correct equilibrium sampling at a significantly lower computational cost compared to all-atom BGs.

**[CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning](cocoedit_content-consistent_image_editing_via_region_regularized_reinforcement_l.md)**

:   To address the issue where "editing models often make unintended changes in non-edited regions," this paper constructs the CoCoEdit-40K local editing dataset, proposes a pixel-level similarity reward to complement MLLM rewards, and designs a region-regularized RL objective (constraining consistency in non-edited regions for high-reward samples and forcing changes in edited regions for low-reward samples). This improves both FLUX.1 Kontext and Qwen-Image-Edit in edit scores and PSNR/SSIM, breaking the existing trade-off where "improving editing capability inevitably hurts consistency."

**[CoFrGeNet: Continued Fraction Architectures for Language Generation](cofrgenet_continued_fraction_architectures_for_language_generation.md)**

:   This paper introduces "continued fractions," a class of functions known for optimal rational approximation, into generative Transformers. The authors design CoFrNet replacement modules (CAttnU/CAttnM/Cffn) for multi-head attention and FFN. By utilizing "continuants" in a closed-form derivation, they reduce $d$ divisions to 1, achieving comparable or superior downstream performance on GPT2-xl and Llama-3.2B using only $\frac{2}{3}\sim\frac{1}{2}$ of the parameters.

**[Compositional Generative Modeling from Decentralized Data](compositional_generative_modeling_from_decentralized_data.md)**

:   When generative factors are partitioned across multiple clients that do not share raw data, this paper proposes **DCFM (Decentralized Compositional Flow Matching)** to enforce global conditional independence constraints on attributes. This allows the model to generate attribute combinations never observed by any single client, significantly outperforming federated learning and mixture-of-experts baselines across conditional image generation, robotic spatial planning, and chest X-ray disease co-occurrence tasks.

**[Compression as Adaptation: Implicit Visual Representation with Diffusion Foundation Models](compression_as_adaptation_implicit_visual_representation_with_diffusion_foundati.md)**

:   Visual signals are encoded as Low-Rank Adaptation (LoRA) parameters on a frozen diffusion foundation model and compressed into a single compact vector via hash mapping. This achieves high perceptual quality video compression at extremely low bitrates while supporting inference-time scaling and generative editing.

**[Conf-Gen: Conformal Uncertainty Quantification for Generative Models](conf-gen_conformal_uncertainty_quantification_for_generative_models.md)**

:   The Conf-Gen framework is proposed to extend Conformal Risk Control (CRC) to generative tasks. Using parameterized selection functions and admissibility functions, it provides formal uncertainty guarantees for tasks such as LLM QA, image generation, dialogue systems, and AI agents, while relaxing theoretical assumptions like the monotonicity of CRC.

**[Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards](conflict-aware_additive_guidance_for_flow_models_under_compositional_rewards.md)**

:   To address the off-manifold drift issue in flow models during inference-time guidance under multi-objective compositional rewards, this paper proposes Conflict-Aware Additive Guidance (CAR). By detecting gradient conflicts and dynamically switching to learnable value gradient corrections, it improves identity preservation by 25.4% and planning success rate by 38.75% with minimal additional computational overhead.

**[Conformal Reliability: A New Evaluation Metric for Conditional Generation](conformal_reliability_a_new_evaluation_metric_for_conditional_generation.md)**

:   The paper proposes CReL, a reliability score based on Conformal Prediction. By constructing convex prediction sets in latent space and optimizing for worst-case metric performance, it achieves uncertainty-aware evaluation for conditional generative models. Experiments on image-to-text and text-to-image tasks reveal reliability differences that traditional single-output metrics fail to capture.

**[Content-Style Identification via Differential Independence](content-style_identification_via_differential_independence.md)**

:   This paper proposes CSDI (content-style differential independence), a novel identifiability condition. It proves that unpaired multi-domain content-style block identifiability is achievable under settings where content and style are statistically **correlated** and Jacobians are **dense**, provided that the column spaces of the generator's Jacobians with respect to content and style are mutually orthogonal on the data manifold. Using Hutchinson noise probing, this condition is implemented as a scalable regularization term $\mathcal{L}_{\rm orth}$ for StyleGAN2-ADA. In counterfactual generation and cross-domain translation on AFHQ / CelebA-HQ, FID is reduced from 5.2 / 4.6 to 4.4 / 4.3, and LPIPS is improved from 0.40 / 0.26 to 0.45 / 0.34.

**[DFlash: Block Diffusion for Flash Speculative Decoding](dflash_block_diffusion_for_flash_speculative_decoding.md)**

:   DFlash replaces autoregressive drafters like EAGLE-3 with a lightweight "Block Diffusion" drafter. By injecting multi-layer hidden features of the target model as KV into every layer of the draft model, it enables parallel drafting of an entire block of tokens in a single forward pass, achieving up to 6× lossless acceleration—approximately 2.5× faster than EAGLE-3.

**[DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection](dgs-net_distillation-guided_gradient_surgery_for_clip_fine-tuning_in_ai-generate.md)**

:   To address the issue where "fine-tuning CLIP for AI-generated image detection causes catastrophic forgetting that destroys transferable priors," this paper proposes DGS-Net. It decomposes the classification loss gradient by coordinates into harmful positive components $g^+$ and beneficial negative components $g^-$. The image gradients of the training network are first orthogonally projected onto the complement space of the **harmful directions of frozen CLIP text gradients** (Orthogonal Suppression, removing task-irrelevant semantics), and then aligned with the **beneficial directions of frozen CLIP image gradients** (Prior Alignment, preserving pre-trained priors). This achieves an average detection accuracy 6.6% higher than the SOTA across 50 generative models.

**[Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers](diagnosing_and_correcting_concept_omission_in_multimodal_diffusion_transformers.md)**

:   The paper utilizes linear probes to discover that in the intermediate layers of MM-DiT (FLUX / SD3.5), the key vectors of text tokens naturally encode a binary signal indicating "whether the target concept will appear." Based on this, the authors propose Omission Signal Intervention (OSI): during inference, the mean difference direction of "omission class - existence class" is injected into the key vectors of the Top-K heads with an intensity of $\alpha\sigma\boldsymbol{\theta}$. This stimulates the model's "self-awareness" of missing concepts to complete the generation. On FLUX, the GenEval 6-object accuracy improves from 0.18 to 0.40 without any fine-tuning.

**[Diffusion Differentiable Resampling](diffusion_differentiable_resampling.md)**

:   This paper proposes **diffusion resampling**: a **training-free** diffusion process that serves as a naturally differentiable reparameterization replacement for the resampling step in Sequential Monte Carlo (SMC). It proves consistent convergence relative to the number of samples $N$ under the Wasserstein distance and outperforms existing differentiable resampling methods such as OT, Gumbel-Softmax, and Soft/Resampling-on-weights across multiple particle filtering and parameter estimation benchmarks.

**[Diffusion Models Are Statistically Optimal for Learning Low-Dimensional Multi-Modal Distributions](diffusion_models_are_statistically_optimal_for_learning_low-dimensional_multi-mo.md)**

:   This paper proves that when the data distribution is supported on a union of $M$ low-dimensional linear subspaces (UoS) and the distribution within each subspace is sub-Gaussian, a kernel density-based score estimator allows the score-based diffusion sampler to achieve a 1-Wasserstein error of $\varepsilon$ with $\widetilde{O}(\varepsilon^{-(k\vee 2)})$ samples (where $k$ is the maximum intrinsic dimension). This results achieves the minimax optimal rate matching the intrinsic dimension under multi-modal settings without assumptions of smoothness, bounded density, or log-concavity, effectively circumventing the curse of dimensionality.

**[Direct 3D-Aware Object Insertion via Decomposed Visual Proxies](direct_3d-aware_object_insertion_via_decomposed_visual_proxies.md)**

:   DIRECT upgrades "object insertion" from 2D inpainting to a **pose-controllable** task: it first lifts a reference image into an interactive 3D proxy using an off-the-shelf image-to-3D model, renders dense geometric condition maps based on user-specified 6-DoF poses, and then injects "geometry, appearance, and context" conditions into the diffusion model via **decomposed independent pathways**. This ensures strict adherence to specified 3D poses while preserving reference appearance and achieving harmonious background integration, outperforming previous methods in both geometric controllability and visual quality.

**[DirectEdit: Step-Level Accurate Inversion for Flow-Based Image Editing](directedit_step-level_accurate_inversion_for_flow-based_image_editing.md)**

:   DirectEdit achieves "step-level accurate reconstruction" without increasing NFE by recording the latent residual $\Delta\mathbf{Z}_t$ for each step during Rectified Flow inversion and injecting it ahead of time in the forward path. This strictly aligns the reconstruction path with the inversion trajectory. Combined with MLLM+SAM multi-branch mask noise blending and attention Value injection, it significantly outperforms all existing training-free methods such as RF-Inversion, FireFlow, FTEdit, and DNAEdit, with comprehensive rankings of 4.0 (FLUX) / 2.43 (SD3.5) on PIE-Bench.

**[DiScoFormer: Plug-In Density and Score Estimation with Transformers](discoformer_plug-in_density_and_score_estimation_with_transformers.md)**

:   This paper proposes DiScoFormer, a Transformer that is equivariant to sample permutation and coordinate affine transformations. It maps any i.i.d. sample set to corresponding densities $f$ and scores $\nabla\log f$ in a single forward pass. Theoretically, it proves that self-attention with appropriate parametrization can exactly replicate normalized Gaussian KDE. Experimentally, it outperforms classical KDE across various distributions (GMM, Laplace, Student-$t$), widespread sample sizes, and a range of dimensions. It serves as a plug-and-play score oracle for Fisher information, entropy estimation, and solving Fokker–Planck-type PDEs.

**[Discrete Diffusion Samplers and Bridges: Off-Policy Algorithms and Applications in Latent Spaces](discrete_diffusion_samplers_and_bridges_off-policy_algorithms_and_applications_i.md)**

:   This work systematically migrates mature off-policy RL training techniques from continuous-space diffusion sampling (replay buffer, importance weighting, MCMC exploration) to discrete diffusion samplers for the first time. It further generalizes these to data-to-energy discrete Schrödinger bridges, significantly mitigating mode collapse on multi-modal distributions like Ising/Potts and discretized GMMs. Finally, it demonstrates data-free conditional image generation (posterior sampling) within the discrete latent space of VQ-VAEs.

**[Divide-and-Denoise: A Game-Theoretic Method for Fairly Composing Diffusion Models](divide-and-denoise_a_game-theoretic_method_for_fairly_composing_diffusion_models.md)**

:   The paper models the "collaborative sampling of multiple pre-trained diffusion models" as a **fair division game**. At each step, game theory is utilized to assign specific image regions to each model (allocation), ensuring that the composite denoising respects these assigned zones. This allows combinations such as "single-dog model + single-cat model" to generate images containing both a dog and a cat without competing for the same space, all without requiring training or weight sharing. GenEval %images improved from 58% (MultiDiffusion) to 88.5%.

**[Divide and Conquer: Reliable Multi-View Evidential Learning for Deepfake Detection](divide_and_conquer_reliable_multi-view_evidential_learning_for_deepfake_detectio.md)**

:   This paper proposes the DiCoME framework, which uses geometric orthogonal projection to force the decoupling of CLIP semantic features and forgery artifact features into two complementary "expert views." It then employs Dempster–Shafer evidence fusion to explicitly model "epistemic conflict" between these views to output reliable uncertainty. The framework improves average AUC from 0.923 to 0.939 (cross-dataset) and 0.976 (cross-manipulation) on deepfake detection benchmarks.

**[Efficient Learning of Deep State Space Models via Importance Smoothing](efficient_learning_of_deep_state_space_models_via_importance_smoothing.md)**

:   This paper proposes Parallel Variational Monte Carlo (PVMC), which utilizes prefix/suffix associative scans to compute the importance-weighted marginal smoothing distributions of Deep State Space Models (DSSM) within an $\mathcal{O}(\log N \times \log T)$ span. Supporting both supervised state estimation and generative modeling, it achieves approximately 10× speedup over the fastest differentiable SMC baselines while providing higher accuracy.

**[E²PO: Embedding-perturbed Exploration Preference Optimization for Flow Models](embedding-perturbed_exploration_preference_optimization_for_flow_models.md)**

:   Addressing the collapse of intra-group variance and the disappearance of training signals in group-based RL alignment (e.g., GRPO, DiffusionNFT) for flow models, E²PO injects a set of learned structured perturbations into the text embedding space to maintain discriminative variance. Combined with a noise-aware schedule and a reference-anchored batch strategy, it improves GenEval from 0.917 to 0.932 on SD3.5-M while significantly enhancing diversity.

**[End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer](end-to-end_autoregressive_image_generation_with_1d_semantic_tokenizer.md)**

:   EOSTok employs a single-stage end-to-end pipeline to jointly train a 1D ViT tokenizer and an autoregressive model. By utilizing a newly proposed APR (Autoregressive Prediction Reconstruction) loss, gradients from "next-token prediction" are effectively propagated back to the pixel space to prevent codebook collapse. Furthermore, "implicit alignment" is introduced to inject DINOv2 semantics into the 1D latent space without compromising the 1D autoregressive structure, achieving a SOTA FID of 1.48 (without guidance) on ImageNet 256.

**[Enhancing Membership Inference Attacks on Diffusion Models from a Frequency-Domain Perspective](enhancing_membership_inference_attacks_on_diffusion_models_from_a_frequency-doma.md)**

:   This paper analyzes failure modes of Membership Inference Attacks (MIA) on diffusion models from a frequency-domain perspective. It identifies that high-frequency content amplifies the standard deviation of scores for both member and hold-out samples, thereby diluting membership advantage. The authors propose a "high-frequency filter" module that requires no training and zero additional inference cost. By applying the same FFT low-pass processing to the predicted and target images before calculating reconstruction error, mainstream MIAs such as Naive/SecMI/PIA achieve performance gains of 4–11 percentage points in ASR/AUC/TPR@1%FPR on DDIM and Stable Diffusion (with TPR@1%FPR jumping from 6% to 41% in specific scenarios).

**[Envisioning Beyond the Few: Disentangled Semantics and Primitives for Few-Shot Atypical Layout-to-Image Generation](envisioning_beyond_the_few_disentangled_semantics_and_primitives_for_few-shot_at.md)**

:   To address "representation fragmentation" in layout-to-image generation within 5-shot atypical domains (aerial / underwater / extreme dark), the authors explicitly decompose the conditional representation of each category into global semantic anchors and local recomposable primitives. By using a saliency-aware loss to enforce foreground consistency, they reduce the Bootstrap FID on DIOR from 82.5 to 74.3 and improve mAP to 26.1.

**[Escaping Mode Collapse in LLM Generation via Geometric Regulation](escaping_mode_collapse_in_llm_generation_via_geometric_regulation.md)**

:   This paper reinterprets "mode collapse" (repetition, cycles, monotony) in long-form LLM generation from a dynamical systems perspective as "geometric collapse" of hidden state trajectories in representation space. It proposes RMR—a lightweight low-rank damping on the Transformer value cache—to suppress the most persistent self-reinforcing directions, maintaining stable, high-quality generation even in extremely low-entropy decoding regimes ($0.8$ nats/step).

**[Esoteric Language Models: A Family of Any-Order Diffusion LLMs](esoteric_language_models_a_family_of_any-order_diffusion_llms.md)**

:   Eso-LMs deeply integrate AR and Masked Diffusion at the loss, attention, and sampling levels. By utilizing a "causal-on-shuffled-sequence" denoising Transformer, it simultaneously supports parallel diffusion and left-to-right AR. This marks the **first time an MDM can utilize exact KV cache during the diffusion phase**, achieving 14–65× speedups over MDLM and 3–4× over BD3-LM on OWT long contexts, while reaching SOTA on the speed–quality Pareto frontier.

**[Evaluating the Representation Space of Diffusion Models via Self-Supervised Principles](evaluating_the_representation_space_of_diffusion_models_via_self-supervised_prin.md)**

:   This paper examines the internal representations of diffusion models through the lens of the two principles of self-supervised learning (SSL): "invariance + expansion." It proposes a label-free scalar metric, **ICR (Invariant Contamination Ratio)**, which predicts the optimal noise levels for downstream classification and provides early warnings for overfitting/memorization during training without requiring sampling or training classifiers.

**[EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)**

:   EvoGM reformulates "searching for task-vector merging coefficients $\bm{\lambda}$" from an evolutionary search with hand-crafted mutation operators into a learnable generative task. It uses a pair of cycle-consistent MLP generators to learn the distribution of high-performance regions from historical winner-loser pairs. By wrapping this in an outer "basis shift" mechanism to refresh the expert pool periodically, EvoGM outperforms the SOTA PSO-Merging by approximately 1.4% on 8 GLUE tasks and significantly leads on unseen tasks using Qwen2.5-1.5B with 10 models.

**[Exploring and Exploiting Stability in Latent Flow Matching](exploring_and_exploiting_stability_in_latent_flow_matching.md)**

:   This paper systematically characterizes "trajectory stability" in Latent Flow Matching (LFM)—demonstrating that pruning 75% of data, varying architecture sizes, or changing training seeds under the same noise seed produces nearly identical images. This property is leveraged into two practical algorithms: (1) balanced-clustering pruning that enables removing 50% of data on CelebA-HQ with slight FID improvements and 75% on ImageNet; (2) a Coarse-to-Fine two-stage generation that joins DiT-XL/2 (675M) and DiT-S/2 (33M), achieving a 2.15× inference speedup.

**[$f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data](f-trajectory_balance_a_loss_family_for_tuning_gflownets_generative_models_and_ll.md)**

:   This work generalizes the $\mathbb{KL}_{sq}$ proxy loss—which computes the "squared difference of log-probabilities" as seen in GFlowNet and Kimi—to the entire family of $f$-divergences. This results in a tunable family of mode-seeking $\leftrightarrow$ mode-covering losses where on-policy gradients equal the true $f$-divergence gradients and off-policy optimality remains consistent. Validations are conducted on synthetic grids, SynFlowNet molecule generation, diffusion model conditional sampling, and asynchronous LLM RL (GSM8k / MATH).

**[Finding DoRI: Discovery of Retained Images in Diffusion Models](finding_dori_discovery_of_retained_images_in_diffusion_models.md)**

:   The authors demonstrate via a simple adversarial text embedding optimization method (DoRI) that diffusion model memory mitigation schemes like NeMo or Wanda—which aim to "prune and locate memory neurons"—merely "hide" memories rather than truly erasing them. This is because memorization is not localized at the embedding, activation, or weight levels. They further propose an adversarial fine-tuning scheme to genuinely extract training samples from the model.

**[ForceForget: Reinforcement Concept Removal for Enhancing Safety in Text-to-Image Models](forceforget_reinforcement_concept_removal_for_enhancing_safety_in_text-to-image_.md)**

:   This work reformulates "erasing unsafe concepts" as a reinforcement learning reward optimization problem. It fine-tunes a diffusion model using a Concept Erasure Reward (CER)—composed of a safety reward and an alignment reward—paired with a "Safety Adapter" that modifies only a few tail text tokens. This approach thoroughly removes pornographic content while maximizing the preservation of benign semantics (especially person-related content) embedded in harmful prompts.

**[Forget-It-All: Multi-Concept Machine Unlearning via Concept-Aware Neuron Masking](forget-it-all_multi-concept_machine_unlearning_via_concept-aware_neuron_masking.md)**

:   This paper proposes FIA, a training-free multi-concept unlearning framework. By utilizing "contrastive concept saliency + spatio-temporal sparse selection," it locates concept-sensitive neurons for each target concept. When fusing multi-concept masks, it explicitly preserves "concept-agnostic neurons" that respond to multiple concepts simultaneously, pruning only concept-exclusive connections. On SD v1.5/v1.4, with a total sparsity rate of <0.3%, it achieves simultaneous unlearning of ten Imagenette classes (average forget accuracy 1.9%, overall score 86%), as well as multiple artistic styles and inappropriate content.

**[From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection](from_talking_to_singing_a_new_challenge_for_audio-visual_deepfake_detection.md)**

:   Targeting the "Singing Head" challenge—a difficult domain neglected by existing deepfake detectors—the authors construct the SHDF dataset to quantify the "Talking → Singing" domain shift. They propose the T-AVFD framework, which uses Alpha-CLIP with multi-granularity real/fake text contrastive learning to extract "semantic patterns of real faces." A differential weight module adaptively fuses lip-audio consistency and facial semantics. Trained solely on real talking videos, it generalizes to singing forgeries, improving SHDF AUC from the ~50% range to 80.2%.

**[GASS: Geometry-Aware Spherical Sampling for Disentangled Diversity Enhancement in Text-to-Image Generation](gass_geometry-aware_spherical_sampling_for_disentangled_diversity_enhancement_in.md)**

:   The authors project sample diversity under the same prompt onto the CLIP unit hypersphere, expanding the projection spread along the "text direction $\mathbf{e}_t$" and the "orthogonal principal residual direction $\mathbf{u}_{\text{ind}}$". This geometric expansion is transferred back to the diffusion/flow sampling trajectory via gradient optimization on the predicted clean image $\hat{x}_{0|t}$, enhancing both prompt-dependent (pose, composition) and prompt-independent (background, style) diversity in SD2.1 and SD3-M with minimal loss in quality or alignment.

**[DynaDiff: Generative Adaptation of Dynamics to Environmental Shifts via Weight-space Diffusion](generative_adaptation_of_dynamics_to_environmental_shifts_via_weight-space_diffu.md)**

:   DynaDiff reformulates the meta-learning problem of "training a predictor for a new environment" into a conditional sampling problem of "directly generating full network weights using a diffusion model." By utilizing a weight graph, functional consistency loss, and a dynamics-aware prompter, it achieves a 10.78% average RMSE reduction over strong baselines across four PDE systems.

**[Generative Visual Code Mobile World Models](generative_visual_code_mobile_world_models.md)**

:   The authors reformulate mobile GUI world modeling into a new paradigm where "VLMs generate renderable web code." They propose an automated data synthesis pipeline that rewrites policy trajectories into (image state, action) $\rightarrow$ (reasoning chain, next-state code) training samples. The resulting gWorld-8B/32B models achieve SOTA across six in/out-of-distribution benchmarks, improving baseline instruction accuracy by 27–46 percentage points and reducing rendering failure rates to $<1\%$.

**[GenExam: A Multidisciplinary Text-to-Image Exam](genexam_a_multidisciplinary_text-to-image_exam.md)**

:   GenExam adopts the "drawing exam" as the gold standard for measuring the integrated reasoning-understanding-generation capabilities of T2I models. By providing ground-truth images and fine-grained scoring points for 1000 questions across 10 disciplines, results reveal that even the strongest closed-source model, Nano Banana Pro, achieves only a 70.2% strict score, while most open-source T2I and unified MLLMs score below 3%.

**[Geometry-Aware Dataset Condensation for Diffusion Model Training](geometry-aware_dataset_condensation_for_diffusion_model_training.md)**

:   Addressing the limitation where existing dataset condensation methods are unsuitable for training diffusion models, this paper reformulates real subset selection as a **geometry-aware distribution alignment problem**. It defines the alignment target via one-sided Partial Optimal Transport (POT) combined with statistical regularization, and solves it through a two-stage discrete optimization (greedy + exchange). On ImageNet, training DiT/SiT with only 0.8% of the data achieves an FID significantly lower than the previous strongest baseline D2C (3.43 vs. 4.20 under a 10K budget).

**[Geometry-Aware Tabular Diffusion](geometry-aware_tabular_diffusion.md)**

:   The authors propose GATD (Geometry-Aware Tabular Diffusion), which explicitly incorporates "angles and lengths between column pairs" as geometric features into the denoising inputs and loss functions as auxiliary supervision signals. Using a small MLP with only 1/3.5 the parameters of TabDiff (and as low as 1/25 for classification tasks), GATD achieves wins in 8/10 Shape, 7/10 Trend, and 9/10 downstream utility metrics across 10 datasets. Furthermore, the same set of default hyperparameters can be directly transferred to GNN and Transformer denoisers, yielding improvements in 27/30 Shape and 25/30 Trend metrics.

**[Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion](geometry-based_schrödinger_bridges_for_trustworthy_multimodal_fusion.md)**

:   Ours proposes GMF: using Diffusion Schrödinger Bridge + Rectified Flow in latent space to estimate the "transport correction cost" for each modality (the squared initial velocity $\|v_\theta(z,0)\|^2$). This serves as a geometric reliability signal **decoupled from classifier confidence** to dynamically weight multimodal fusion, thereby breaking the circular dependency of "the model judging itself." It significantly outperforms confidence-based trustworthy fusion baselines under sensor noise and semantic conflict.

**[Gradient Preconditioning for Efficient and Reliable Reward-Guided Generation](gradient_preconditioning_for_efficient_and_reliable_reward-guided_generation.md)**

:   By projecting the reward gradient onto a "white Gaussian noise feasible set" characterized by block-wise $\ell_1/\ell_2$ norms in the DFT domain, the authors make test-time latent optimization for one-step generation models both fast and stable: reaching SOTA MPGR's Aesthetic Score on FLUX in only 30% of the wall-clock time and completely avoiding reward hacking.

**[GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)**

:   GUDA reformulates "group-wise training data attribution" as a counterfactual question: "How much would the model's log-likelihood for a given sample drop if a specific group were absent during training?" It uses machine unlearning to "erase a group" from the full model to approximate the counterfactual model obtained via Leave-One-Group-Out (LOGO) retraining. Using the difference in ELBO as the attribution score, GUDA proves more accurate than CLIP similarity and instance-level gradient attribution on CIFAR-10 and Stable Diffusion artistic style attribution, while being approximately 100x faster than LOGO retraining.

**[GuidedBridge: Training-freely Improving Bridge Models with Prior Guidance](guidedbridge_training-freely_improving_bridge_models_with_prior_guidance.md)**

:   Addressing diffusion bridge models (data-to-data generation), the paper proposes training-free Prior Guidance (PG): by perturbing a clean prior to construct a "weak prior," the model extrapolates between the denoising results of strong and weak priors to amplify prior utilization. Further incorporating a U-shaped Frequency-modulated PG (FMPG) and a CFG-FMPG cascaded framework, the method consistently improves the FID of pre-trained bridge models such as DDBM / DBIM on tasks like Edges→Handbags, DIODE, and ImageNet inpainting without additional training or increased NFE.

**[(HB-ARFM) History-Bootstrapped Flow Matching for Inverse Boiling Reconstruction](hb-arfm_history-bootstrapped_flow_matching_for_inverse_boiling_reconstruction.md)**

:   HB-ARFM solves the inverse problem of reconstructing multiphase boiling flow fields using "history-observation-guided" conditional flow matching. It bootstraps an initial latent state from a historical observation window and then advances the reconstruction autoregressively using the same conditional velocity field. Observing only interface geometry and velocity, it achieves the first spatiotemporally consistent reconstruction of complete temperature and velocity fields.

**[HoloFair: Unified T2I Fairness Evaluation and Fair-GRPO Debiasing](holofair_unified_t2i_fairness_evaluation_and_fair-grpo_debiasing.md)**

:   This paper constructs HoloFair, a unified fairness benchmark for T2I models (comprising a SpaFreq dual-stream attribute classifier + MGBI multi-attribute geometric mean metric). Based on this, it proposes Fair-GRPO: using log-ratio multi-attribute per-prompt rewards + KL-regularized GRPO, it improves MGBI from 0.5211 to 0.6772 (+29.9%) on SD3.5-Medium while maintaining or slightly improving image quality.

**[Hölder++: Improving the Quality-Coherence Trade-off in Multimodal VAEs](hölder_improving_the_quality-coherence_trade-off_in_multimodal_vaes.md)**

:   Addressing the long-standing "generation quality vs. cross-modal coherence" trade-off in multimodal VAEs, this paper proposes Hölder++. It introduces the first exact implementation of symmetric Hölder pooling ($\alpha=0.5$) as a modality aggregator, combined with shared-private subspace separation and top-down hierarchical inference. These architectural improvements push the quality-coherence Pareto frontier to SOTA across four benchmarks.

**[Image Restoration via Diffusion Models with Dynamic Resolution](image_restoration_via_diffusion_models_with_dynamic_resolution.md)**

:   SubDAPS / SubDAPS++ integrates pixel-space diffusion restoration methods (such as DPS and DAPS) into a "dynamic resolution diffusion" framework—sampling in $64^2 / 128^2$ subspaces during early stages and returning to the $256^2$ full resolution later. By replacing Langevin with Conjugate Gradient (CG), using threshold-based switching between stochastic/deterministic sampling, and adding a corrector step that requires no extra network evaluations, it outperforms most pixel and latent diffusion methods in 4 linear + 2 non-linear restoration tasks with faster inference.

**[Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)**

:   This paper treats the sampling trajectory of the reverse SDE in graph diffusion as a parametric curve on a Riemannian statistical manifold. Using the Fisher-Rao metric, a training-free Drift Variation Score (DVS) is derived to measure the local "information curvature" of the trajectory. The step size is adaptively scaled to ensure equal-length progression on the information manifold, achieving higher FCD/MMD fidelity with fewer steps in molecular (QM9/ZINC250k) and graph (Planar/SBM/Ego) generation.

**[Initialization is Half the Battle: Generating Diverse Images from a Guidance Potential Posterior](initialization_is_half_the_battle_generating_diverse_images_from_a_guidance_pote.md)**

:   This paper views "initial noise" as a random variable to be sampled from a posterior defined by a conditional guidance potential. It proposes DivIn: a method using one-step Langevin dynamics to push standard Gaussian noise toward "low-potential, flat" regions. This significantly alleviates mode collapse in diffusion and flow matching models with almost no added inference overhead and is orthogonally compatible with existing trajectory-based diversity methods.

**[Krause Synchronization Transformers](krause_synchronization_transformers.md)**

:   The authors incorporate the Krause bounded confidence consensus model into Transformers, replacing global softmax similarity with "distance-RBF + local window + top-k sparsity." They theoretically prove that this encourages multi-cluster synchronization rather than global collapse, achieving superior performance and 30%+ computational savings across ViT, autoregressive image generation, and LLMs.

**[Latent Diffusion Pretraining for Crystal Property Prediction](latent_diffusion_pretraining_for_crystal_property_prediction.md)**

:   CrysLDNet migrates "diffusion pretraining" from the raw crystal feature space to a smooth latent space learned by a VAE. This allows the PDDFormer encoder to learn more compact and symmetry-aware structural semantics on 380,000 unlabeled GNoME crystals. Downstream property predictions on JARVIS / MP show an average MAE reduction of 4.26% / 4.90% compared to strong supervised SOTA models, with even more significant advantages in low-data and experimental data correction scenarios.

**[Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis](learning_general_causal_structures_with_hidden_dynamic_process_for_climate_analy.md)**

:   This paper proposes CaDRe, which utilizes a structurally constrained temporal VAE to jointly identify the "causal graph between observed variables" and the "latent dynamic processes driving observations" within a single non-parametric framework. It provides identifiability theorems for recovering both from temporal data simultaneously. The theory is validated on synthetic data, while the model achieves causal graphs consistent with domain experts and competitive temperature prediction accuracy on CESM2 climate data.

**[Let EEG Models Learn EEG](let_eeg_models_learn_eeg.md)**

:   JET redefines multi-channel EEG generation as a "continuous trajectory on the neural manifold," utilizing Conditional Flow Matching with a standard Transformer to directly model raw waveforms. Coupled with three structural constraints characterizing EEG spectra, stationarity, and statistics, JET reduces TS-FID by over 40% against strong baselines across three clinical TUH benchmarks.

**[Linearizing Vision Transformer with Test-Time Training](linearizing_vision_transformer_with_test-time_training.md)**

:   The authors discover that a two-layer TTT inner model is structurally equivalent to Softmax attention (where Softmax can be viewed as a two-layer dynamic MLP). This facilitates direct weight inheritance of Q/K/V/MLP. By incorporating key Instance Normalization for shift-invariance and depthwise convolutions on Q/K for locality, they linearized and accelerated Stable Diffusion 3.5 by 1.32×–1.47× with only 1 hour of fine-tuning.

**[LithoGRPO: Fast Inverse Lithography via GRPO Reinforced Flow Matching](lithogrpo_fast_inverse_lithography_via_grpo_reinforced_flow_matching.md)**

:   LithoGRPO models lithography mask generation as a rectified flow conditioned on target layouts and fine-tunes it using GRPO reinforcement learning. This allows a single forward pass to simultaneously optimize L2/PVB (differentiable) and EPE/Shot (non-differentiable) metrics. With a 130×–490× accelerated fast shot-count algorithm, it improves the comprehensive rank from 5.6 to 4.3 on LithoBench, with a per-sample inference time of only 0.1 s.

**[Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation](local_hessian_spectral_filtering_for_robust_intrinsic_dimension_estimation.md)**

:   This paper proposes LHSD, which applies a Hill-type spectral filter to the log-density Hessian of a score model to retain only near-zero eigenvalues for counting tangent space dimensions. By leveraging Stochastic Lanczos Quadrature, it reduces the computational cost from $\mathcal{O}(D^3)$ to $\mathcal{O}(D)$, enabling stable estimation of Local Intrinsic Dimension (LID) in 3072-dimensional image spaces and diagnosing training sample memorization in diffusion models.

**[Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences](localizing_memorized_regions_in_diffusion_models_via_coordinate-wise_curvature_d.md)**

:   This paper characterizes "local memory in diffusion models" as **variance collapse (high curvature)** of log-density on specific coordinates. By computing the **coordinate-wise curvature difference** between a conditional model and an underfitted baseline (unconditional model or early checkpoint), the method subtracts "pseudo-memory" caused by the inherent low variance of the data manifold. This isolates **overfitting-driven memorized regions**, improving localization IoU on Stable Diffusion ground-truth masks from 0.75 (BE) to approximately 0.92.

**[MIRO: Multi-Reward Conditioned Pretraining for Simultaneously Improving T2I Quality and Efficiency](miro_multi-reward_conditioned_pretraining_improves_t2i_quality_and_efficiency.md)**

:   MIRO moves "alignment" from the RLHF post-training stage directly into pretraining: it assigns 7 reward scores (aesthetics, user preference, text-alignment, visual reasoning, scientific correctness, etc.) to each training image, enabling the Flow Matching model to learn $p(x|c, s)$. During inference, multi-reward CFG guides the generation toward high-reward regions. A 0.36B parameter model exceeds the 12B FLUX-dev on GenEval with 370× less training compute, and its single-sample inference quality surpasses the baseline performed 128 times.

**[Mitigating the Contractivity Trap in Diffusion ODEs via Stein Stabilization](mitigating_the_contractivity_trap_in_diffusion_odes_via_stein_stabilization.md)**

:   To address the issue in diffusion model Probability Flow ODE (PF-ODE) where "high-expressivity denoisers + aggressive step sizes destroy the contractivity stability certificate, leading to amplified errors and trajectory divergence" (named the **contractivity trap** by the authors), SteinDiff utilizes Stein's identity to transform the uncomputable "alignment with clean target" term into a computable divergence term. It derives a closed-form, reference-free, and training-free step-wise correction coefficient $\gamma_k$ for geometry-aware residual correction of solver candidates. SteinDiff significantly reduces the FID of large-step sampling on CIFAR-10 / ImageNet-64 / LSUN-Bedrooms (up to a 45.8% reduction).

**[OcclusionFormer: Arranging Z-Order for Layout-Grounded Image Generation](occlusionformer_arranging_z-order_for_layout-grounded_image_generation.md)**

:   Addressing texture entanglement and hierarchical confusion in overlapping regions of layout-to-image generation, the authors construct a large-scale dataset SA-Z with explicit Z-order and amodal annotations. They propose OcclusionFormer, which explicitly models occlusion priority via instance decoupling and volume rendering, and strengthens spatial consistency with a queried alignment loss. Occlusion-aware metrics on the OverLayBench Complex subset and the self-constructed SA-Z Eval significantly outperform strong baselines such as Eligen, Creatilayout, and InstanceAssemble.

**[Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition](offline_multi-agent_reinforcement_learning_via_sequential_score_decomposition.md)**

:   OMSD replaces the traditional "independent marginal regression" behavioral constraint in offline MARL with a "chain conditional decomposition + one conditional diffusion model per agent" approach. By regularizing each agent's policy conditioned on the actions already selected by prefix agents, it avoids Out-of-Distribution (OOD) mismatches caused by "aligned marginals but misaligned joints" under multimodal joint behavior distributions. It improves average returns by +33% to +74% over existing SOTA on multiple MPE and MaMuJoCo datasets.

**[Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs](offline_preference_optimization_for_rectified_flow_with_noise-tracked_pairs.md)**

:   This paper proposes PNAPO for Rectified Flow (RF) text-to-image models—an offline preference optimization framework that saves the "prior noise used during generation" alongside "winner/loser images" as sextuplets. By leveraging the RF linear trajectory hypothesis for trajectory estimation and dynamic regularization coefficient scheduling, it achieves higher performance on SD3-M/FLUX while reducing training compute to $1/12$ compared to Diffusion-DPO.

**[OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild](omniaid_decoupling_semantic_and_artifacts_for_universal_ai-generated_image_detec.md)**

:   OmniAID employs a decoupled MoE architecture consisting of "Semantic Experts + a Universal Artifact Expert" to learn two types of forgery cues—"content-related flaws" and "universal generation artifacts"—within a low-rank residual subspace derived from CLIP-ViT attention weight SVD. Coupled with the modern Mirage dataset, it achieves state-of-the-art average accuracies of 95.9%, 91.4%, and 88.4% on GenImage, Chameleon, and Mirage-Test benchmarks, respectively.

**[OMP: One-step Meanflow Policy with Directional Alignment](omp_one-step_meanflow_policy_with_directional_alignment.md)**

:   This paper addresses three theoretical "pathologies" exposed when applying the MeanFlow paradigm directly to robotic manipulation (spectral bias, gradient starvation in low-speed zones, and nested JVP memory explosion). It proposes OMP: a cosine-style directional alignment loss is used to "lock" the predicted mean velocity direction to the ground truth, and a Finite Difference DDE is utilized to approximate the Jacobian-Vector Product (JVP), decoupling forward and backward passes. This allows the one-step (NFE=1) generative policy to achieve a 6.8ms latency while outperforming MP1 by an average of 3.4% on Adroit/Meta-World and by 10.6% on Meta-World "Very Hard" tasks.

**[Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization](order_within_chaos_capturing_intrinsic_energy_anomalies_for_ai-manipulated_image.md)**

:   Starting from the spectral bias of diffusion models, this paper theoretically proves that the local Gibbs energy of diffusion-generated regions is inevitably lower than that of real imaging regions. Accordingly, a LAD (Local Adjacency Discrepancy) energy map is constructed as an intrinsic forensic fingerprint. A lightweight adapter then injects LAD cues into SAM to achieve pixel-level forgery localization. Coupled with the EditStream multi-agent engine that automatically pulls the latest editing models from HuggingFace to continuously refresh training data, the method improves the average IoU from the previous SOTA of ~0.25 to 0.46 across 7 AI editing datasets.

**[Orthogonal Concept Erasure for Diffusion Models](orthogonal_concept_erasure_for_diffusion_models.md)**

:   This paper reformulates "additive parameter editing" concept erasure (e.g., UCE/SPEED) in T2I diffusion models as a multiplicative "layered orthogonal rotation $W^* = QW$". By combining a subspace-level erasure target with a Procrustes closed-form solution, it calculates $Q$ in a single step—erasing 100 celebrity concepts in 4.3 seconds with near-zero damage to non-target concepts.

**[Pareto-Guided Optimal Transport for Multi-Reward Alignment](pareto-guided_optimal_transport_for_multi-reward_alignment.md)**

:   PG-OT transforms "multi-reward T2I alignment" from a "weighted global summation" into "constructing a Pareto frontier for each individual prompt and employing Sinkhorn Optimal Transport to move dominated samples toward that frontier." By introducing Joint Domination Rate (JDR) and Joint Collapse Rate (JCR), the method exposes reward hacking hidden by average scores, achieving a 47.98% $\text{JDR}_2$ on Parti-Prompts (an 11% improvement over strong baselines) and a human win rate of nearly 80%.

**[Path-Coupled Bellman Flows for Distributional Reinforcement Learning](path-coupled_bellman_flows_for_distributional_reinforcement_learning.md)**

:   The geometric "affine transport" property of the distributional Bellman equation is explicitly woven into the flow matching path: a shared base noise drives the paths of both the current and successor states simultaneously, while a $\lambda$ control variate balances bias and variance. This results in a stable distributional critic that is consistent with both the source distribution and the Bellman endpoints.

**[PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World](physforge_generating_physics-grounded_3d_assets_for_interactive_virtual_world.md)**

:   Interactive 3D object creation is reframed as a two-stage "physical planning followed by physical generation" problem. A VLM acts as a physical architect to generate a "Hierarchical Physical Blueprint" containing hierarchy, materials, and kinematic constraints. Subsequently, a diffusion model utilizes KineVoxel Injection to jointly denoise articulation parameters and geometric voxels. Combined with the PhysDB dataset—comprising 150k assets with four-tier annotations—this approach achieves the first generation of 3D assets from a single view that are directly graspable, pushable, and articulatable within physics engines.

**[PolyFlow: Safe and Efficient Polytope-Constrained Flow Matching with Constraint Embedding and Projection-free Update](polyflow_safe_and_efficient_polytope-constrained_flow_matching_with_constraint_e.md)**

:   PolyFlow "welds" hard polytope constraint satisfaction directly into the network architecture and flow definition of Flow Matching models—using discrete-time flows to eliminate numerical integration errors and a Frank-Wolfe-style "ray shooting to boundary + learnable step size" to replace expensive projection solvers. This achieves **zero constraint violation** in planning and control tasks while reducing inference latency by one to two orders of magnitude.

**[Position: Adopting AI in Practice Does Not Guarantee the Productivity Boost](position_adopting_ai_in_practice_does_not_guarantee_the_productivity_boost.md)**

:   This position paper argues that "introducing AI into an organization does not automatically equate to productivity gains." It identifies five human and environmental moderators ignored by traditional economic models (personnel composition, individual baseline capability, learning curves, equitable use incentives, and goal flexibility). By incorporating organizational effectiveness $\Omega$, capability adjustment $\phi(z,\kappa_i)$, learning curves $\lambda_i(\tau)$, and effective automation thresholds $\tilde N_{IT}$ into the Gries-Naudé (2022) partial equilibrium model, the authors derive a revised production function that explains the massive output gap between organizations with identical AI investments.

**[Position: AI Evaluations Should be Grounded on a Theory of Capability](position_ai_evaluations_should_be_grounded_on_a_theory_of_capability.md)**

:   The authors argue that "benchmark score = capability" is an **implicit inference** rather than a direct measurement. They advocate for explicitly modeling AI evaluation as a statistical inference task and suggest utilizing four psychometric capability theories (CTT/IRT/CDM/BNSM) as templates, introducing an "Evaluation Card" for evaluators to justify their modeling assumptions.

**[Principled RL for Flow Matching Emerges from the Chunk-level Policy Optimization](principled_rl_for_flow_matching_emerges_from_the_chunk-level_policy_optimization.md)**

:   GCPO transitions the step-level optimization in flow matching post-training—where GRPO assigns the "same final reward as advantage to every step"—into "chunk-level" optimization. By adaptively grouping consecutive steps into chunks based on flow matching's own temporal dynamics $L1_{rel}(x,t)$ and utilizing normalized chunk-level importance ratios $r^i_j$ for policy updates, it smooths out erroneous gradients caused by the "final success $\neq$ step-wise optimal" mismatch. This achieves a relative gain of up to 43% over GRPO on HPSv3, ImageReward, GenEval, and DPG.

**[Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution](q-dit4sr_exploration_of_detail-preserving_diffusion_transformer_quantization_for.md)**

:   This paper introduces Q-DiT4SR, the first PTQ framework designed for DiT-based Real-World Image Super-Resolution (Real-ISR). It preserves high-frequency details through a "global low-rank + local block-wise rank-1" hierarchical SVD decomposition (H-SVD). Furthermore, it proposes data-free inter-layer weight bit-width allocation (VaSMP) based on rate-distortion theory and dynamic programming-based timestep activation bit-width scheduling (VaTMP). Q-DiT4SR achieves SOTA performance under ultra-low bit settings of W4A6 / W4A4, compressing the model by 5.8× and reducing computations by 6.14×.

**[Quantifying Error Propagation and Model Collapse in Diffusion Models](quantifying_error_propagation_and_model_collapse_in_diffusion_models.md)**

:   This work provides the first set of paired upper and lower bounds for the phenomenon of "model collapse induced by recursive training with synthetic data" in score-based diffusion models. Specifically, it establishes that the single-generation divergence satisfies $\chi^2(\hat p^{i+1}\|q_i)\asymp \varepsilon_{\star,i}^2$, and the multi-generation cumulative divergence $D_N$ is a weighted sum of previous score error energies decaying geometrically by $(1-\alpha)^{2m}$. This effectively transforms the empirical observation that "adding fresh data alleviates collapse" into a precise decay law.

**[RAIGen: Rare Attribute Identification in Text-to-Image Generative Models](raigen_rare_attribute_identification_in_text-to-image_generative_models.md)**

:   RAIGen utilizes Matryoshka Sparse Autoencoders (MSAE) to decompose the bottleneck representations of T2I diffusion models into interpretable neurons. By applying a combined score of "activation rarity $\times$ CLIP semantic distinctiveness," it identifies minority neurons that are "internally encoded but rarely appear in generation," extending bias auditing from "predefined categories" and "salient majority patterns" to label-free rare attribute discovery.

**[Rao-Blackwellized Score Matching on Manifolds](rao-blackwellized_score_matching_on_manifolds.md)**

:   When the data distribution lies on an embedded manifold $M\subset\mathbb{R}^D$, the tangential target learned by ambient Gaussian Denoising Score Matching (DSM) contains normal noise channels with variance diverging as $d/\sigma^2$. This paper proves that a single Rao-Blackwell conditioning step on the nearest-point projection $\pi(X)$ cleanly removes this singular channel and expands the remaining target precisely as "intrinsic Riemannian score + $\sigma^2$-order Tweedie correction + $\sigma^2$-order Weingarten/Ricci extrinsic curvature correction."

**[Recovering Hidden Reward in Diffusion-Based Policies](recovering_hidden_reward_in_diffusion-based_policies.md)**

:   EnergyFlow explicitly parameterizes the score field of a diffusion policy as the negative gradient of a scalar energy function. It proves that under maximum-entropy optimality, the score equals the gradient of the soft Q-function, thereby providing a "free" scalar signal for downstream RL reward shaping without adversarial optimization, while the conservative field constraint improves OOD generalization.

**[Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment](restoring_initial_noise_sensitivity_in_text-to-image_distillation_via_geometric_.md)**

:   This paper points out that existing T2I diffusion distillation methods, which focus only on "pointwise output alignment," lead to the collapse of the student model's sensitivity to initial noise. It proposes GAD: using a finite-difference approximation of the Jacobian-Vector Product (JVP) under a pair of perturbed inputs to force the student to match the teacher's directional response to noise perturbations, thereby restoring layout controllability and generation diversity without sacrificing fidelity.

**[Rethinking FID Through the Geometry of the Reference Dataset](rethinking_fid_through_the_geometry_of_the_reference_dataset.md)**

:   This paper demonstrates that the "lower-is-better" assumption of FID systematically fails across different reference datasets. By introducing two geometric descriptors—distribution density $\langle -\log d_k\rangle$ and effective rank $\mathrm{erank}(A)$—the authors use hierarchical linear modeling to prove these descriptors explain ~70% of the cross-dataset variance in the "sample quality → FID" slope, providing the first quantitative attribution of FID's fragility to the reference set itself.

**[Riemannian MeanFlow for One-Step Generation on Manifolds](riemannian_meanflow_for_one-step_generation_on_manifolds.md)**

:   The paper extends the "average velocity one-step generation" of MeanFlow to Riemannian manifolds. By using **parallel transport** to move instantaneous velocities from different tangent spaces to a common one before averaging, it defines the average velocity on manifolds and derives the **Riemannian MeanFlow Identity**. It employs intrinsic training via log maps in the common tangent space (avoiding trajectory simulation and Christoffel symbols), decomposes the objective into two terms, and resolves gradient conflicts using PCGrad. The method achieves one-step sampling quality comparable to the strongest baselines on spheres, tori, SO(3), and SE(3), while significantly reducing sampling costs.

**[RT-Lynx: Putting the GEMM Sparsity In a Right Way for Diffusion Models](rt-lynx_putting_the_gemm_sparsity_in_a_right_way_for_diffusion_models.md)**

:   Authors find that DiT **activations** are more naturally sparse than **weights** (only 5–10% of channels per token are activated). They migrate 2:4 semi-structured sparsity from the weight side to the activation side, utilizing norm scaling, LoRA residual compensation, and selective layer skipping to recover quality. A fused CUDA pipeline for "Online Top-K + Sparse GEMM" is implemented, achieving an average 1.55× speedup for linear layers on Qwen-Image / FLUX / Z-Image without FID/IR degradation.

**[SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders](saemnesia_erasing_concepts_in_diffusion_models_with_supervised_sparse_autoencode.md)**

:   By incorporating a supervised "concept-latent" assignment loss during the Sparse Autoencoder (SAE) training phase, this work enforces each target concept to concentrate into a single neuron (feature centralization). This reduces concept erasure in diffusion models from a two-dimensional hyperparameter search of "searching multiple neurons + tuning intensity" to "tuning a single multiplier." On UnlearnCanvas, it achieves an average improvement of 9.22 points over the SOTA SAeUron, reduces hyperparameter search costs by 96.67%, and demonstrates higher robustness against adversarial attacks.

**[Saving Foundation Flow-Matching Priors for Inverse Problems](saving_foundation_flow-matching_priors_for_inverse_problems.md)**

:   Addressing the observation that foundation flow-matching (FM) models like Stable Diffusion / Flux significantly underperform compared to domain-specific or even untrained priors in solving inverse problems, the authors propose FMPlug. By using a time-learnable warm-start guided by approximate samples and a sharp Gaussian shell constraint, FMPlug forces the latent variables of the foundation FM back onto the thin shell it truly "understands," significantly restoring its capability as an inverse problem prior.

**[Scalable GANs with Transformers](scalable_gans_with_transformers.md)**

:   This paper proposes GAT (Generative Adversarial Transformers), a scalable GAN framework constructed with pure Transformer generators and discriminators in the VAE latent space. By activating early generator layers through Multi-level Noise-perturbed Guidance (MNG) and stabilizing large-scale training with width-aware learning rate scaling, GAT-XL/2 achieves a single-step SOTA FID of 2.18 on ImageNet-256 class-conditional generation in only 60 epochs, using $4\times$ fewer epochs than comparable 1-NFE diffusion/flow baselines.

**[SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes](scenesmith_agentic_generation_of_simulation-ready_indoor_scenes.md)**

:   SceneSmith utilizes a designer-critic-orchestrator VLM agent triangle to construct indoor scenes layer-by-layer on a hierarchical tree of "layout $\rightarrow$ furniture $\rightarrow$ small objects." It deeply couples text-to-3D generation, articulated object retrieval, and physical property estimation into the agent toolchain. Generating directly from a single natural language prompt, it produces dense, actionable environments ready for physical simulators. Each room averages 71 objects (compared to 11–23 in baselines), with an inter-object collision rate $< 2\%$ and a gravity-based stability rate of $96\%$, significantly outperforming all prior methods.

**[Self-Prompting Diffusion Transformer for Open-Vocabulary Scene Text Editing via In-Context Learning](self-prompting_diffusion_transformer_for_open-vocabulary_scene_text_editing_via_.md)**

:   Ours proposes a self-prompting scene text editing method based on FLUX-Fill (MM-DiT). It directly crops style prompts from the original image and renders glyph prompts using Pillow. These prompts are concatenated channel-wise with the masked image and fed into the diffusion backbone. Following this, cooldown training is performed on 4,000 high-quality paired images generated by Nano Banana Pro. This approach achieves open-vocabulary text replacement while maintaining consistency with original styles across 13 languages.

**[Semantic-Aware Motion Encoding for Topology-Agnostic Character Animation](semantic-aware_motion_encoding_for_topology-agnostic_character_animation.md)**

:   SATA utilizes joint semantic labels generated by MLLMs for FiLM-style feature modulation, combined with spatio-temporal interleaved graph auto-encoders, to compress BVH motions of arbitrary skeletal topologies into a shared latent space. This achieves high-fidelity reconstruction and zero-shot cross-species motion retargeting without paired data.

**[Semantic Granularity Navigation in Image Editing](semantic_granularity_navigation_in_image_editing.md)**

:   NaviEdit decouples the implicit coupling where "model scale coordinate = editing progress clock" in diffusion/flow editors. Under a fixed step budget, it uses a training-free inference-time controller to concentrate computational effort on the density within an effective scale window rather than expanding the range into high-noise regions, improving both background fidelity and semantic consistency across PIE-Bench / ImgEdit-Bench and multiple flow backbones.

**[Shifting the Breaking Point of Flow Matching for Multi-Instance Editing](shifting_the_breaking_point_of_flow_matching_for_multi-instance_editing.md)**

:   To address the persistent issue of "attribute leakage" during simultaneous multi-instance editing in MMDiT-based models (e.g., FLUX.1 Kontext) utilizing Rectified Flow Matching, this paper proposes Instance-Disentangled Attention (IDAttn). By applying structured masks to joint attention, each editing instruction is bound to its corresponding bounding box. Combined with a hierarchical disentanglement/harmonization schedule and efficient independent multi-prompt encoding, the method enables $N$ non-interfering edits in a single forward pass. It significantly outperforms multi-turn and concatenation baselines on the newly proposed Infographic text editing benchmark.

**[Simple Approximation and Derivative Free Inference-Time Scaling for Diffusion Models via Sequential Monte Carlo on Path Measures](simple_approximation_and_derivative_free_inference-time_scaling_for_diffusion_mo.md)**

:   The authors upgrade inference-time reward guidance for diffusion models from "particle-space SMC + high-order derivatives" to "path-space SMC + Girsanov likelihood ratios," resulting in the URGE algorithm. Each trajectory only requires a first-order gradient of the guidance $G$ and an accumulated simple Itô term as weight, completely eliminating the need for derivatives of the reward $r$, the Hessian, or score estimation. It matches or exceeds FK-Corrector / AFDPS / FK-Steering on GMM, inverse problems, and text-to-image tasks.

**[Skipping the Zeros in Diffusion Models for Sparse Data Generation](skipping_the_zeros_in_diffusion_models_for_sparse_data_generation.md)**

:   SED transforms diffusion models from "full dense denoising across all dimensions" to "diffusion only on non-zero dimensions + autoregressive decoding of dimension-value pairs," making the computational cost nearly constant relative to the number of non-zeros instead of linear with dimension, while strictly preserving the semantic information of "explicit zeros" in scientific data.

**[SpatialReward: Bridging the Perception Gap in Online RL for Image Editing via Explicit Spatial Reasoning](spatialreward_bridging_the_perception_gap_in_online_rl_for_image_editing_via_exp.md)**

:   The authors identify an "attention collapse" issue in MLLM-based editing reward models—where the model focuses on sink tokens rather than comparing the original and edited images—and propose SpatialReward. It directs an 8B model to first predict bounding boxes for edited regions and then use these box tokens as anchors for interleaved cross-image reasoning. Combined with a 260K-sample spatial-aware dataset and a two-stage GRPO training process, it achieves SOTA on three reward benchmarks and improves the GEdit-Bench score of OmniGen2 by +0.90 (double the improvement of GPT-4.1).

**[Spectral Guidance for Flexible and Efficient Control of Diffusion Models](spectral_guidance_for_flexible_and_efficient_control_of_diffusion_models.md)**

:   This paper proposes Spectral Guidance: by self-supervising the learning of the left singular functions of the conditional expectation operator in the diffusion process, arbitrary guidance signals (labels / CLIP / masks) are projected onto a set of spectral bases aligned with diffusion dynamics. This bypasses denoiser backpropagation, achieving a 37 percentage point accuracy improvement over the strongest training-free baseline on CIFAR-10 while being 4x faster in sampling.

**[Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation](speculative_coupled_decoding_for_training-free_lossless_acceleration_of_autoregr.md)**

:   This paper identifies the root cause of limited acceleration in Speculative Jacobi Decoding (SJD) for autoregressive visual generation: independent sampling of draft tokens between successive iterations leads to a collision probability near zero. By replacing independent sampling with Maximal or Gumbel Coupling (a one-line modification with zero additional training), image generation is accelerated by up to $4.2\times$ and video generation by up to $13.6\times$, while strictly maintaining the output distribution consistency with original AR decoding.

**[Stable Velocity: A Variance Perspective on Flow Matching](stable_velocity_a_variance_perspective_on_flow_matching.md)**

:   This paper re-examines flow matching from the overlooked perspective of "conditional velocity variance." It discovers that training trajectories naturally split into a high-variance zone near the prior and a low-variance zone near the data. Based on this, a unified framework, Stable Velocity, is proposed, featuring an unbiased multi-sample variance reduction loss (StableVM), a VA-REPA module that enables representation alignment only in low-variance zones, and a training-free sampler (StableVS) that utilizes closed-form solutions in low-variance zones. The method achieves improved training efficiency and >2× sampling acceleration on ImageNet 256 and SD3.5/Flux/Qwen-Image/Wan2.2.

**[Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models](stage-wise_distortion-perception_traversal_in_zero-shot_inverse_problems_with_di.md)**

:   A two-stage framework, MAP-RPS, is proposed: it first uses the diffusion model's score for Maximum A Posteriori (MAP) estimation to approach the Minimum Mean Square Error (MMSE) solution (low-distortion anchor), then re-noises the MAP result to time $t_0$ followed by posterior sampling (sliding along the D-P curve towards high perceptual quality). A single pre-trained diffusion model enables flexible traversal of the distortion-perception trade-off during inference and achieves SOTA multi-task performance on MS-COCO when extended to latent diffusion.

**[STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack](stare_step-wise_temporal_alignment_and_red-teaming_engine_for_multi-modal_toxici.md)**

:   This paper treats the entire denoising trajectory of T2I models as the "attack surface" for VLM red-teaming. By utilizing a hierarchical RL framework (STARE) comprising a high-level prompt editor and low-level GRPO fine-tuning for rectified-flow models, the authors not only improve the attack success rate by 68% over SOTA but also reveal a novel phenomenon—Optimization-Induced Phase Alignment. This phenomenon shows that adversarial optimization automatically binds "conceptual toxicity" to early denoising stages and "detailed toxicity" to later stages, transforming a chaotic toxicity formation process into predictable "vulnerability windows."

**[Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](support-proximity_augmented_diffusion_estimation_for_offline_black-box_optimizat.md)**

:   SPADE replaces traditional regression surrogates with a conditional diffusion model to model $p(y\mid\boldsymbol{x})$. By incorporating "mean/ranking calibration" and "kNN support regularization (mean shrinkage + variance expansion)," it implicitly injects data priors into the surrogate, enabling offline black-box optimization to consistently achieve SOTA performance on Design-Bench and LLM data mixture tasks.

**[SURF: Separation via Unsupervised Remixing Flow](surf_separation_via_unsupervised_remixing_flow.md)**

:   SURF combines the supervised flow matching framework FLOSS with the unsupervised ReMixIT / Self-Remixing teacher-student remixing training strategy. This allows a generative flow matching separator to be trained **entirely from mixture observations** (without any clean source samples). It nearly matches the performance of supervised flows on MNIST/CIFAR10 image separation and LibriSpeech / FUSS audio separation, establishing a new unsupervised SOTA.

**[SURGE: Approximation and Training Free Particle Filter for Diffusion Surrogate](surge_approximation_and_training_free_particle_filter_for_diffusion_surrogate.md)**

:   SURGE treats the guided sampling of diffusion surrogate models as a biased distribution on path measures. It utilizes the Girsanov formula to calculate importance weights for SMC resampling, thereby obtaining an approximation-free data assimilation filter for diffusion surrogates without retraining or approximating the Doob $h$-transform. It consistently outperforms BPF/EnKF/SDA/FlowDAS on Lorenz, Navier-Stokes, and SEVIR weather forecasting tasks.

**[Temporal Difference Learning for Diffusion Models](temporal_difference_learning_for_diffusion_models.md)**

:   The paper reformulates the diffusion denoising process as a Markov Reward Process (MRP) and treats training as policy evaluation in reinforcement learning. It proposes a Temporal Difference (TD) objective that enforces the model's "multi-step drift" along the denoising trajectory to match the true diffusion drift. As a plug-and-play regularizer added to baseline losses like EDM or Consistency Training, it significantly improves FID, particularly in few-step sampling (low NFE) scenarios.

**[The Coupling Within: Flow Matching via Distilled Normalizing Flows](the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)**

:   This paper introduces Normalized Flow Matching (NFM), which utilizes the "accurate deterministic data-to-noise bijection" produced by a pre-trained TarFlow (an autoregressive normalizing flow) as the noise-data coupling for Flow Matching. This advances FM's convergence speed and low-step FID to new levels while significantly exceeding the inference speed of the teacher NF model.

**[The Latent Color Subspace: Emergent Order in High-Dimensional Chaos](the_latent_color_subspace_emergent_order_in_high-dimensional_chaos.md)**

:   The authors discover that "color" occupies only a three-dimensional subspace (Latent Color Subspace, LCS) within the VAE latent space of FLUX.1. Its geometry closely mirrors the bicone of the HSL color model. Based on this, they propose a **completely training-free, pure closed-form latent space transformation** method that can both "read out" the emergent colors during generation and precisely modify specific objects to a target color.

**[Threshold-Guided Optimization for Visual Generative Models](threshold-guided_optimization_for_visual_generative_models.md)**

:   The authors dismantle the paired preference assumption of DPO, proving that the KL-regularized optimal policy inherently compares the reward of each sample to an uncomputable instance-dependent baseline $\tau^*(x)=\beta\log Z(x)$. Consequently, they replace it with a global threshold $\tau$ estimated from score quantiles and incorporate a confidence weight proportional to $|s-\tau|$. This enables stable alignment for Diffusion models and MaskGIT using only scalar scores (without paired preferences), consistently outperforming Diffusion-DPO, KTO, and DSPO across five reward models and three test sets.

**[Timestep Rescheduling in Diffusion Inversion](timestep_rescheduling_in_diffusion_inversion.md)**

:   The authors discover that diffusion inversion error strongly depends on timestep size and follows a "high at both ends, low in the middle" parabolic distribution across timestep indices. They propose TRDI, a training-free, zero-overhead non-uniform timestep scheduler. TRDI first performs global timestep stretching and subsequently employs dynamic programming for local rearrangement, concentrating computational resources on segments with higher errors. It serves as a plug-and-play plugin to consistently improve reconstruction and editing precision across various inversion methods.

**[Transferable Multi-Bit Watermarking Across Frozen Diffusion Models via Latent Consistency Bridges](transferable_multi-bit_watermarking_across_frozen_diffusion_models_via_latent_co.md)**

:   DiffMark continuously injects a learned latent space perturbation $\delta$ into each denoising step of a frozen diffusion model, allowing the watermark signal to accumulate in the final latent variable $z_0$. By utilizing a Latent Consistency Model (LCM) as a differentiable training bridge to bypass the backpropagation of 50 DDIM steps, the scheme achieves a 64-bit decoding in 16.4 ms via a single forward pass, while remaining plug-and-play across models without retraining.

**[UnHype: CLIP-Guided Hypernetworks for Dynamic LoRA Unlearning](unhype_clip-guided_hypernetworks_for_dynamic_lora_unlearning.md)**

:   UnHype utilizes a hypernetwork with CLIP text embeddings as input to dynamically generate LoRA weights at inference time—generating LoRAs that suppress a concept when encountered while generating near-zero LoRAs for normal concepts. This transforms static unlearning (training one LoRA per concept) into amortized unlearning (on-the-fly adapter generation), supporting both Stable Diffusion and Flux.

**[Unified Masked Diffusion Models with Diverse Generation Orders](unifying_masked_diffusion_models_with_various_generation_orders_and_beyond.md)**

:   This paper proposes a unified framework OeMDM and its learnable version LoMDM—unifying random masking, autoregressive, and block diffusion models under a single NELBO by explicitly modeling "velocity" (generation priority), enabling joint learning of generation order and the diffusion backbone from scratch.

**[Forgetting in Diffusion Models: A Unified Framework via KL Divergence and Likelihood Constraints](unlearning_in_diffusion_models_a_unified_framework_with_kl_divergence_and_likeli.md)**

:   This paper proposes a unified constrained optimization framework that formalizes machine unlearning in diffusion models as minimizing the deviation from a pre-trained model subject to explicit separation conditions from the unlearning distribution. Through three constraint forms (Reverse KL, Forward KL, and Likelihood constraints), it uniformly handles concept and data unlearning while proving strong duality.

**[ViewMask-1-to-3: Multi-View Consistent Image Generation via Multimodal Discrete Diffusion Models](viewmask-1-to-3_multi-view_consistent_image_generation_via_multimodal_discrete_d.md)**

:   By utilizing discrete diffusion models and visual tokenization, multi-view generation is reformulated as a discrete sequence prediction task. A simple random masking strategy combined with self-attention naturally achieves cross-view consistency, significantly outperforming continuous diffusion methods.

**[Visual Implicit Autoregressive Modeling](visual_implicit_autoregressive_modeling.md)**

:   This paper integrates Deep Equilibrium (DEQ) implicit fixed-point layers into the next-scale autoregressive framework of VAR. By utilizing Stochastic Jacobian-Free Backpropagation to achieve constant-memory training, the authors compress the 2 billion parameters of VAR-d30 to 770 million. At inference, the number of iterations per scale becomes a "tunable knob"—maintaining an FID of 2.16 and sFID of 8.07 on ImageNet-256, while reducing the peak memory on a single 4090 from 19.24GB to 8.53GB and increasing throughput from 15.16 to 32.08 img/s.

**[Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding](watch_your_step_information_injection_in_diffusion_models_via_shadow_timestep_em.md)**

:   This paper reveals that the long-overlooked "timestep embedding" in diffusion models serves as an unoccupied information side channel. By extending the training timestep range to a "shadow interval" (shadow timestep) and binding an alternative data distribution to it, the same diffusion model can generate normal images in the explicit interval and "hidden" images in the shadow interval without changing the scheduler interface. This can be used for both covert backdoor attacks and model watermark verification. The paper also provides a theoretical analysis of mutual coherence based on sinusoidal position encoding to explain why two disjoint intervals can carry independent information.

**[Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance](weak_diffusion_priors_can_still_achieve_strong_inverse-problem_performance.md)**

:   The paper discovers that diffusion model priors with low fidelity or domain mismatch can still achieve robust performance in information-rich inverse problems. This seemingly contradictory phenomenon is explained through Bayesian consistency theory and local correlation analysis, providing explicit conditions for when weak priors remain effective.

**[When Preference Labels Fall Short: Aligning Diffusion Models from Real Data](when_preference_labels_fall_short_aligning_diffusion_models_from_real_data.md)**

:   This paper argues that preference labels consisting of generated images tend to guide models toward "relatively better but still flawed" samples. It proposes automatically constructing preference signals using real images and their controllable degraded versions. By using only 512 pairs of samples, it aligns SD-1.5 and SD-3.5-M, achieving performance that is comparable to or supplements Diffusion-DPO / FlowGRPO.

**[WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation](wise_a_world_knowledge-informed_semantic_evaluation_for_text-to-image_generation.md)**

:   WISE constructs a text-to-image evaluation benchmark containing 1000 knowledge-dense prompts. It examines whether models can transform implicit semantics—such as cultural common sense, spatio-temporal reasoning, and natural science knowledge—into correct visual content. The study reveals significant shortcomings in world knowledge generation for existing T2I and unified multimodal models.

**[You Don't Need All That Attention: Surgical Memorization Mitigation in Text-to-Image Diffusion Models](you_dont_need_all_that_attention_surgical_memorization_mitigation_in_text-to-ima.md)**

:   This paper proposes GUARD, an inference-time framework for mitigating memorization in text-to-image diffusion models. By introducing "repulsion" from the original memorized prompt and "attraction" toward a safe conditional prediction into the standard classifier-free guidance, and instantiating the positive target through dynamic cross-attention spike detection and attenuation, GUARD reduces the replication of training images while maintaining image quality and prompt alignment.

**[Zeroth-Order Non-Log-Concave Sampling with Variance Reduction and Applications to Inverse Problems](zeroth-order_non-log-concave_sampling_with_variance_reduction_and_applications_t.md)**

:   This paper proposes a zeroth-order Langevin sampling method with variance reduction, replacing the $O(d)$ function queries per step with intermittent large-batch estimates and recursive small-batch updates. It extends the method to ZO-APMC, utilizing pre-trained score-based priors for posterior sampling with convergence guarantees in black-box inverse problems where only forward model queries are available.
