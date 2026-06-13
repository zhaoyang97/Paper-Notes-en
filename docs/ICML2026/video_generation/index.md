---
title: >-
  ICML2026 Video Generation Papers · 28 Notes
description: >-
  28 ICML2026 papers in the Video Generation area, covering Video Generation, Diffusion Models, Model Compression and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ICML2026"
  - "Video Generation"
  - "AI paper notes"
  - "paper summaries"
  - "Diffusion Models"
  - "Model Compression"
item_list:
  - u: "aad-1_asymmetric_adversarial_distillation_for_one-step_autoregressive_video_gene/"
    t: "AAD-1: Asymmetric Adversarial Distillation for One-Step Autoregressive Video Generation"
  - u: "attention_sparsity_is_input-stable_training-free_sparse_attention_for_video_gene/"
    t: "Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering"
  - u: "camgeo_sparse_camera-conditioned_image-to-video_generation_with_3d_geometry_prio/"
    t: "CamGeo: Sparse Camera-Conditioned Image-to-Video Generation with 3D Geometry Prior"
  - u: "dfsattn_dynamic_fine-grained_sparse_attention_for_efficient_video_generation/"
    t: "DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation"
  - u: "enhancing_train-free_infinite-frame_generation_for_consistent_long_videos/"
    t: "Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos"
  - u: "epic_efficient_video_camera_control_learning_with_precise_anchor-video_guidance/"
    t: "EPiC: Efficient Video Camera Control Learning with Precise Anchor-Video Guidance"
  - u: "explainable_forensics_of_manipulated_segments_in_untrimmed_long_videos/"
    t: "Explainable Forensics of Manipulated Segments in Untrimmed Long Videos"
  - u: "exploring_data-free_lora_transferability_for_video_diffusion_models/"
    t: "Exploring Data-Free LoRA Transferability for Video Diffusion Models"
  - u: "itryon_mastering_interactive_video_virtual_try-on_with_spatial-semantic_guidance/"
    t: "iTryOn: Mastering Interactive Video Virtual Try-On with Spatial-Semantic Guidance"
  - u: "light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention/"
    t: "Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention"
  - u: "lightning_unified_video_editing_via_in-context_sparse_attention/"
    t: "Lightning Unified Video Editing via In-Context Sparse Attention"
  - u: "locot2v-bench_benchmarking_long-form_and_complex_text-to-video_generation/"
    t: "LocoT2V-Bench: Benchmarking Long-form and Complex Text-to-Video Generation"
  - u: "luve_latent-cascaded_ultra-high-resolution_video_generation_with_dual_frequency_/"
    t: "LuVe: Latent-Cascaded Ultra-High-Resolution Video Generation with Dual Frequency Experts"
  - u: "mive_multiscale_vision-language_features_for_reference-guided_video_editing/"
    t: "MiVE: Multiscale Vision-language features for reference-guided video Editing"
  - u: "motimotion_motion-controlled_video_generation_with_visual_reasoning/"
    t: "MotiMotion: Motion-Controlled Video Generation with Visual Reasoning"
  - u: "olaf-world_orienting_latent_actions_for_video_world_modeling/"
    t: "OLAF-World: Orienting Latent Actions for Video World Modeling"
  - u: "quant_videogen_auto-regressive_long_video_generation_via_2-bit_kv-cache_quantiza/"
    t: "Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization"
  - u: "quantized_keys_steal_attention_bias_correction_for_kv-cache_compression_in_video/"
    t: "Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Generation"
  - u: "rays_as_pixels_learning_a_joint_distribution_of_videos_and_camera_trajectories/"
    t: "Rays as Pixels: Learning A Joint Distribution of Videos and Camera Trajectories"
  - u: "self-refining_video_sampling/"
    t: "Self-Refining Video Sampling"
  - u: "sgmd_score_gradient_matching_distillation_for_few-step_video_diffusion_distillat/"
    t: "SGMD: Score Gradient Matching Distillation for Few-Step Video Diffusion"
  - u: "t2av-compass_towards_unified_evaluation_for_text-to-audio-video_generation/"
    t: "T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation"
  - u: "vanim_rendering-aware_sparse_state_modeling_for_structure-preserving_vector_anim/"
    t: "VAnim: Rendering-Aware Sparse State Modeling for Structure-Preserving Vector Animation"
  - u: "veda_scalable_video_diffusion_via_distilled_sparse_attention/"
    t: "VEDA: Scalable Video Diffusion via Distilled Sparse Attention"
  - u: "where_concept_erasure_should_occur_concept-layer_alignment_in_text-to-video_diff/"
    t: "Where Concept Erasure Should Occur: Concept-Layer Alignment in Text-to-Video Diffusion Models"
  - u: "wind_weather_inverse_diffusion_for_zero-shot_atmospheric_modeling/"
    t: "WIND: Weather Inverse Diffusion for Zero-Shot Atmospheric Modeling"
  - u: "world-r1_reinforcing_3d_constraints_for_text-to-video_generation/"
    t: "World-R1: Reinforcing 3D Constraints for Text-to-Video Generation"
  - u: "worldcache_accelerating_world_models_for_free_via_heterogeneous_token_caching/"
    t: "WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching"
item_total: 28
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# 🎬 Video Generation

**🧪 ICML2026** · **28** paper notes

📌 **Same area in other venues:** [💬 ACL2026 (4)](../../ACL2026/video_generation/index.md) · [📷 CVPR2026 (66)](../../CVPR2026/video_generation/index.md) · [🔬 ICLR2026 (19)](../../ICLR2026/video_generation/index.md) · [🤖 AAAI2026 (11)](../../AAAI2026/video_generation/index.md) · [🧠 NeurIPS2025 (23)](../../NeurIPS2025/video_generation/index.md) · [📹 ICCV2025 (49)](../../ICCV2025/video_generation/index.md)

🔥 **Top topics:** Video Generation ×12 · Diffusion Models ×6 · Model Compression ×2

**[AAD-1: Asymmetric Adversarial Distillation for One-Step Autoregressive Video Generation](aad-1_asymmetric_adversarial_distillation_for_one-step_autoregressive_video_gene.md)**

:   AAD-1 employs asymmetric adversarial distillation using a "causal generator + bidirectional video-level discriminator" alongside DMD warmup to compress autoregressive image-to-video generation into a single sampling step per chunk, while mitigating motion collapse and long-range drift.

**[Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering](attention_sparsity_is_input-stable_training-free_sparse_attention_for_video_gene.md)**

:   SVOO discovers that the attention sparsity of video DiT layers is an intrinsic property that is "input-independent within layers and significantly heterogeneous across layers." Based on this, it performs offline layer-wise sparsity calibration followed by online bidirectional QK co-clustering for block partitioning. It achieves training-free acceleration of up to 1.93× across 7 models (e.g., Wan/HunyuanVideo) while maintaining PSNR at 29 dB.

**[CamGeo: Sparse Camera-Conditioned Image-to-Video Generation with 3D Geometry Prior](camgeo_sparse_camera-conditioned_image-to-video_generation_with_3d_geometry_prio.md)**

:   CamGeo distills 3D geometric knowledge from a pre-trained 3D video model (VGGT) through **training-only distillation**. By providing supervision signals only during the training phase, the diffusion model can generate high-quality videos that are geometrically consistent and motion-smooth under **sparse camera input** conditions. During inference, VGGT is completely removed to maintain efficiency.

**[DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation](dfsattn_dynamic_fine-grained_sparse_attention_for_efficient_video_generation.md)**

:   DFSAttn achieves a **2.1× end-to-end speedup** with quality comparable to full attention through **3D Hilbert curve reordering** + **hierarchical block scoring** + **adaptive mask caching**—addressing the core issue of quality degradation in block-sparse attention at high sparsity rates (>80%).

**[Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos](enhancing_train-free_infinite-frame_generation_for_consistent_long_videos.md)**

:   MIGA enables base video models to generate **infinite-length** and **highly temporally consistent** videos without training through two core mechanisms: **Two-stage Training-inference Alignment** (TTA) and **Dual Consistency Enhancement** (DCE: Self-Reflection + Long-range Frame Guidance). Its comprehensive VBench score improves by 2.8% compared to FIFO-Diffusion (97.82 vs 95.02).

**[EPiC: Efficient Video Camera Control Learning with Precise Anchor-Video Guidance](epic_efficient_video_camera_control_learning_with_precise_anchor-video_guidance.md)**

:   EPiC utilizes a "first-frame visibility mask" approach to directly construct pixel-aligned anchor videos from arbitrary in-the-wild videos. Combined with a lightweight Anchor-ControlNet (26M parameters, <1% of the backbone) that operates exclusively on visible regions, it achieves SOTA I2V camera control accuracy while freezing the CogVideoX-5B-I2V backbone. It requires only 5K videos and 500 training steps, generalizing zero-shot to V2V tasks.

**[Explainable Forensics of Manipulated Segments in Untrimmed Long Videos](explainable_forensics_of_manipulated_segments_in_untrimmed_long_videos.md)**

:   Ours proposes the task of **temporal localization and explainable analysis of AI-generated segments in long videos**, introducing the **TASLE large-scale dataset** and the **two-stage MSLoc baseline method**. By employing boundary-aware proposal generation and MLLM-based refinement, it achieves precise localization and explainable reasoning for manipulated segments in mixed real-fake videos.

**[Exploring Data-Free LoRA Transferability for Video Diffusion Models](exploring_data-free_lora_transferability_for_video_diffusion_models.md)**

:   This paper provides the first weight-space analysis of full fine-tuning (FFT) and LoRA for Video Diffusion Models (VDMs). It discovers that both "preserve the singular spectrum and only rotate singular subspaces," yet they exhibit conflicting routing directions on head clusters. Consequently, the authors propose CASA—a data-free "spectral arbitration by cluster" LoRA transfer method. This approach allows LoRAs trained on the Wan2.1 base model to be directly migrated to distilled variants, such as FastWan, without requiring user data or retraining.

**[iTryOn: Mastering Interactive Video Virtual Try-On with Spatial-Semantic Guidance](itryon_mastering_interactive_video_virtual_try-on_with_spatial-semantic_guidance.md)**

:   iTryOn defines the "Interactive Video Virtual Try-On" task for the first time—enabling individuals in videos to **actively manipulate garments** (zipping, lifting hemlines, stretching fabric) rather than merely displaying them passively. By addressing spatial ambiguity with **3D hand priors**, strictly aligning timestamped action captions with corresponding frames via **Action-Aware RoPE (A-RoPE)**, and amplifying learning signals for sparse interaction frames through **Action-Aware Constraint Loss (AC Loss)**, it improves the ISR (Interaction Success Rate) from 0.397 (baseline) to 0.610 (+54%) on the self-constructed VVT-Interact dataset.

**[Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)**

:   Light Forcing is the first sparse attention scheme customized for autoregressive (AR) video diffusion models—Chunk-Aware Growth (CAG) dynamically allocates sparsity by quantifying the cumulative error contribution of each generated chunk, and Hierarchical Sparse Attention (HSA) flexibly captures historical dependencies through frame-level → chunk-level two-stage mask selection. It achieves 1.30× end-to-end / 3.79× attention acceleration on Self Forcing, with a VBench total score of 84.5 > dense baseline 84.1.

**[Lightning Unified Video Editing via In-Context Sparse Attention](lightning_unified_video_editing_via_in-context_sparse_attention.md)**

:   Addressing the quadratic attention bottleneck in In-Context Learning (ICL) video editing, the authors design In-context Sparse Attention (ISA) based on two insights: "context tokens are less significant than source tokens" and "Query sharpness is proportional to Taylor approximation error." They train LIVEditor, which achieves a ~60% speedup while outperforming SOTA full-attention models across multiple benchmarks.

**[LocoT2V-Bench: Benchmarking Long-form and Complex Text-to-Video Generation](locot2v-bench_benchmarking_long-form_and_complex_text-to-video_generation.md)**

:   LocoT2V-Bench is a professional benchmark for **long-form + complex scene** generation, featuring 234 real-world videos across 18 themes with prompts averaging 249 words. It introduces the LoCoT2V-Eval framework with 5 dimensions and 17 sub-dimensions (including hierarchical VQA, conditional gating, and an Auditor-Evaluator dual-agent HERD). Systematic evaluation of 17 models reveals a common performance bottleneck: "Strong perceptual quality, weak fine-grained alignment, and poor character consistency."

**[LuVe: Latent-Cascaded Ultra-High-Resolution Video Generation with Dual Frequency Experts](luve_latent-cascaded_ultra-high-resolution_video_generation_with_dual_frequency_.md)**

:   LuVe redefines UHR video generation from "passive detail enhancement" to "active content completion." Through a three-stage cascade (low-resolution motion → latent space upsampling → high-resolution refinement) and frequency-domain-driven Dual Frequency Experts (Low-Frequency Expert for global semantic consistency, High-Frequency Expert for texture refinement), it achieves a total score of 84.03 on VBench 4K, surpassing UltraWan-4K's 83.75.

**[MiVE: Multiscale Vision-language features for reference-guided video Editing](mive_multiscale_vision-language_features_for_reference-guided_video_editing.md)**

:   MiVE simultaneously extracts the **first and last layer** hidden states from Qwen3-VL as multiscale condition tokens, concatenates them with VAE visual latents into a long sequence, and performs reference-guided video editing using unified self-attention in DiT. On a 60-clip 720P benchmark, it achieves top human preference and six VLM-based automatic scores, outperforming open-source Wan-Animate and commercial Kling O1.

**[MotiMotion: Motion-Controlled Video Generation with Visual Reasoning](motimotion_motion-controlled_video_generation_with_visual_reasoning.md)**

:   MotiMotion transforms sparse and imprecise user trajectories and text prompts into physically plausible and causally consistent motion trajectories and text descriptions through VLM reasoning. It then employs a **confidence-weighted** control strategy to guide a diffusion model to generate natural videos that conform to world knowledge and physical principles—achieving a physical realism score of 0.302 on MotiBench, significantly surpassing Wan-Move's 0.218 (+38%).

**[OLAF-World: Orienting Latent Actions for Video World Modeling](olaf-world_orienting_latent_actions_for_video_world_modeling.md)**

:   OLAF-World learns transferable latent actions through **Sequence-level Control-Effect Alignment** (Seq∆-REPA)—converting unlabeled videos into action-controllable video world models to achieve zero-shot action transfer across contexts. It matches the performance of AdaWorld (trained on 2 hours of data) using only 1 minute of labeled data (rotation control precision 0.4680 vs. 0.6420).

**[Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization](quant_videogen_auto-regressive_long_video_generation_via_2-bit_kv-cache_quantiza.md)**

:   QVG is a training-free KV-cache quantization framework for autoregressive video diffusion. By employing semantic-aware clustering for token smoothing and progressive residual multi-stage compression, it reduces KV memory consumption to 1/7 on LongCat-Video, HY-WorldPlay, and Self-Forcing. It maintains an end-to-end latency overhead of <4% and significantly outperforms LLM quantization baselines like KIVI and QuaRot in 2-bit scenarios.

**[Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Generation](quantized_keys_steal_attention_bias_correction_for_kv-cache_compression_in_video.md)**

:   This paper discovers that KV cache quantization in chunked autoregressive video diffusion models causes a **systematic shift in attention weights** ("Quantized Keys Steal Attention"). By deriving a per-score correction term based on Jensen's Inequality, it restores near-BF16 video quality under aggressive INT2 quantization (VBench 78.02 vs. 78.27) while saving 50% memory.

**[Rays as Pixels: Learning A Joint Distribution of Videos and Camera Trajectories](rays_as_pixels_learning_a_joint_distribution_of_videos_and_camera_trajectories.md)**

:   The authors represent per-pixel camera rays (origin + direction) as "raxel maps"—3-channel tensors with the same shape as RGB images. By processing these maps through a pre-trained video VAE and using Decoupled Self-Cross Attention within a Flow Matching DiT, the model learns a joint distribution that simultaneously supports pose estimation, camera-controlled generation, and joint video-trajectory generation with a single set of weights.

**[Self-Refining Video Sampling](self-refining_video_sampling.md)**

:   The pretrained flow matching video generator is reinterpreted as a "denoising autoencoder." During inference, a "Predict-and-Perturb" inner loop is used within the same noise level to repeatedly correct latents. An uncertainty mask derived from model self-consistency is applied to refine only dynamic regions. This approach significantly improves motion coherence and physical plausibility without any external verifiers or additional training, achieving over 70% human preference.

**[SGMD: Score Gradient Matching Distillation for Few-Step Video Diffusion](sgmd_score_gradient_matching_distillation_for_few-step_video_diffusion_distillat.md)**

:   SGMD introduces a **stable teacher stop-gradient Fisher objective** and a **dual-potential (NR/RC) mechanism**. It addresses the high cost of fake score tracking (5 updates per round in DMD2) and motion suppression issues in few-step video diffusion distillation. It achieves ~3× training speedup while improving motion quality from 0.65 to 0.78 (VideoAlign) under 4-step distillation.

**[T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation](t2av-compass_towards_unified_evaluation_for_text-to-audio-video_generation.md)**

:   T2AV-Compass is the first comprehensive evaluation benchmark for text-to-audio-video (T2AV) generation, featuring 500 complex prompts and a dual-layer evaluation framework (low-level signal metrics + high-level MLLM diagnostics). It systematically evaluates 15 cutting-edge T2AV systems, quantitatively revealing an "audio realism bottleneck" where even top-tier models achieve 85%+ realism in video but only 50% in audio.

**[VAnim: Rendering-Aware Sparse State Modeling for Structure-Preserving Vector Animation](vanim_rendering-aware_sparse_state_modeling_for_structure-preserving_vector_anim.md)**

:   VAnim models open-domain text-to-SVG animation as "sparse state updates on a persistent DOM tree" + "Identification-First motion planning" + "GRPO rendering-aware reinforcement learning." This approach compresses sequence length by $9.86\times$ while maintaining topological consistency, significantly outperforming GPT-5.2, Gemini 3 Pro, and LiveSketch.

**[VEDA: Scalable Video Diffusion via Distilled Sparse Attention](veda_scalable_video_diffusion_via_distilled_sparse_attention.md)**

:   VEDA reformulates the sparse attention problem in video DiTs as "explicit distillation of the full-attention structure"—utilizing statistics-aware tile scoring + head-aware grouping search + hardware-efficient kernels to maintain generation quality at extreme 90-95% sparsity, delivering 5.1× end-to-end speedup and 10.5× attention acceleration for Waver-12B 720P 10-second videos.

**[Where Concept Erasure Should Occur: Concept-Layer Alignment in Text-to-Video Diffusion Models](where_concept_erasure_should_occur_concept-layer_alignment_in_text-to-video_diff.md)**

:   This paper discovers that target concepts in text-to-video diffusion models are most separable only at specific depths. It proposes CLEAR, which utilizes Gumbel-Softmax to learn "where to erase" and Sparse Autoencoders (SAE) to learn "which concept direction to erase," enabling more precise suppression of target concepts while preserving video quality without modifying diffusion model weights.

**[WIND: Weather Inverse Diffusion for Zero-Shot Atmospheric Modeling](wind_weather_inverse_diffusion_for_zero-shot_atmospheric_modeling.md)**

:   WIND models the global atmospheric sequence as an unconditional video diffusion prior. During inference, it formulates forecasting, downscaling, sparse reconstruction, mass conservation, and warming scenarios as differentiable inverse problems, solving multiple meteorological and climate tasks zero-shot using a single frozen model.

**[World-R1: Reinforcing 3D Constraints for Text-to-Video Generation](world-r1_reinforcing_3d_constraints_for_text-to-video_generation.md)**

:   World-R1 formulates the 3D consistency problem of text-to-video models as reinforcement learning (RL) post-training. By employing implicit camera conditioning and 3D-aware rewards for Flow-GRPO alignment on video foundation models like Wan 2.1, it significantly reduces geometric hallucinations while maintaining general video generation quality without altering the model architecture or inference pipeline.

**[WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching](worldcache_accelerating_world_models_for_free_via_heterogeneous_token_caching.md)**

:   WorldCache addresses the issue of non-uniform evolution of multi-modal tokens (e.g., RGB/depth) in diffusion world models. By categorizing tokens into stable, linear, and chaotic types based on curvature and adaptively triggering full forward passes, it achieves up to 3.65x to 3.7x end-to-end acceleration on models like HunyuanVoyager and Aether while maintaining the quality of world generation and 3D reconstruction.
