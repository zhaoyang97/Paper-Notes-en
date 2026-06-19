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

📌 **Same area in other venues:** [📷 CVPR2026 (152)](../../CVPR2026/video_generation/index.md) · [💬 ACL2026 (4)](../../ACL2026/video_generation/index.md) · [🔬 ICLR2026 (19)](../../ICLR2026/video_generation/index.md) · [🤖 AAAI2026 (11)](../../AAAI2026/video_generation/index.md) · [🧠 NeurIPS2025 (23)](../../NeurIPS2025/video_generation/index.md) · [📹 ICCV2025 (49)](../../ICCV2025/video_generation/index.md)

🔥 **Top topics:** Video Generation ×12 · Diffusion Models ×6 · Model Compression ×2

**[AAD-1: Asymmetric Adversarial Distillation for One-Step Autoregressive Video Generation](aad-1_asymmetric_adversarial_distillation_for_one-step_autoregressive_video_gene.md)**

:   AAD-1 utilizes asymmetric adversarial distillation with a "causal generator + bidirectional video-level discriminator" alongside DMD warmup to compress autoregressive image-to-video generation to a single sampling step per chunk, while mitigating motion collapse and long-range drift.

**[Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering](attention_sparsity_is_input-stable_training-free_sparse_attention_for_video_gene.md)**

:   SVOO discovers that the attention sparsity of each layer in video DiT is an intrinsic property that is "input-independent within layers and significantly heterogeneous between layers." Based on this, it performs offline per-layer sparsity calibration followed by online QK bidirectional co-clustering for block partitioning. It achieves up to 1.93× speedup while maintaining a PSNR of 29 dB across 7 models (e.g., Wan, HunyuanVideo) without any training.

**[CamGeo: Sparse Camera-Conditioned Image-to-Video Generation with 3D Geometry Prior](camgeo_sparse_camera-conditioned_image-to-video_generation_with_3d_geometry_prio.md)**

:   CamGeo distills 3D geometric knowledge from a pre-trained 3D video model (VGGT) through **training-only distillation**. By providing supervision signals only during the training phase, the diffusion model generates high-quality videos with geometric consistency and smooth motion under **sparse camera inputs**, while the VGGT is completely removed during inference to maintain efficiency.

**[DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation](dfsattn_dynamic_fine-grained_sparse_attention_for_efficient_video_generation.md)**

:   DFSAttn achieves **2.1× end-to-end acceleration** with quality comparable to full attention through **3D Hilbert curve reordering** + **hierarchical block scoring** + **adaptive mask caching**. It addresses the core issue of quality degradation in block-sparse attention at high sparsity ratios (>80%).

**[Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos](enhancing_train-free_infinite-frame_generation_for_consistent_long_videos.md)**

:   MIGA enables base video models to generate **infinite-length** and **highly temporally consistent** videos without training through two core mechanisms: **Two-stage Training-inference Alignment** (TTA) and **Dual Consistency Enhancement** (DCE: self-reflection + long-range frame guidance). Its VBench comprehensive score improves by 2.8% compared to FIFO-Diffusion (97.82 vs 95.02).

**[EPiC: Efficient Video Camera Control Learning with Precise Anchor-Video Guidance](epic_efficient_video_camera_control_learning_with_precise_anchor-video_guidance.md)**

:   EPiC utilizes a "first-frame visibility mask-based" approach to directly construct pixel-aligned anchor videos from arbitrary in-the-wild videos. By pairing this with Anchor-ControlNet—comprising only 26M parameters (<1% of the backbone) and operating exclusively on visible regions—it achieves SOTA I2V camera control and zero-shot generalization to V2V while freezing the CogVideoX-5B-I2V backbone and training on only 5K videos for 500 steps.

**[Explainable Forensics of Manipulated Segments in Untrimmed Long Videos](explainable_forensics_of_manipulated_segments_in_untrimmed_long_videos.md)**

:   This paper proposes the task of **temporal localization and explainable analysis of AI-generated segments in long videos**, introducing the **large-scale TASLE dataset** and the **two-stage MSLoc baseline method**—achieving precise localization and explainable reasoning of manipulated segments in mixed real-fake videos through boundary-aware proposal generation and MLLM refinement.

**[Exploring Data-Free LoRA Transferability for Video Diffusion Models](exploring_data-free_lora_transferability_for_video_diffusion_models.md)**

:   This paper presents the first weight-space analysis of Full Fine-Tuning (FFT) and LoRA for Video Diffusion Models (VDMs). It discovers that both "preserve the singular spectrum and only rotate the singular subspaces," but exhibit conflicting routing directions on head clusters. Based on this, the authors propose CASA—a data-free "spectral arbitration by clustering" LoRA transfer method that allows LoRA trained on base models like Wan2.1 to be directly transferred to distilled variants like FastWan without requiring user data or retraining.

**[iTryOn: Mastering Interactive Video Virtual Try-On with Spatial-Semantic Guidance](itryon_mastering_interactive_video_virtual_try-on_with_spatial-semantic_guidance.md)**

:   iTryOn defines the "Interactive Video Virtual Try-On" task for the first time—enabling individuals in videos to **actively manipulate garments** (zipping, lifting hems, stretching fabric) rather than merely displaying them passively. By addressing spatial ambiguity with **3D hand priors**, strictly aligning timestamped action captions with corresponding frames via **Action-aware RoPE (A-RoPE)**, and amplifying learning signals for sparse interaction frames with **Action-aware Constraint Loss (AC Loss)**, it improves the Interactive Success Rate (ISR) from 0.397 to 0.610 (+54%) on the self-constructed VVT-Interact dataset.

**[Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)**

:   Light Forcing is the first sparse attention scheme customized for autoregressive (AR) video diffusion models. Chunk-Aware Growth (CAG) quantifies the cumulative error contribution of each generated chunk to dynamically allocate sparsity, while Hierarchical Sparse Attention (HSA) flexibly captures historical dependencies through frame-level → chunk-level dual-mask selection. It achieves 1.30× end-to-end / 3.79× attention speedup on Self Forcing, with a VBench total score of 84.5 > dense baseline 84.1.

**[Lightning Unified Video Editing via In-Context Sparse Attention](lightning_unified_video_editing_via_in-context_sparse_attention.md)**

:   Addressing the quadratic attention bottleneck in In-Context Learning (ICL) video editing, the authors design In-context Sparse Attention (ISA) based on two insights: "context token saliency is lower than source tokens" and "Query sharpness is proportional to Taylor approximation error." They train LIVEditor, which achieves a ~60% speedup while surpassing SOTA full-attention models on multiple benchmarks.

**[LocoT2V-Bench: Benchmarking Long-form and Complex Text-to-Video Generation](locot2v-bench_benchmarking_long-form_and_complex_text-to-video_generation.md)**

:   LocoT2V-Bench is a professional benchmark for **long-form + complex scene** generation—featuring 234 real videos × 18 themes × 249-word average prompts. It includes the LoCoT2V-Eval framework across 5 dimensions and 17 sub-dimensions (incorporating hierarchical VQA + conditional gating + Auditor-Evaluator dual-agent HERD). Systematic evaluation of 17 long-form video generation models reveals a universal bottleneck: "strong perceptual quality, weak fine-grained alignment, and poor character consistency."

**[LuVe: Latent-Cascaded Ultra-High-Resolution Video Generation with Dual Frequency Experts](luve_latent-cascaded_ultra-high-resolution_video_generation_with_dual_frequency_.md)**

:   LuVe redefines UHR video generation from "passive detail enhancement" to "active content completion." Through a three-stage cascade (Low-res Motion → Latent Upsampling → High-res Refinement) and dual frequency experts driven by frequency domain analysis (Low-Frequency Expert for global semantic consistency and High-Frequency Expert for texture refinement), LuVe achieves a total score of 84.03 on VBench 4K, surpassing UltraWan-4K's 83.75.

**[MiVE: Multiscale Vision-language features for reference-guided video Editing](mive_multiscale_vision-language_features_for_reference-guided_video_editing.md)**

:   MiVE extracts the **first + last layer** hidden states of Qwen3-VL simultaneously as multiscale condition tokens. These are concatenated with VAE visual latents into a long sequence for reference-guided video editing within a unified self-attention DiT. On a 60-segment 720P benchmark, it achieved first place in both human preference and 6 VLM auto-scores, surpassing open-source Wan-Animate and commercial Kling O1.

**[MotiMotion: Motion-Controlled Video Generation with Visual Reasoning](motimotion_motion-controlled_video_generation_with_visual_reasoning.md)**

:   MotiMotion transforms sparse, imprecise user trajectories and text prompts into physically plausible and causally consistent motion trajectories and text descriptions via VLM reasoning. It employs a **confidence-weighted** control strategy to guide a diffusion model, generating natural videos aligned with world knowledge and physical principles—achieving a physical authenticity score of 0.302 on MotiBench, significantly outperforming Wan-Move’s 0.218 (+38%).

**[OLAF-World: Orienting Latent Actions for Video World Modeling](olaf-world_orienting_latent_actions_for_video_world_modeling.md)**

:   OLAF-World enables transferable latent action learning through **sequence-level control-effect alignment** (Seq∆-REPA). It transforms unlabeled videos into action-controllable world models, achieving zero-shot action transfer across contexts. With only 1 minute of labeled data, it reaches performance comparable to AdaWorld trained on 2 hours of data (rotation control precision 0.4680 vs. 0.6420).

**[Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization](quant_videogen_auto-regressive_long_video_generation_via_2-bit_kv-cache_quantiza.md)**

:   QVG is a training-free KV-cache quantization framework for autoregressive video diffusion. By employing semantic-aware clustering for token smoothing and progressive residual multi-stage compression, it reduces KV memory footprint to 1/7 of the original on LongCat-Video/HY-WorldPlay/Self-Forcing with <4% end-to-end latency overhead. At 2-bit, its quality significantly outperforms LLM quantization baselines like KIVI and QuaRot.

**[Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Generation](quantized_keys_steal_attention_bias_correction_for_kv-cache_compression_in_video.md)**

:   This paper discovers that KV-cache quantization in chunked autoregressive video diffusion models causes a **systematic shift in attention weights** ("quantized keys steal attention"). By deriving a per-score correction term based on Jensen's Inequality, it restores video quality close to BF16 (VBench 78.02 vs 78.27) under aggressive INT2 quantization while saving 50% memory.

**[Rays as Pixels: Learning A Joint Distribution of Videos and Camera Trajectories](rays_as_pixels_learning_a_joint_distribution_of_videos_and_camera_trajectories.md)**

:   This work packs the per-pixel ray "origin + direction" of each camera into a 3-channel "raxel" map with the same shape as RGB images. It leverages a pre-trained video VAE as a camera encoder and employs Decoupled Self-Cross Attention to integrate raxels and video frames into a single Flow Matching DiT for joint denoising. For the first time, a single set of weights supports pose estimation, camera-controlled video generation, and joint "video + trajectory" generation.

**[Self-Refining Video Sampling](self-refining_video_sampling.md)**

:   The pretrained flow matching video generator is reinterpreted as a "denoising autoencoder." During inference, a Predict-and-Perturb inner loop iteratively corrects latent deviations within the same noise level. An uncertainty mask derived from model self-consistency is applied to refine only dynamic regions. This approach significantly enhances motion coherence and physical plausibility without any external verifier or additional training, achieving a human preference rate exceeding 70%.

**[SGMD: Score Gradient Matching Distillation for Few-Step Video Diffusion](sgmd_score_gradient_matching_distillation_for_few-step_video_diffusion_distillat.md)**

:   SGMD introduces a **stable teacher stop-gradient Fisher objective** and a **dual potential (NR/RC) mechanism** to address high tracking costs of fake scores (e.g., DMD2 requires 5 updates per iteration) and motion suppression in few-step video diffusion distillation. It achieves ~3× training acceleration while improving motion quality from 0.65 to 0.78 (VideoAlign) under 4-step distillation.

**[T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation](t2av-compass_towards_unified_evaluation_for_text-to-audio-video_generation.md)**

:   T2AV-Compass is the first comprehensive evaluation benchmark for Text-to-Audio-Video (T2AV) generation, featuring 500 complex prompts and a dual-layer evaluation framework (low-level signal metrics + high-level MLLM diagnostics). It systematically evaluates 15 cutting-edge T2AV systems, quantitatively revealing an "audio realism bottleneck" where even top-tier models achieve 85%+ realism in the video dimension versus only 50% in audio.

**[VAnim: Rendering-Aware Sparse State Modeling for Structure-Preserving Vector Animation](vanim_rendering-aware_sparse_state_modeling_for_structure-preserving_vector_anim.md)**

:   VAnim models open-domain text-to-SVG animation as "sparse state updates on a persistent DOM tree" + "Identification-First motion planning" + "GRPO rendering-aware reinforcement learning." This approach compresses sequence lengths by $9.86\times$ while maintaining topological consistency, significantly outperforming GPT-5.2, Gemini 3 Pro, and LiveSketch.

**[VEDA: Scalable Video Diffusion via Distilled Sparse Attention](veda_scalable_video_diffusion_via_distilled_sparse_attention.md)**

:   VEDA reformulates the sparse attention problem in video DiT as "explicit distillation of the full attention structure." By combining statistic-aware tile scoring, head-aware grouping search, and hardware-efficient kernels, it maintains generation quality at extreme 90-95% sparsity. It achieves a 5.1× end-to-end speedup and 10.5× attention acceleration for Waver-12B generating 720P 10-second videos.

**[Where Concept Erasure Should Occur: Concept-Layer Alignment in Text-to-Video Diffusion Models](where_concept_erasure_should_occur_concept-layer_alignment_in_text-to-video_diff.md)**

:   This paper discovers that target concepts in text-to-video (T2V) diffusion models are most separable only at specific depths. It proposes CLEAR, which utilizes Gumbel-Softmax to learn "where to erase" and Sparse Autoencoders (SAE) to learn "which concept direction to erase," enabling precise suppression of target concepts while preserving video quality without modifying diffusion model weights.

**[WIND: Weather Inverse Diffusion for Zero-Shot Atmospheric Modeling](wind_weather_inverse_diffusion_for_zero-shot_atmospheric_modeling.md)**

:   WIND models global atmospheric sequences as an unconditional video diffusion prior. During inference, it treats forecasting, downscaling, sparse reconstruction, mass conservation, and warming scenarios as differentiable inverse problems, enabling a single frozen model to solve multiple weather and climate tasks zero-shot.

**[World-R1: Reinforcing 3D Constraints for Text-to-Video Generation](world-r1_reinforcing_3d_constraints_for_text-to-video_generation.md)**

:   World-R1 reformulates the 3D consistency problem in text-to-video models as a reinforcement learning post-training task. It utilizes Flow-GRPO to align video foundation models, such as Wan 2.1, with implicit camera conditioning and 3D-aware rewards. This approach significantly reduces geometric hallucinations without altering model architecture or inference pipelines while maintaining general video generation quality.

**[WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching](worldcache_accelerating_world_models_for_free_via_heterogeneous_token_caching.md)**

:   WorldCache addresses the issue of non-uniform evolution of multimodal tokens (such as RGB and depth) in diffusion world models. By categorizing tokens into stable, linear, and chaotic types based on curvature and adaptively triggering full forward passes, it achieves up to 3.65x to 3.7x end-to-end acceleration on models like HunyuanVoyager and Aether, while substantially maintaining the quality of world generation and 3D reconstruction.
