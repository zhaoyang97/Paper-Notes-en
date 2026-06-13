---
title: >-
  NeurIPS2025 LLM Efficiency Papers · 34 Notes
description: >-
  34 NeurIPS2025 papers in the LLM Efficiency area, covering LLM, Layout & Composition, Adversarial Robustness, Alignment/RLHF and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "NeurIPS2025"
  - "LLM Efficiency"
  - "AI paper notes"
  - "paper summaries"
  - "LLM"
  - "Layout & Composition"
  - "Adversarial Robustness"
  - "Alignment/RLHF"
item_list:
  - u: "3model_speculative_decoding/"
    t: "3-Model Speculative Decoding (PyramidSD)"
  - u: "a_unified_framework_for_establishing_the_universal_approxima/"
    t: "A Unified Framework for Establishing the Universal Approximation of Transformer-Type Architectures"
  - u: "advancing_expert_specialization_for_better_moe/"
    t: "Advancing Expert Specialization for Better MoE"
  - u: "approximately_aligned_decoding/"
    t: "Approximately Aligned Decoding"
  - u: "constant_bit-size_transformers_are_turing_complete/"
    t: "Constant Bit-Size Transformers Are Turing Complete"
  - u: "critical_batch_size_revisited_a_simple_empirical_approach_to_large-batch_languag/"
    t: "Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training"
  - u: "disc_dynamic_decomposition_improves_llm_inference_scaling/"
    t: "DISC: Dynamic Decomposition Improves LLM Inference Scaling"
  - u: "efficient_training-free_online_routing_for_high-volume_multi-llm_serving/"
    t: "Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving"
  - u: "flowmoe_a_scalable_pipeline_scheduling_framework_for_distributed_mixture-of-expe/"
    t: "FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed MoE Training"
  - u: "from_shortcut_to_induction_head_how_data_diversity_shapes_algorithm_selection_in/"
    t: "From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers"
  - u: "hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac/"
    t: "Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access"
  - u: "harnessing_the_computation_redundancy_in_vits_to_boost_adversarial_transferabili/"
    t: "Harnessing the Computation Redundancy in ViTs to Boost Adversarial Transferability"
  - u: "hierarchical_balance_packing_towards_efficient_supervised_fine-tuning_for_long-c/"
    t: "Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM"
  - u: "jet-nemotron_efficient_language_model_with_post_neural_architecture_search/"
    t: "Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search"
  - u: "l-mtp_leap_multi-token_prediction_beyond_adjacent_context_for_large_language_mod/"
    t: "L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models"
  - u: "let_the_experts_speak_improving_survival_prediction_calibration_via_mixture-of-e/"
    t: "Let the Experts Speak: Improving Survival Prediction & Calibration via Mixture-of-Experts Heads"
  - u: "long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l/"
    t: "Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs"
  - u: "loogle_v2_are_llms_ready_for_real_world_long_dependency_challenges/"
    t: "LooGLE v2: Are LLMs Ready for Real World Long Dependency Challenges?"
  - u: "moesd_unveil_speculative_decodings_potential_for_accelerating_sparse_moe/"
    t: "MoESD: Revealing the Potential of Speculative Decoding to Accelerate Sparse MoE"
  - u: "mozart_modularized_and_efficient_moe_training_on_35d_wafer-scale_chiplet_archite/"
    t: "Mozart: Modularized and Efficient MoE Training on 3.5D Wafer-Scale Chiplet Architectures"
  - u: "omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d/"
    t: "OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding"
  - u: "on_the_expressive_power_of_mixture-of-experts_for_structured_complex_tasks/"
    t: "On the Expressive Power of Mixture-of-Experts for Structured Complex Tasks"
  - u: "scale-invariant_attention/"
    t: "Scale-invariant Attention"
  - u: "silent_tokens_loud_effects_padding_in_llms/"
    t: "Silent Tokens, Loud Effects: Padding in LLMs"
  - u: "skyladder_better_and_faster_pretraining_via_context_window_scheduling/"
    t: "SkyLadder: Better and Faster Pretraining via Context Window Scheduling"
  - u: "sparta_alignment_collectively_aligning_multiple_language_models_through_combat/"
    t: "SPARTA Alignment: Collectively Aligning Multiple Language Models through Combat"
  - u: "technical_debt_in_in-context_learning_diminishing_efficiency_in_long_context/"
    t: "Technical Debt in In-Context Learning: Diminishing Efficiency in Long Context"
  - u: "tensor_product_attention_is_all_you_need/"
    t: "Tensor Product Attention Is All You Need"
  - u: "the_emergence_of_sparse_attention_impact_of_data_distribution_and_benefits_of_re/"
    t: "The Emergence of Sparse Attention: Impact of Data Distribution and Benefits of Repetition"
  - u: "tiled_flash_linear_attention_more_efficient_linear_rnn_and_xlstm_kernels/"
    t: "Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels"
item_total: 34
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# ⚡ LLM Efficiency

**🧠 NeurIPS2025** · **34** paper notes

📌 **Same area in other venues:** [🧪 ICML2026 (30)](../../ICML2026/llm_efficiency/index.md) · [💬 ACL2026 (22)](../../ACL2026/llm_efficiency/index.md) · [🔬 ICLR2026 (20)](../../ICLR2026/llm_efficiency/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_efficiency/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/llm_efficiency/index.md)

🔥 **Top topics:** LLM ×5

**[3-Model Speculative Decoding (PyramidSD)](3model_speculative_decoding.md)**

:   PyramidSD introduces a three-tier pyramid decoding architecture by inserting an intermediate "qualifier" model between the draft model ($M_D$) and target model ($M_T$) in standard speculative decoding. The method exploits the natural entropy gradient across model scales within a model family to hierarchically filter tokens, and employs a fuzzy acceptance criterion to relax the matching threshold, achieving up to 1.91× speedup (reaching 124 tok/s on an RTX 4090).

**[A Unified Framework for Establishing the Universal Approximation of Transformer-Type Architectures](a_unified_framework_for_establishing_the_universal_approxima.md)**

:   A unified theoretical framework is established for proving the universal approximation property (UAP) of diverse Transformer architectures. The framework rests on two core conditions — nonlinear affine invariance of the feed-forward layer and token distinguishability of the attention layer — and leverages an analyticity assumption to reduce the latter to verification on only two-sample cases. The framework successfully covers a wide range of practical architectures, including softmax, RBF kernel, Performer, BigBird, Linformer, and others.

**[Advancing Expert Specialization for Better MoE](advancing_expert_specialization_for_better_moe.md)**

:   By jointly optimizing an orthogonality loss (reducing projection overlap among experts) and a variance loss (increasing routing score diversity), the proposed method reduces expert overlap by 45% and improves routing variance by 150% without modifying the MoE architecture, achieving an average gain of 23.79% across 11 benchmarks while fully preserving load balance.

**[Approximately Aligned Decoding](approximately_aligned_decoding.md)**

:   This paper proposes Approximately Aligned Decoding (AprAD), a method for constrained generation in LLMs that leverages the prefix-selection algorithm from speculative decoding. Upon encountering a constraint violation, AprAD neither reverts only one token (as in constrained generation, which causes extreme probability amplification) nor resamples entirely from scratch (as in ASAp, which incurs prohibitive computational cost). Instead, it intelligently selects a rollback position via speculative sampling, achieving a favorable trade-off between output distribution distortion and computational efficiency.

**[Constant Bit-Size Transformers Are Turing Complete](constant_bit-size_transformers_are_turing_complete.md)**

:   This paper provides the first proof that a Transformer with constant bit-size precision and a fixed number of parameters — permitting only context window growth — is Turing complete. It establishes the exact complexity equivalence WINDOW[s(n)] = SPACE[s(n)], demonstrating that expanding the context window, rather than model size, suffices for universal computation.

**[Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training](critical_batch_size_revisited_a_simple_empirical_approach_to_large-batch_languag.md)**

:   This paper proposes a branched training method to directly measure the critical batch size (CBS) empirically, finding that CBS grows rapidly in early training before plateauing and is independent of model scale. Based on this insight, a batch size warmup strategy is designed that achieves equivalent or superior training loss with 43% fewer gradient steps.

**[DISC: Dynamic Decomposition Improves LLM Inference Scaling](disc_dynamic_decomposition_improves_llm_inference_scaling.md)**

:   DISC proposes a dynamic decomposition algorithm that automatically and recursively adjusts the granularity of reasoning steps at inference time based on the z-score (normalized maximum of sampled rewards) at each step — decomposing difficult steps more finely while taking larger strides over easy ones. It can be plugged into greedy search, Beam Search, and MCTS, achieving higher pass@k with fewer token budgets on APPS, MATH, and LiveCodeBench.

**[Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](efficient_training-free_online_routing_for_high-volume_multi-llm_serving.md)**

:   This paper proposes PORT, the first training-free online LLM routing algorithm. PORT estimates query features via approximate nearest neighbor search (ANNS) and performs a one-shot optimization of dual variables as routing weights on a small set of initial queries. Under a limited token budget, PORT achieves near-offline-optimal routing performance with a $1-o(1)$ competitive ratio, delivering on average 3.55× performance improvement, 1.85× cost efficiency, and 4.25× throughput over baselines.

**[FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed MoE Training](flowmoe_a_scalable_pipeline_scheduling_framework_for_distributed_mixture-of-expe.md)**

:   FlowMoE proposes a unified pipeline scheduling framework that integrates MHA computation, gating, expert computation, and A2A communication into a single pipeline. A priority-driven all-reduce tensor chunking mechanism maximizes communication–computation overlap, achieving 1.13×–1.82× speedup, 10–39% energy reduction, and 7–32% memory savings across multiple real-world MoE models.

**[From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers](from_shortcut_to_induction_head_how_data_diversity_shapes_algorithm_selection_in.md)**

:   This paper provides rigorous theoretical analysis demonstrating that the diversity of pretraining data—characterized by the *max-sum ratio*—determines whether a single-layer Transformer learns a generalizable induction head or a non-OOD-generalizing positional shortcut, and derives a closed-form optimal pretraining distribution that promotes induction head formation.

**[Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)**

:   This paper proposes Hierarchical Sparse Attention (HSA) and the RAMba architecture, which enable Mamba to perform efficient long-range random access through a two-stage token-to-chunk relevance learning mechanism and hardware-aligned kernel design. Pretrained on only 4K context, RAMba achieves 100% accuracy on 64M passkey retrieval.

**[Harnessing the Computation Redundancy in ViTs to Boost Adversarial Transferability](harnessing_the_computation_redundancy_in_vits_to_boost_adversarial_transferabili.md)**

:   By systematically exploiting data-level and model-level computation redundancy in ViTs, this paper proposes five techniques—attention sparsification, attention head permutation, clean token regularization, Ghost MoE diversification, and robust tokens—combined with an online learning strategy that dynamically selects operations. The method achieves an average fooling rate of 86.9% on ImageNet-1K, substantially outperforming all baselines.

**[Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM](hierarchical_balance_packing_towards_efficient_supervised_fine-tuning_for_long-c.md)**

:   This paper proposes Hierarchical Balance Packing (HBP), which addresses attention computation imbalance and communication waste in mixed long/short-context SFT through multi-level packing groups, balanced batching, adaptive sequence parallelism, and stable loss normalization. HBP achieves a 2.4× training speedup on DeepSeek-V2 (236B) without performance degradation.

**[Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search](jet-nemotron_efficient_language_model_with_post_neural_architecture_search.md)**

:   NVIDIA proposes the PostNAS pipeline — starting from a pretrained full-attention model, freezing MLP weights, and applying a four-step search (full-attention layer placement → linear attention block selection → novel JetBlock design → hardware-aware hyperparameter search) to yield the hybrid Jet-Nemotron architecture. The 2B model surpasses Qwen3-1.7B on MMLU-Pro while achieving 47× higher generation throughput.

**[L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models](l-mtp_leap_multi-token_prediction_beyond_adjacent_context_for_large_language_mod.md)**

:   L-MTP introduces a leap mechanism into multi-token prediction (MTP) by predicting tokens at non-adjacent positions (e.g., positions 1, 3, 5, 7 instead of 1, 2, 3, 4). A "looking backward" decoding strategy reuses prior predictions to fill the gaps, achieving a 22% inference speedup on 3B–12B models while maintaining or improving task performance.

**[Let the Experts Speak: Improving Survival Prediction & Calibration via Mixture-of-Experts Heads](let_the_experts_speak_improving_survival_prediction_calibration_via_mixture-of-e.md)**

:   Three discrete-time deep Mixture-of-Experts (MoE) survival analysis architectures are proposed, among which Personalized MoE achieves superior clustering, calibration, and predictive accuracy simultaneously by allowing each expert to generate a patient-specific event distribution.

**[Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)**

:   This paper proposes Dynamic Hierarchical Sparse Attention (DHSA), a hierarchical framework that replaces dense attention with sparse attention via adaptive chunk segmentation, chunk-level similarity prediction, and upsampling to token level — without retraining the base model. On Gemma2/3, DHSA achieves accuracy on par with dense attention while reducing prefill latency by 20–60% and peak memory by 35%.

**[LooGLE v2: Are LLMs Ready for Real World Long Dependency Challenges?](loogle_v2_are_llms_ready_for_real_world_long_dependency_challenges.md)**

:   LooGLE v2 is a long-dependency reasoning benchmark spanning four real-world domains—legal, financial, gaming, and code—with context lengths ranging from 16K to 2M tokens. It comprises 10 domain-specific task types and 1,934 QA instances. Evaluation of 10 LLMs reveals that the strongest model, GPT-4.1, achieves only 59.2%, exposing fundamental deficiencies of current LLMs in real-world long-dependency scenarios.

**[MoESD: Revealing the Potential of Speculative Decoding to Accelerate Sparse MoE](moesd_unveil_speculative_decodings_potential_for_accelerating_sparse_moe.md)**

:   This work challenges the prevailing belief that speculative decoding (SD) is ineffective for MoE models. Through theoretical analysis and experiments, it demonstrates that MoE models benefit more from SD than dense models at medium batch sizes. The paper introduces *target efficiency* as a system-level metric to quantify acceleration bottlenecks, constructs a reliable performance prediction model, and achieves up to 2.29× speedup on Qwen2-57B-A14B.

**[Mozart: Modularized and Efficient MoE Training on 3.5D Wafer-Scale Chiplet Architectures](mozart_modularized_and_efficient_moe_training_on_35d_wafer-scale_chiplet_archite.md)**

:   Mozart is an algorithm-hardware co-design framework that achieves over 1.9× training speedup on three MoE-LLMs via expert clustering and allocation, fine-grained streaming scheduling, and a 3.5D chiplet architecture (NoP-Tree + hierarchical memory).

**[OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding](omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d.md)**

:   This paper proposes the OmniDraft framework, which achieves cross-vocabulary speculative decoding via an online n-gram cache, aligns the draft model with the target model through a hybrid distillation loss, and dynamically adjusts the proposal length with an adaptive drafting head. A single lightweight Llama-68M model thereby provides speculative decoding acceleration (1.5–2×) for diverse target models such as Vicuna-7B, Qwen2-7B, and Llama3-8B.

**[On the Expressive Power of Mixture-of-Experts for Structured Complex Tasks](on_the_expressive_power_of_mixture-of-experts_for_structured_complex_tasks.md)**

:   This paper presents the first systematic analysis of MoE expressive power on structured complex tasks. It proves that shallow MoE can overcome the curse of dimensionality on low-dimensional manifolds (approximation rate governed by intrinsic dimension $d$ rather than ambient dimension $D$), and that deep MoE with $E$ experts × $L$ layers can efficiently approximate piecewise functions with $E^L$ pieces through hierarchical composition—far exceeding the naive upper bound of $LE$.

**[Scale-invariant Attention](scale-invariant_attention.md)**

:   Drawing inspiration from the scale invariance of natural images, this paper proposes a position-dependent affine transformation on attention logits—comprising a multiplicative scaling and an additive shift—such that the total attention weight and sparsity over any token range satisfy scale invariance. This enables zero-shot generalization from short-context training to long-context inference (4k→64k) with a single hyperparameter $\tau$.

**[Silent Tokens, Loud Effects: Padding in LLMs](silent_tokens_loud_effects_padding_in_llms.md)**

:   This paper systematically investigates the effects of padding tokens on LLMs when they are not properly masked. The study finds that even a small number of padding tokens can drift hidden-layer representations, degrade generation quality, and unpredictably shift social biases. Critically, 128 padding tokens raise the harmful prompt attack success rate of Llama-3.1-8B from 8% to 77.5%, effectively constituting a jailbreak.

**[SkyLadder: Better and Faster Pretraining via Context Window Scheduling](skyladder_better_and_faster_pretraining_via_context_window_scheduling.md)**

:   SkyLadder, a progressive short-to-long context window scheduling strategy, achieves superior pretraining efficiency (22% training time saved) and improved model performance (+3.7%) under a fixed compute budget, challenging the prevailing belief that "longer context = better performance."

**[SPARTA Alignment: Collectively Aligning Multiple Language Models through Combat](sparta_alignment_collectively_aligning_multiple_language_models_through_combat.md)**

:   Multiple LLMs form a "Spartan tribe" to engage in mutual competition and peer evaluation. Preference pairs are generated via reputation-weighted judgment aggregation, and all models are iteratively trained with DPO. The approach surpasses self-alignment baselines such as Self-Rewarding on 10 out of 12 tasks, with an average improvement of 7%.

**[Technical Debt in In-Context Learning: Diminishing Efficiency in Long Context](technical_debt_in_in-context_learning_diminishing_efficiency_in_long_context.md)**

:   Drawing on the methodology of optimization software benchmarking, this work precisely quantifies the sample efficiency of ICL relative to the Bayes-optimal estimator via performance ratios. A clear dichotomy is identified: in the few-shot regime (≤15 demonstrations), efficiency is near-optimal (only ~10% overhead), whereas in the many-shot regime (>40 demonstrations) it degrades sharply (>45% overhead). Information-theoretic analysis establishes that this phenomenon stems from a non-decreasing excess risk that is irreducible—an intrinsic limitation of the ICL mechanism.

**[Tensor Product Attention Is All You Need](tensor_product_attention_is_all_you_need.md)**

:   By decomposing Q/K/V into weighted sums of low-rank factors via contextual tensor products, TPA compresses the KV cache by 10–16×, while surpassing standard MHA/MQA/GQA/MLA on both validation loss and downstream task accuracy.

**[The Emergence of Sparse Attention: Impact of Data Distribution and Benefits of Repetition](the_emergence_of_sparse_attention_impact_of_data_distribution_and_benefits_of_re.md)**

:   This paper investigates the emergence mechanism of sparse attention through theoretical analysis and controlled experiments, revealing that the emergence time follows a power-law relationship with respect to sequence length and dimensionality, $T_\epsilon \propto \sqrt{d} \cdot T$. It further demonstrates that both in-context and cross-sample data repetition strategies accelerate emergence, offering a unified sparse attention perspective for understanding capability emergence in LLMs.

**[Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels](tiled_flash_linear_attention_more_efficient_linear_rnn_and_xlstm_kernels.md)**

:   This paper proposes TFLA (Tiled Flash Linear Attention), which achieves efficient linear RNN/mLSTM kernels through two-level sequence parallelism and tiling optimization, delivering significant wall-clock speedups over FlashAttention 3 and Mamba 2 (>2× in training vs. Mamba 2) while maintaining equivalent model accuracy.

**[UMoE: Unifying Attention and FFN with Shared Experts](umoe_unifying_attention_and_ffn_with_shared_experts.md)**

:   By reformulating the multi-head attention mechanism, this work reveals that attention shares the same "two-layer matrix multiplication" structure as FFN layers. Based on this insight, UMoE is proposed as a unified architecture that employs identically designed experts for both attention and FFN layers with parameter sharing, outperforming existing FFN-MoE and Attention-MoE baselines on both Base (134M) and Large (1.1B) models.

**[Unmasking COVID-19 Vulnerability in Nigeria: Mapping Risks Beyond Urban Hotspots](unmasking_covid-19_vulnerability_in_nigeria_mapping_risks_beyond_urban_hotspots.md)**

:   This paper constructs a comprehensive COVID-19 vulnerability risk scoring system for Nigerian states, integrating four dimensions — population density, poverty, healthcare accessibility, and age risk — and visualizes hotspot regions via GIS mapping, providing a data-driven decision tool for public health resource allocation.

**[Yggdrasil: Bridging Dynamic Speculation and Static Runtime for Latency-Optimal Tree-Based LLM Decoding](yggdrasil_bridging_dynamic_speculation_and_static_runtime_for_latency-optimal_tr.md)**

:   This paper proposes Yggdrasil, a latency-optimal speculative decoding system that achieves compiler-friendly dynamic drafting via the Equal-Growth Tree (EGT) structure, replaces the conventional AAL metric with a latency-aware optimization objective, and reduces CPU-GPU coordination overhead through a stage-based scheduling runtime, achieving up to 3.98× end-to-end speedup on A100/A40 GPUs.

**[ZeroS: Zero-Sum Linear Attention for Efficient Transformers](zeros_zero-sum_linear_attention_for_efficient_transformers.md)**

:   By removing the zeroth-order uniform term $1/t$ from softmax, ZeroS constructs a linear attention mechanism with zero-sum weights, breaking the limitation of convex combinations to purely additive mixing. This enables differential/contrastive operations within a single layer while maintaining $O(Nd^2)$ linear complexity, matching or surpassing standard softmax attention across multiple sequence modeling benchmarks.
