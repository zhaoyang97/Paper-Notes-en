---
title: >-
  ICML2026 LLM Efficiency Papers · 32 Notes
description: >-
  32 ICML2026 papers in the LLM Efficiency area, covering LLM, Diffusion Models, Compression and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ICML2026"
  - "LLM Efficiency"
  - "AI paper notes"
  - "paper summaries"
  - "LLM"
  - "Diffusion Models"
  - "Compression"
item_list:
  - u: "beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_/"
    t: "Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts"
  - u: "criticalkv_optimizing_kv_cache_eviction_from_an_output_perturbation_perspective/"
    t: "CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective"
  - u: "dllm-cache_accelerating_diffusion_large_language_models_with_adaptive_caching/"
    t: "dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching"
  - u: "do_transformers_need_three_projections_systematic_study_of_qkv_variants/"
    t: "Do Transformers Need Three Projections? A Systematic Study of QKV Sharing Systems"
  - u: "dot-moe_differentiable_optimal_transport_for_moefication/"
    t: "DOT-MoE: Transforming Dense LLMs into MoE with Differentiable Optimal Transport"
  - u: "efficient_training-free_multi-token_prediction_via_embedding-space_probing/"
    t: "Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing"
  - u: "ekka_automated_diagnosis_of_silent_errors_in_llm_inference/"
    t: "Ekka: Automated Diagnosis of Silent Errors in LLM Inference"
  - u: "fast-dllm_fréchet_profile_decoding_for_faster_diffusion_llm_inference/"
    t: "Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference"
  - u: "graphflow_a_graph-based_workflow_management_for_efficient_llm-agent_serving/"
    t: "GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving"
  - u: "hyperparameter_transfer_with_mixture-of-expert_layers/"
    t: "Hyperparameter Transfer with Mixture-of-Experts Layers"
  - u: "knapspec_self-speculative_decoding_via_adaptive_layer_selection_as_a_knapsack_pr/"
    t: "KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem"
  - u: "l3_large_lookup_layers/"
    t: "L$^3$: Large Lookup Layers"
  - u: "minedraft_a_framework_for_batch_parallel_speculative_decoding/"
    t: "MineDraft: A Framework for Batch Parallel Speculative Decoding"
  - u: "obcache_optimal_brain_kv_cache_pruning_for_efficient_long-context_llm_inference/"
    t: "OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference"
  - u: "optimal_bayesian_stopping_for_efficient_inference_of_consistent_llm_answers/"
    t: "Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers"
  - u: "oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration/"
    t: "OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration"
  - u: "prism_spectral-aware_block-sparse_attention/"
    t: "Prism: Spectral-Aware Block-Sparse Attention"
  - u: "proactivellm_learning_active_interaction_for_streaming_large_language_models/"
    t: "ProactiveLLM: Learning Active Interaction for Streaming Large Language Models"
  - u: "probmoe_differentiable_probabilistic_routing_for_mixture-of-experts/"
    t: "ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts"
  - u: "proxy_compression_for_language_modeling/"
    t: "Proxy Compression for Language Modeling"
  - u: "remoe_boosting_expert_reuse_through_router_fine-tuning_in_memory-constrained_moe/"
    t: "ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference"
  - u: "repetitioncurse_measuring_and_understanding_router_imbalance_in_mixture-of-exper/"
    t: "RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress"
  - u: "scout_active_information_foraging_for_long-text_understanding_with_decoupled_epi/"
    t: "Scout: Active Information Foraging for Long-Text Understanding with Decoupled Epistemic States"
  - u: "siamesenorm_breaking_the_barrier_to_reconciling_prepost-norm/"
    t: "SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm"
  - u: "skill-based_mixture-of-experts_adaptive_routing_for_heterogeneous_reasoning_via_/"
    t: "Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills"
  - u: "sparser_block-sparse_attention_via_token_permutation/"
    t: "Sparser Block-Sparse Attention via Token Permutation"
  - u: "stochastic_sparse_attention_for_memory-bound_inference/"
    t: "Stochastic Sparse Attention for Memory-Bound Inference"
  - u: "team_temporal-spatial_consistency_guided_expert_activation_for_moe_diffusion_lan/"
    t: "TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration"
  - u: "theoretically_optimal_attentionffn_ratios_in_disaggregated_llm_serving/"
    t: "Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving"
  - u: "training-inference_consistent_segmented_execution_for_long-context_llms/"
    t: "Training-Inference Consistent Segmented Execution for Long-Context LLMs"
item_total: 32
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# ⚡ LLM Efficiency

**🧪 ICML2026** · **32** paper notes

📌 **Same area in other venues:** [📷 CVPR2026 (5)](../../CVPR2026/llm_efficiency/index.md) · [💬 ACL2026 (22)](../../ACL2026/llm_efficiency/index.md) · [🔬 ICLR2026 (20)](../../ICLR2026/llm_efficiency/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_efficiency/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/llm_efficiency/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/llm_efficiency/index.md)

🔥 **Top topics:** LLM ×11 · Diffusion Models ×3 · Compression ×2

**[Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)**

:   The authors propose an "orthogonal growth" strategy for converged MoE models—using interpositional layer replication for depth and noisy expert cloning for width—scaling a 17B model to 70B. This achieve a 10.6% accuracy improvement over training from scratch under the same additional compute budget.

**[CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective](criticalkv_optimizing_kv_cache_eviction_from_an_output_perturbation_perspective.md)**

:   The authors reformulate the empirical problem of "which KV cache entries are critical" as an optimization problem of "minimizing attention output perturbation." They derive an analytical upper bound for the perturbation (involving both attention weights and value norms projected by $W^O$) and design a plug-and-play two-stage greedy selection algorithm. This approach reduces the compression loss of three SOTA eviction methods (SnapKV/AdaKV/HeadKV) by more than half on average across 29 long-context datasets.

**[dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](dllm-cache_accelerating_diffusion_large_language_models_with_adaptive_caching.md)**

:   To address the significant inference slowdown in Diffusion Large Language Models (dLLMs) caused by the inability of bidirectional attention to reuse KV cache, this paper proposes dLLM-Cache—a training-free method. It employs long-interval caching for static prompts and short-interval refreshing for dynamic responses, utilizing Value cosine similarity to select the 25% most "active" tokens for local recomputation. On LLaDA 8B and Dream 7B, it achieves up to 9.1× FLOPs acceleration with minimal performance degradation.

**[Do Transformers Need Three Projections? A Systematic Study of QKV Sharing Systems](do_transformers_need_three_projections_systematic_study_of_qkv_variants.md)**

:   The paper systematically compares three QKV projection sharing schemes: Q=K-V (shared query and key), Q-K=V (shared key and value), and Q=K=V (all three shared). It finds that for Language Modeling (LM), Q-K=V increases Perplexity (PPL) by only 3.1% while reducing the KV cache by 50%. This approach is orthogonal to GQA/MQA, enabling a total cache reduction of 87.5%–96.9%, providing quantifiable memory benefits for edge inference.

**[DOT-MoE: Transforming Dense LLMs into MoE with Differentiable Optimal Transport](dot-moe_differentiable_optimal_transport_for_moefication.md)**

:   DOT-MoE models the "allocation of neurons to experts when converting a dense FFN to an MoE" as a differentiable optimal transport problem. It employs Sinkhorn-Knopp iterations to solve entropic-regularized balanced transport combined with a Straight-Through Estimator, allowing joint end-to-end learning of neuron-to-expert assignment and the router. It retains 90% of dense performance under 50% active parameters on LLaMA-2/3 and Qwen2.5, outperforming all baselines including structured pruning, random allocation, and clustering.

**[Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing](efficient_training-free_multi-token_prediction_via_embedding-space_probing.md)**

:   This paper proposes ESP (Embedding-Space Probing): without modifying weights or training auxiliary models, it injects "mean prompt embeddings" as mask tokens into the input sequence of a frozen LLM. This probes multiple future tokens in a single forward pass, followed by lossless speculative verification using the base model itself. On LLaMA3 / Qwen3, it achieves 7–11% higher average acceptance length and 15–19% higher throughput compared to training-free baselines like LADE, STAND, and PLD.

**[Ekka: Automated Diagnosis of Silent Errors in LLM Inference](ekka_automated_diagnosis_of_silent_errors_in_llm_inference.md)**

:   Ekka models the diagnosis of silent errors in LLM serving frameworks—where outputs degrade without explicit errors—as a differential debugging task using reference implementations like HuggingFace as an oracle. By employing an agentic pipeline of "component mapping $\rightarrow$ activation alignment $\rightarrow$ change-point analysis," it automatically localizes problematic modules. Ekka achieves a diagnosis accuracy of 80% pass@1 / 88% pass@5 across 17 real-world vLLM/SGLang issues and discovered 4 hidden bugs confirmed by developers.

**[Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference](fast-dllm_fréchet_profile_decoding_for_faster_diffusion_llm_inference.md)**

:   Addressing the parallel decoding bottleneck of diffusion language models (dLLMs), this paper proposes training-free Fréchet Profile Decoding: using the entire sorted confidence profile rather than just the "weakest selected token" to determine how many tokens to commit in each parallel step. This strictly generalizes the factor rule of Fast-dLLM to heterogeneous confidence scenarios, achieving an average throughput of 1.36× and a 29% reduction in NFE on LLaDA-8B across four benchmarks with almost no loss in precision.

**[GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving](graphflow_a_graph-based_workflow_management_for_efficient_llm-agent_serving.md)**

:   GraphFlow unifies multiple agent workflows into a global operational Directed Acyclic Graph (wGraph). It generates task-adaptive subgraph workflows online using GNN+MLP. By replacing independent workflow caching with differential caching ("base KV + sparse prefix residual + path pruning"), it achieves an average improvement of 4.95pp across five reasoning/coding/QA benchmarks while reducing KV memory consumption to approximately 1/4.

**[Hyperparameter Transfer with Mixture-of-Experts Layers](hyperparameter_transfer_with_mixture-of-expert_layers.md)**

:   This paper extends the maximal update parametrization (μP/CompleteP) to sparse MoE Transformers. It defines initialization and learning rate (LR) scaling rules for routers, expert up/down projections, and expert biases when model width, depth, number of experts, and expert width are simultaneously scaled. Using a three-level Mean-Field Dynamical Mean Field Theory (DMFT), the authors prove that this parametrization possesses a scale-invariant limit as $n_{\text{embd}}, n_{\text{exp}}, n_{\text{hid}}, L \to \infty$ (at fixed activation sparsity $\kappa$). Optimal LRs and initializations can be directly reused from 38M active parameter base models to 2B parameter MoEs. MoEs trained with zero-shot hyperparameters achieve performance comparable to or better than dense GPT2 speedrun models at equivalent active parameter counts.

**[KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem](knapspec_self-speculative_decoding_via_adaptive_layer_selection_as_a_knapsack_pr.md)**

:   KnapSpec reformulates draft layer selection in Self-Speculative Decoding (SSD) as a 0/1 knapsack problem. By decoupling Attention and MLP modules, using context-length-dependent hardware latency as "weight" and hidden state cosine similarity (with the first rigorous proof provided) as "value," it adaptively identifies the subnetwork that maximizes Tokens-per-Time via parallel DP at each step. It achieves up to 1.47× wall-clock speedup on Qwen3 / Llama3 in long-context scenarios without additional training.

**[L$^3$: Large Lookup Layers](l3_large_lookup_layers.md)**

:   This paper proposes L$^3$ (Large Lookup Layer), which generalizes the tokenizer embedding table into a "large lookup layer" that can be inserted into the decoder. By using **static routing** based on token IDs to retrieve a set of learned key/value embeddings, and then using the current hidden state for attention-based aggregation, the model achieves a higher degree of sparsity without the common MoE pitfalls of dynamic routing, auxiliary losses, and offloading difficulties. It outperforms dense models of the same compute power and MoE models of the same sparsity at 800M–2.6B active parameter scales.

**[MineDraft: A Framework for Batch Parallel Speculative Decoding](minedraft_a_framework_for_batch_parallel_speculative_decoding.md)**

:   MineDraft achieves **overlapped execution** of drafting for one batch and verification for another across two independent sets of GPUs by maintaining two request batches. This transforms the traditionally serial "draft-verify" pipeline of speculative decoding into batch-parallel PSD. At the cost of only one additional GPU, it increases throughput by up to 75% and reduces end-to-end latency by up to 39% compared to standard SD, and is implemented as a plug-and-play vLLM plugin.

**[OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference](obcache_optimal_brain_kv_cache_pruning_for_efficient_long-context_llm_inference.md)**

:   This paper reformulates KV cache eviction as a "layer-wise structured pruning" problem. Leveraging the second-order Taylor approximation from Optimal Brain Damage, it derives closed-form saliency scores for independent value, independent key, and joint key-value pruning units. These serve as plug-and-play scoring components for existing attention-only eviction frameworks like H2O, TOVA, SnapKV, and AdaKV, achieving consistent improvements on LLaMA-3.1 and Qwen-2.5 across RULER and LongBench benchmarks (e.g., AdaKV's performance increases by nearly 15% on query-agnostic RULER-4K with a 30% budget).

**[Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers](optimal_bayesian_stopping_for_efficient_inference_of_consistent_llm_answers.md)**

:   This paper models the "Self-Consistency (multiple sampling for majority vote)" problem as a Bayesian optimal stopping problem with prior information. It proposes an $L$-aggregated posterior approximation that tracks only three types of counts: "top-1 frequency, top-2 frequency, and others." The authors theoretically prove that $L=3$ achieves the same asymptotically optimal stopping time as the exact posterior as $\delta \to 0$. Experimentally, it saves 30%–80% of LLM calls on GSM8K and CommonsenseQA at approximately 1.4x the speed of ASC.

**[OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration](oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration.md)**

:   OServe jointly models "resource allocation + parallel strategy + request routing" for LLM serving as a two-level maximum flow problem on flow networks. Combined with LSTM workload prediction and ad hoc model switching based on GPU interconnects, it addresses the heterogeneity of real-world traffic in both spatial (different request types) and temporal (varying composition over time) dimensions. End-to-end P99 latency and throughput are improved by an average of 1.5× and a maximum of 2× compared to vLLM.

**[Prism: Spectral-Aware Block-Sparse Attention](prism_spectral-aware_block-sparse_attention.md)**

:   Prism decomposes "block importance estimation" into high-frequency and low-frequency bands of RoPE, applying mean-pooling and softmax separately. It uses temperature automatically calibrated by energy ratios to align logit scales, enabling purely block-level operations (eliminating token-level search) while maintaining accuracy nearly identical to full attention. It achieves a 5.1× speedup over FlashAttention-2 at 128K.

**[ProactiveLLM: Learning Active Interaction for Streaming Large Language Models](proactivellm_learning_active_interaction_for_streaming_large_language_models.md)**

:   ProactiveLLM enables streaming LLMs to use their internal states (attention or prediction entropy) to decide "when to speak." By employing Masked Streaming Modeling + Synchronous Privileged Self-Distillation, it learns to perceive "semantic sufficiency" without relying on any external alignment annotations. This significantly reduces interaction latency with minimal performance degradation.

**[ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)**

:   ProbMoE reformulates MoE top-$k$ routing as "probabilistic inference over a cardinality-constrained subset distribution." It employs the SIMPLE estimator for sampling from an exact-$k$ subset distribution during the forward pass and uses analytically computed conditional marginal probabilities $m_j=\partial \log Z_k/\partial \log p_j$ as a differentiable proxy for discrete selection during the backward pass. This approach significantly improves performance on tasks like GSM8K, Law, and Translation for OLMoE and Qwen1.5-MoE while enhancing expert utilization. It also naturally extends to a Dynamic-$k$ variant that adaptively activates the number of experts based on token difficulty.

**[Proxy Compression for Language Modeling](proxy_compression_for_language_modeling.md)**

:   The authors propose "proxy compression"—training a model where 90% of the data is fed as short sequences produced by a tokenizer or neural compressor and 10% as raw UTF-8 bytes, combined with sentinel tokens and a brief in-context translation warm-up. During inference, all compressors are discarded, and the model operates solely on raw bytes. This approach significantly outperforms pure byte-level models under fixed compute and matches or exceeds tokenizer baselines at large scales.

**[ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference](remoe_boosting_expert_reuse_through_router_fine-tuning_in_memory-constrained_moe.md)**

:   ReMoE freezes all non-router parameters and fine-tunes only the gate using a compound loss of "temporal locality regularization + Trust-KL semantic anchor." This shapes the routing trajectory to be more "cache-friendly." Without changing the architecture or adding runtime overhead, it improves the expert reuse rate of adjacent tokens by approximately 26% and reduces TPOT by 43.6–49.8% on Jetson Orin NX (achieving a 1.77–1.99× decoding speedup).

**[RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress](repetitioncurse_measuring_and_understanding_router_imbalance_in_mixture-of-exper.md)**

:   By providing MoE LLMs with extremely simple OOD prompts that repeat the same token $N$ times, the authors find that the router directs almost all tokens to a fixed small set of top-$k$ experts. Under Expert Parallelism (EP) deployment, this creates a single-GPU bottleneck while idling other GPUs, increasing TTFT by 20%–148% on an 8-GPU cluster and turning the MoE parallel accelerator into a DoS attack surface.

**[Scout: Active Information Foraging for Long-Text Understanding with Decoupled Epistemic States](scout_active_information_foraging_for_long-text_understanding_with_decoupled_epi.md)**

:   Scout remodels million-token long-text understanding (LTU) as an "active information foraging" process. It introduces a provenance-anchored epistemic state $\mathcal{E}_t$, decoupled from the interaction trajectory, as the sole base for reasoning. Through gap-diagnosed self-evaluation, it iteratively converges to a sufficient subset of information. On LooGLE-v2 and $\infty$Bench, Scout matches or exceeds frontier models like Gemini-3-Pro while reducing token costs to approximately $1/8$.

**[SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm](siamesenorm_breaking_the_barrier_to_reconciling_prepost-norm.md)**

:   Addressing the structural conflict where Pre-Norm and Post-Norm cannot coexist within a single-stream architecture, the authors propose SiameseNorm, a dual-stream residual architecture. It maintains one unnormalized stream to preserve the Pre-Norm identity gradient highway and one normalized stream to retain Post-Norm representation control. By coupling these streams through shared residual blocks, it consistently outperforms Pre-Norm baselines across 400M~15B dense/MoE language models, ViT, and DiT with negligible overhead.

**[Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills](skill-based_mixture-of-experts_adaptive_routing_for_heterogeneous_reasoning_via_.md)**

:   SKILL-MOE proposes a training-free symbolic MoE framework that uses "skills" as routing signals. It extracts required skills for each question, dynamically recruits $k$ experts from 16 pre-trained LLMs based on skill-model profiles, and merges multiple CoT solutions into a final answer using a task-level optimal aggregator. Coupled with expert-bucketed batch inference, it allows 16 7-8B models to run on a single GPU, outperforming the strongest multi-agent baselines by 8.15% on average.

**[Sparser Block-Sparse Attention via Token Permutation](sparser_block-sparse_attention_via_token_permutation.md)**

:   Ours proposes PBS-Attn, which leverages the permutation invariance of attention to rearrange Keys within segments based on "global importance." This clusters scattered "heavy hitters" into continuous high-density blocks before performing block-sparse computation, achieving up to 2.75x end-to-end acceleration for long-context prefilling while nearly matching full attention accuracy.

**[Stochastic Sparse Attention for Memory-Bound Inference](stochastic_sparse_attention_for_memory-bound_inference.md)**

:   SANTA reinterprets the value aggregation $AV$ of attention as "weighted summation of value rows $V$ based on softmax probabilities $A$." It transforms this into an unbiased estimate by sampling $S \ll n_k$ indices from $A$ without replacement and directly averaging the corresponding $V$ rows. By utilizing stratified/systematic sampling to reduce variance and implementing a GPU kernel aligned with FlashDecoding, it achieves a 1.5× end-to-end speedup over FlashInfer/FlashDecoding under 32k context without accuracy degradation.

**[TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration](team_temporal-spatial_consistency_guided_expert_activation_for_moe_diffusion_lan.md)**

:   TEAM addresses the inherent mismatch in MoE Diffusion Language Models (dLLM) where "many experts are activated for only a few accepted tokens." By leveraging temporal and spatial consistency within block decoding, it designs differentiated expert activation and decoding strategies for decoded, hot, and cold tokens, achieving up to 2.2× speedup on SDAR 30B-A3B with near-zero accuracy loss.

**[Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](theoretically_optimal_attentionffn_ratios_in_disaggregated_llm_serving.md)**

:   This work provides the first theoretical framework for the emerging Attention-FFN Disaggregation (AFD) inference architecture. Based on a probabilistic workload model where prefill lengths have a finite mean and decode lengths follow a geometric distribution, it derives a closed-form solution for the optimal A/F ratio $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$ under an rA-1F topology. Theoretical predictions are validated with a trace-calibrated simulator, showing a deviation of <10% from measured optimal values.

**[Training-Inference Consistent Segmented Execution for Long-Context LLMs](training-inference_consistent_segmented_execution_for_long-context_llms.md)**

:   This paper proposes a long-context LLM framework that shares identical segmented forward execution semantics for both training and inference: it maintains a fixed-length differentiable KV tail across segments plus a forward-only retrieval bypass. On LLaMA2-7B 32K/80K, it achieves LongBench/RULER performance comparable to or better than full attention with approximately $6\times$ lower peak prefill memory.

**[Variational Routing: A Scalable Bayesian Framework for Calibrated MoE Transformers](variational_routing_a_scalable_bayesian_framework_for_calibrated_mixture-of-expe.md)**

:   Proposes the variational routing framework VMoER—by performing variational inference on the routing decisions of MoE layers rather than weight inference, it achieves efficient Bayesian uncertainty modeling. It reduces calibration error by 94% and improves routing stability by 38% while maintaining <1% additional FLOPs overhead.

**[WarmServe: Multi-model GPU Warm-up Mechanism via Load-Once-Many](warmserve_enabling_one-for-many_gpu_prewarming_for_multi-llm_serving.md)**

:   WarmServe reduces tail TTFT by 50.8x compared to existing systems by analyzing long-term periodic patterns in LLM serving workloads. It proactively preloads multiple model parameters into GPUs, using optimized placement algorithms and dynamic KV cache reservation strategies to quickly launch new instances during request bursts.
