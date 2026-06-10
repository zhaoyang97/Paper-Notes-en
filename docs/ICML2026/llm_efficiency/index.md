---
title: >-
  ICML2026 LLM Efficiency Papers · 37 Notes
description: >-
  37 ICML2026 papers in the LLM Efficiency area, covering LLM, Diffusion Models, Compression and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ICML2026"
  - "LLM Efficiency"
  - "AI paper notes"
  - "paper summaries"
  - "LLM"
  - "Diffusion Models"
  - "Compression"
item_list:
  - u: "a_queueing-theoretic_framework_for_stability_analysis_of_llm_inference_with_kv_c/"
    t: "A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints"
  - u: "beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_/"
    t: "Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts"
  - u: "criticalkv_optimizing_kv_cache_eviction_from_an_output_perturbation_perspective/"
    t: "CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective"
  - u: "do_transformers_need_three_projections_systematic_study_of_qkv_variants/"
    t: "Do Transformers Need Three Projections? A Systematic Study of QKV Sharing Schemes"
  - u: "dot-moe_differentiable_optimal_transport_for_moefication/"
    t: "DOT-MoE: Converting Dense LLMs to MoE with Differentiable Optimal Transport"
  - u: "efficient_training-free_multi-token_prediction_via_embedding-space_probing/"
    t: "Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing"
  - u: "ekka_automated_diagnosis_of_silent_errors_in_llm_inference/"
    t: "Ekka: Automated Diagnosis of Silent Errors in LLM Inference"
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
  - u: "not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving/"
    t: "Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving"
  - u: "obcache_optimal_brain_kv_cache_pruning_for_efficient_long-context_llm_inference/"
    t: "OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference"
  - u: "optimal_bayesian_stopping_for_efficient_inference_of_consistent_llm_answers/"
    t: "Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers"
  - u: "oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration/"
    t: "OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration"
  - u: "pipesd_an_efficient_cloud-edge_collaborative_pipeline_inference_framework_with_s/"
    t: "PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding"
  - u: "plan_for_speed_dilated_scheduling_for_masked_diffusion_language_models/"
    t: "Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models"
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
  - u: "slay_geometry-aware_spherical_linearized_attention_with_yat-kernel/"
    t: "SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel"
  - u: "sparser_block-sparse_attention_via_token_permutation/"
    t: "Sparser Block-Sparse Attention via Token Permutation"
  - u: "stochastic_sparse_attention_for_memory-bound_inference/"
    t: "Stochastic Sparse Attention for Memory-Bound Inference"
item_total: 37
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# ⚡ LLM Efficiency

**🧪 ICML2026** · **37** paper notes

📌 **Same area in other venues:** [💬 ACL2026 (22)](../../ACL2026/llm_efficiency/index.md) · [📷 CVPR2026 (4)](../../CVPR2026/llm_efficiency/index.md) · [🔬 ICLR2026 (19)](../../ICLR2026/llm_efficiency/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_efficiency/index.md) · [🧠 NeurIPS2025 (35)](../../NeurIPS2025/llm_efficiency/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/llm_efficiency/index.md)

🔥 **Top topics:** LLM ×12 · Diffusion Models ×2 · Compression ×2

**[A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints](a_queueing-theoretic_framework_for_stability_analysis_of_llm_inference_with_kv_c.md)**

:   This work establishes the first queueing model for LLM inference that explicitly incorporates the dynamic behavior of KV cache memory, deriving a closed-form stability condition $\lambda < \mu(1-\delta)$, enabling operators to directly compute the required number of GPUs. Validation on single GPU, 8-GPU clusters, and LongBench real data shows prediction error within $10\%$.

**[Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)**

:   This work proposes an "orthogonal growth" strategy for converged MoE models—utilizing interpositional layer replication for the depth dimension and noisy expert cloning for the width dimension. By expanding a 17B model to 70B, it achieves a 10.6% accuracy improvement over training from scratch under the same additional computational budget.

**[CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective](criticalkv_optimizing_kv_cache_eviction_from_an_output_perturbation_perspective.md)**

:   The authors formalize the empirical question of "which KV cache entries are critical" as an optimization problem of minimizing attention output perturbation. They derive an analytical upper bound for perturbation (incorporating both attention weights and value norms projected via $W^O$) and design a plug-and-play two-stage greedy selection algorithm. This approach reduces the average compression loss of three SOTA methods—SnapKV, AdaKV, and HeadKV—by more than half across 29 long-context datasets.

**[Do Transformers Need Three Projections? A Systematic Study of QKV Sharing Schemes](do_transformers_need_three_projections_systematic_study_of_qkv_variants.md)**

:   The paper systematically compares three QKV projection sharing schemes: Q=K-V (shared query and key), Q-K=V (shared key and value), and Q=K=V (all three shared). It finds that Q-K=V in Language Modeling (LM) increases Perplexity (PPL) by only 3.1% while reducing KV cache by 50%. This approach is orthogonal to GQA/MQA, allowing for a combined 87.5%-96.9% cache reduction, providing quantifiable memory benefits for edge inference.

**[DOT-MoE: Converting Dense LLMs to MoE with Differentiable Optimal Transport](dot-moe_differentiable_optimal_transport_for_moefication.md)**

:   DOT-MoE models the problem of "how to allocate neurons to experts when converting a dense FFN to an MoE" as differentiable optimal transport. By using Sinkhorn-Knopp iterations to solve entropic-regularized balanced transport and a Straight-Through Estimator, it enables the joint end-to-end learning of neuron-to-expert assignment and the router. It retains 90% of dense performance with 50% active parameters on LLaMA-2/3 and Qwen2.5, outperforming all baselines such as structured pruning, random allocation, and clustering.

**[Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing](efficient_training-free_multi-token_prediction_via_embedding-space_probing.md)**

:   This paper proposes ESP (Embedding-Space Probing): without modifying any weights or training auxiliary models, it injects the "mean prompt embedding" as mask tokens into the input sequence of a frozen LLM. By probing multiple future tokens in a single forward pass and performing lossless speculative verification using the base model itself, ESP achieves 7–11% higher average acceptance lengths and 15–19% higher throughput than training-free baselines (LADE / STAND / PLD) on LLaMA3 and Qwen3.

**[Ekka: Automated Diagnosis of Silent Errors in LLM Inference](ekka_automated_diagnosis_of_silent_errors_in_llm_inference.md)**

:   Ekka reformulates the diagnosis of "silent errors"—where LLM serving frameworks produce degraded outputs without throwing errors—into a differential debugging task using reference implementations like HuggingFace as an oracle. By utilizing an agentic pipeline of "component mapping $\rightarrow$ activation alignment $\rightarrow$ change-point analysis," Ekka automatically locates specific faulty modules. It achieves 80% pass@1 and 88% pass@5 accuracy across 17 real-world vLLM/SGLang issues and discovered 4 new hidden bugs confirmed by developers.

**[GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving](graphflow_a_graph-based_workflow_management_for_efficient_llm-agent_serving.md)**

:   GraphFlow unifies multiple agent workflows into a single global operational DAG (wGraph). It generates task-specific subgraph workflows online using GNN+MLP and replaces traditional independent workflow caching with a differential caching strategy ("Base KV + Sparse Prefix Residual + Path Pruning"). Across 5 reasoning/code/QA benchmarks, it achieves an average improvement of 4.95pp while compressing KV memory to approximately 1/4.

**[Hyperparameter Transfer with Mixture-of-Experts Layers](hyperparameter_transfer_with_mixture-of-expert_layers.md)**

:   This paper extends the maximal update parametrization (mUP/CompleteP) to sparse MoE Transformers. It introduces initialization and learning rate (LR) scaling rules for the router, expert up/down projections, and expert biases when width, depth, number of experts, and expert width are simultaneously scaled. Using a triple-layer mean-field Dynamical Mean-Field Theory (DMFT), the authors prove that this parametrization possesses a scale-invariant limit as $n_{\text{embd}}, n_{\text{exp}}, n_{\text{hid}}, L \to \infty$ (under fixed active sparsity $\kappa$). Optimal LR and initialization can be directly reused from a 38M active parameter base model up to a 2B total parameter MoE. Zero-shot HP-tuned MoEs match or outperform dense GPT2 speedrun results at equivalent active parameter counts.

**[KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem](knapspec_self-speculative_decoding_via_adaptive_layer_selection_as_a_knapsack_pr.md)**

:   KnapSpec reformulates draft layer selection in Self-Speculative Decoding (SSD) as a 0/1 knapsack problem. It decouples Attention and MLP modules, utilizes context-length-dependent hardware latency as "weight," and employs hidden state cosine similarity (with the first rigorous proof provided) as "value." Through parallel dynamic programming (DP), it adaptively identifies sub-networks that maximize Tokens-per-Time at each step, achieving up to 1.47× real-world wall-clock speedup on Qwen3/Llama3 in long-context scenarios without additional training.

**[L$^3$: Large Lookup Layers](l3_large_lookup_layers.md)**

:   This paper proposes L$^3$ (Large Lookup Layer), which generalizes the tokenizer embedding table into a "large lookup layer" that can be inserted into the decoder. It utilizes **static routing** based on token IDs to retrieve a set of learned key/value embeddings, which are then aggregated by current hidden states via attention. This increases model sparsity by an additional order of magnitude without the typical MoE issues of dynamic routing, auxiliary losses, and offloading difficulties. It outperforms dense models of comparable compute and MoEs of comparable sparsity across 800M–2.6B active parameters.

**[MineDraft: A Framework for Batch Parallel Speculative Decoding](minedraft_a_framework_for_batch_parallel_speculative_decoding.md)**

:   MineDraft transforms the originally serial "draft-verify" pipeline of speculative decoding into batch parallel PSD by maintaining two batches of requests and allowing the drafting of one batch to **overlap execution** with the verification of the other on independent GPUs. At the cost of only one additional GPU, it increases throughput by up to 75% and reduces end-to-end latency by up to 39% compared to standard SD, and is implemented as a plug-and-play vLLM plugin.

**[Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)**

:   This paper identifies that in multi-turn dialogue scenarios, the traditional Prefill-Decode (PD) disaggregation architecture is highly inefficient due to repeated P→D recomputation and KV transmission at every turn. It proposes the PPD (Prefill-capable Decode) dynamic routing system, allowing decode nodes to decide—based on SLO weights—whether to locally process Turn 2+ append-prefill. This reduces Turn 2+ TTFT by approximately 68%.

**[OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference](obcache_optimal_brain_kv_cache_pruning_for_efficient_long-context_llm_inference.md)**

:   Ours reformulates KV cache eviction as a "layer-wise structured pruning" problem. By leveraging the second-order Taylor approximation from Optimal Brain Damage, the authors derive closed-form saliency scores for independent value units, independent key units, and joint key-value units. These scores serve as plug-and-play "saliency replacements" for existing attention-only eviction frameworks like H2O, TOVA, SnapKV, and AdaKV. Ours achieves consistent gains on RULER and LongBench for LLaMA-3.1 / Qwen-2.5 (e.g., AdaKV's accuracy increases by nearly 15% on query-agnostic RULER-4K with a 30% budget).

**[Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers](optimal_bayesian_stopping_for_efficient_inference_of_consistent_llm_answers.md)**

:   This paper models the "Self-Consistency (SC) majority voting via multiple sampling" problem as a Bayesian optimal stopping problem with prior information. It proposes an $L$-aggregated posterior approximation that tracks only three categories of counts: "most frequent / second most frequent / others." The authors theoretically prove that $L=3$ achieves an asymptotic optimal stopping time identical to the exact posterior as $\delta \to 0$. Experimentally, it saves 30%–80% of LLM calls on GSM8K and CommonsenseQA at approximately 1.4x the speed of ASC.

**[OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration](oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration.md)**

:   OServe models the joint optimization of "resource allocation + parallel strategy + request routing" for LLM serving as a bilevel maximum flow problem on a flow network. Combined with LSTM workload prediction and ad hoc model switching based on GPU interconnects, it addresses the heterogeneity of real-world traffic in both spatial (different request types) and temporal (varying composition over time) dimensions. Compared to vLLM, it improves end-to-end P99 latency and throughput by an average of 1.5$\times$ and up to 2$\times$.

**[PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](pipesd_an_efficient_cloud-edge_collaborative_pipeline_inference_framework_with_s.md)**

:   This paper proposes PipeSD: transforming speculative decoding from sequential cloud-edge execution to a token-batch pipeline, replacing fixed draft length with dual-threshold NAV triggering and Bayesian autotuning. On a real 5G cloud-edge testbed, PipeSD achieves 1.16×–2.16× speedup and 14–25% reduction in cloud energy consumption.

**[Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models](plan_for_speed_dilated_scheduling_for_masked_diffusion_language_models.md)**

:   This work proposes the Dilated Unmasking Scheduler (DUS): by using a "dilated, equidistant" predefined unmasking order that does not rely on model confidence, the number of denoiser calls per block of $B$ tokens is reduced from $\mathcal O(B)$ to $\mathcal O(\log B)$. On LLaDA / Dream / DiffuCoder, this achieves a 5.8× wall-clock speedup with quality surpassing confidence-based parallel planners.

**[Prism: Spectral-Aware Block-Sparse Attention](prism_spectral-aware_block-sparse_attention.md)**

:   Prism decomposes "block importance estimation" into high-frequency and low-frequency bands of RoPE using mean-pooling and softmax separately. It automatically calibrates logit magnitudes using a temperature derived from energy ratios, enabling purely block-level operations (eliminating token-level search) to achieve accuracy nearly identical to full attention, while reaching a 5.1× speedup over FlashAttention-2 at 128K context.

**[ProactiveLLM: Learning Active Interaction for Streaming Large Language Models](proactivellm_learning_active_interaction_for_streaming_large_language_models.md)**

:   ProactiveLLM enables streaming LLMs to use their internal states (attention or predictive entropy) to decide "when to speak." By employing masked streaming modeling and synchronous privileged self-distillation, it learns to perceive "semantic sufficiency" without relying on any external alignment annotations, significantly reducing interaction latency while maintaining performance.

**[ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)**

:   ProbMoE reformulates top-$k$ routing as "probabilistic inference over cardinality-constrained subset distributions." It uses the SIMPLE estimator to sample from the exact-$k$ subset distribution during the forward pass and employs analytically computed expert marginal probabilities $m_j=\partial \log Z_k/\partial \log p_j$ as a differentiable proxy for discrete selection during the backward pass. This approach significantly improves performance on GSM/Law/Translation tasks for OLMoE and Qwen1.5-MoE while notably enhancing expert utilization. It also naturally extends to a Dynamic-$k$ variant that adaptively activates expert counts based on token difficulty.

**[Proxy Compression for Language Modeling](proxy_compression_for_language_modeling.md)**

:   The authors propose "proxy compression"—where 90% of the training data is fed as short sequences produced by a tokenizer or neural compressor, and 10% is fed as raw UTF-8 bytes, complemented by sentinel tokens and brief in-context translation warm-up. During inference, all compressors are discarded, and the model operates solely on raw bytes. This approach significantly outperforms pure byte-level models under fixed compute budgets and matches or exceeds tokenizer baselines at scale.

**[ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference](remoe_boosting_expert_reuse_through_router_fine-tuning_in_memory-constrained_moe.md)**

:   ReMoE freezes all non-router parameters and fine-tunes only the gate using a compound loss comprising "temporal locality regularization + Trust-KL semantic anchor." This reshapes the routing trajectory to be more "cache-friendly." Without altering the architecture or adding runtime overhead, it increases the expert reuse rate of adjacent tokens by approximately 26%, reducing TPOT by 43.6–49.8% (1.77–1.99× decoding speedup) on Jetson Orin NX.

**[RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress](repetitioncurse_measuring_and_understanding_router_imbalance_in_mixture-of-exper.md)**

:   By feeding Mixture-of-Experts (MoE) LLMs minimalist OOD prompts that repeat the same token $N$ times, the authors discover that the router directs nearly all tokens to a fixed set of few top-$k$ experts. Under Expert Parallelism (EP) deployment, this creates a single-GPU bottleneck while idling other GPUs, increasing TTFT by 20%–148% on 8-GPU clusters. This effectively turns the MoE parallel accelerator into a DoS attack surface.

**[Scout: Active Information Foraging for Long-Text Understanding with Decoupled Epistemic States](scout_active_information_foraging_for_long-text_understanding_with_decoupled_epi.md)**

:   Scout remodels million-token long-text understanding (LTU) as an "active information foraging" process. It introduces an epistemic state $\mathcal{E}_t$, decoupled from the interaction trajectory and anchored with provenance points, as the sole reasoning foundation. Through gap-diagnosed self-evaluation, it iteratively converges to a query-sufficient subset. It matches or exceeds frontier models like Gemini-3-Pro on LooGLE-v2 and $\infty$Bench while reducing token costs to approximately $1/8$.

**[SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm](siamesenorm_breaking_the_barrier_to_reconciling_prepost-norm.md)**

:   Addressing the structural contradiction where Pre-Norm and Post-Norm cannot coexist within a single-stream architecture, the authors propose SiameseNorm, a dual-stream residual architecture. It maintains an unnormalized stream as the Pre-Norm identity gradient highway and a normalized stream for Post-Norm representation control. By coupling both streams via shared residual blocks, it consistently outperforms Pre-Norm baselines across 400M~15B dense/MoE language models, ViT, and DiT with negligible overhead.

**[Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills](skill-based_mixture-of-experts_adaptive_routing_for_heterogeneous_reasoning_via_.md)**

:   SKILL-MOE proposes a training-free, symbolic MoE framework that uses "skills" as routing signals: it extracts required skills from each problem, dynamically recruits $k$ experts from 16 pre-trained LLMs based on skill-model profiles, and merges multiple CoT paths into a final answer using a task-level optimal aggregator. Coupled with expert-wise batch inference, it enables running 16 7-8B models on a single GPU, achieving an average accuracy 8.15% higher than the strongest multi-agent baselines.

**[SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel](slay_geometry-aware_spherical_linearized_attention_with_yat-kernel.md)**

:   SLAY linearizes the Yat-kernel, inspired by the physical "inverse-square interaction," through four steps: (1) spherical normalization, (2) Laplace integral representation via Bernstein theorem, (3) Gauss-Laguerre quadrature, and (4) tensor product positive random features for polynomial+exponential kernels. This yields an $O(L)$ attention mechanism nearly indistinguishable from softmax.

**[Sparser Block-Sparse Attention via Token Permutation](sparser_block-sparse_attention_via_token_permutation.md)**

:   This paper proposes PBS-Attn, which leverages the permutation invariance of attention. It first reorders keys within segments based on "global importance" to aggregate scattered heavy hitters into contiguous high-density blocks. This allows for block-sparse computation that near-perfectly maintains full attention accuracy while achieving up to a 2.75x end-to-end acceleration in long-context prefilling.

**[Stochastic Sparse Attention for Memory-Bound Inference](stochastic_sparse_attention_for_memory-bound_inference.md)**

:   SANTA treats the value aggregation $AV$ in attention as "a weighted sum of value rows $V$ according to softmax probabilities $A$." It transforms this into an unbiased estimate by "sampling $S \ll n_k$ indices from $A$ without replacement and directly averaging the corresponding $V$ rows." Using stratified/systematic sampling to reduce variance and implemented as a GPU kernel aligned with FlashDecoding, it achieves a 1.5× end-to-end speedup over FlashInfer/FlashDecoding under a 32k context without accuracy degradation.

**[TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration](team_temporal-spatial_consistency_guided_expert_activation_for_moe_diffusion_lan.md)**

:   TEAM addresses the inherent mismatch in MoE Diffusion Language Models (dLLM) where "a large number of experts are activated while only few tokens are accepted." By utilizing temporal and spatial consistency during in-block decoding, TEAM designs differentiated expert activation and decoding strategies for decoded, hot, and cold tokens. It achieves up to 2.2× speedup on SDAR 30B-A3B with near-zero precision loss.

**[Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](theoretically_optimal_attentionffn_ratios_in_disaggregated_llm_serving.md)**

:   This work provides the first theoretical framework for the emerging Attention-FFN Disaggregation (AFD) inference architecture. Based on a probabilistic workload model where prefill lengths have a finite mean and decode lengths follow a geometric distribution, it derives a closed-form solution for the optimal A/F ratio $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$ under an rA-1F topology. Theoretical predictions are validated with a trace-calibrated simulator, showing a deviation of <10% from measured optimal values.

**[Towards Resource-Efficient LLMs: End-to-End Energy Accounting of Distillation Pipelines](towards_resource-efficient_llms_end-to-end_energy_accounting_of_distillation_pip.md)**

:   The authors built a staged GPU energy measurement framework based on NVML, decomposing the distillation pipeline into "teacher side + student side + evaluation" for segment-wise accounting. They found that one-off teacher logit caching/synthetic data generation dominates energy use, causing KD and synthetic SFT to consume about $2.4\times$ more energy than direct SFT for 1B–13B OLMo-2 students. They provide a closed-form break-even formula, showing distillation is only truly "energy-saving" when teacher outputs are reused more than $N^*$ times.

**[Training-Inference Consistent Segmented Execution for Long-Context LLMs](training-inference_consistent_segmented_execution_for_long-context_llms.md)**

:   This paper proposes a long-context LLM framework where training and inference share identical segmented forward execution semantics: retaining only a fixed-length differentiable KV tail across segments plus a forward-only retrieval bypass. On LLaMA2-7B 32K/80K, it achieves LongBench/RULER performance comparable to or even better than full attention with approximately $6\times$ lower peak prefill VRAM.

**[Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference](understand_and_accelerate_memory_processing_pipeline_for_disaggregated_llm_infer.md)**

:   This paper unifies optimizations in modern LLM long-context inference—such as sparse attention, RAG, and compressed context memory—into a four-stage "Prepare Memory → Compute Relevancy → Retrieval → Apply to Inference" memory processing pipeline. It quantitatively demonstrates that this pipeline accounts for 22%-97% of total latency and that each stage exhibits highly heterogeneous computational characteristics. Based on this, a GPU-FPGA heterogeneous system is proposed: regular/compute-intensive operations remain on the GPU, while sparse/irregular/memory-intensive operations are offloaded to the FPGA. On MI210 + Alveo U55C, up to 2.2× end-to-end speedup and 4.7× energy reduction are achieved.

**[Variational Routing: A Scalable Bayesian Framework for Calibrated MoE Transformers](variational_routing_a_scalable_bayesian_framework_for_calibrated_mixture-of-expe.md)**

:   The variational routing framework VMoER is proposed—performing variational inference on the routing decisions of MoE layers rather than weight inference leads to efficient Bayesian uncertainty modeling. It reduces calibration error by 94% and improves routing stability by 38% while maintaining <1% additional FLOPs overhead.

**[WarmServe: A Multi-Model Loading GPU Warm-up Mechanism](warmserve_enabling_one-for-many_gpu_prewarming_for_multi-llm_serving.md)**

:   WarmServe proactively pre-loads multiple model parameters onto GPUs by analyzing long-term periodic patterns of LLM workloads. Combined with optimized placement algorithms and dynamic KV cache reservation strategies, the system enables rapid instantiation of new instances during request bursts—reducing tail TTFT by 50.8× compared to existing systems.
