---
title: >-
  ICML2026 LLM Efficiency Papers · 48 Notes
description: >-
  48 ICML2026 papers in the LLM Efficiency area, covering LLM, Diffusion Models, Compression, Reasoning and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ICML2026"
  - "LLM Efficiency"
  - "AI paper notes"
  - "paper summaries"
  - "LLM"
  - "Diffusion Models"
  - "Compression"
  - "Reasoning"
item_list:
  - u: "a_risk_decomposition_framework_for_pre-hoc_fine-tuning_prediction/"
    t: "A Risk Decomposition Framework for Pre-Hoc Fine-Tuning Prediction"
  - u: "beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_/"
    t: "Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts"
  - u: "criticalkv_optimizing_kv_cache_eviction_from_an_output_perturbation_perspective/"
    t: "CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective"
  - u: "diffusion_language_model_parallel_decoding_via_product-of-experts_bridge/"
    t: "Diffusion Language Model Parallel Decoding via Product-of-Experts Bridge"
  - u: "dllm-cache_accelerating_diffusion_large_language_models_with_adaptive_caching/"
    t: "dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching"
  - u: "do_transformers_need_three_projections_systematic_study_of_qkv_variants/"
    t: "Do Transformers Need Three Projections? A Systematic Study of QKV Sharing Systems"
  - u: "dot-moe_differentiable_optimal_transport_for_moefication/"
    t: "DOT-MoE: Transforming Dense LLMs into MoE with Differentiable Optimal Transport"
  - u: "dynamic_linear_attention/"
    t: "Dynamic Linear Attention"
  - u: "efficient_training-free_multi-token_prediction_via_embedding-space_probing/"
    t: "Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing"
  - u: "ekka_automated_diagnosis_of_silent_errors_in_llm_inference/"
    t: "Ekka: Automated Diagnosis of Silent Errors in LLM Inference"
  - u: "fast-dllm_fréchet_profile_decoding_for_faster_diffusion_llm_inference/"
    t: "Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference"
  - u: "focus_dllms_know_how_to_tame_their_compute_bound/"
    t: "FOCUS: DLLMs Know How to Tame Their Compute Bound"
  - u: "graphflow_a_graph-based_workflow_management_for_efficient_llm-agent_serving/"
    t: "GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving"
  - u: "hyperparameter_transfer_with_mixture-of-expert_layers/"
    t: "Hyperparameter Transfer with Mixture-of-Experts Layers"
  - u: "ir3de_a_linear_router_for_large_language_models/"
    t: "IR3DE: A Linear Router for Large Language Models"
  - u: "kalman_linear_attention_parallel_bayesian_filtering_for_efficient_language_model/"
    t: "Kalman Linear Attention: Parallel Bayesian Filtering For Efficient Language Modelling and State Tracking"
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
  - u: "q-delta_beyond_key-value_associative_state_evolution/"
    t: "Q-Delta: Beyond Key–Value Associative State Evolution"
  - u: "remoe_boosting_expert_reuse_through_router_fine-tuning_in_memory-constrained_moe/"
    t: "ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference"
  - u: "repetitioncurse_measuring_and_understanding_router_imbalance_in_mixture-of-exper/"
    t: "RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress"
  - u: "repo_language_models_with_context_re-positioning/"
    t: "RePo: Language Models with Context Re-Positioning"
item_total: 48
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# ⚡ LLM Efficiency

**🧪 ICML2026** · **48** paper notes

📌 **Same area in other venues:** [📷 CVPR2026 (8)](../../CVPR2026/llm_efficiency/index.md) · [🔬 ICLR2026 (171)](../../ICLR2026/llm_efficiency/index.md) · [💬 ACL2026 (23)](../../ACL2026/llm_efficiency/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_efficiency/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/llm_efficiency/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/llm_efficiency/index.md)

🔥 **Top topics:** LLM ×14 · Diffusion Models ×5 · Compression ×2 · Reasoning ×2

**[A Risk Decomposition Framework for Pre-Hoc Fine-Tuning Prediction](a_risk_decomposition_framework_for_pre-hoc_fine-tuning_prediction.md)**

:   Fine-tuning LLMs is expensive and hard to predict. This paper formalizes "predicting final fine-tuning performance before or during early training" as a stochastic estimation problem under information constraints. It decomposes prediction risk into an **irreducible intrinsic limit (static data-model compatibility) + reducible optimization variance**. It proves a mandatory lower bound of $c^{-\alpha}$ for the decay rate of optimization variance (no predictor can exceed this speed), derives budget-optimal stopping conditions, and organizes tasks into three predictability regimes—Static-Sufficient, Dynamic-Critical, and Noise-Dominant—using the "intrinsic limit × decay rate" axes, explaining why shallow probing suffices for SST-2 but fails for GSM8K.

**[Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)**

:   The authors propose an "orthogonal growth" strategy for converged MoE models—using interpositional layer replication for depth and noisy expert cloning for width—scaling a 17B model to 70B. This achieve a 10.6% accuracy improvement over training from scratch under the same additional compute budget.

**[CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective](criticalkv_optimizing_kv_cache_eviction_from_an_output_perturbation_perspective.md)**

:   The authors reframe the heuristic-based problem of "identifying critical KV cache entries" as an optimization problem of "minimizing attention output perturbation." They derive an analytical upper bound for perturbation (weighted by both attention weights and value norms projected via $W^O$) and design a plug-and-play two-stage greedy selection algorithm. This method reduces the compression loss of SOTA eviction approaches like SnapKV, AdaKV, and HeadKV by more than half on average across 29 long-context datasets.

**[Diffusion Language Model Parallel Decoding via Product-of-Experts Bridge](diffusion_language_model_parallel_decoding_via_product-of-experts_bridge.md)**

:   Diffusion Language Models (DLMs) enable parallel decoding but suffer from poor quality. Directly using Monte Carlo methods to correct DLM drafts toward an Autoregressive (AR) target is computationally expensive due to the massive distribution gap. This paper proposes PoE-Bridge, which inserts a Product-of-Experts intermediate bridge distribution between the DLM and AR models. This decomposes the difficult "DLM $\to$ AR" correction into two easier "DLM $\to$ PoE $\to$ AR" steps. Combined with mixed-temperature sampling and elastic rejection windows, it accelerates standard DLM decoding by up to 5$\times$ on mathematical reasoning and coding tasks while recovering at least 95% of AR accuracy.

**[dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](dllm-cache_accelerating_diffusion_large_language_models_with_adaptive_caching.md)**

:   To address the bottleneck where diffusion Large Language Models (dLLMs) suffer from extremely slow inference due to bidirectional attention and the inability to reuse KV caches, this paper proposes dLLM-Cache. This training-free method applies long-interval caching for static prompts and short-interval refreshing for dynamic responses. By using Value cosine similarity (V-verify) to select and recompute the top 25% most "active" tokens, it achieves up to 9.1× FLOPs acceleration on LLaDA 8B / Dream 7B with almost no drop in performance.

**[Do Transformers Need Three Projections? A Systematic Study of QKV Sharing Systems](do_transformers_need_three_projections_systematic_study_of_qkv_variants.md)**

:   The paper systematically compares three QKV projection sharing schemes: Q=K-V (shared query and key), Q-K=V (shared key and value), and Q=K=V (all three shared). It finds that for Language Modeling (LM), Q-K=V increases Perplexity (PPL) by only 3.1% while reducing the KV cache by 50%. This approach is orthogonal to GQA/MQA, enabling a total cache reduction of 87.5%–96.9%, providing quantifiable memory benefits for edge inference.

**[DOT-MoE: Transforming Dense LLMs into MoE with Differentiable Optimal Transport](dot-moe_differentiable_optimal_transport_for_moefication.md)**

:   DOT-MoE models the "allocation of neurons to experts when converting a dense FFN to an MoE" as a differentiable optimal transport problem. It employs Sinkhorn-Knopp iterations to solve entropic-regularized balanced transport combined with a Straight-Through Estimator, allowing joint end-to-end learning of neuron-to-expert assignment and the router. It retains 90% of dense performance under 50% active parameters on LLaMA-2/3 and Qwen2.5, outperforming all baselines including structured pruning, random allocation, and clustering.

**[Dynamic Linear Attention](dynamic_linear_attention.md)**

:   Addressing the issue where existing "multi-state linear attention" mechanisms merge memory using fixed rules, causing critical tokens to be compressed into coarse summaries prematurely and accumulating errors, DLA proposes an **information-aware + capacity-constrained** dynamic memory framework. By using a lightweight "state information score" to adaptively determine when to create or merge memory states based on token-level information changes, and employing a fixed-size temporal cache to suppress state explosion, DLA consistently outperforms SOTA Log-Linear Attention across 16 datasets. Furthermore, the DLA version of Mamba-2 matches the performance of a full-attention Transformer with an equivalent number of parameters.

**[Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing](efficient_training-free_multi-token_prediction_via_embedding-space_probing.md)**

:   This paper proposes ESP (Embedding-Space Probing): without modifying any weights or training auxiliary models, it injects "mean prompt embeddings" as mask tokens into the input sequence of a frozen LLM. It probes multiple future tokens simultaneously in a single forward pass and uses the base model itself for lossless speculative verification. On LLaMA3 / Qwen3, it achieves 7–11% higher average acceptance length and 15–19% higher throughput than similar training-free baselines (LADE / STAND / PLD).

**[Ekka: Automated Diagnosis of Silent Errors in LLM Inference](ekka_automated_diagnosis_of_silent_errors_in_llm_inference.md)**

:   Ekka models the diagnosis of silent errors in LLM serving frameworks—where outputs degrade without explicit errors—as a differential debugging task using reference implementations like HuggingFace as an oracle. By employing an agentic pipeline of "component mapping $\rightarrow$ activation alignment $\rightarrow$ change-point analysis," it automatically localizes problematic modules. Ekka achieves a diagnosis accuracy of 80% pass@1 / 88% pass@5 across 17 real-world vLLM/SGLang issues and discovered 4 hidden bugs confirmed by developers.

**[Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference](fast-dllm_fréchet_profile_decoding_for_faster_diffusion_llm_inference.md)**

:   Addressing the parallel decoding bottleneck of diffusion language models (dLLMs), this paper proposes training-free Fréchet Profile Decoding. It uses the entire sorted confidence profile—rather than just the "weakest selected token"—to determine the number of tokens to commit per step. This strictly generalizes the factor rule of Fast-dLLM to heterogeneous confidence scenarios, achieving a 1.36× average throughput increase and 29% NFE reduction on LLaDA-8B across four benchmarks with almost no loss in accuracy.

**[FOCUS: DLLMs Know How to Tame Their Compute Bound](focus_dllms_know_how_to_tame_their_compute_bound.md)**

:   FOCUS finds that in Diffusion Large Language Models (DLLMs), only ~10% of tokens in a block are actually decoded per step, leaving 90% of the compute wasted. It reveals that the "incremental attention importance from the first two layers" highly predicts which tokens are decodable. Based on this, it designs a training-free inference system that evicts non-decodable tokens after Layer 1, allowing for larger effective batches. Compared to the production-grade engine LMDeploy, FOCUS achieves up to a 3.52× throughput increase under large batches with no loss (and even slight improvements) in generation quality.

**[GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving](graphflow_a_graph-based_workflow_management_for_efficient_llm-agent_serving.md)**

:   GraphFlow unifies multiple agent workflows into a global operational DAG (wGraph). It uses GNN+MLP to generate task-adaptive subgraph workflows online and replaces traditional independent caching with a differential KV cache strategy ("Base KV + Sparse Prefix Residual + Path Pruning"). This achieves an average improvement of 4.95pp across five benchmarks while reducing KV memory to approximately 1/4.

**[Hyperparameter Transfer with Mixture-of-Experts Layers](hyperparameter_transfer_with_mixture-of-expert_layers.md)**

:   This paper extends the maximal update parametrization (μP/CompleteP) to sparse MoE Transformers. It defines initialization and learning rate (LR) scaling rules for routers, expert up/down projections, and expert biases when model width, depth, number of experts, and expert width are simultaneously scaled. Using a three-level Mean-Field Dynamical Mean Field Theory (DMFT), the authors prove that this parametrization possesses a scale-invariant limit as $n_{\text{embd}}, n_{\text{exp}}, n_{\text{hid}}, L \to \infty$ (at fixed activation sparsity $\kappa$). Optimal LRs and initializations can be directly reused from 38M active parameter base models to 2B parameter MoEs. MoEs trained with zero-shot hyperparameters achieve performance comparable to or better than dense GPT2 speedrun models at equivalent active parameter counts.

**[IR3DE: A Linear Router for Large Language Models](ir3de_a_linear_router_for_large_language_models.md)**

:   This paper proposes IR3DE—a linear LLM router constructed using the **closed-form solution of ridge regression**. It routes each prompt to the most suitable domain expert based solely on token embeddings, eliminating the need to train additional language models or centralize datasets. It allows experts to be added or removed dynamically without retraining the router. Despite its linear nature, it achieves 98.4% normalized performance on reasoning tasks, surpassing all baselines.

**[Kalman Linear Attention: Parallel Bayesian Filtering For Efficient Language Modelling and State Tracking](kalman_linear_attention_parallel_bayesian_filtering_for_efficient_language_model.md)**

:   This work reinterprets sequence mixing as exact Bayesian filtering. By utilizing the "information form" of the Kalman filter, it reformulates the sequential recursive update into a parallelizable prefix scan using Möbius (fractional linear) mappings. The resulting KLA is a plug-and-play, linear-complexity sequence mixing layer that is more expressive than GLA and provides explicit state uncertainty.

**[KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem](knapspec_self-speculative_decoding_via_adaptive_layer_selection_as_a_knapsack_pr.md)**

:   KnapSpec reformulates draft layer selection in Self-Speculative Decoding (SSD) as a 0/1 knapsack problem. By decoupling Attention and MLP modules, using context-length-dependent hardware latency as "weight" and hidden state cosine similarity (with the first rigorous proof provided) as "value," it adaptively identifies the subnetwork that maximizes Tokens-per-Time via parallel DP at each step. It achieves up to 1.47× wall-clock speedup on Qwen3 / Llama3 in long-context scenarios without additional training.

**[L$^3$: Large Lookup Layers](l3_large_lookup_layers.md)**

:   The paper proposes L$^3$ (Large Lookup Layer), generalizing the tokenizer embedding table into "large lookup layers" that can be inserted into the decoder. By using **static routing** based on token IDs to retrieve a set of learned key/value embeddings and performing attention-based aggregation using the current hidden state, it achieves a higher level of model sparsity without the pain points of MoE (dynamic routing, auxiliary losses, offloading challenges). It outperforms dense models with equivalent compute and MoEs with equivalent sparsity across 800M–2.6B active parameters.

**[MineDraft: A Framework for Batch Parallel Speculative Decoding](minedraft_a_framework_for_batch_parallel_speculative_decoding.md)**

:   MineDraft maintains two batches of requests and **overlaps the execution** of drafting for one batch with the verification of another on two independent sets of GPUs. This transforms the sequential "draft-verify" pipeline of speculative decoding into batch-parallel PSD. At the cost of only one additional GPU, it increases throughput by up to 75% and reduces end-to-end latency by up to 39% compared to standard SD, and is implemented as a plug-and-play vLLM plugin.

**[OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference](obcache_optimal_brain_kv_cache_pruning_for_efficient_long-context_llm_inference.md)**

:   This paper reformulates KV cache eviction as a "layer-wise structural pruning" problem. By leveraging the second-order Taylor approximation from Optimal Brain Damage, it derives closed-form saliency scores for independent value pruning, independent key pruning, and joint key-value pruning units. These serve as plug-and-play "score replacements" for existing attention-only eviction frameworks such as H2O, TOVA, SnapKV, and AdaKV, achieving consistent improvements on LLaMA-3.1 and Qwen-2.5 across RULER and LongBench (e.g., AdaKV's performance increases by nearly 15% on query-agnostic RULER-4K with a 30% budget).

**[Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers](optimal_bayesian_stopping_for_efficient_inference_of_consistent_llm_answers.md)**

:   This paper models the "Self-Consistency (multiple sampling for majority vote)" problem as a Bayesian optimal stopping problem with prior information. It proposes an $L$-aggregated posterior approximation that tracks only three types of counts: "top-1 frequency, top-2 frequency, and others." The authors theoretically prove that $L=3$ achieves the same asymptotically optimal stopping time as the exact posterior as $\delta \to 0$. Experimentally, it saves 30%–80% of LLM calls on GSM8K and CommonsenseQA at approximately 1.4x the speed of ASC.

**[OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration](oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration.md)**

:   OServe jointly models LLM serving "resource allocation + parallel strategy + request routing" as a bi-level maximum flow problem on a flow network. Combined with LSTM-based workload prediction and ad-hoc model switching via GPU interconnects, it addresses the heterogeneity of real-world traffic in both spatial (different request types) and temporal (varying composition over time) dimensions. End-to-end P99 latency and throughput improved by an average of 1.5× and a maximum of 2× compared to vLLM.

**[Prism: Spectral-Aware Block-Sparse Attention](prism_spectral-aware_block-sparse_attention.md)**

:   Prism decomposes "block importance estimation" into high-frequency and low-frequency bands of RoPE, performing mean-pooling and softmax separately. It automatically calibrates logit magnitudes using a temperature derived from energy ratios. This approach relies entirely on block-level operations (eliminating token-level search), achieving accuracy comparable to full attention and a 5.1× speedup over FlashAttention-2 at 128K context.

**[ProactiveLLM: Learning Active Interaction for Streaming Large Language Models](proactivellm_learning_active_interaction_for_streaming_large_language_models.md)**

:   ProactiveLLM enables streaming LLMs to decide "when to speak" using their own internal states (attention or prediction entropy). By leveraging masked streaming modeling and synchronous privileged self-distillation, the model learns to perceive whether the "semantics are sufficient" without relying on any external alignment labels, significantly compressing interaction latency while maintaining performance.

**[ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)**

:   ProbMoE reformulates MoE top-$k$ routing as "probabilistic inference over a cardinality-constrained subset distribution." The forward pass uses the SIMPLE estimator to sample from an exact-$k$ subset distribution, while the backward pass uses analytically computed conditional marginal expert probabilities $m_j=\partial \log Z_k/\partial \log p_j$ as a differentiable proxy for discrete selection. It significantly improves GSM/Law/Translation tasks on OLMoE/Qwen1.5-MoE and enhances expert utilization, while naturally extending to a Dynamic-$k$ variant that adaptively activates experts based on token difficulty.

**[Proxy Compression for Language Modeling](proxy_compression_for_language_modeling.md)**

:   The authors propose "proxy compression"—training where 90% of data is fed as short sequences produced by a tokenizer/neural compressor and 10% as raw UTF-8 bytes, coupled with sentinel tokens and a brief in-context translation warm-up. During inference, all compressors are discarded, and the model processes only raw bytes; yet, it significantly outperforms pure byte-level models under fixed compute and matches or exceeds tokenizer baselines at larger scales.

**[Q-Delta: Beyond Key–Value Associative State Evolution](q-delta_beyond_key-value_associative_state_evolution.md)**

:   This paper challenges the implicit assumption in linear attention that "queries are only responsible for read-out and do not participate in state evolution." It demonstrates that query read-out $\hat{o}_t=S_{t-1}q_t$ itself constitutes a structured value prediction (complementary to key retrieval). Based on this, it proposes **Q-Delta**, which injects the hybrid prediction errors of both keys and queries into the delta rule state update. While maintaining linear time complexity and chunkwise parallel efficiency, Q-Delta consistently outperforms strong baselines such as DeltaNet and GatedDeltaNet in language modeling and long-context retrieval (S-NIAH average 90.0% vs. GatedDeltaNet 83.5%).

**[ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference](remoe_boosting_expert_reuse_through_router_fine-tuning_in_memory-constrained_moe.md)**

:   ReMoE freezes all non-router parameters and fine-tunes only the gates using a composite loss of "Temporal Locality Regularization + Trust-KL Semantic Anchor." This shapes the routing trajectories to be more "cache-friendly," increasing the expert reuse rate of adjacent tokens by approximately 26% without changing the architecture or adding runtime overhead. It reduces TPOT by 43.6–49.8% (achieving 1.77–1.99× decoding acceleration) on Jetson Orin NX.

**[RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress](repetitioncurse_measuring_and_understanding_router_imbalance_in_mixture-of-exper.md)**

:   By providing MoE large models with minimalist OOD prompts such as "repeating the same token N times," the authors discover that the router directs almost all tokens to a fixed small set of top-$k$ experts. Under Expert Parallelism (EP) deployment, this creates a bottleneck on a single card while leaving other GPUs idle, increasing TTFT by 20%–148% on 8-GPU clusters. This effectively turns the MoE parallel accelerator into a DoS attack surface.

**[RePo: Language Models with Context Re-Positioning](repo_language_models_with_context_re-positioning.md)**

:   Existing LLMs force tokens into linear integer indices $0 \dots L-1$, making the attention layers bear the heavy burden of "organizing context structure." RePo introduces a lightweight differentiable module $f_\phi$ that assigns **continuous, non-linear** position values based on token hidden states. This offloads the "extra cognitive load," leading to consistent gains in noisy contexts, structured data, and long-context tasks, with almost no degradation in standard short-context tasks.

**[RKSC: Reasoning-Aware KV Cache Sharing and Confident Early Exit for Multi-Step LLM Inference](rksc_reasoning-aware_kv_cache_sharing_and_confident_early_exit_for_multi-step_ll.md)**

:   RKSC is a **training-free** inference framework addressing two types of structural waste in "multi-branch inference" (running multiple reasoning trajectories followed by voting/verification). By employing "KV sharing via hidden state similarity" and "dual-level confidence early exit," it eliminates redundant prefix KV computations and excessive verification forwards. It achieves an average speedup of $3.008\times$ compared to a no-cache baseline across 5 models (7B–10B) and 4 benchmarks, with an early-exit-induced error rate of only $0.37\%$.

**[Scout: Active Information Foraging for Long-Text Understanding with Decoupled Epistemic States](scout_active_information_foraging_for_long-text_understanding_with_decoupled_epi.md)**

:   Scout remodels million-token long-text understanding (LTU) as an "active information foraging" process. It introduces a provenance-anchored epistemic state $\mathcal{E}_t$, decoupled from the interaction trajectory, as the sole base for reasoning. Through gap-diagnosed self-evaluation, it iteratively converges to a sufficient subset of information. On LooGLE-v2 and $\infty$Bench, Scout matches or exceeds frontier models like Gemini-3-Pro while reducing token costs to approximately $1/8$.

**[SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm](siamesenorm_breaking_the_barrier_to_reconciling_prepost-norm.md)**

:   To address the structural conflict where Pre-Norm and Post-Norm cannot coexist within a single-stream architecture, the authors propose SiameseNorm, a dual-stream residual architecture. It maintains an unnormalized stream as an identity gradient highway (Pre-Norm) and a normalized stream for main-path representation control (Post-Norm). By coupling these two streams via shared residual blocks, SiameseNorm consistently outperforms Pre-Norm baselines across 400M~15B dense/MoE language models, ViT, and DiT with negligible overhead.

**[Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills](skill-based_mixture-of-experts_adaptive_routing_for_heterogeneous_reasoning_via_.md)**

:   SKILL-MOE proposes a training-free symbolic MoE framework that uses "skills" as routing signals. It extracts required skills for each problem, dynamically recruits $k$ experts from 16 pre-trained LLMs based on skill-model profiles, and fuses multiple CoT responses via a task-level optimal aggregator. Combined with expert-batched inference, it runs 16 7-8B models on a single GPU, outperforming the strongest multi-agent baseline by 8.15% on average.

**[Skip a Layer or Loop It? Learning Program-of-Layers in LLMs](skip_a_layer_or_loop_it_learning_program-of-layers_in_llms.md)**

:   This paper treats each layer of a pre-trained LLM as an "atomic function" that can be arbitrarily invoked. It proposes "Program-of-Layers" (PoLar)—customizing an execution program for each input that can **skip or loop** layers. The authors first empirically demonstrate via MCTS that such optimal programs exist training-free for almost every input. They then train a lightweight predictor to produce the execution program in a single shot. On mathematical reasoning benchmarks, PoLar achieves higher accuracy than standard forward passes and existing dynamic depth methods, often while executing fewer layers.

**[SoftMoE: Soft Differentiable Routing for Mixture-of-Experts in LLMs](softmoe_soft_differentiable_routing_for_mixture-of-experts_in_llms.md)**

:   SoftMoE replaces the non-differentiable hard top-$k$ selection in MoE with a LapSum-based differentiable "soft top-$k$" operator. This enables gradient optimization for routing and allows the number of activated experts to adapt per token. Furthermore, a global budget constraint allows the model to **self-learn the optimal expert allocation per layer**. Results show that SoftMoE matches or exceeds sparse MoE while using experts more efficiently, revealing an intriguing pattern: **deeper layers tend to activate more experts**.

**[Sparser Block-Sparse Attention via Token Permutation](sparser_block-sparse_attention_via_token_permutation.md)**

:   This paper proposes PBS-Attn, which leverages the permutation invariance of attention to reorder keys within segments based on "global importance." This聚拢 (gathers) scattered heavy hitters into continuous high-density blocks before performing block-sparse computation, achieving up to 2.75x end-to-end acceleration for long-context prefilling while maintaining accuracy nearly equal to full attention.

**[STAR: Rethinking MoE Routing as Structure-Aware Subspace Learning](star_rethinking_moe_routing_as_structure-aware_subspace_learning.md)**

:   STAR reinterprets MoE routing as a "subspace learning" problem. Beyond the traditional shallow linear router, it employs the Generalized Hebbian Algorithm (GHA) to online-learn a set of orthogonal bases tracking the principal directions of the input. This aligns routing decisions directly with the input structure, achieving more stable expert specialization and superior downstream performance across synthetic tasks, LLaMA-MoE pre-training, BERT-GLUE fine-tuning, and ViT ImageNet-C.

**[Stochastic Sparse Attention for Memory-Bound Inference](stochastic_sparse_attention_for_memory-bound_inference.md)**

:   SANTA treats the value aggregation $AV$ of attention as an "expectation of value rows $V$ weighted by softmax probabilities $A$." It transforms this into an unbiased estimator that samples $S \ll n_k$ indices from $A$ without replacement and directly averages the corresponding $V$ rows. By employing stratified/systematic sampling to reduce variance and implementing GPU kernels aligned with FlashDecoding, SANTA achieves a 1.5× end-to-end speedup compared to FlashInfer/FlashDecoding in 32k context scenarios without accuracy degradation.

**[Structuring The Future: Diffusion LLM Speculative Decoding via Calibrated Draft Graphs](structuring_the_future_diffusion_llm_speculative_decoding_via_calibrated_draft_g.md)**

:   Spiffy adapts speculative decoding for Diffusion Language Models (dLLMs): instead of training a separate draft model, it utilizes the target model's own distribution for "auto-speculation." It organizes multi-step denoising states into a **Directed Draft Graph** and maximizes the acceptance rate using an offline-calibrated graph structure. This achieves up to a 8.6× reduction in model forward passes and a 6.3× speedup in token throughput on LLaDA / Dream / SDAR, while provably maintaining lossless output distributions.

**[TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration](team_temporal-spatial_consistency_guided_expert_activation_for_moe_diffusion_lan.md)**

:   TEAM addresses the inherent mismatch in MoE Diffusion Language Models (dLLM) where "a large number of experts are activated but only a few tokens are accepted." By leveraging the temporal and spatial consistency of in-block decoding, TEAM designs differentiated expert activation and decoding strategies for three types of tokens: decoded, hot, and cold. This achieves up to a 2.2× speedup on SDAR 30B-A3B with near-zero precision loss.

**[Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](theoretically_optimal_attentionffn_ratios_in_disaggregated_llm_serving.md)**

:   This paper provides the first theoretical framework for the emerging Attention-FFN Disaggregation (AFD) inference architecture. Based on a probabilistic workload model where "prefill length has a finite mean and decode length follows a geometric distribution," it derives a closed-form solution for the optimal A/F ratio in an rA-1F topology: $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$. The theory is validated using a trace-calibrated simulator, showing a deviation of <10% from measured optimal values.

**[Training-Inference Consistent Segmented Execution for Long-Context LLMs](training-inference_consistent_segmented_execution_for_long-context_llms.md)**

:   This paper proposes a long-context LLM framework that shares identical segmented forward execution semantics for both training and inference: it maintains a fixed-length differentiable KV tail across segments plus a forward-only retrieval bypass. On LLaMA2-7B 32K/80K, it achieves LongBench/RULER performance comparable to or better than full attention with approximately $6\times$ lower peak prefill memory.

**[TuneAhead: Predicting Fine-tuning Performance Before Full Training Begins](tuneahead_predicting_fine-tuning_performance_before_full_training_begins.md)**

:   Addressing the pain point where failure is realized only after full training (wasting hundreds of GPU hours), TuneAhead encodes each candidate fine-tuning task into a meta-feature vector consisting of "static dataset descriptors + 100-step dynamic probe features." It uses LightGBM to predict final performance before training (RMSE 1.47pp on 370 test tasks, with 95.1% within ±3pp) and provides diagnosable explanations for "why it might fail" using SHAP.

**[Turning Back Without Forgetting: Selective Backward Refinement for Parameter-Efficient Continual Learning](turning_back_without_forgetting_selective_backward_refinement_for_parameter-effi.md)**

:   SABER is the first to achieve "no-replay forward and backward transfer" in prompt-based continual learning. It utilizes dual correlation criteria—gradient geometry and loss distribution—to decide whether to "go back and refine old task prompts," and restricts updates to an orthogonal subspace that does not interfere with old tasks to perform "safe refinement," allowing subsequent tasks to actively improve the accuracy of prior tasks.

**[Variational Routing: A Scalable Bayesian Framework for Calibrated MoE Transformers](variational_routing_a_scalable_bayesian_framework_for_calibrated_mixture-of-expe.md)**

:   This paper proposes VMoER, a variational routing framework that achieves efficient Bayesian uncertainty modeling by performing variational inference on MoE routing decisions rather than weights. It reduces calibration error by 94% and improves routing stability by 38% while maintaining <1% extra FLOPs overhead.

**[VIA-SD: Verification via Intra-Model Routing for Speculative Decoding](via-sd_verification_via_intra-model_routing_for_speculative_decoding.md)**

:   Addressing the binary decision bottleneck of "either accept or recompute with the target model" in speculative decoding, VIA-SD routes a lightweight "slim-verifier" from within the full verifier to handle "medium-confidence" tokens. This forms a draft → slim-verifier → full-verifier multi-stage process, reducing rejection rates by 0.10–0.22 and achieving an additional 10–20% speedup over strong speculative decoding baselines across four tasks and multiple model families.

**[WarmServe: Multi-model GPU Warm-up Mechanism via Load-Once-Many](warmserve_enabling_one-for-many_gpu_prewarming_for_multi-llm_serving.md)**

:   WarmServe reduces tail TTFT by 50.8x compared to existing systems by analyzing long-term periodic patterns in LLM serving workloads. It proactively preloads multiple model parameters into GPUs, using optimized placement algorithms and dynamic KV cache reservation strategies to quickly launch new instances during request bursts.
