---
title: >-
  ICML2026 Pretraining Papers · 27 Notes
description: >-
  27 ICML2026 papers in the Pretraining area, covering LLM, Agents, Layout & Composition, Diffusion Models, Few-/Zero-Shot Learning and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ICML2026"
  - "Pretraining"
  - "AI paper notes"
  - "paper summaries"
  - "LLM"
  - "Agents"
  - "Layout & Composition"
  - "Diffusion Models"
  - "Few-/Zero-Shot Learning"
item_list:
  - u: "ac-odm_actor--critic_online_data_mixing_for_sample-efficient_llm_pretraining/"
    t: "AC-ODM: Actor–Critic Online Data Mixing for Sample-Efficient LLM Pretraining"
  - u: "annotations_mitigate_post-training_mode_collapse/"
    t: "Annotations Mitigate Post-Training Mode Collapse"
  - u: "beyond_structural_symmetries_linear_mode_connectivity_via_neuron_identifiability/"
    t: "Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability"
  - u: "constrained_bayesian_experimental_design_via_online_planning/"
    t: "Constrained Bayesian Experimental Design via Online Planning"
  - u: "data_difficulty_and_the_generalization--extrapolation_tradeoff_in_llm_fine-tunin/"
    t: "Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning"
  - u: "decoupling_the_what_and_where_with_polar_coordinate_positional_embeddings/"
    t: "Decoupling the \"What\" and \"Where\" With Polar Coordinate Positional Embeddings"
  - u: "different_layers_different_manifolds_module-wise_weight-space_geometry_in_transf/"
    t: "Different Layers, Different Manifolds: Module-Wise Weight-Space Geometry in Transformer Optimization"
  - u: "dropout_universality_scaling_laws_and_optimal_scheduling_at_the_edge-of-chaos/"
    t: "Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos"
  - u: "explaining_data_mixing_scaling_laws/"
    t: "Explaining Data Mixing Scaling Laws"
  - u: "flexrank_nested_low-rank_knowledge_decomposition_for_adaptive_model_deployment/"
    t: "FlexRank: Nested Low-Rank Knowledge Decomposition for Adaptive Model Deployment"
  - u: "focus_and_dilution_the_multi-stage_learning_process_of_attention/"
    t: "Focus and Dilution: The Multi-stage Learning Process of Attention"
  - u: "if_open_source_is_to_win_it_must_go_public/"
    t: "If open source is to win, it must go public"
  - u: "incremental_bpe_tokenization/"
    t: "Incremental BPE Tokenization"
  - u: "infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted/"
    t: "InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition"
  - u: "inverse_depth_scaling_from_most_layers_being_similar/"
    t: "Inverse Depth Scaling From Most Layers Being Similar"
  - u: "moose-star_unlocking_tractable_training_for_scientific_discovery_by_breaking_the/"
    t: "MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier"
  - u: "names_dont_matter_symbol-invariant_transformer_for_open-vocabulary_learning/"
    t: "Names Don't Matter: Symbol-Invariant Transformer for Open-Vocabulary Learning"
  - u: "on_the_expressive_power_of_permutation-equivariant_weight-space_networks/"
    t: "On the Expressive Power of Permutation-Equivariant Weight-Space Networks"
  - u: "on_training_large_language_models_for_long-horizon_tasks_an_empirical_study_of_h/"
    t: "On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length"
  - u: "poet-x_memory-efficient_llm_training_by_scaling_orthogonal_transformation/"
    t: "POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation"
  - u: "predicting_large_model_test_losses_with_a_noisy_quadratic_system/"
    t: "Predicting Large Model Test Losses with a Noisy Quadratic System"
  - u: "scaling_depth_capacity_via_zeroone-layer_model_expansion/"
    t: "Scaling Depth Capacity via Zero/One-Layer Model Expansion"
  - u: "spare_stacked_parallelism_with_adaptive_reordering_for_fault-tolerant_llm_pretra/"
    t: "SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant LLM Pretraining Systems with 100k+ GPUs"
  - u: "the_devil_is_in_the_condition_numbers_why_is_glu_better_than_non-glu_structure/"
    t: "The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?"
  - u: "trust_functions_near-lossless_weak-to-strong_generalization_by_learning_when_to_/"
    t: "Trust Functions: Near-Lossless Weak-to-Strong Generalization by Learning When to Trust the Weak Teacher"
  - u: "tuning_the_implicit_regularizer_of_masked_diffusion_language_models_enhancing_ge/"
    t: "Tuning the Implicit Regularizer of Masked Diffusion Language Models: Enhancing Generalization via Insights from k-Parity"
  - u: "xtransfer_modality-agnostic_few-shot_model_transfer_for_human_sensing_at_the_edg/"
    t: "XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge"
item_total: 27
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# 📚 Pretraining

**🧪 ICML2026** · **27** paper notes

📌 **Same area in other venues:** [📷 CVPR2026 (5)](../../CVPR2026/llm_pretraining/index.md) · [🔬 ICLR2026 (79)](../../ICLR2026/llm_pretraining/index.md) · [💬 ACL2026 (12)](../../ACL2026/llm_pretraining/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_pretraining/index.md) · [🧠 NeurIPS2025 (51)](../../NeurIPS2025/llm_pretraining/index.md) · [📹 ICCV2025 (9)](../../ICCV2025/llm_pretraining/index.md)

🔥 **Top topics:** LLM ×6

**[AC-ODM: Actor–Critic Online Data Mixing for Sample-Efficient LLM Pretraining](ac-odm_actor--critic_online_data_mixing_for_sample-efficient_llm_pretraining.md)**

:   AC-ODM formulates the dynamic adjustment of pre-training data domain weights as a continuous control problem in reinforcement learning. Using the DDPG Actor-Critic framework, it perceives the model state in real-time, outputs sampling weights for each domain, and employs "inter-domain gradient alignment" as the reward. Theoretically, this is proven equivalent to maximizing constructive interference of gradients (effective descent step size). On Pythia-1B, it achieves optimal perplexity with approximately 66% fewer steps than strong baselines, scores a 27.5% relative improvement on MMLU, and increases HumanEval pass@1 by 2.23 times, with only a 0.4% increase in wall-clock time per step and 2% extra memory.

**[Annotations Mitigate Post-Training Mode Collapse](annotations_mitigate_post-training_mode_collapse.md)**

:   The authors observe that SFT aligns models with a low-entropy semantic prior, leading to "inverse scaling" where larger instruction-tuned models become increasingly repetitive. They propose "Annotation-Anchored Training"—tagging documents with semantic tags during pre-training and masking the loss on these tags during SFT—enabling the model to sample semantics before generating responses, which reduces the semantic diversity gap by 85% while maintaining instruction-following performance.

**[Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability](beyond_structural_symmetries_linear_mode_connectivity_via_neuron_identifiability.md)**

:   This paper proposes a theoretical framework of "effective function classes" and "neuron identifiability," revealing that breaking structural symmetry does not equate to breaking effective symmetry—even if permutation symmetry in the parameter space is eliminated, data-dependent approximate symmetries may still make neuron swapping costs extremely low. Based on this, it provides sufficient conditions for achieving Linear Mode Connectivity (LMC) without the need for alignment.

**[Constrained Bayesian Experimental Design via Online Planning](constrained_bayesian_experimental_design_via_online_planning.md)**

:   This paper proposes COPEx: a semi-amortized scheme combining "offline pre-trained amortized posterior networks + design policies + online multi-step lookahead scenario trees." This allows Bayesian experimental design (BED) to dynamically adapt to budget, cost, and transition constraints at test time. COPEx consistently outperforms baselines such as VPCE, ALINE, and RL-BOED in EIG/RMSE across three types of tasks: constrained location finding, CES, and cost-aware AL.

**[Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning](data_difficulty_and_the_generalization--extrapolation_tradeoff_in_llm_fine-tunin.md)**

:   This paper systematically investigates the role of data difficulty in SFT, discovering that there is no "universally optimal difficulty." Instead, an optimal difficulty exists that **drifts toward harder samples as the data scale increases**. This is explained through a PAC-Bayes framework as a tradeoff between the "in-distribution generalization gap" and the "extrapolation gap."

**[Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings](decoupling_the_what_and_where_with_polar_coordinate_positional_embeddings.md)**

:   The authors point out that the mainstream positional encoding, RoPE, couples "content (what)" and "position (where)" into the same phase, leading to poor performance on tasks requiring "finding content by position" or "locating position by content." They propose PoPE, which uses softplus to separate magnitude (controlling what) and pure positional phase (controlling where). As a minor modification to RoPE, PoPE consistently outperforms it in diagnostic tasks, music/genomic/language modeling, and achieves **length extrapolation to 10x the training length without any fine-tuning**, surpassing YaRN which is specifically designed for extrapolation.

**[Different Layers, Different Manifolds: Module-Wise Weight-Space Geometry in Transformer Optimization](different_layers_different_manifolds_module-wise_weight-space_geometry_in_transf.md)**

:   This workshop paper systematically compares "module-wise manifold constraint" schemes during GPT-2 small pre-training. It discovers that applying strong spectral constraints (Stiefel) to Attention layers while applying weak constraints (DGram) to MLP layers achieves the best performance. Conversely, training Attention layers with DGram leads to divergence, for which the authors provide a mechanistic explanation: "Singular value swelling $\rightarrow$ Logit inflation $\rightarrow$ Softmax saturation $\rightarrow$ Gradient degradation."

**[Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos](dropout_universality_scaling_laws_and_optimal_scheduling_at_the_edge-of-chaos.md)**

:   The authors interpret dropout as an "external field" $h$ that breaks the $c^*=1$ perfect alignment fixed point in mean field signal propagation theory. They derive the Landau equation, two-parameter scaling collapse, and identify two distinct universality classes for smooth and kinked activations. This leads to a "zero-overhead" practical conclusion: a **front-loaded schedule** reduces test loss by 18–35% in MLPs and ViTs compared to constant dropout under the same budget.

**[Explaining Data Mixing Scaling Laws](explaining_data_mixing_scaling_laws.md)**

:   This paper provides the long-missing theoretical explanation for "multi-domain data mixing scaling laws." By extending two classic theories of single-domain scaling laws (the quantization model and the projection linear regression model) to multiple domains, it proposes a "shared head, disjoint tail" distribution hypothesis. It identifies two mechanisms governing the loss of each domain: **capacity competition** (limited model capacity is contested by domain-specific skills, globally coupling all domain losses) and **data quantity noise** (losses in harder-to-learn domains decrease more slowly, biasing the optimal ratio toward them). The resulting model achieves lower fitting errors using fewer parameters and enables cross-scale extrapolation, using small-scale fitted parameters to predict optimal ratios for large models.

**[FlexRank: Nested Low-Rank Knowledge Decomposition for Adaptive Model Deployment](flexrank_nested_low-rank_knowledge_decomposition_for_adaptive_model_deployment.md)**

:   FlexRank performs activation-aware low-rank decomposition (DataSVD) on each linear layer of a pre-trained large model, uses dynamic programming to select a set of **strictly nested** sub-models corresponding to different compute budgets in $O(L\cdot K)$ time, and jointly trains this shared weight set using knowledge distillation. Finally, via Gauge-Aligned Reparametrization, rank savings are translated into actual FLOPs savings—yielding a "family" of deployable models for LLMs and ViTs that approach the true Pareto frontier with a single training run.

**[Focus and Dilution: The Multi-stage Learning Process of Attention](focus_and_dilution_the_multi-stage_learning_process_of_attention.md)**

:   By performing phased linearization of gradient flow around a series of critical points in a simplified setting where a single-layer Transformer learns Markovian data, this paper reveals and rigorously characterizes the recurring "focus–dilution" cycles in attention training, observing consistent phenomena on WikiText and TinyStories.

**[If open source is to win, it must go public](if_open_source_is_to_win_it_must_go_public.md)**

:   This ICML 2026 position paper argues that "open-source AI" in its current form cannot truly democratize AI access or provide public goods in the same way Linux or PyTorch did. It posits that open source can only succeed if embedded within "Public AI"—infrastructure for compute, inference, post-training, and data provided by governments, national labs, universities, and non-profit institutions.

**[Incremental BPE Tokenization](incremental_bpe_tokenization.md)**

:   This paper proposes the first incremental BPE tokenization algorithm with a strict $\mathcal{O}(\log^2 t)$ worst-case per-byte complexity. By utilizing an Aho–Corasick automaton to locate the search space and binary search on a Centroid Decomposition of the "Suffix-Successor Tree" to identify the "last token," it serves as a drop-in replacement achieving up to $\sim 3\times$ speedup over Hugging Face tokenizers. Furthermore, it eliminates the $\mathcal{O}(n^2)$ degradation of tiktoken on pathological inputs.

**[InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)**

:   The authors propose InfoLaw: redefining "pre-training" as a process of "accumulating information in buckets." The information volume per bucket equals "quality density $f_d \times$ unique tokens $M_d \times \log K$" multiplied by an exponential decay factor associated with repetition counts $R_d$. By fitting validation loss as $L = \alpha\cdot\text{info}^{-\beta}$ on 252M-1.2B models, the law extrapolates to 7B models and 425B tokens with an average error of 0.15% (max 0.96%) and directly enables searching for optimal data recipes.

**[Inverse Depth Scaling From Most Layers Being Similar](inverse_depth_scaling_from_most_layers_being_similar.md)**

:   By measuring LLM hidden state dynamics and conducting controlled experiments with a teacher-student toy model, this paper proves that LLM loss is approximately inversely proportional to depth ($\alpha_\ell \approx 1$). This is attributed to an inefficient but robust "ensemble averaging" mode where the vast majority of layers perform functionally similar small-step updates to cancel out errors.

**[MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier](moose-star_unlocking_tractable_training_for_scientific_discovery_by_breaking_the.md)**

:   MOOSE-Star decomposes the problem of "training an LLM to directly generate scientific hypotheses"—originally an $\mathcal{O}(N^k)$ combinatorial search—into two sequential subtasks: "Inspiration Retrieval + Hypothesis Synthesis." By integrating hierarchical tree retrieval, bounded composition, and motivation planning, it reduces optimal complexity from exponential to $\mathcal{O}(\log N)$ and releases the TOMATO-Star dataset containing 108,717 papers with decomposition annotations.

**[Names Don't Matter: Symbol-Invariant Transformer for Open-Vocabulary Learning](names_dont_matter_symbol-invariant_transformer_for_open-vocabulary_learning.md)**

:   The authors modify the Transformer into a structure with "a shared-weight parallel embedding stream for each interchangeable symbol + cross-stream aggregated attention." This architecture-level design guarantees identical outputs for variable renaming (alpha-equivalence) and allows the inclusion of new symbols not seen during training into the vocabulary during testing. It outperforms comparable baselines and even GPT-5.2 on propositional logic and LTL witness generation tasks.

**[On the Expressive Power of Permutation-Equivariant Weight-Space Networks](on_the_expressive_power_of_permutation-equivariant_weight-space_networks.md)**

:   This paper establishes the first systematic theory of expressive power for permutation-equivariant weight-space networks (e.g., DWS, NFN, GMN, NG-GNN) operating on MLP weights. It proves these architectures are nearly equivalent in expressivity and characterizes their universality across four approximation scenarios (function-space functionals/operators, permutation-invariant functionals, and permutation-equivariant operators) under the "general position" assumption. A theoretically derived modification, OCE (Output Capacity Expansion via ensembling multiple MLPs), achieves a 34% improvement over SOTA on INR editing benchmarks.

**[On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](on_training_large_language_models_for_long-horizon_tasks_an_empirical_study_of_h.md)**

:   Using a set of carefully controlled Sudoku/Rush Hour tasks where "reasoning difficulty remains constant while only the horizon length varies," this paper systematically proves that **task horizon itself is an independent root cause for LLM agent RL training collapse**. The authors propose two horizon-reduction mechanisms—macro actions and subgoal decomposition—which not only stabilize training but also enable strong zero-shot generalization across longer horizons (horizon generalization).

**[POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation](poet-x_memory-efficient_llm_training_by_scaling_orthogonal_transformation.md)**

:   POET-X implements a system-level acceleration and memory optimization for POET (reParameterized Orthogonal Equivalence Training), which is training-stable but slow and memory-intensive. By combining input-centric reconstruction, permutation kernel acceleration, block-diagonal batch parallelism, half-storage CNP, and Triton fusion, it achieves a 3× memory reduction and 8× speedup compared to the original POET. This allows for pre-training 8B~13B LLMs on a single H100, while AdamW triggers OOM under identical settings.

**[Predicting Large Model Test Losses with a Noisy Quadratic System](predicting_large_model_test_losses_with_a_noisy_quadratic_system.md)**

:   This paper proposes the Noisy Quadratic System (NQS)—a mechanistic loss model that frames LLM test loss as $L(N, B, K)$ (model size / batch size / update steps). It is the first to explicitly model batch size within a scaling law, improving extrapolation capabilities on Pythia + OWT2 from Chinchilla's ~20× compute range to ~4000×.

**[Scaling Depth Capacity via Zero/One-Layer Model Expansion](scaling_depth_capacity_via_zeroone-layer_model_expansion.md)**

:   This paper proposes "Zero/One-Layer Progressive Training"—first training an extremely shallow model with almost no Transformer layers, then expanding the depth to the target number of layers at a late stage of training ($\approx 80\%$ iterations). Combined with a Warmup-Stable-Decay (WSD) learning rate schedule and muP hyperparameter transfer, this approach saves approximately $80\%$ of computation ($\approx 5\times$ speedup) across GPT-2, Llama-3, and DeepSeek-V3 while maintaining terminal loss parity.

**[SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant LLM Pretraining Systems with 100k+ GPUs](spare_stacked_parallelism_with_adaptive_reordering_for_fault-tolerant_llm_pretra.md)**

:   SPARe cyclically stacks $r$ layers of data shards across groups in the data parallelism dimension. Upon node failure, it employs Hopcroft-Karp and min-cost max-flow algorithms for adaptive reordering of the "all-reduce stack number." In restart-dominant scenarios with 600k GPUs, it achieves availability comparable to $r\times$ traditional replication with only $2\sim 3\times$ computational overhead, reducing time-to-train by $40\sim 50\%$ compared to Rep+CKPT.

**[The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?](the_devil_is_in_the_condition_numbers_why_is_glu_better_than_non-glu_structure.md)**

:   From an NTK perspective, this work proves that GLU rewrites the kernel matrix of a two-layer network as the "Hadamard product of the original NTK and the data Gram matrix," which significantly compresses the condition number and accelerates convergence. Empirical results demonstrate that GLU does not improve the generalization gap; its benefits derive entirely from superior optimization.

**[Trust Functions: Near-Lossless Weak-to-Strong Generalization by Learning When to Trust the Weak Teacher](trust_functions_near-lossless_weak-to-strong_generalization_by_learning_when_to_.md)**

:   This paper reframes "Weak-to-Strong Generalization" as a **data selection** problem and proposes the "Trust Function." By using a lightweight MLP to read the hidden states of the weak teacher's final layer and predicting the reliability of weak labels, the method selects only high-trust samples to train the strong student. This achieves near-lossless or even super-ground-truth performance across multiple tasks and can be iterated into a "Weak-to-Strong Chain" to amplify gains.

**[Tuning the Implicit Regularizer of Masked Diffusion Language Models: Enhancing Generalization via Insights from k-Parity](tuning_the_implicit_regularizer_of_masked_diffusion_language_models_enhancing_ge.md)**

:   This paper decomposes the training objective of Masked Diffusion Language Models (MDLM) into a "signal term + noise term" using the analytically solvable $k$-parity task. It theoretically proves that the noise term acts as an **implicit regularizer** that suppresses grokking and avoids memory traps. Based on this, the authors propose **Signal-Rich Mask Sampling**, narrowing the training mask rate $t$ from a uniform $\mathcal{U}[0,1]$ to a middle-range window. This approach significantly reduces perplexity on 50M models and yields an 8.8% improvement in pre-training and 5.8% in SFT for 8B models.

**[XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge](xtransfer_modality-agnostic_few-shot_model_transfer_for_human_sensing_at_the_edg.md)**

:   XTransfer targets human sensing tasks on edge devices by transferring pre-trained models from any modality (image, text, audio, or sensors) using limited target sensor data. It mitigates cross-modality feature misalignment through layer-wise model repairing and resource-constrained layer recombining, simultaneously improving few-shot accuracy and edge deployment efficiency.
