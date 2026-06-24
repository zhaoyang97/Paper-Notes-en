---
title: >-
  ACL2025 Model Compression Papers · 78 Notes
description: >-
  78 ACL2025 papers in the Model Compression area, covering Model Compression, LLM, Compression, Knowledge Distillation, Adversarial Robustness, Layout & Composition and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ACL2025"
  - "Model Compression"
  - "AI paper notes"
  - "paper summaries"
  - "LLM"
  - "Compression"
  - "Knowledge Distillation"
  - "Adversarial Robustness"
  - "Layout & Composition"
item_list:
  - u: "500xcompressor_generalized_prompt_compression_for_large_language_models/"
    t: "500xCompressor: Generalized Prompt Compression for Large Language Models"
  - u: "accurate_kv_cache_quantization_with_outlier_tokens_tracing/"
    t: "Accurate KV Cache Quantization with Outlier Tokens Tracing"
  - u: "aligndistil_token_level_alignment/"
    t: "AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation"
  - u: "apb_distributed_long_context/"
    t: "APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs"
  - u: "assigning_distinct_roles_to_quantized_and_low-rank_matrices_toward_optimal_weigh/"
    t: "Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition"
  - u: "basic_reading_distillation/"
    t: "Basic Reading Distillation"
  - u: "beamlora_beam_constraint_lora/"
    t: "BeamLoRA: Beam-Constraint Low-Rank Adaptation"
  - u: "beyond_logits_aligning_feature_dynamics_for_effective_knowledge_distillation/"
    t: "Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation"
  - u: "beyond_text_compression_tokenizers/"
    t: "Beyond Text Compression: Evaluating Tokenizers Across Scales"
  - u: "bf16_or_death_quantization_tradeoffs/"
    t: "\"Give Me BF16 or Give Me Death\"? Accuracy-Performance Trade-Offs in LLM Quantization"
  - u: "blockpruner_fine-grained_pruning_for_large_language_models/"
    t: "BlockPruner: Fine-grained Pruning for Large Language Models"
  - u: "bone_soups_multi_objective_gen/"
    t: "Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation"
  - u: "brainecho_semantic_brain_signal_decoding_through_vector-quantized_spectrogram_re/"
    t: "BrainECHO: Semantic Brain Signal Decoding through Vector-Quantized Spectrogram Reconstruction for Whisper-Enhanced Text Generation"
  - u: "cami_a_counselor_agent_supporting_motivational_interviewing_through_state_infere/"
    t: "CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration"
  - u: "capture_key_cot_distillation/"
    t: "Capture the Key in Reasoning to Enhance CoT Distillation Generalization"
  - u: "cfsp_an_efficient_structured_pruning_framework_for_llms_with_coarse-to-fine_acti/"
    t: "CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information"
  - u: "claimpkg_enhancing_claim_verification_via_pseudo-subgraph_generation_with_lightw/"
    t: "ClaimPKG: Enhancing Claim Verification via Pseudo-Subgraph Generation with Lightweight Specialized LLM"
  - u: "cola_collaborative_low-rank_adaptation/"
    t: "CoLA: Collaborative Low-Rank Adaptation"
  - u: "compact_and_compressible_representations_for_llms_using_structured_sparse_decom/"
    t: "Compact and Compressible Representations for LLMs Using Structured Sparse Decomposition"
  - u: "compression_in_transformer_language_models_has_a_surprising_relationship_with_pe/"
    t: "Compression in Transformer Language Models Has a Surprising Relationship with Performance"
  - u: "dac_prompt_compression/"
    t: "DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression"
  - u: "data_laundering_artificially_boosting_benchmark_results_through_knowledge_distil/"
    t: "Data Laundering: Artificially Boosting Benchmark Results through Knowledge Distillation"
  - u: "deepsolution_boosting_complex_engineering_solution_design_via_tree-based_explora/"
    t: "DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking"
  - u: "direct_behavior_optimization_unlocking_the_potential_of_lightweight_llms/"
    t: "Direct Behavior Optimization: Unlocking the Potential of Lightweight LLMs"
  - u: "disentangling_the_roles_of_representation_and_selection_in_data_pruning/"
    t: "Disentangling the Roles of Representation and Selection in Data Pruning"
  - u: "domix_an_efficient_framework_for_exploiting/"
    t: "DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning"
  - u: "drpruning_robust_pruning/"
    t: "DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization"
  - u: "eac_moe_expert_aware_compression/"
    t: "EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models"
  - u: "efficient_long_context_language_model_retrieval_with_compression/"
    t: "Efficient Long Context Language Model Retrieval with Compression"
  - u: "efficient_opamp_adaptation_for_zoom_attention_to_golden_contexts/"
    t: "Efficient OpAmp Adaptation for Zoom Attention to Golden Contexts"
item_total: 78
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# 📦 Model Compression

**💬 ACL2025** · **78** paper notes

📌 **Same area in other venues:** [📷 CVPR2026 (108)](../../CVPR2026/model_compression/index.md) · [🔬 ICLR2026 (240)](../../ICLR2026/model_compression/index.md) · [💬 ACL2026 (59)](../../ACL2026/model_compression/index.md) · [🧪 ICML2026 (117)](../../ICML2026/model_compression/index.md) · [🤖 AAAI2026 (60)](../../AAAI2026/model_compression/index.md) · [🧠 NeurIPS2025 (143)](../../NeurIPS2025/model_compression/index.md)

🔥 **Top topics:** Model Compression ×20 · LLM ×19 · Compression ×15 · Knowledge Distillation ×4 · Adversarial Robustness ×4

**[500xCompressor: Generalized Prompt Compression for Large Language Models](500xcompressor_generalized_prompt_compression_for_large_language_models.md)**

:   The paper proposes 500xCompressor, which compresses up to around 500 natural language tokens into the KV values of as few as 1 special token, achieving compression ratios from 6x to 480x. It introduces only about 0.25% of additional parameters, while the LLM retains 62.26%–72.89% of its original capabilities after compression, significantly outperforming the ICAE baseline.

**[Accurate KV Cache Quantization with Outlier Tokens Tracing](accurate_kv_cache_quantization_with_outlier_tokens_tracing.md)**

:   It is discovered that a small number of outlier tokens in the outlier channels of KV Cache deviate from the previously assumed uniform distribution. To address this, the Outlier Tokens Tracing (OTT) method is proposed to dynamically trace and exclude these tokens during the quantization process. Under 2-bit quantization, this approach achieves a 6.4x memory compression and a 2.3x throughput speedup while significantly improving accuracy.

**[AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](aligndistil_token_level_alignment.md)**

:   AlignDistil theoretically proves the equivalence between the RLHF objective and a token-level distillation process. Based on this, it designs a simple distillation method: constructing a teacher distribution through a linear combination of logit distributions from a DPO model and a reverse DPO model, and combining this with a token-adaptive extrapolation mechanism to achieve token-level reward optimization. It outperforms existing methods on AlpacaEval 2.0, MT-Bench, and Arena-Hard while achieving faster convergence.

**[APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs](apb_distributed_long_context.md)**

:   APB proposes a distributed long-context inference framework. By introducing local KV cache compression and a mechanism to pass compressed context blocks across GPUs into the sequence parallelism framework, it achieves up to 9.2x, 4.2x, and 1.6x prefill speedup compared to FlashAttn, RingAttn, and StarAttn, respectively, without compromising task performance.

**[Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition](assigning_distinct_roles_to_quantized_and_low-rank_matrices_toward_optimal_weigh.md)**

:   Proposes ODLRI (Outlier-Driven Low-Rank Initialization) to assign an explicit role to the low-rank component in the joint quantization and low-rank optimization (Q+LR) framework—capturing activation outlier-sensitive weights, allowing the quantized component to handle a smoother residual. This consistently reduces perplexity and improves zero-shot accuracy in 2-bit extreme quantization scenarios for Llama2/3 and Mistral.

**[Basic Reading Distillation](basic_reading_distillation.md)**

:   This paper proposes Basic Reading Distillation (BRD). By having a teacher LLM generate basic reading behavior data (including NER and QA) on general corpora, a small student model is trained to mimic these behaviors. This allows a 564M-parameter small model to reach or exceed the performance of a teacher model 20 times its size across various NLP tasks, without being exposed to downstream task data.

**[BeamLoRA: Beam-Constraint Low-Rank Adaptation](beamlora_beam_constraint_lora.md)**

:   BeamLoRA observes that the importance of different ranks in LoRA modules varies significantly and evolves dynamically during training. Inspired by beam search, it proposes to dynamically evaluate rank importance inside the training process, prune unimportant ranks, and expand the parameter space for important ranks. This improves performance under a fixed total rank, consistently outperforming LoRA and its variants across 12 datasets on three base models.

**[Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation](beyond_logits_aligning_feature_dynamics_for_effective_knowledge_distillation.md)**

:   This paper proposes a knowledge distillation method that goes beyond logit matching. By aligning the dynamics of feature changes (rather than static feature snapshots) of the teacher and student models during the training process, it achieves more effective knowledge transfer, significantly improving distillation performance on NLP tasks.

**[Beyond Text Compression: Evaluating Tokenizers Across Scales](beyond_text_compression_tokenizers.md)**

:   This paper systematically evaluates the impact of 6 tokenizers on 350M and 2.7B parameter models. It finds that tokenizer selection has an extremely minor impact on English tasks but has a significant and scale-consistent impact on multilingual tasks (such as machine translation). The paper also proposes a novel family of intrinsic evaluation metrics based on Zipf's law, which predict downstream performance in multilingual scenarios significantly better than text compression rates.

**["Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization](bf16_or_death_quantization_tradeoffs.md)**

:   This is the most comprehensive empirical study of LLM quantization to date, conducting over 500k evaluations of FP8/INT8/INT4 on the entire Llama-3.1 family (8B/70B/405B). It finds that FP8 is nearly lossless, INT8 incurs only a 1-3% drop, and INT4 is surprisingly competitive, while providing recommendations for selecting quantization formats in different deployment scenarios.

**[BlockPruner: Fine-grained Pruning for Large Language Models](blockpruner_fine-grained_pruning_for_large_language_models.md)**

:   Proposes BlockPruner, which decomposes Transformer layers into two minimal residual blocks (MHA and MLP), evaluates block importance based on perplexity, and performs fine-grained pruning through iterative search, achieving superior compression performance compared to layer-level pruning.

**[Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation](bone_soups_multi_objective_gen.md)**

:   This paper proposes the Bone Soup model merging approach, which addresses the suboptimality of single-objective model merging in Rewarded Soup by first constructing "backbone rewards" (combinations of multi-objective rewards) to train backbone models, and then using a symmetric circulant matrix mapping to determine merging coefficients. This achieves superior Pareto frontiers and better controllability across three multi-objective generation tasks.

**[BrainECHO: Semantic Brain Signal Decoding through Vector-Quantized Spectrogram Reconstruction for Whisper-Enhanced Text Generation](brainecho_semantic_brain_signal_decoding_through_vector-quantized_spectrogram_re.md)**

:   This paper proposes BrainECHO, a three-stage framework (Autoencoding-Alignment-Finetuning) that maps brain signals to the Mel-spectrogram space via vector-quantized discrete representations, enabling high-quality non-invasive brain-to-text decoding with Whisper.

**[CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration](cami_a_counselor_agent_supporting_motivational_interviewing_through_state_infere.md)**

:   This paper proposes CAMI (Counselor Agent for Motivational Interviewing), a counseling agent based on the principles of Motivational Interviewing (MI). It utilizes the STAR framework (State inference, Topic exploration, Action & Response generation) to guide clients to generate change talk, outperforming existing methods in both automated and human evaluations.

**[Capture the Key in Reasoning to Enhance CoT Distillation Generalization](capture_key_cot_distillation.md)**

:   Proposed EDIT (mistakE-Driven key reasonIng step distillaTion), which constructs paired positive/negative dual CoTs data, employs the minimum edit distance algorithm to locate key reasoning steps, and guides smaller models to focus on learning these key steps through a token-level fine-grained loss function, rather than simply mimicking the teacher's reasoning format.

**[CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information](cfsp_an_efficient_structured_pruning_framework_for_llms_with_coarse-to-fine_acti.md)**

:   This paper proposes the CFSP framework, which utilizes coarse-grained (inter-block) and fine-grained (intra-block) activation information as importance criteria to guide structured pruning of LLMs. It requires only a single forward pass to complete the pruning and outperforms existing methods across multiple models and sparsity budgets.

**[ClaimPKG: Enhancing Claim Verification via Pseudo-Subgraph Generation with Lightweight Specialized LLM](claimpkg_enhancing_claim_verification_via_pseudo-subgraph_generation_with_lightw.md)**

:   Proposes the ClaimPKG framework, which utilizes a lightweight specialized LLM to convert textual claims into pseudo-subgraph representations, retrieves relevant subgraphs from a knowledge graph as evidence, and finally performs reasoning and verification using a general LLM, outperforming SOTA by 9%-12% accuracy on the FactKG dataset.

**[CoLA: Collaborative Low-Rank Adaptation](cola_collaborative_low-rank_adaptation.md)**

:   Proposes CoLA, a flexible LoRA architecture that breaks the fixed quantity constraint between matrices A and B (#A=M, #B=N), and designs three collaborative strategies (full collaboration / random collaboration / heuristic collaboration). Combined with an extended PiSSA initialization, it significantly outperforms existing PEFT methods in low-sample scenarios.

**[Compact and Compressible Representations for LLMs Using Structured Sparse Decomposition](compact_and_compressible_representations_for_llms_using_structured_sparse_decom.md)**

:   This paper proposes a structured sparse decomposition method that decomposes the LLM weight matrix into a combination of a low-rank component and a structured sparse component. This achieves high compression ratios while maintaining model performance, enabling efficient deployment of large language models in resource-constrained environments.

**[Compression in Transformer Language Models Has a Surprising Relationship with Performance](compression_in_transformer_language_models_has_a_surprising_relationship_with_pe.md)**

:   This paper investigates the relationship between compression (weight compressibility) and model performance in Transformer language models from an information-theoretic perspective. It uncovers a counter-intuitive phenomenon: within a certain range, models that are easier to compress actually exhibit better generalization performance, which aligns with the prediction of the Minimum Description Length (MDL) principle.

**[DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression](dac_prompt_compression.md)**

:   DAC proposes a dynamic attention-aware prompt compression method. By integrating information entropy and attention scores as token importance metrics, and dynamically perceiving the entropy shift during the compression process for fine-grained compression, it improves the average score by 1.33 points over SOTA methods on LongBench.

**[Data Laundering: Artificially Boosting Benchmark Results through Knowledge Distillation](data_laundering_artificially_boosting_benchmark_results_through_knowledge_distil.md)**

:   This paper exposes a vulnerability where knowledge distillation can be abused to artificially inflate benchmark scores. Through "Data Laundering," knowledge learned by a teacher model on a test set is covertly transferred to a student model via seemingly legitimate intermediate training steps. This allows a 2-layer BERT to achieve 73.94% on GPQA (close to OpenAI o1's 77.30%) without actually learning how to reason.

**[DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking](deepsolution_boosting_complex_engineering_solution_design_via_tree-based_explora.md)**

:   This paper proposes SolutionBench, a new benchmark, and SolutionRAG, a new framework, for complex engineering solution design. By leveraging tree-based exploration and bi-point thinking (alternating design and review) within a RAG framework, it progressively generates reliable engineering solutions satisfying multiple constraints, achieving state-of-the-art (SOTA) results across 8 engineering domains.

**[Direct Behavior Optimization: Unlocking the Potential of Lightweight LLMs](direct_behavior_optimization_unlocking_the_potential_of_lightweight_llms.md)**

:   The DeBoP paradigm is proposed to transform the behavior optimization of lightweight LLMs (LwLLM) into the optimization of discrete execution sequences. By employing gradient-free Monte Carlo Tree Search (MCTS) to automatically find the optimal demonstration, LLaMA3-8B outperforms GPT-3.5 on most tasks while reducing computation time by approximately 60%.

**[Disentangling the Roles of Representation and Selection in Data Pruning](disentangling_the_roles_of_representation_and_selection_in_data_pruning.md)**

:   This paper systematically decomposes data pruning into two independent dimensions: "data representation" and "selection algorithms." Through theoretical analysis and large-scale experiments, it is found that representation quality (especially training gradient) plays a decisive role in pruning performance, while different selection algorithms have their own strengths and weaknesses across different scenarios and often deviate from their design goals.

**[DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning](domix_an_efficient_framework_for_exploiting.md)**

:   This paper proposes DoMIX, which stores domain-specific knowledge in independent LoRA modules and flexibly combines them during fine-tuning using a diagonally initialized bridge matrix. Under continual domain-adaptive pretraining scenarios, it reduces pretraining time by 58% and GPU memory by 87% while outperforming state-of-the-art (SOTA) methods.

**[DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization](drpruning_robust_pruning.md)**

:   DRPruning introduces distributionally robust optimization (DRO) to LLM structured pruning. By leveraging scaling laws to predict the final loss of each domain as a reference and dynamically adjusting the training data distribution to balance post-pruning domain performance, it surpasses Sheared LLaMA by -5.59% PPL and +2.95% on downstream tasks in monolingual and multilingual settings, respectively.

**[EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models](eac_moe_expert_aware_compression.md)**

:   EAC-MoE provides an in-depth analysis of the expert selection characteristics of MoE models and proposes two complementary modules: Quantization with Expert Selection Calibration (QESC) to alleviate the expert-shift issue by calibrating routers layer-by-layer during quantization, and Pruning based on Expert Selection Frequency (PESF) to dynamically prune unimportant experts during inference based on selection frequency. It achieves significant memory compression and inference acceleration with minimal accuracy loss across 4 MoE models.

**[Efficient Long Context Language Model Retrieval with Compression](efficient_long_context_language_model_retrieval_with_compression.md)**

:   CoLoR (Compression for Long context Retrieval) is proposed to jointly train a passage compression model using preference optimization and length regularization, compressing the context length by 1.91x while improving the retrieval performance of long-context language models by 6%.

**[Efficient OpAmp Adaptation for Zoom Attention to Golden Contexts](efficient_opamp_adaptation_for_zoom_attention_to_golden_contexts.md)**

:   Inspired by operational amplifier (OpAmp) circuits, this work proposes the OpAmp Adaptation method, which efficiently modifies the attention mechanism of pre-trained Transformers using adapters. This enables LLMs to focus more precisely on golden documents in noisy context scenarios. Qwen2.5-OpAmp-72B outperforms DeepSeek-V3 and GPT-4o on multiple noisy context benchmarks.

**[EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](efficientqat.md)**

:   EfficientQAT proposes a two-stage QAT framework: first performing Block-wise All-Parameter training (Block-AP) to provide a good initialization, and then executing End-to-End Quantization-Parameter fine-tuning (E2E-QP) to capture cross-block interactions. It achieves 2-bit quantization of Llama-2-70B in 41 hours on a single A100 GPU, with only a 3-point accuracy degradation.

**[Explaining Puzzle Solutions in Natural Language: An Exploratory Study on 6×6 Sudoku](explaining_puzzle_solutions_in_natural_language_an_exploratory_study_on_6x6_sudo.md)**

:   Evaluating the capabilities of five LLMs in solving and explaining $6\times6$ Sudoku puzzles reveals that even though o1-preview can solve 65% of the puzzles, its generated explanation chains remain severely lacking in faithfulness, clarity, and educational value.

**[Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients](federated_lora_heterogeneous.md)**

:   Proposes LoRA-A² (Low Rank Adaptation with Alternating freeze and Adaptive rank selection), which addresses the aggregation discordance problem in federated LoRA by alternately freezing matrices A and B. Combined with an adaptive rank selection mechanism, it significantly compresses upload parameter volume (up to 99.8% reduction) while maintaining robustness, outperforming existing methods significantly, especially in low-rank and high data heterogeneity scenarios.

**[FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models](fedex_lora_federated_exact_aggregation.md)**

:   FedEx-LoRA identifies that independently averaging the A and B matrices of LoRA in federated learning yields inaccurate global updates ("the mean of products $\neq$ the product of means"). By incorporating a residual error term into the frozen weight matrix to achieve exact aggregation, FedEx-LoRA consistently outperforms FedIT and FFA-LoRA across multiple reasoning and NLU tasks.

**[Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching](flipping_kd_small_to_large.md)**

:   This paper proposes a "reverse knowledge distillation" paradigm—allowing LLMs to learn domain expertise in text matching from fine-tuned small models. This is achieved by reinterpreting a decoder-only LLM as an encoder-decoder architecture (using the compression matrix of LoRA as the encoder) and designing a Margin-aware Contrastive Loss to align representation similarities.

**[A Silver Bullet or a Compromise for Full Attention? A Comprehensive Study of Gist Token-based Context Compression](gist_token_context_compression.md)**

:   This paper conducts a comprehensive and systematic study of Gist Token-based context compression methods, finding that while the fine-grained KV Cache architecture is near-lossless on tasks like RAG and QA, a significant gap remains in exact recall tasks. It also identifies three critical failure modes and proposes two effective improvement strategies.

**[Graph-guided Cross-composition Feature Disentanglement for Compositional Zero-shot Learning](graph-guided_cross-composition_feature_disentanglement_for_compositional_zero-sh.md)**

:   DCDA proposes a graph-guided cross-composition feature disentanglement scheme. By injecting dual adapters (L-Adapter for text-side GNN feature aggregation and V-Adapter for vision-side cross-attention disentanglement) into the frozen CLIP, it significantly outperforms existing methods on compositional zero-shot learning tasks.

**[GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning](gsq-tuning_group-shared_exponents_integer_in_fully_quantized_training_for_llms_o.md)**

:   GSQ-Tuning proposes a fully quantized fine-tuning framework based on the "Group-Shared Exponents Integer" (GSEI) format. By completely eliminating floating-point operations in both inference and training, it is combined with LoRA adapters to achieve on-device LLM fine-tuning that is close to BF16 fine-tuning in accuracy, while reducing memory by 1.85x, power consumption by 5x, and silicon area by 11x.

**[IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)**

:   Having discovered high similarity between the attention matrices of LLMs of different scales, this work proposes the IAM framework. During the prefill stage, IAM establishes a cosine-similarity mapping between the attention heads of a small language model (SLM) and those of a large language model (LLM). During the decode stage, it replaces the attention computation of the LLM's mapped layers with the attention matrices of the SLM. This achieves a 22% reduction in KV cache and an 11% inference speedup, while remaining orthogonal to existing KV cache compression methods.

**[L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](l4q_parameter_efficient_quantization_aware_finetuning.md)**

:   L4Q is proposed to deeply integrate Quantization-Aware Training (QAT) with LoRA: it first merges weights with LoRA parameters and then applies unified quantization. By customizing the backpropagation path, it eliminates the memory overhead of storing weight gradients, enabling joint optimization of quantization and fine-tuning parameters, which significantly outperforms existing methods under 4-bit and 3-bit quantization.

**[Language Models Resist Alignment: Evidence From Data Compression](language_models_resist_alignment.md)**

:   This paper proposes the concept of LLM "elasticity" from a data compression perspective, proving that the change in compression rate under fine-tuning perturbations is inversely proportional to the dataset size. Because the pre-training data is vastly larger than the alignment data, alignment effects are preferentially "forgotten." This fundamentally explains the fragility of LLM alignment from an information-theoretic standpoint.

**[Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders](language_specific_features.md)**

:   This work utilizes Sparse Autoencoders (SAEs) to analyze the internal representations of multilingual LLMs. It reveals the presence of strong language-specific SAE features, which are correlated not only with language-specific tokens but also with language contexts. Ablating these features only impacts performance on the corresponding language, and synergistic effects are observed among multiple language-specific features. Furthermore, these features are applied to enhance steering vectors, enabling precise control of the generated language.

**[Towards the Law of Capacity Gap in Distilling Language Models](law_of_capacity_gap_distilling_language_models.md)**

:   Unveils the "Law of Capacity Gap" in language model distillation, which states that the size of the optimal teacher model scales linearly with the student model size (approximately 2.5x). This turns the "impossible triangle" in LLM distillation into a solvable problem, leading to the successful distillation of the 3B MiniMA model.

**[Limited-Resource Adapters Are Regularizers, Not Linguists](limited-resource_adapters_are_regularizers_not_linguists.md)**

:   This paper combines adapter souping (weight averaging) with cross-attention fine-tuning for low-resource Creole machine translation. While the method yields significant improvements (up to +8 BLEU), linguistic relatedness does not meaningfully co-vary with adapter performance—randomly initialized, untrained adapters perform equally well. This indicates that the role of adapters in this setting is essentially **parameter regularization rather than linguistic information transfer**.

**[LLMSR@XLLM25: Less is More: Enhancing Structured Multi-Agent Reasoning via Quality-Guided Distillation](llmsrxllm25_less_is_more_enhancing_structured_multi-agent_reasoning_via_quality-.md)**

:   This paper proposes the *Less is More* framework. Under the extreme low-resource conditions of only 24 annotated samples, it distills high-quality structured reasoning data to fine-tune a LLaMA3-8B multi-agent system via three stages: reverse-prompt induction, GPT-4o-enhanced retrieval-augmented reasoning synthesis, and dual-stage reward-guided filtering. It achieved third place in the XLLM@ACL2025 Shared Task.

**[LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation](longred_mitigating_short-text_degradation_of_long-context_large_language_models_.md)**

:   This paper systematically analyzes two causes of short-text performance degradation in long-context LLMs (distribution drift and catastrophic forgetting), and proposes LongReD. Through two training objectives, namely short-text distillation and short-to-long distillation, LongReD minimizes the distribution discrepancy between the extended model and the original model, preserving short-text performance up to 99.4% of the original model while maintaining long-text modeling capabilities.

**[Low-Rank Interconnected Adaptation across Layers](low-rank_interconnected_adaptation_across_layers.md)**

:   Lily (Low-rank Interconnected Adaptation across Layers) is proposed, which decouples and interconnects/shares LoRA's A/B adapters across layers, combined with a data-dependent routing mechanism, to achieve high-rank weight updates with equivalent or fewer parameters, consistently outperforming LoRA across multimodal, multi-architecture, and multi-scale scenarios.

**[MaCP: Minimal yet Mighty Adaptation via Hierarchical Cosine Projection](macp_minimal_yet_mighty_adaptation_via_hierarchical_cosine_projection.md)**

:   This paper proposes MaCP—a parameter-efficient fine-tuning (PEFT) method based on the Discrete Cosine Transform (DCT). By projecting weight updates into the cosine frequency domain and hierarchically selecting the most critical frequency components, MaCP achieves performance superior to or comparable to existing PEFT methods with an extremely low parameter count (99.7% fewer parameters than LoRA).

**[Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation](magnet_multi-turn_tool-use_data_synthesis_and_distillation_via_graph_translation.md)**

:   The Magnet framework is proposed to construct high-quality multi-turn function calling (FC) training trajectories based on random walks and node operations (Insert/Merge/Split) on a function dependency graph. Combined with prompt-based context distillation to generate positive/negative contrastive trajectories for SFT + mDPO training, the 14B model Magnet-14B-mDPO achieves a score of 68.01 on BFCL-v3 (ranking 4th), significantly outperforming the teacher model Gemini-1.5-pro-002 in multi-turn scenarios.

**[Memorization: A Close Look at Books](memorization_a_close_look_at_books.md)**

:   This work systematically investigates the memorization of complete books in the Llama 3 model family, demonstrating that book extraction rates strongly correlate with their popularity (a proxy for training data duplication). Furthermore, through LoRA fine-tuning, it reveals that instruction-tuning mitigates memorization via minimal weight updates concentrated in the bottom transformer blocks.

**[MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](moqae_mixed_precision_kv_cache.md)**

:   MoQAE creatively treats different quantization bit-width configurations as "experts" in MoE, employing a lightweight router to learn the optimal quantization strategy for each chunk. Combined with routing freezing and routing sharing mechanisms, it significantly reduces the KV cache memory of long-context inference with almost zero accuracy loss.

**[MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)**

:   MoRE (Mixture of Low-Rank Experts) is proposed to treat different ranks in LoRA as distinct experts. Through an adaptive rank selector, the most suitable rank is dynamically chosen for each task. Combined with task embeddings optimized by contrastive learning and a balanced data sampling strategy, efficient multi-task fine-tuning is achieved using a single LoRA module.

**[mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding](mplug_docowl2_doc_compress.md)**

:   This paper proposes a layout-aware High-resolution DocCompressor module, which employs global low-resolution visual features as queries and sub-image features as keys/values for grouped cross-attention. This compresses each high-resolution document image from thousands of tokens down to 324 tokens. Combined with a three-stage training framework, it achieves SOTA performance in multi-page document understanding while reducing First Token Latency by over 50%.

**[One QuantLLM for ALL: Fine-tuning Quantized LLMs Once for Efficient Deployments](one_quantllm_for_all_fine-tuning_quantized_llms_once_for_efficient_deployments.md)**

**[Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging](osrm_lora_merging_orthogonal.md)**

:   OSRM identifies that the failure of LoRA model merging stems from interaction interference between parameters and data distributions (rather than merely parameter conflicts). It proposes initializing the LoRA A matrix prior to fine-tuning via eigenvalue decomposition of the data covariance matrix, making its subspace orthogonal to the data distributions of other tasks. This minimizes cross-task interference during merging, significantly improving merging performance across 8 datasets and 5 models.

**[Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models](outlier-safe_pre-training_for_robust_4-bit_quantization_of_large_language_models.md)**

:   The OSP (Outlier-Safe Pre-Training) framework proactively prevents outlier formation during the pre-training phase through three key innovations: the Muon optimizer (eliminating privileged basis directions), Single-Scale RMSNorm (preventing channel magnification), and a learnable embedding projection layer (redistributing embedding layer activations). A 1.4B model trained on 1T tokens achieves near-zero excess kurtosis (0.04 vs. 1818.56 in standard models) and scores an average of 35.7 (compared to 26.5 for Adam) under aggressive 4-bit quantization, with only 2% training overhead.

**[C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](parameter-efficient_fine-tuning_via_circular_convolution.md)**

:   This paper proposes C3A, a method that replaces the low-rank matrix decomposition of LoRA with a circular convolution operator to achieve parameter-efficient fine-tuning. Its key advantage is the decoupling of matrix rank and parameter size, enabling high-rank adaptation with few parameters. Meanwhile, it maintains computational and memory efficiency comparable to LoRA via FFT, consistently outperforming LoRA and its variants across multiple fine-tuning tasks.

**[Pre-training Distillation for Large Language Models: A Design Space Exploration](pre-training_distillation_for_large_language_models_a_design_space_exploration.md)**

:   This work systematically explores the design space of pre-training distillation (PD) for large language models across four dimensions: logits processing, loss function selection, scaling laws, and offline/online logits. Through extensive experiments, optimal configurations and valuable insights are provided.

**[Predicting Through Generation: Why Generation Is Better for Prediction](predicting_through_generation_why_generation_is_better_for_prediction.md)**

:   This paper proves from an information-theoretic perspective that token-level generation retains more mutual information than pooled representations. It proposes the PredGen framework, which addresses the exposure bias and format mismatch issues in generative prediction through scheduled sampling and a task adapter. Additionally, a Writer-Director Alignment Loss is designed to unify the generation and prediction objectives.

**[Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation](prompt_distill_teacher_student.md)**

:   This paper proposes the CanDist framework. Drawing inspiration from human "ambiguity aversion" behavior under uncertainty, it guides the LLM to output multiple candidate labels instead of a single label (candidate annotation). It then distills these annotations into a small language model (SLM) via a Distribution Refinery strategy to obtain final labels. Both theoretical and experimental results demonstrate that candidate annotation distillation outperforms single-label distillation.

**[PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models](ptq161_low_bit_quantization.md)**

:   PTQ1.61 is proposed as the first post-training quantization method that effectively compresses LLM weights to a true sub-2-bit (1.61-bit) format. It achieves state-of-the-art (SOTA) performance through three key techniques: a 1D structured mask (introducing an overhead of only 0.0002-bit), block-wise scaling factor optimization, and quantization preprocessing.

**[Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis](quaff_quantized_peft.md)**

:   This paper proposes the Outlier Spatial Stability Hypothesis (OSSH)—that the spatial locations of activation outlier channels remain stable during fine-tuning—and designs the Quaff framework based on this hypothesis. By handling only a few persistent outlier channels using targeted momentum scaling, Quaff achievements a 1.73× latency reduction and a 30% memory saving, while also improving accuracy on GPQA by 0.6%.

**[Quantification of Large Language Model Distillation](quantification_of_large_language_model_distillation.md)**

:   This paper proposes two complementary LLM distillation quantification methods: Identity Consistency Evaluation (ICE) and Response Similarity Evaluation (RSE). By utilizing jailbreak attacks to uncover identity leakage and multi-granular response similarity, these methods measure the degree of model distillation. The results show that most well-known LLMs (except Claude, Doubao, and Gemini) exhibit a high degree of distillation.

**[Revisiting LoRA through the Lens of Parameter Redundancy: Spectral Encoding Helps](revisiting_lora_through_the_lens_of_parameter_redundancy_spectral_encoding_helps.md)**

:   This paper systematically investigates the parameter redundancy issue in LoRA fine-tuning, discovering that reducing density redundancy does not compromise expressiveness (sparsity property). The authors propose SeLoRA, which reparameterizes LoRA matrices from a sparse spectral subspace using spectral transformations (Fourier/Wavelet) to achieve superior performance with fewer parameters, while offering plug-and-play integration with various LoRA variants.

**[Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models](scaling_laws_and_efficient_inference_for_ternary_language_models.md)**

:   This paper systematically studies the scaling laws of Ternary Language Models (TriLM) and finds that TriLM benefits significantly more from increasing training data than from scaling parameter size. Guided by this insight, the Spectra-1.1 model family (1B/2B/3B) is trained on 1.2T tokens. The authors also propose 1.6-bit and 2-bit weight packing schemes along with the TriRun GPU kernel, achieving up to an 8x inference acceleration.

**[Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing](sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)**

:   This paper proposes Sci-LoRA, a framework that mixes multi-domain LoRAs. By employing a text encoder trained via contrastive learning, a dynamic weight generator, and a LoRA fusion module, it achieves cross-domain lay paraphrasing of scientific texts across 12 disciplines without requiring domain labels, outperforming the state-of-the-art (SOTA) on 10 metrics across 5 datasets.

**[SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation](scope_optimizing_key-value_cache_compression_in_long-context_generation.md)**

:   This paper proposes the SCOPE framework, which separately optimizes Key-Value (KV) cache compression strategies for the prefill and decoding stages in long-context generation tasks. Specifically, the prefill stage preserves the full cache to maintain understanding capability, while the decoding stage utilizes a sliding window to select heavy hitters, further optimizing memory and transmission efficiency through adaptive and discontinuous strategies.

**[Mitigating Selection Bias with Node Pruning and Auxiliary Options](selection_bias_node_pruning.md)**

:   This paper proposes two complementary methods, Bias Node Pruning (BNP) and Auxiliary Option Injection (AOI), to concurrently mitigate the selection bias of LLMs in multiple-choice questions (MCQs) from both internal and external perspectives. This is achieved by localizing and pruning 0.002% of the biased parameters in the model's output layer (white-box) and injecting an "I don't know" auxiliary option (widely applicable to black-box models). Additionally, a distribution-level bias metric, CKLD, is introduced. The combined approach improves the ARC-Challenge accuracy on Llama-3 from 52.3% to 65.3%.

**[Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs](sparse_logit_sampling_accelerating_knowledge_distillation_in_llms.md)**

:   Demonstrates that naive Top-K sparse knowledge distillation yields biased estimation and proposes Random Sampling Knowledge Distillation (RSKD) based on importance sampling. RSKD provides unbiased gradient estimation while requiring the storage of only extremely sparse logits. The training overhead is increased by less than 10% compared to cross-entropy, while maintaining performance on par with full knowledge distillation.

**[State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](state_offset_tuning_ssm_peft.md)**

:   This paper proposes State-offset Tuning, a novel family of "state-based" PEFT methods for SSMs (such as Mamba). By directly injecting a trainable state offset $h'$ at each time step rather than virtual tokens used in Prefix-Tuning, it overcomes the issue of limited expressivity of prompt-based approaches on SSMs, consistently outperforming LoRA and Prefix-Tuning with fewer parameters.

**[STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning](stun_moe_pruning.md)**

:   STUN proposes a two-stage MoE pruning paradigm of "structured-then-unstructured": the first stage utilizes routing weight behavioral similarity to cluster redundant experts, completing expert-level pruning with $O(1)$ GPU forward passes; the second stage performs unstructured weight pruning within the remaining experts. Co-designing these two stages allows 40% sparsification on the 480B Snowflake Arctic with almost no performance loss.

**[TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering](tada_training-free_recipe_for_decoding_with_adaptive_kv_cache_compression_and_me.md)**

:   The authors propose TaDA—a training-free KV cache compression method. By performing head-wise mean-centering on K/V activations and then quantizing the deviations (instead of raw activations), TaDA automatically eliminates the outlier problem. Combined with layer-wise adaptive quantization bit-width search, it compresses the KV cache to 27% of the original 16-bit size while preserving near-baseline accuracy.

**[TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition](teamlora_boosting_low-rank_adaptation_with_expert_collaboration_and_competition.md)**

:   TeamLoRA is proposed to optimize the Multi-LoRA architecture through an asymmetric collaboration module (a "plug-and-play" structure with shared A matrices and multiple expert B matrices) and a competition module based on Shapley values. This achieves a better performance-efficiency trade-off in multi-task learning—reducing training time by 30% and increasing inference speed by 40% compared to MoELoRA, while achieving superior performance.

**[Trans-PEFT: Transferable Parameter-Efficient Fine-Tuning on Evolving Base Models](trans_peft_transferable.md)**

:   Trans-PEFT discovers that base model updates (e.g., Qwen2→Qwen2.5) primarily alter task knowledge stored in FFN layers while minimally affecting task patterns in Attention layers. Based on this insight, it proposes two strategies—intra-layer knowledge masking and cross-layer knowledge dropping—enabling PEFT modules trained on older versions to be directly transferred to newer versions without re-fine-tuning, yielding performance gains of up to 30%.

**[UniICL: An Efficient ICL Framework Unifying Compression, Selection, and Generation](uniicl_icl_framework.md)**

:   This work proposes the UniICL framework, which utilizes **a single frozen LLM** to concurrently accomplish three tasks: demonstration compression (compress $\rightarrow$ virtual tokens), demonstration selection (ranking based on the similarity of compressed virtual tokens), and final response generation. It requires only 17M trainable parameters (projection layer + learnable embedding). Coupled with a Demonstration Bank caching mechanism to avoid redundant compression, UniICL scales from 4-shot to 64-shot ICL under a 12$\times$ compression ratio (within 24GB VRAM), outperforming baselines like AutoCompressor, ICAE, and LLMLingua on multiple out-of-domain datasets.

**[UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models](uniquanf_unified_quantization.md)**

:   UniQuanF unifies the strengths of Uniform Quantization (UQ, high optimizability but low representational capacity) and Binary-Coding Quantization (BCQ, high representational capacity but low optimizability). Through unified initialization, local periodic mapping, and a unification theorem, it achieves highly accurate LLM quantization without any extra deployment overhead, yielding up to 4.60% improvement on GSM8K.

**[Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)**

:   Wanda++ is proposed: a lightweight LLM pruning framework based on decoder block-level regional gradients. It improves the pruning criterion with Regional Gradient Score (RGS) and minimizes the output discrepancies between dense and sparse blocks via Regional Optimization (RO). Under 2:4 sparsity, it reduces WikiText perplexity by up to 32% compared to Wanda, while pruning a 7B model within 10 minutes on a single H100 GPU.

**[Who Taught You That? Tracing Teachers in Model Distillation](who_taught_you_that_tracing_teachers_in_model_distillation.md)**

:   This paper introduces a novel problem of "teacher model attribution": given a distilled student model, can its training teacher be identified from a pool of candidate teachers? It is found that n-gram similarity and perplexity are unreliable, whereas Part-of-Speech (PoS) syntactic templates provide effective signals for teacher identification.
