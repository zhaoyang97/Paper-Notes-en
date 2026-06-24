---
title: >-
  ACL2025 LLM Efficiency Papers · 42 Notes
description: >-
  42 ACL2025 papers in the LLM Efficiency area, covering LLM, Summarization, Dialogue, Question Answering, Reasoning, Adversarial Robustness and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ACL2025"
  - "LLM Efficiency"
  - "AI paper notes"
  - "paper summaries"
  - "LLM"
  - "Summarization"
  - "Dialogue"
  - "Question Answering"
  - "Reasoning"
  - "Adversarial Robustness"
item_list:
  - u: "a_drop-in_solution_for_on-the-fly_adaptation_of_speculative_decoding_in_large_la/"
    t: "A Drop-In Solution for On-the-Fly Adaptation of Speculative Decoding in Large Language Models"
  - u: "accelerating_speculative_decoding_via_efficient_context-aware_draft_generation/"
    t: "Accelerating Speculative Decoding via Efficient Context-Aware Draft Generation"
  - u: "adaptive_grouped_pe_context_window/"
    t: "LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive Long-context Scaling Without Training"
  - u: "boosting_long-context_information_seeking_via_query-guided_activation_refilling/"
    t: "Boosting Long-Context Information Seeking via Query-Guided Activation Refilling"
  - u: "clasp_self_speculative_decoding/"
    t: "CLaSp: In-Context Layer Skip for Self-Speculative Decoding"
  - u: "cnnsum_exploring_long-context_summarization_with_large_language_models_in_chines/"
    t: "CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels"
  - u: "consistency-preserving_contrastive_decoding_for_faithful_document-grounded_dial/"
    t: "Consistency-Preserving Contrastive Decoding for Faithful Document-Grounded Dialogue"
  - u: "consultant_decoding_yet_another_synergistic_mechanism/"
    t: "Consultant Decoding: Yet Another Synergistic Mechanism"
  - u: "decoding_knowledge_attribution_in_mixture-of-experts_a_framework_of_basic-refine/"
    t: "Decoding Knowledge Attribution in Mixture-of-Experts: A Framework of Basic-Refinement Collaboration and Efficiency Analysis"
  - u: "design_choices_for_extending_the_context_length_of_visual_language_models/"
    t: "Giraffe: Design Choices for Extending the Context Length of Visual Language Models"
  - u: "distance_between_relevant_information_pieces_causes_bias_in_long-context_llms/"
    t: "Distance between Relevant Information Pieces Causes Bias in Long-Context LLMs"
  - u: "dive_moe_reconstruction/"
    t: "DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts"
  - u: "dynamic_chunking_and_selection_for_reading_comprehension_of_ultra-long_context_i/"
    t: "Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models"
  - u: "efficient_many-shot_in-context_learning_with_dynamic_block-sparse_attention/"
    t: "Efficient Many-Shot In-Context Learning with Dynamic Block-Sparse Attention"
  - u: "entailment-preserving_first-order_logic_representations_in_natural_language_enta/"
    t: "Entailment-Preserving First-order Logic Representations in Natural Language Entailment"
  - u: "epman_episodic_memory_attention_for_generalizing_to_longer_contexts/"
    t: "EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts"
  - u: "fastdraft_how_to_train_your_draft/"
    t: "FastDraft: How to Train Your Draft"
  - u: "focusllm_precise_understanding_of_long_context_by_dynamic_condensing/"
    t: "FocusLLM: Precise Understanding of Long Context by Dynamic Condensing"
  - u: "fuel_unveiling_environmental_impacts_of_llm_serving/"
    t: "FUEL: Unveiling Environmental Impacts of Large Language Model Serving: A Functional Unit View"
  - u: "gigachat_family_efficient_russian_language_modeling_through_mixture_of_experts_a/"
    t: "GigaChat Family: Efficient Russian Language Modeling Through Mixture of Experts Architecture"
  - u: "kv_latent_cache_reduction/"
    t: "KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding"
  - u: "ladm_long_context_data/"
    t: "LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs"
  - u: "literary_evidence_retrieval_via_long-context_language_models/"
    t: "Literary Evidence Retrieval via Long-Context Language Models"
  - u: "longbench_v2_towards_deeper_understanding_and_reasoning_on_realistic_long-contex/"
    t: "LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks"
  - u: "longreward_improving_long-context_large_language_models_with_ai_feedback/"
    t: "LongReward: Improving Long-context Large Language Models with AI Feedback"
  - u: "longsafety_evaluating_long-context_safety_of_large_language_models/"
    t: "LongSafety: Evaluating Long-Context Safety of Large Language Models"
  - u: "many_shot_attacks_long_context/"
    t: "What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs"
  - u: "mitigating_posterior_salience_attenuation_in_long-context_llms_with_positional_c/"
    t: "Mitigating Posterior Salience Attenuation in Long-Context LLMs with Positional Contrastive Decoding"
  - u: "native_sparse_attention/"
    t: "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"
  - u: "on_many-shot_in-context_learning_for_long-context_evaluation/"
    t: "On Many-Shot In-Context Learning for Long-Context Evaluation"
item_total: 42
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# ⚡ LLM Efficiency

**💬 ACL2025** · **42** paper notes

📌 **Same area in other venues:** [📷 CVPR2026 (8)](../../CVPR2026/llm_efficiency/index.md) · [🔬 ICLR2026 (171)](../../ICLR2026/llm_efficiency/index.md) · [💬 ACL2026 (23)](../../ACL2026/llm_efficiency/index.md) · [🧪 ICML2026 (48)](../../ICML2026/llm_efficiency/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_efficiency/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/llm_efficiency/index.md)

🔥 **Top topics:** LLM ×9

**[A Drop-In Solution for On-the-Fly Adaptation of Speculative Decoding in Large Language Models](a_drop-in_solution_for_on-the-fly_adaptation_of_speculative_decoding_in_large_la.md)**

:   This paper proposes a drop-in adaptive solution for speculative decoding that dynamically adjusts the speculative window size $\gamma$ (and potentially the choice of draft models) during inference, thereby maximizing the end-to-end speedup of speculative decoding under diverse input distributions.

**[Accelerating Speculative Decoding via Efficient Context-Aware Draft Generation](accelerating_speculative_decoding_via_efficient_context-aware_draft_generation.md)**

:   This paper proposes an efficient context-aware draft generation strategy to accelerate speculative decoding. By enabling the draft model to dynamically adjust the generation quality based on the current context, it significantly improves LLM inference throughput while maintaining output consistency.

**[LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive Long-context Scaling Without Training](adaptive_grouped_pe_context_window.md)**

:   Ours proposes LaMPE (Length-aware Multi-grained Positional Encoding), which adaptively determines the optimal mapping length using a **parameterized scaled sigmoid function**, and designs a **three-region multi-grained attention mechanism** (fine-grained local head + linearly normalized and compressed middle + tail that restores long-range dependencies) to achieve training-free plug-and-play context window extrapolation for LLMs, comprehensively outperforming existing methods on five major long-context benchmarks.

**[Boosting Long-Context Information Seeking via Query-Guided Activation Refilling](boosting_long-context_information_seeking_via_query-guided_activation_refilling.md)**

:   This paper proposes ACRE (Activation Refilling), which constructs a bi-layer KV Cache architecture—consisting of an L1 layer to compactly capture global information and an L2 layer to provide detailed local information. By using the input query to dynamically replenish relevant items from L2 to L1, ACRE achieves highly efficient processing of long-context information retrieval tasks, with significant improvements in both performance and efficiency.

**[CLaSp: In-Context Layer Skip for Self-Speculative Decoding](clasp_self_speculative_decoding.md)**

:   CLaSp proposes a training-free self-speculative decoding method that dynamically adjusts the layer skipping strategy based on context after each verification step using a dynamic programming algorithm. By utilizing the full hidden states of the previous verification step as the target to select the optimal set of skipped layers, it achieves $1.3-1.7\times$ speedup on the LLaMA3 series without altering the generation distribution.

**[CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels](cnnsum_exploring_long-context_summarization_with_large_language_models_in_chines.md)**

:   The authors construct CNNSum—a multi-scale long-text summarization benchmark based on Chinese novels (695 samples, 16k-128k tokens)—ensuring quality via human annotation. Through a systematic evaluation of 20+ LLMs, they discover that advanced LLMs tend to generate subjective commentary leading to vague summaries, smaller models offer better cost-effectiveness, fine-tuning Base versions yields superior results over Chat versions, and fine-tuning on short-context data alone can significantly enhance long-text summarization capabilities.

**[Consistency-Preserving Contrastive Decoding for Faithful Document-Grounded Dialogue](consistency-preserving_contrastive_decoding_for_faithful_document-grounded_dial.md)**

:   This paper proposes Consistency-Preserving Contrastive Decoding (CPCD), a method that contrasts document-conditioned and document-free generation distributions during the decoding phase. This strategy enhances the faithfulness of document-grounded dialogue systems to source documents while maintaining response fluency and dialogue consistency.

**[Consultant Decoding: Yet Another Synergistic Mechanism](consultant_decoding_yet_another_synergistic_mechanism.md)**

:   This paper proposes Consultant Decoding (CD), a novel cooperative decoding mechanism that verifies draft tokens based on the target model's negative log-likelihood (NLL). Compared to the likelihood-ratio verification methods of traditional speculative decoding, CD significantly improves the acceptance rate, reduces the frequency of target model calls, and maintains or even exceeds the generation quality of the target model.

**[Decoding Knowledge Attribution in Mixture-of-Experts: A Framework of Basic-Refinement Collaboration and Efficiency Analysis](decoding_knowledge_attribution_in_mixture-of-experts_a_framework_of_basic-refine.md)**

:   Proposes a cross-layer knowledge attribution algorithm to systematically analyze the "basic-refinement" collaboration framework of shared experts and routed experts in MoE models, revealing that MoEs achieve 31% higher layer-wise efficiency compared to dense models, and validating the decisive impact of architectural depth on robustness through a semantic-driven routing mechanism (attention head-expert correlation $r=0.68$) and expert blocking experiments.

**[Giraffe: Design Choices for Extending the Context Length of Visual Language Models](design_choices_for_extending_the_context_length_of_visual_language_models.md)**

:   This work systematically explores the design space for extending the context window of existing Visual Language Models (VLMs) to 128K. It proposes best practices across three dimensions—data recipe, positional encoding extension, and context utilization—and introduces two techniques: M-RoPE++ and hybrid-resolution training. The resulting Giraffe model achieves state-of-the-art (SOTA) performance among long-context VLMs.

**[Distance between Relevant Information Pieces Causes Bias in Long-Context LLMs](distance_between_relevant_information_pieces_causes_bias_in_long-context_llms.md)**

:   This paper proposes the LongPiBench benchmark to systematically investigate, for the first time, LLMs' sensitivity to the distance (spacing) between multiple relevant information pieces in long contexts. It reveals that while current models have largely overcome the "lost-in-the-middle" problem, they still exhibit significant positional bias when the spacing between relevant information pieces varies.

**[DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts](dive_moe_reconstruction.md)**

:   This paper proposes DIVE, a method to reconstruct dense LLMs into MoE architectures. The core insight is that calibration datasets from different domains lead structured pruning to produce distinct pruning candidates, which can be leveraged to build domain-specific experts. Combined with an efficient two-stage retraining strategy (dense router training + sparse expert LoRA training), DIVE outperforms existing pruning and MoE reconstruction methods while updating less than 1% of the parameters.

**[Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models](dynamic_chunking_and_selection_for_reading_comprehension_of_ultra-long_context_i.md)**

:   Proposes Dynamic Chunking and Selection (DCS), which addresses semantic fragmentation caused by fixed chunking in long texts through semantic similarity-based dynamic chunking and question-aware classifier-based chunk selection. Using Llama3 as the base model, it achieves a single-hop average of 35.50 (+28.6%) and a multi-hop average of 29.07 (+20.0%) across 12 long-text QA datasets, while maintaining robustness under 256k token inputs.

**[Efficient Many-Shot In-Context Learning with Dynamic Block-Sparse Attention](efficient_many-shot_in-context_learning_with_dynamic_block-sparse_attention.md)**

:   Proposes Dynamic Block-Sparse Attention (DBSA), a training-free inference framework that achieves near-fine-tuning inference latency in many-shot in-context learning through structured block-sparse attention encoding and dynamic Retrieval-based KV cache while maintaining >95% of the accuracy of the best-performing methods.

**[Entailment-Preserving First-order Logic Representations in Natural Language Entailment](entailment-preserving_first-order_logic_representations_in_natural_language_enta.md)**

:   This paper formally defines the Entailment-Preserving First-Order Logic Representation (EPF) task and a set of reference-free evaluation metrics (EPR family). It proposes an iterative learning-to-rank training method that optimizes the NL→FOL translation of T5 models via the BRIO loss, enabling the generated FOL representations to be verified for entailment relations by automated theorem provers. It improves EPR by 1.8–2.7% and EPR@16 by 17.4–20.6% across three datasets.

**[EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts](epman_episodic_memory_attention_for_generalizing_to_longer_contexts.md)**

:   This paper proposes the EpMAN method, which estimates the relative relevance of context chunks through an episodic memory module, uses this relevance to re-weight the decoder's self-attention (differentiating attention), and combines it with noisy training and attention range expansion strategies. It achieves stronger and more robust performance than long-context LLMs and RAG in the 16k-256k context length range.

**[FastDraft: How to Train Your Draft](fastdraft_how_to_train_your_draft.md)**

:   This work proposes FastDraft, an efficient draft model pre-training and alignment pipeline. It can train a draft model of approximately 50M parameters on a single node with 8 GPUs within 24 hours. When paired with Speculative Decoding, it achieves up to a 3x memory bandwidth speedup and a 2x actual inference speedup.

**[FocusLLM: Precise Understanding of Long Context by Dynamic Condensing](focusllm_precise_understanding_of_long_context_by_dynamic_condensing.md)**

:   This paper proposes FocusLLM, a framework that dynamically condenses key information from long text by partitioning it into chunks and injecting dynamic prompts into each chunk. These chunks are condensed into candidate tokens using a trainable mechanism, which are then aggregated into the local context via a parallel decoding mechanism to generate the next token. Only requiring an 8K training length and a 0.5B token training budget, it successfully extends LLaMA-2 to a 400K context, outperforming all baselines on LongBench and ∞-Bench.

**[FUEL: Unveiling Environmental Impacts of Large Language Model Serving: A Functional Unit View](fuel_unveiling_environmental_impacts_of_llm_serving.md)**

:   Proposes the FUEL framework, which for the first time introduces the concept of "Functional Unit" from life cycle assessment (LCA) as a standardized baseline for comparison. It evaluates carbon emissions of different LLM serving configurations under unified quality, performance, and workload constraints, revealing several counter-intuitive green AI insights through three case studies: model size, quantization strategy, and hardware selection.

**[GigaChat Family: Efficient Russian Language Modeling Through Mixture of Experts Architecture](gigachat_family_efficient_russian_language_modeling_through_mixture_of_experts_a.md)**

:   Introduces the GigaChat family—the first MoE-architecture LLM family designed and pre-trained from scratch for the Russian language. It includes base and instruction-tuned models with 20B total and 3.3B active parameters, achieving SOTA performance among models of the same scale on Russian benchmarks, with a training speed $2\times$ faster than dense models of equivalent capacity and a 40% reduction in inference latency.

**[KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding](kv_latent_cache_reduction.md)**

:   KV-Latent achieves 50-87% KV Cache compression with less than 1% of the pre-training tokens' computational cost while maintaining performance. It achieves this by directly shrinking Key/Value attention head dimensions (mapping KV vectors to a low-dimensional latent space) and adapting a two-stage fine-tuning strategy along with frequency-aware RoPE modifications.

**[LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs](ladm_long_context_data.md)**

:   LADM proposes an attention-based long-context training data selection framework. By training a small-scale Long Attention Calculator to compute attention dependency scores between spans (PFS → AFS → CDS), it efficiently screens high-quality samples with strong long-range dependency from large-scale corpora for continual pre-training. Using only 1B tokens, it significantly enhances the long-context capabilities of LLMs.

**[Literary Evidence Retrieval via Long-Context Language Models](literary_evidence_retrieval_via_long-context_language_models.md)**

:   The RELiC dataset is adapted into a long-context literary evidence retrieval benchmark (292 high-quality samples), requiring models to find missing citations for literary analyses within full novel texts (45k-125k tokens). Gemini Pro 2.5 achieves a 62.5% accuracy, surpassing human experts (55%) for the first time, whereas the best open-source model, DeepSeek-R1, reaches only 29.1%, highlighting a huge gap in interpretative reasoning between closed-source and open-source models.

**[LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks](longbench_v2_towards_deeper_understanding_and_reasoning_on_realistic_long-contex.md)**

:   LongBench v2 is a challenging long-context evaluation benchmark consisting of 503 hard multiple-choice questions, with context lengths ranging from 8k to 2M tokens, covering six major task types. Under a 15-minute time limit, human experts only achieve an accuracy of 53.7%, while the strongest direct-generation model (GPT-4o 2024-08) achieves only 50.1%, and the reasoning model o1-preview reaches 57.7%, highlighting the critical importance of test-time compute scaling for deep long-context understanding.

**[LongReward: Improving Long-context Large Language Models with AI Feedback](longreward_improving_long-context_large_language_models_with_ai_feedback.md)**

**[LongSafety: Evaluating Long-Context Safety of Large Language Models](longsafety_evaluating_long-context_safety_of_large_language_models.md)**

:   Proposes LongSafety, the first LLM safety evaluation benchmark specifically tailored for open-ended long-context tasks. It covers 7 safety categories and 6 task types across 1,543 test cases. The evaluation reveals that most models achieve a safety rate below 55%, and safety capabilities in short contexts do not transfer well to long-context scenarios.

**[What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs](many_shot_attacks_long_context.md)**

:   This study systematically analyzes the key factors of Many-Shot Jailbreaking (MSJ) attacks, finding that context length is the decisive factor in attack success, while content harmfulness, topic, and format are nearly irrelevant—even repeating safe content or random meaningless text (Lorem Ipsum) can break the safety alignment of the model in long contexts.

**[Mitigating Posterior Salience Attenuation in Long-Context LLMs with Positional Contrastive Decoding](mitigating_posterior_salience_attenuation_in_long-context_llms_with_positional_c.md)**

:   This work identifies the phenomenon of Posterior Salience Attenuation (PSA) in long-context LLMs, where the salience of gold tokens decreases as context length grows while they still maintain top ranks. Consequently, a training-free Positional Contrastive Decoding (PCD) method is proposed to amplify long-range signals by contrasting logits from long-range-aware attention and local-aware attention, achieving SOTA results across multiple long-context benchmarks.

**[Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](native_sparse_attention.md)**

:   DeepSeek proposes NSA—a natively trainable hierarchical sparse attention mechanism that achieves efficient long-context modeling through three parallel attention paths: token compression, token selection, and sliding window. After pre-training on a 27B parameter model, its performance matched or even surpassed Full Attention across all metrics, while delivering significant acceleration on 64k sequences.

**[On Many-Shot In-Context Learning for Long-Context Evaluation](on_many-shot_in-context_learning_for_long-context_evaluation.md)**

:   This paper conducts an in-depth study of many-shot ICL for evaluating long-context language models, proposes the Sample Learning Ratio (SLR) metric to distinguish between SSL and ASL tasks, and constructs the ManyICLBench benchmark to comprehensively evaluate 12 LCLMs.

**[Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)**

:   This paper proposes the Ref-Long benchmark to evaluate long-context language models (LCLMs) from the overlooked dimension of "referencing attribution" (identifying which documents reference a given key and returning their indices). It contains 3 subsets (ranging from synthetic to real) with a total of 4,300 tasks. The findings reveal that even GPT-4o achieves only 19% ExAcc on Multi-Hard-24K, far below the human baseline of 92%, and neither prompt engineering nor specialized fine-tuning can fundamentally resolve this issue.

**[RefreshKV: Updating Small KV Cache During Long-form Generation](refreshkv_updating_small_kv_cache_during_long-form_generation.md)**

:   Proposes RefreshKV, an inference method that periodically alternates between full KV cache attention and small KV cache attention during generation, dynamically updating the small KV cache based on the attention patterns of full attention steps. Without permanently discarding any tokens, it achieves acceleration comparable to eviction-based methods while significantly improving performance on long-form generation tasks.

**[SAM Decoding: Speculative Decoding via Suffix Automaton](sam_decoding_speculative_decoding_via_suffix_automaton.md)**

:   This paper proposes SAM-Decoding, which utilizes a Suffix Automaton (SAM) to perform longest suffix matching on general text corpora and the current text sequence for efficient draft generation in speculative decoding. Achieving an average $O(1)$ time complexity, it outperforms existing retrieval-based methods on Spec-Bench by over 18%, and can be combined complementarily with methods like EAGLE-2 to yield a further speedup of 3.28%–11.13%.

**[Scaling Context, Not Parameters: Training a Compact 7B Language Model for Efficient Long-Context Processing](scaling_context_not_parameters_training_a_compact_7b_language_model_for_efficien.md)**

:   Proposes MegaBeam-Mistral-7B, a 7B language model supporting a 512K token context length. Through engineering practices such as four-stage progressive training, RoPE theta tuning, bfloat16 precision correction, and XLA compiler memory optimization, this compact model achieves and even surpasses the performance of larger parameter models (such as Llama-3.1-70B, GPT-4) on long-context tasks.

**[SEAL: Scaling to Emphasize Attention for Long-Context Retrieval](seal_scaling_to_emphasize_attention_for_long-context_retrieval.md)**

:   By identifying that specific attention heads/channels have positive or negative impacts on long-context retrieval, SEAL designs learnable head-level and channel-level scaling factors. Fine-tuning with only 50 synthetic samples significantly improves the long-context retrieval performance of LLMs, and these scaling factors can be merged into model weights offline to achieve zero inference overhead.

**[Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models](sliding_windows_full_ranking.md)**

:   This paper systematically investigates the application of long-context LLMs in passage ranking, proposing the use of full ranking (ranking all passages at once) to replace traditional sliding window strategies. By designing a multi-pass sliding window label construction method and an importance-aware loss function to fine-tune the full ranking model, this approach achieves comprehensive improvements in ranking performance while enhancing efficiency by approximately 30-65%.

**[Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](smarter_better_faster_longer_a_modern_bidirectional_encoder_for_fast_memory_effi.md)**

:   ModernBERT is proposed, systematically introducing modern LLM architectural optimizations (RoPE, GeGLU, alternating local/global attention, and unpadding) into encoder-only models. Trained on 2T tokens and natively supporting a context length of 8192, it outperforms BERT/RoBERTa/DeBERTaV3 across classification and retrieval tasks while achieving significantly faster inference speeds and superior memory efficiency.

**[SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers](spindlekv_layered_kv_cache.md)**

:   SpindleKV proposes a layer-aware KV cache compression strategy: active attention-driven token eviction in deep layers (leveraging sparse attention), and similarity-based codebook substitution in shallow layers (leveraging high token similarity), while resolving GQA compatibility issues to achieve a 50% KV cache reduction without performance loss.

**[Squeezed Attention: Accelerating Long Context Length LLM Inference](squeezed_attention_accelerating_long_context_length_llm_inference.md)**

:   Ours proposes Squeezed Attention, which compresses the Key vectors of fixed contexts via offline K-means clustering. During inference, centroid matching is used to predict important Keys and compute exact attention solely on them. This achieves a 3.1x reduction in the KV budget with no loss in accuracy, yielding over 4x speedup in both prefill and generation stages.

**[Tetris: Optimal Draft Token Selection for Batch Speculative Decoding](tetris_optimal_draft_token_selection_for_batch_speculative_decoding.md)**

:   Tetris proposes a method to dynamically select the optimal draft tokens across requests in batch speculative decoding scenarios, maximizing inference throughput under limited computational resources by greedily choosing tokens with the highest cumulative acceptance probability.

**[How to Train Long-Context Language Models (Effectively)](train_long_context_effectively.md)**

:   This paper systematically studies how to effectively train long-context language models through continual pre-training and SFT. It proposes a series of key designs, including data mixing ratios, training length scaling, and evaluation protocols. The resulting ProLong-8B achieves the same-scale SOTA on 128K length using only **5%** of Llama-3.1's long-context training data.

**[What are the Essential Factors in Crafting Effective Long Context Multi-Hop Instruction Datasets? Insights and Best Practices](what_are_the_essential_factors_in_crafting_effective_long_context_multi-hop_inst.md)**

:   A Multi-agent Interactive Multi-hop Generation (MIMG) framework is proposed to systematically synthesize high-quality long-context multi-hop instruction data through four modules: quality validation, single-hop question generation, multi-question sampling, and multi-hop merging. The trained models achieve an average improvement of 7.54%, even surpassing larger-scale human-annotated datasets.
