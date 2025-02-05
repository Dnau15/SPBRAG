**Project Report: Development of a Retrieval Classification Module for RAG Pipelines**  
*Author: Dnau15*  

---

### **1. Introduction**  
**Objective**: This project aims to develop a classification module to determine whether retrieval augmentation (RAG) is necessary for a given user query. By selectively bypassing the retrieval step in cases where it is unnecessary (e.g., simple arithmetic or trivial requests), the system reduces computational overhead and improves response latency.  

**Background**:  
Retrieval-Augmented Generation (RAG) pipelines typically involve:  
1. **Chunking**: Segmenting documents into smaller units.  
2. **Embedding & Vector Database**: Storing and retrieving contextually relevant text chunks.  
3. **Reranking**: Prioritizing retrieved documents by relevance.  
4. **LLM Generation**: Synthesizing answers using retrieved context.  

While RAG enhances factual accuracy, it introduces latency due to retrieval and processing of large contexts. Many queries (e.g., "2+2," "Count the letters 'r' in 'strawberry'") do not require external knowledge. A classifier that predicts whether retrieval is needed can optimize this process.  

---

### **2. Methodology**  

#### **2.1 Dataset Collection & Preparation**  
Three datasets were used to train the classifier:  

1. **Merge/Rewrite Dataset** ([positivethoughts/merge_rewrite_13.3k](https://huggingface.co/datasets/positivethoughts/merge_rewrite_13.3k)):  
   - **Purpose**: Examples of prompts *not* requiring retrieval (e.g., text merging/rewriting tasks).  
   - **Columns**: 
        - Rewrite prompt (user prompt)
        - Rewritten text (model output)
        - Original text (text provided by user)
        - Id (just id ) 😃  

2. **Databricks Dolly 15k** ([databricks/dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)):  
   - **Purpose**: Mixed prompts requiring/not requiring retrieval.  
   - **Labeling Strategy**: Prompts labeled as requiring retrieval if they involved open-ended questions, brainstorming, or factual queries (e.g., "Why do camels survive without water?"). Non-retrieval examples included creative writing and instructions. 
   - **Columns**:
        - Instruction (user prompt)
        - Context (context provided by user)
        - Response (model output)
        - Category (category of prompt)

3. **SQuAD v2** ([rajpurkar/squad_v2](https://huggingface.co/datasets/rajpurkar/squad_v2)):  
   - **Purpose**: Questions explicitly requiring retrieval (e.g., Wikipedia-based Q&A). 
   - **Columns**:
        - Id (just id) 😃 
        - Title (belong to some topic, e.g. Beyonce, Dog, etc.)
        - Context (context provided by user)
        - Question (user prompt)
        - Answers (model output)

**Data Processing**:  
- Removed NaN values and duplicates.  
- Balanced classes by undersampling majority categories.  
- Final dataset: ~3,000 samples (train/val/test split: 70%/15%/15%).  

#### **2.2 Model Training**  
**Model Choice**: BERT-base multilingual model (fine-tuned for binary classification: "retrieval needed" vs. "no retrieval needed").  

**Hyperparameters**:  
- Learning rate: 5e-5  
- Batch size: 4 (constrained by GPU memory)  
- Epochs: 3  
- Optimizer: AdamW  

**Training Workflow**:  
1. Tokenized prompts using BERT tokenizer.  
2. Trained on a balanced subset of 3,000 examples.  
3. Monitored validation loss and accuracy to prevent overfitting.  

#### **2.3 Pipeline Implementation**  
A simplified RAG pipeline was developed to test the classifier:  

1. **Chunking**:  
   - Chunk size: 512 tokens  
   - Overlap: 20 tokens  

2. **Vector Database**:  
   - **Database**: Milvus (open-source, scalable)  
   - **Index**: IVF_FLAT with L2 distance metric  

3. **LLM**:  
   - Initial choice: Mistral-7B-Instruct-v0.2 (via Hugging Face API).  
   - Fallback: Llama3-3B (via Ollama due to API limitations).  

**Rationale for Omissions**:  
- **Reranker**: Excluded to focus on classifier performance; reranker integration is orthogonal to retrieval time savings.  
- **Simplified LLM**: Smaller models (e.g., Llama3-3B) were prioritized for faster prototyping.  

---

### **3. Results & Analysis**  
1. **Classifier Performance**:  
   - Achieved **92% accuracy** on the test set (Figure 1).  
   - Validation loss stabilized after 3 epochs

   **Pipeline Efficiency**:  
   - Queries flagged as "no retrieval" bypassed the vector database, reducing latency by **~30%** (average).  

---

### **4. Limitations & Future Work**  
**Limitations**:  
- Small training dataset (~3k samples) compared to the article’s 100k.  
- Simplified pipeline lacking reranker and advanced LLMs.  

**Future Improvements**:  
1. Expand training data with synthetic prompts (e.g., GPT-generated examples).  
2. Integrate reranker for improved document relevance.  
3. Benchmark against larger LLMs (e.g., Llama3-70B).  

---

### **5. Conclusion**  
This project demonstrates that a lightweight classifier can optimize RAG pipelines by dynamically toggling retrieval. The approach reduces latency without compromising accuracy for simple queries. Future work will focus on scaling the classifier and integrating advanced components for production-grade performance.  

---  
**Figures**:  
- *Figure 1*: Classifier accuracy on test set.  
- *Figure 2*: Training/validation loss curves.  

[← Back to Main Documentation](../README.md)
