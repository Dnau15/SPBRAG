# SPBRAG - Optimizing Retrieval Augmented Generation Pipelines 🚀


Advanced RAG system with dynamic query classification, supporting Llama-3.2 models through Ollama integration.

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Fine Tuning](#-fine-tuning)
- [Usage](#-usage)
---

## 🌟 Overview

SPBRAG enhances RAG pipelines using a hybrid approach:
1. **BERT-based Query Classification** - Determines context requirement
2. **Llama-3.2 LLM** - Generates context-aware responses
3. **Evaluation Framework** - Measures precision/recall of retrieval components

---

## ✨ Key Features

- **Multi-Model Support** 🤖  
  `llama3.2:1B` (fast) and `llama3.2:3B` (high-quality) variants
- **Automatic Model Handling** ⚙️  
  One-line model downloads via Ollama
- **Secure Configuration** 🔑  
  Environment-based API key management
- **Flexible Training** 🏋️  
  Custom BERT fine-tuning capabilities

---

## 📊 Evaluation Metrics

For detailed explanations of our evaluation metrics and interpretation guidelines, see:
[Metrics Documentation](./docs/metrics.md)

Key tracked metrics include:
- F1 score and EM score
- Metrics For Evaluation RAG
- Classification Accuracy For BERT
- Latency Benchmarks

---
## 📚
To review the progress of my work and my thoughts, I suggest you read:
[Workflow](./docs/workflow.md)
---

## 🛠 Installation

### Step 1: System Preparation

#### Install micromamba if missing
```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

#### For fish users
```bash
curl -L micro.mamba.pm/install.sh | sh
```
---

### Step 2: Repository Setup
#### Clone the project

```bash
git clone https://github.com/Dnau15/SPBRAG.git
cd SPBRAG
```

---

### Step 3: Environment Configuration

#### Create and activate environment, install dependencies

```bash
micromamba create -n spbrag python=3.11 -y
micromamba activate spbrag

./setup.sh
```

#### If you have problems with executing .sh
```bash
chmod +x ./setup.sh
```

---

### Step 4: LLM Setup

#### Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3.2:1B  # Lightweight version (1B params)
ollama pull llama3.2:3B  # High-quality version (3B params)
```

---

## 🔧 Configuration

#### If you want to use Huggingface api and don't want to use Ollama 😞
### API Keys Setup

1. Create `.env` file:
```bash
touch .env
```

2. Add Hugging Face credentials:
```env
HUGGINGFACE_API_KEY=your_hf_api_key_here
```

---

### Python Path Configuration

#### Bash/Zsh
```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
```

#### Fish
```bash
set -x PYTHONPATH (pwd)/src $PYTHONPATH
```

```bash
#### Windows (Powershell)
$env:PYTHONPATH = "$(pwd)/src;$env:PYTHONPATH"
```

---

## 🤖 Fine Tuning

### BERT Models Setup For Training
 
#### Create model directory

```bash
mkdir -p models/bert-text-classification-model

```
### Generate Base Dataset for BERT training 

```bash
python src/rag_system/data/data_creation.py
```

### Fine-tune BERT Classifier

```bash
python src/rag_system/training/fine_tune_bert.py \
  --file_path=your_path or default \
  --num_samples_per_class=1500 \
  --num_epochs=5 \
  --learning_rate=2e-5
```

### Dataset Schema
| Column | Type | Description |
|--------|------|-------------|
| query | text | User input |
| requires_context | bool | Context flag |
| reference_text | text | Ground truth |

---

## 🚀 Usage

### Standard Evaluation

```bash
python src/rag_system/evaluation/evaluator.py \
  --collection_name=TestCollection5 \
  --model_type=llama \ # or mistral
```

### Advanced Options

```bash
# Model Configuration
--bert_path="google-bert/bert-base-uncased"  # Path to BERT model
--tokenizer_path="google-bert/bert-base-uncased"  # Custom tokenizer
--embedding_model_path="sentence-transformers/all-mpnet-base-v2"  # Embedding model

# Vector Database Settings
--milvus_uri="./data/milvus_demo.db"  # Local Milvus instance path
--collection_name="rag_eval"  # Collection name for stored embeddings

# LLM Configuration
--llm_repo_id="mistralai/Mistral-7B-Instruct-v0.2"  # Alternative LLM
--llm_max_new_tokens=100  # Maximum response length
--model_type="mistral"  # [llama3|mistral] LLM variant

# Retrieval Parameters
--top_k=30  # Number of context chunks to retrieve
--context_len=1000  # Context window size (in tokens)

# Evaluation Settings
--num_test_samples=5  # Number of test cases to evaluate
--use_classifier=True  # Enable/disable query classification
```


