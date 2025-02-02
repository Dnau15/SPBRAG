# SPBRAG Searching for Best Practices in Retrieval Augmented Generation
The main idea of the project is to demonstrate how to use Query Classification model in order to speed up RAG pipeline and increase quality.
## How it works?
- **Input**: User query.
- **Classification Model**: Determines if context is required.
    - If context is required, the system retrieves relevant documents.
    - If not, it proceeds directly to generating a response.
- **Retrieval**: Fetch relevant context from a knowledge base.
- **Generation**: Generate a response using the query and (if applicable) retrieved context.
## 🚀 Getting Started

Follow these step-by-step instructions to set up and run the project on your local machine.

---

### **Prerequisites**

Before you begin, ensure you have the following installed on your system:

- **Python**: Version 3.11 or higher. You can check your Python version by running:
  ```bash
  python --version
  ```
  If not installed, download it from [python.org](https://www.python.org/downloads/).
  The project uses python version 3.11

- **Git**: To clone the repository. Verify installation with:
  ```bash
  git --version
  ```

- **Virtual Environment**: It is recommended to use a virtual environment to manage dependencies. You can use `venv` (built-in) or `conda`, `micromamba`.

---

### **Step 1: Clone the Repository**

Clone this repository to your local machine using the following command:
```bash
git clone https://github.com/Dnau15/SPBRAG.git
```

Navigate into the project directory:
```bash
cd SPBRAG
```

---

### **Step 2: Set Up a Virtual Environment**

Create and activate a virtual environment to isolate the project's dependencies:

- **Using `venv`**:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
  ```

- **Using `conda`** (if applicable):
  ```bash
  conda create -n your-env-name python=3.11
  conda activate your-env-name
  ```

- **Using `micromamba`** (if applicable):
  ```bash
  micromamba create -n your-env-name python=3.11
  micromamba activate your-env-name
  ```
---

### **Step 3: Install Dependencies**

Install the required Python packages by running:
```bash
pip install -r requirements.txt
```

If you encounter any issues during installation, ensure your `pip` is up-to-date:
```bash
pip install --upgrade pip
```

---

### **Step 4: Download Pre-trained Models (If Required)**

Some components of this project rely on pre-trained models. Follow these steps to download them:

1. **BERT Model**:
   - If you want to use trained BERT ensure the path specified in the code (`./models/bert-text-classification-model`) exists.
   - If you want to train BERT, use the following path `bert-base-uncased`
   - Download the model weights and place them in the appropriate directory.

2. **Sentence Transformers**:
   - The `sentence-transformers/all-mpnet-base-v2` model will be downloaded automatically when the script runs for the first time. Ensure you have an active internet connection.

---

### **Step 5: Configure API Keys**

This project requires API keys for external services. Create a `.env` file in the root directory and add the necessary keys:

```env
HUGGINGFACE_API_KEY=your-huggingface-api-key
```

Replace `your-huggingface-api-key` with your actual API key from [Hugging Face](https://huggingface.co/docs/hub/security-tokens).

---

### Step 6: Generate the Dataset

Before running the project, you need to generate the dataset. Use the provided `dataset.py` script to create the dataset:

```bash
python dataset.py
```

This will generate the necessary dataset files in the `./data/` directory. Ensure that the dataset file (`test.csv`) is created successfully.

---

### Optional Step: Fine-Tune BERT (If Needed)

If you want to fine-tune the BERT model for your specific task, you can use the `fine_tune_bert.py` script:

```bash
python fine_tune_bert.py
```

This step is optional but recommended if you need a more tailored model for your dataset. The fine-tuned model will be saved in the `./models/` directory.

---

### **Step 7: Run the Project**

Once everything is set up, you can run the project using the following command:
```bash
python rag.py
```

For more advanced usage, refer to the CLI options provided by the `fire` library:
```bash
python test_rag_system.py --help
```

---

### **Additional Notes**

- **Dataset**: Ensure the dataset file (`test.csv`) is available in the `./data/` directory. If not, download it from the provided link or generate it using the preprocessing scripts.

- **Logging**: The project uses Python's built-in `logging` module. Logs are printed to the console and can be redirected to a file if needed.

- **Troubleshooting**:
  - If you encounter any errors, check the logs for detailed information.
  - Ensure all dependencies are correctly installed and compatible with your Python version.

---

By following these steps, you should have the project up and running smoothly. If you face any issues, feel free to open an issue on the [GitHub Issues page](https://github.com/your-username/your-repo-name/issues). Happy coding! 😊
