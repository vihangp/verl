# Reasoning Trace Analysis for Large Language Models

This directory contains tools for studying reasoning traces of large language models. The scripts in this directory allow you to:

1. Download and use models from Hugging Face
2. Run batch inference on a set of queries
3. Extract and store reasoning tokens and final outputs
4. Generate and store embeddings of the reasoning tokens
5. Analyze and visualize the reasoning traces and embeddings

## Scripts Overview

- **visualize_reasoning_traces.py**: Main script for downloading models, running inference, and generating embeddings
- **analyze_reasoning_traces.py**: Script for analyzing and visualizing the reasoning traces and embeddings
- **run_reasoning_analysis.py**: Convenience script that runs the complete workflow
- **example_queries.json**: Example dataset with math and logic problems

## Requirements

Make sure you have the following packages installed:

```bash
pip install torch transformers datasets tqdm numpy
```

## Quick Start

Here's a simple example to get started with the complete workflow:

```bash
# Run the complete workflow
python examples/exploration/run_reasoning_analysis.py \
  --model_name "google/flan-t5-small" \
  --device "cuda" \
  --fp16 \
  --output_dir "./reasoning_outputs"
```

Or you can run the scripts individually:

```bash
# Step 1: Generate reasoning traces and embeddings
python examples/exploration/visualize_reasoning_traces.py \
  --model_name "google/flan-t5-small" \
  --dataset_path "examples/exploration/example_queries.json" \
  --query_column "query" \
  --output_dir "./reasoning_outputs"

# Step 2: Analyze and visualize the results
python examples/exploration/analyze_reasoning_traces.py \
  --results_path "./reasoning_outputs/flan-t5-small_results.json" \
  --embeddings_path "./reasoning_outputs/flan-t5-small_embeddings.npy" \
  --output_dir "./reasoning_outputs/analysis"
```

## Features

### 1. Model Loading

The script can download and prepare any model from Hugging Face:

```python
model, tokenizer = download_model(
    "google/flan-t5-base",
    device="cuda",
    torch_dtype=torch.float16  # Use half precision to save memory
)
```

### 2. Query Loading

You can load queries from:
- Local JSON/JSONL files
- Hugging Face datasets
- Other formats supported by the `datasets` library

```python
queries = load_queries(
    "examples/exploration/example_queries.json",
    query_column="query",
    num_samples=10  # Limit number of samples (optional)
)
```

### 3. Batch Inference

Run inference on multiple queries efficiently:

```python
results = run_batch_inference(
    model,
    tokenizer,
    queries,
    prompt_template="{query}\nLet me think step by step.",
    max_new_tokens=512,
    batch_size=1  # Increase for faster processing if you have enough memory
)
```

### 4. Reasoning Extraction

The script automatically extracts reasoning steps and final answers:

```python
reasoning_text, final_answer = extract_reasoning_tokens(
    generated_text,
    reasoning_start_marker="Let me think step by step.",
    reasoning_end_marker="Therefore, the answer is"
)
```

### 5. Embedding Generation

Generate embeddings for reasoning traces to enable further analysis:

```python
embeddings = generate_embeddings(
    reasoning_texts,
    embedding_model,
    embedding_tokenizer,
    batch_size=32
)
```

## Command Line Arguments

The script supports various command line arguments:

### Model Parameters
- `--model_name`: Name of the model on Hugging Face Hub (required)
- `--embedding_model`: Name of the embedding model (default: "sentence-transformers/all-MiniLM-L6-v2")
- `--cache_dir`: Directory to store downloaded models
- `--use_auth_token`: Use Hugging Face auth token

### Dataset Parameters
- `--dataset_path`: Path or name of the dataset (required)
- `--query_column`: Column name containing the queries (default: "query")
- `--split`: Dataset split to use (default: "test")
- `--num_samples`: Number of samples to load (default: all)

### Generation Parameters
- `--prompt_template`: Template for formatting queries (default: "{query}\nLet me think step by step.")
- `--max_new_tokens`: Maximum number of tokens to generate (default: 512)
- `--batch_size`: Batch size for inference (default: 1)

### Output Parameters
- `--output_dir`: Directory to save outputs (default: "./outputs")
- `--output_name`: Name for output files (default: model name)

### Device Parameters
- `--device`: Device to run inference on (default: "cuda")
- `--fp16`: Use half precision (fp16)

## Output Format

The script generates two output files:

1. `{model_name}_results.json`: Contains all results including:
   - Original queries
   - Full model outputs
   - Extracted reasoning steps
   - Final answers
   - Embeddings for each reasoning trace

2. `{model_name}_embeddings.npy`: NumPy array of embeddings for easier loading and analysis

## Analysis Features

The `analyze_reasoning_traces.py` script provides several analysis features:

1. **t-SNE Visualization**: Visualize embeddings in 2D space using t-SNE
2. **Clustering**: Group similar reasoning traces using K-means clustering
3. **Common Phrases Analysis**: Extract and analyze common phrases in reasoning traces
4. **Length Analysis**: Analyze the distribution of reasoning trace lengths
5. **Interactive 3D Visualization**: Create an interactive 3D visualization of embeddings

You can run specific analyses using the `--analysis_type` parameter:

```bash
# Run only t-SNE visualization
python examples/exploration/analyze_reasoning_traces.py \
  --results_path "./reasoning_outputs/flan-t5-small_results.json" \
  --embeddings_path "./reasoning_outputs/flan-t5-small_embeddings.npy" \
  --analysis_type "tsne"
```

Available analysis types: `tsne`, `clusters`, `phrases`, `length`, `interactive`, `all`

## Example Analysis Code

If you want to perform custom analysis, you can load the results and embeddings in your own code:

```python
import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load results
with open("./reasoning_outputs/flan-t5-small_results.json", "r") as f:
    results = json.load(f)

# Load embeddings
embeddings = np.load("./reasoning_outputs/flan-t5-small_embeddings.npy")

# Visualize embeddings with t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.title("t-SNE Visualization of Reasoning Traces")
plt.savefig("reasoning_tsne.png")
```

## Customization

You can customize the scripts in various ways:

### For Data Collection (`visualize_reasoning_traces.py`):

- Change the prompt template to elicit different reasoning patterns
- Modify the reasoning extraction markers to match your model's output format
- Use different embedding models for various analysis needs
- Adjust batch size and other parameters for performance optimization

### For Analysis (`analyze_reasoning_traces.py`):

- Change the number of clusters for K-means clustering
- Adjust t-SNE parameters like perplexity
- Modify the n-gram range for phrase extraction
- Create custom visualizations based on the embeddings

### For the Complete Workflow (`run_reasoning_analysis.py`):

- Chain multiple models and datasets in a single run
- Add custom pre-processing or post-processing steps
- Integrate with other analysis tools or frameworks
