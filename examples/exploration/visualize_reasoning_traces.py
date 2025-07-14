#!/usr/bin/env python3
"""
Script to study reasoning traces of large language models.
This script allows downloading models from Hugging Face, running batch inference,
and storing reasoning tokens, final outputs, and embeddings.
"""

import os
import json
import torch
import argparse
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from datasets import load_dataset
import numpy as np


def download_model(
    model_name: str,
    cache_dir: Optional[str] = None,
    use_auth_token: bool = False,
    device: str = "cuda",
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Download and prepare a model from Hugging Face model hub.
    
    Args:
        model_name: Name of the model on Hugging Face Hub (e.g., "google/flan-t5-base")
        cache_dir: Directory to store downloaded models
        use_auth_token: Whether to use the Hugging Face auth token
        device: Device to load the model on ("cuda", "cpu", etc.)
        torch_dtype: Data type for model weights (e.g., torch.float16 for half precision)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Downloading model: {model_name}")
    
    # Set default dtype if not specified
    if torch_dtype is None and device == "cuda":
        torch_dtype = torch.float16  # Default to fp16 for GPU to save memory
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    
    if device != "cuda" and device != "auto":
        model = model.to(device)
    
    print(f"Model loaded successfully: {model_name}")
    return model, tokenizer


def get_embedding_model(
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cuda",
    cache_dir: Optional[str] = None,
) -> PreTrainedModel:
    """
    Load a model for generating embeddings.
    
    Args:
        embedding_model_name: Name of the embedding model
        device: Device to load the model on
        cache_dir: Directory to store downloaded models
        
    Returns:
        Embedding model
    """
    print(f"Loading embedding model: {embedding_model_name}")
    embedding_model = AutoModel.from_pretrained(
        embedding_model_name, 
        cache_dir=cache_dir,
        device_map="auto" if device == "cuda" else None,
    )
    
    if device != "cuda" and device != "auto":
        embedding_model = embedding_model.to(device)
        
    return embedding_model


def mean_pooling(model_output, attention_mask):
    """
    Mean pooling to get sentence embeddings.
    
    Args:
        model_output: Output from the embedding model
        attention_mask: Attention mask for the tokens
        
    Returns:
        Pooled embeddings
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def generate_embeddings(
    texts: List[str],
    embedding_model: PreTrainedModel,
    embedding_tokenizer: PreTrainedTokenizer,
    device: str = "cuda",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of texts to generate embeddings for
        embedding_model: Model to use for generating embeddings
        embedding_tokenizer: Tokenizer for the embedding model
        device: Device to run inference on
        batch_size: Batch size for generating embeddings
        
    Returns:
        Array of embeddings
    """
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        encoded_input = embedding_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = embedding_model(**encoded_input)
            
        # Mean pooling
        embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def load_queries(
    dataset_path: str,
    query_column: str = "query",
    split: str = "test",
    num_samples: Optional[int] = None,
) -> List[str]:
    """
    Load queries from a dataset.
    
    Args:
        dataset_path: Path or name of the dataset
        query_column: Column name containing the queries
        split: Dataset split to use
        num_samples: Number of samples to load (None for all)
        
    Returns:
        List of query strings
    """
    print(f"Loading dataset: {dataset_path}")
    
    # Check if it's a local file or a HF dataset
    if os.path.exists(dataset_path):
        # Load from local file
        if dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
            with open(dataset_path, 'r') as f:
                data = [json.loads(line) for line in f]
            queries = [item[query_column] for item in data if query_column in item]
        else:
            # Try loading as a CSV or other format
            dataset = load_dataset(dataset_path, split=split)
            queries = dataset[query_column]
    else:
        # Load from Hugging Face datasets
        dataset = load_dataset(dataset_path, split=split)
        queries = dataset[query_column]
    
    # Limit number of samples if specified
    if num_samples is not None and num_samples < len(queries):
        queries = queries[:num_samples]
    
    print(f"Loaded {len(queries)} queries")
    return queries


def extract_reasoning_tokens(
    generated_text: str,
    reasoning_start_marker: str = "Let me think step by step.",
    reasoning_end_marker: str = "Therefore, the answer is",
) -> Tuple[str, str]:
    """
    Extract reasoning tokens and final answer from generated text.
    
    Args:
        generated_text: Text generated by the model
        reasoning_start_marker: Marker indicating the start of reasoning
        reasoning_end_marker: Marker indicating the end of reasoning
        
    Returns:
        Tuple of (reasoning_text, final_answer)
    """
    # Find reasoning section
    reasoning_start = generated_text.find(reasoning_start_marker)
    if reasoning_start == -1:
        # If start marker not found, consider everything as reasoning
        reasoning_text = generated_text
        final_answer = ""
    else:
        reasoning_start += len(reasoning_start_marker)
        
        # Find end of reasoning
        reasoning_end = generated_text.find(reasoning_end_marker, reasoning_start)
        if reasoning_end == -1:
            # If end marker not found, consider everything after start as reasoning
            reasoning_text = generated_text[reasoning_start:].strip()
            final_answer = ""
        else:
            reasoning_text = generated_text[reasoning_start:reasoning_end].strip()
            final_answer = generated_text[reasoning_end:].strip()
    
    return reasoning_text, final_answer


def run_batch_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    queries: List[str],
    prompt_template: str = "{query}\nLet me think step by step.",
    max_new_tokens: int = 512,
    batch_size: int = 1,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """
    Run batch inference on a list of queries.
    
    Args:
        model: Model to use for inference
        tokenizer: Tokenizer for the model
        queries: List of query strings
        prompt_template: Template for formatting queries
        max_new_tokens: Maximum number of tokens to generate
        batch_size: Batch size for inference
        device: Device to run inference on
        
    Returns:
        List of dictionaries containing query, full_output, reasoning, final_answer
    """
    results = []
    
    for i in tqdm(range(0, len(queries), batch_size), desc="Running inference"):
        batch_queries = queries[i:i+batch_size]
        batch_results = []
        
        for query in batch_queries:
            # Format prompt
            prompt = prompt_template.format(query=query)
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Use greedy decoding for deterministic output
                )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract reasoning and final answer
            reasoning_text, final_answer = extract_reasoning_tokens(generated_text)
            
            # Store results
            result = {
                "query": query,
                "full_output": generated_text,
                "reasoning": reasoning_text,
                "final_answer": final_answer,
            }
            
            batch_results.append(result)
        
        results.extend(batch_results)
    
    return results


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Study reasoning traces of large language models")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Name of the model on Hugging Face Hub")
    parser.add_argument("--embedding_model", type=str, 
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Name of the embedding model")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to store downloaded models")
    parser.add_argument("--use_auth_token", action="store_true",
                        help="Use Hugging Face auth token")
    
    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path or name of the dataset")
    parser.add_argument("--query_column", type=str, default="query",
                        help="Column name containing the queries")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to load (None for all)")
    
    # Generation parameters
    parser.add_argument("--prompt_template", type=str, 
                        default="{query}\nLet me think step by step.",
                        help="Template for formatting queries")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save outputs")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Name for output files (default: model name)")
    
    # Device parameters
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (cuda, cpu)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use half precision (fp16)")
    
    args = parser.parse_args()
    
    # Set output name if not specified
    if args.output_name is None:
        args.output_name = args.model_name.split("/")[-1]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set torch dtype
    torch_dtype = torch.float16 if args.fp16 else None
    
    # Download model
    model, tokenizer = download_model(
        args.model_name,
        cache_dir=args.cache_dir,
        use_auth_token=args.use_auth_token,
        device=args.device,
        torch_dtype=torch_dtype,
    )
    
    # Load embedding model
    embedding_model = get_embedding_model(
        args.embedding_model,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    embedding_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, cache_dir=args.cache_dir)
    
    # Load queries
    queries = load_queries(
        args.dataset_path,
        query_column=args.query_column,
        split=args.split,
        num_samples=args.num_samples,
    )
    
    # Run inference
    results = run_batch_inference(
        model,
        tokenizer,
        queries,
        prompt_template=args.prompt_template,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # Generate embeddings for reasoning traces
    reasoning_texts = [result["reasoning"] for result in results]
    embeddings = generate_embeddings(
        reasoning_texts,
        embedding_model,
        embedding_tokenizer,
        device=args.device,
        batch_size=args.batch_size,
    )
    
    # Add embeddings to results
    for i, result in enumerate(results):
        result["embedding"] = embeddings[i].tolist()
    
    # Save results
    output_path = os.path.join(args.output_dir, f"{args.output_name}_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    
    # Save embeddings separately (as numpy array for easier loading)
    embeddings_path = os.path.join(args.output_dir, f"{args.output_name}_embeddings.npy")
    np.save(embeddings_path, embeddings)
    
    print(f"Embeddings saved to {embeddings_path}")


if __name__ == "__main__":
    main()
