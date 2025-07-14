#!/usr/bin/env python3
"""
Example script demonstrating the complete workflow for analyzing reasoning traces.
This script runs both visualize_reasoning_traces.py and analyze_reasoning_traces.py
in sequence to demonstrate the full pipeline.
"""

import os
import argparse
import subprocess
import sys


def run_command(command):
    """Run a command and print its output."""
    print(f"\n=== Running: {' '.join(command)} ===\n")
    process = subprocess.run(command, capture_output=True, text=True)
    print(process.stdout)
    if process.stderr:
        print(f"Errors:\n{process.stderr}")
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        sys.exit(process.returncode)
    return process.returncode == 0


def main():
    """Main function to run the workflow."""
    parser = argparse.ArgumentParser(description="Run the complete reasoning trace analysis workflow")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="google/flan-t5-small",
                        help="Name of the model on Hugging Face Hub")
    parser.add_argument("--embedding_model", type=str, 
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Name of the embedding model")
    
    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, 
                        default="examples/exploration/example_queries.json",
                        help="Path or name of the dataset")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./reasoning_outputs",
                        help="Directory to save outputs")
    
    # Device parameters
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (cuda, cpu)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use half precision (fp16)")
    
    # Analysis parameters
    parser.add_argument("--n_clusters", type=int, default=3,
                        help="Number of clusters for clustering analysis")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set model name for output files
    model_short_name = args.model_name.split("/")[-1]
    
    # Step 1: Run the visualization script to generate reasoning traces and embeddings
    print("\n=== Step 1: Generating reasoning traces and embeddings ===\n")
    
    visualize_cmd = [
        "python", "examples/exploration/visualize_reasoning_traces.py",
        "--model_name", args.model_name,
        "--embedding_model", args.embedding_model,
        "--dataset_path", args.dataset_path,
        "--output_dir", args.output_dir,
        "--device", args.device,
    ]
    
    if args.fp16:
        visualize_cmd.append("--fp16")
    
    success = run_command(visualize_cmd)
    if not success:
        print("Failed to generate reasoning traces. Exiting.")
        return
    
    # Step 2: Run the analysis script to analyze the reasoning traces
    print("\n=== Step 2: Analyzing reasoning traces ===\n")
    
    results_path = os.path.join(args.output_dir, f"{model_short_name}_results.json")
    embeddings_path = os.path.join(args.output_dir, f"{model_short_name}_embeddings.npy")
    analysis_output_dir = os.path.join(args.output_dir, "analysis")
    
    analyze_cmd = [
        "python", "examples/exploration/analyze_reasoning_traces.py",
        "--results_path", results_path,
        "--embeddings_path", embeddings_path,
        "--output_dir", analysis_output_dir,
        "--n_clusters", str(args.n_clusters),
        "--analysis_type", "all",
    ]
    
    success = run_command(analyze_cmd)
    if not success:
        print("Failed to analyze reasoning traces. Exiting.")
        return
    
    print("\n=== Workflow completed successfully! ===\n")
    print(f"Results are saved in: {args.output_dir}")
    print(f"Analysis outputs are saved in: {analysis_output_dir}")
    print("\nYou can view the interactive visualization by opening:")
    print(f"{analysis_output_dir}/interactive_visualization.html")


if __name__ == "__main__":
    main()
