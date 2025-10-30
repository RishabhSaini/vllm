#!/usr/bin/env python3
"""
Test script to verify GPU memory cleanup when creating multiple LLM instances
sequentially with VLLM_ENABLE_V1_MULTIPROCESSING=0
"""
import gc
import os
import torch
from vllm import LLM, SamplingParams

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def get_gpu_memory():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return {
            "free_gb": free / 1024**3,
            "total_gb": total / 1024**3,
            "used_gb": used / 1024**3,
        }
    return None


def print_memory(label):
    """Print GPU memory with a label"""
    mem = get_gpu_memory()
    if mem:
        print(f"\n{label}:")
        print(f"  Used: {mem['used_gb']:.2f} GiB")
        print(f"  Free: {mem['free_gb']:.2f} GiB")
        print(f"  Total: {mem['total_gb']:.2f} GiB")


def test_sequential_llms():
    """Test creating and destroying multiple LLMs sequentially"""
    model = "meta-llama/Llama-3.1-8b"
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The largest ocean is",
        "Python is a",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20)

    print("="*80)
    print("Testing Sequential LLM Creation with Memory Cleanup")
    print("="*80)

    # Initial memory state
    print_memory("1. Initial GPU memory")

    # First LLM
    print("\n--- Creating First LLM ---")
    llm1 = LLM(
        model=model,
        gpu_memory_utilization=0.6,
        disable_log_stats=True,
    )
    print_memory("2. After creating first LLM")

    # Use first LLM
    print("\nRunning inference with first LLM...")
    outputs = llm1.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated: {generated_text!r}")

    # Explicitly shutdown and delete first LLM
    print("\n--- Shutting down First LLM ---")
    print("Calling llm1.shutdown()...")
    llm1.shutdown()
    print("Deleting llm1...")
    del llm1

    # Force garbage collection
    print("Forcing garbage collection...")
    gc.collect()
    torch.cuda.empty_cache()

    print_memory("3. After shutting down and deleting first LLM")

    # Wait a moment for cleanup to complete
    import time
    time.sleep(2)

    print_memory("4. After waiting 2 seconds")

    # Second LLM with same settings
    print("\n--- Creating Second LLM ---")
    try:
        llm2 = LLM(
            model=model,
            gpu_memory_utilization=0.6,  # Same as first LLM
            disable_log_stats=True,
        )
        print_memory("5. After creating second LLM")

        # Use second LLM
        print("\nRunning inference with second LLM...")
        outputs = llm2.generate(prompts, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated: {generated_text!r}")

        # Cleanup second LLM
        print("\n--- Shutting down Second LLM ---")
        llm2.shutdown()
        del llm2
        gc.collect()
        torch.cuda.empty_cache()

        print_memory("6. After shutting down and deleting second LLM")

        print("\n" + "="*80)
        print("SUCCESS! Both LLMs were created and cleaned up successfully!")
        print("="*80)

    except ValueError as e:
        print("\n" + "="*80)
        print("FAILURE! Second LLM creation failed with error:")
        print(f"  {e}")
        print("="*80)
        print("\nThis indicates that GPU memory was not properly freed.")
        raise


if __name__ == "__main__":
    test_sequential_llms()
