from datasets import load_dataset, load_from_disk
import random
import sglang as sgl
from sglang import function, system, user, assistant, gen, set_default_backend
import argparse
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
import json
from pathlib import Path
import requests
import glob
import os
from tqdm import tqdm


# Define the processing function outside the class
@function
def process_sample(s, question, solution):
    """Process a single math problem using sglang"""
    # Set up system prompt
    s += sgl.system(
        "You are a helpful AI assistant specialized in solving mathematical problems. "
        "Provide clear, step-by-step solutions with mathematical reasoning."
    )

    # Format user prompt
    s += sgl.user(
        f"Please solve this mathematical problem step by step:\n\n"
        f"Problem: {question}\n\n Expected Answer: {solution}\n\n"
        f"Show your complete solution with clear mathematical reasoning and calculations."
    )

    # Generate response
    s += sgl.assistant(sgl.gen("response", max_tokens=256))

    return s["response"]


def process_sample_wrapper(sample):
    """Process a single sample"""
    question = sample.get("input") or sample.get("problem") or sample.get("instruction")
    solution = (
        sample.get("output") or sample.get("expected_answer") or sample.get("response")
    )

    if not question or not solution:
        print(f"Missing required fields in sample: {sample}")
        return None

    model_response = process_sample.run(question=question, solution=solution)

    return {"question": question, "response": model_response, "solution": solution}


class BaseDatasetProcessor:
    def __init__(self, args):
        # Initialize sglang backend
        set_default_backend(select_sglang_backend(args))
        self.args = args

    def batch_process(self, samples, dataset_name, output_file=None):
        """Process multiple problems in batch"""
        results = []
        with tqdm(
            total=len(samples), desc=f"Processing {dataset_name} problems"
        ) as pbar:
            for sample in samples:
                result = process_sample_wrapper(sample)
                if result:
                    processed_result = {
                        "dataset": dataset_name,
                        "question": result["question"],
                        "model_response": result["response"],
                        "ground_truth": result["solution"],
                    }
                    results.append(processed_result)
                    if output_file:
                        with open(output_file, "a") as f:
                            json.dump(processed_result, f)
                            f.write("\n")
                pbar.update(1)
        return results


class AIMEDatasetProcessor(BaseDatasetProcessor):
    def __init__(self, args):
        super().__init__(args)
        self.dataset = load_from_disk(
            "/cpfs01/shared/llm_razor/dongpeijie/workspace/dps-moe-expert-pruner/EASYEP/dataset/aime23_full"
        )
        self.dataset_samples = list(self.dataset)

    def batch_process(self, num_samples=5, output_file=None):
        """Process multiple AIME problems in batch"""
        if not self.dataset_samples:
            print("Error: No dataset samples available")
            return []

        selected_samples = random.sample(
            self.dataset_samples, min(num_samples, len(self.dataset_samples))
        )
        return super().batch_process(selected_samples, "AIME", output_file)


class OpenMathDatasetProcessor(BaseDatasetProcessor):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_path = (
            "/cpfs01/shared/llm_razor/dongpeijie/hf_data/OpenMathReasoning/data"
        )
        self.parquet_files = glob.glob(os.path.join(self.dataset_path, "*.parquet"))
        if not self.parquet_files:
            raise ValueError(f"No parquet files found in {self.dataset_path}")

        self.dataset_samples = []
        for parquet_file in self.parquet_files[:10]:
            dataset = load_dataset("parquet", data_files=parquet_file)
            self.dataset_samples.extend(dataset["train"])

    def batch_process(self, num_samples=5, output_file=None):
        """Process multiple OpenMath problems in batch"""
        if not self.dataset_samples:
            print("Error: No dataset samples available")
            return []

        selected_samples = random.sample(
            self.dataset_samples, min(num_samples, len(self.dataset_samples))
        )
        return super().batch_process(selected_samples, "OpenMath", output_file)


class OpenCodeDatasetProcessor(BaseDatasetProcessor):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_path = (
            "/cpfs01/shared/llm_razor/dongpeijie/hf_data/OpenCodeReasoning/split_1"
        )
        self.parquet_files = glob.glob(os.path.join(self.dataset_path, "*.parquet"))
        if not self.parquet_files:
            raise ValueError(f"No parquet files found in {self.dataset_path}")

        self.dataset_samples = []
        for parquet_file in self.parquet_files[:10]:
            dataset = load_dataset("parquet", data_files=parquet_file)
            self.dataset_samples.extend(dataset["train"])

    def batch_process(self, num_samples=5, output_file=None):
        """Process multiple OpenCode problems in batch"""
        if not self.dataset_samples:
            print("Error: No dataset samples available")
            return []

        selected_samples = random.sample(
            self.dataset_samples, min(num_samples, len(self.dataset_samples))
        )
        return super().batch_process(selected_samples, "OpenCode", output_file)


class AlpacaDatasetProcessor(BaseDatasetProcessor):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_path = (
            "/cpfs01/shared/llm_razor/dongpeijie/hf_data/tatsu-lab/alpaca/data"
        )
        self.parquet_files = glob.glob(os.path.join(self.dataset_path, "*.parquet"))
        if not self.parquet_files:
            raise ValueError(f"No parquet files found in {self.dataset_path}")

        self.dataset_samples = []
        for parquet_file in self.parquet_files[:10]:
            dataset = load_dataset("parquet", data_files=parquet_file)
            self.dataset_samples.extend(dataset["train"])

    def batch_process(self, num_samples=5, output_file=None):
        """Process multiple Alpaca problems in batch"""
        if not self.dataset_samples:
            print("Error: No dataset samples available")
            return []

        selected_samples = random.sample(
            self.dataset_samples, min(num_samples, len(self.dataset_samples))
        )
        return super().batch_process(selected_samples, "Alpaca", output_file)


def main():
    parser = argparse.ArgumentParser(description="Process math problems using an LLM")
    parser.add_argument(
        "--num-shots", type=int, default=5, help="Number of few-shot examples"
    )
    parser.add_argument("--output-file", type=str, help="Path to save results")
    parser.add_argument(
        "--num-problems", type=int, default=100, help="Number of problems to process"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["aime", "openmath", "opencode", "alpaca", "all"],
        default="all",
        help="Dataset to process (aime, openmath, opencode, alpaca, or all)",
    )
    args = add_common_sglang_args_and_parse(parser)

    # Start recording
    response = requests.post(
        f"{args.host}:{args.port}/start_expert_distribution_record"
    )
    print(response.text)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

    all_results = []

    if args.dataset in ["aime", "all"]:
        print("\nProcessing AIME problems...")
        aime_processor = AIMEDatasetProcessor(args)
        aime_results = aime_processor.batch_process(
            num_samples=args.num_problems, output_file=args.output_file
        )
        all_results.extend(aime_results)

    if args.dataset in ["openmath", "all"]:
        print("\nProcessing OpenMath problems...")
        openmath_processor = OpenMathDatasetProcessor(args)
        openmath_results = openmath_processor.batch_process(
            num_samples=args.num_problems, output_file=args.output_file
        )
        all_results.extend(openmath_results)

    if args.dataset in ["opencode", "all"]:
        print("\nProcessing OpenCode problems...")
        opencode_processor = OpenCodeDatasetProcessor(args)
        opencode_results = opencode_processor.batch_process(
            num_samples=args.num_problems, output_file=args.output_file
        )
        all_results.extend(opencode_results)

    if args.dataset in ["alpaca", "all"]:
        print("\nProcessing Alpaca problems...")
        alpaca_processor = AlpacaDatasetProcessor(args)
        alpaca_results = alpaca_processor.batch_process(
            num_samples=args.num_problems, output_file=args.output_file
        )
        all_results.extend(alpaca_results)

    for i, result in enumerate(all_results[:10]):
        print(f"\nProblem {i+1} ({result['dataset']}):")
        print("-" * 80)
        print("Question:", result["question"])
        print("\nModel's Solution:", result["model_response"])
        print("\nGround Truth Solution:", result["ground_truth"])
        print("-" * 80)

    response = requests.post(f"{args.host}:{args.port}/stop_expert_distribution_record")
    print(response.text)

    response = requests.post(f"{args.host}:{args.port}/dump_expert_distribution_record")
    print(response.text)


if __name__ == "__main__":
    main()
