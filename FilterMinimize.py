import multiprocessing
import os
from loguru import logger
from datasets import load_dataset, Dataset
from thefuzz import fuzz
from thefuzz.process import extractBests

num_processes = 20
threshold = 90

def chunk_list_by_parts(lst: list[str], parts: int):
    chunksize = len(lst) // parts
    leftovers = len(lst) % parts
    start = 0
    for i in range(parts):
        if i < leftovers:
            end = start + chunksize + 1
        else:
            end = start + chunksize
        yield lst[start:end]
        start = end

def chunk_list_by_size(lst: list[str], chunk_size: int):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size if i + chunk_size <= len(lst) else len(lst) - i]

def process_chunk(chunk: list[str]):
    deduped = set()
    for item in chunk:
        matches = extractBests(item, chunk, scorer=fuzz.token_set_ratio, score_cutoff=threshold, limit=None)
        if matches:
            deduped.add(max(matches, key=lambda x: (len(x[0]), x[0]))[0])
        else:
            logger.info("No matches in chunk...")
            return chunk
        
    logger.info("Processed chunk...")
    return list(deduped) if len(deduped) != len(chunk) else chunk

def process_in_parallel(lst: list[str], chunk_size: int):
    chunks = list(chunk_list_by_size(lst, chunk_size))
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(process_chunk, chunks)
    pool.close()
    pool.join()
    return results

def minimize_dataset(dataset_in: Dataset):
    chunk_size = 256
    processed_results = process_in_parallel(dataset_in["input"], chunk_size)
    flattened_results = [item for sublist in processed_results for item in sublist]
    pass_dataset = dataset_in.filter(lambda x: x["input"] in flattened_results)
    logger.info(f"1st Pass: Reduced dataset size from {len(dataset_in['input'])} to {len(flattened_results)}...")
    
    chunk_size *= 4
    processed_results = process_in_parallel(pass_dataset["input"], chunk_size)
    flattened_results = [item for sublist in processed_results for item in sublist]
    logger.info(f"2nd Pass: Reduced dataset size from {len(dataset_in['input'])} to {len(flattened_results)}...")
    return pass_dataset.filter(lambda x: x["input"] in flattened_results)

def filter_dataset(dataset_in: Dataset, chatformat: str = "alpaca", sourcedelete=None) -> Dataset:
    data = []
    count = 0
    for item in dataset_in:
        if sourcedelete is not None:
            if any(word in item["source"] for word in sourcedelete):
                continue
        if item["source"].endswith("_" + chatformat):
            count += 1
            logger.info(f"Adding data (#{count}) to dataset...")
            data.append(item)
    return Dataset.from_list(data)

if __name__ == "__main__":
    logger.info("Start processing dataset...")

    if not os.path.exists("bagel-filtered-v0.4.parquet"):
        dataset = load_dataset("parquet", data_files="./bagel-input-output-v0.4.parquet", split="train")
        logger.info("Filtering dataset based on prompt format...")
        result = filter_dataset(dataset, "alpaca", ["slimorca", "coding", "mmlu", "ai2_arc"])
        result.to_parquet(f"bagel-filtered-v0.4.parquet")
    else:
        logger.info("bagel-filtered-v0.4.parquet is already present, please delete to regenerate.")
    
    exit()
    
    if not os.path.exists("bagel-filtered-minimized-v0.4.parquet"):
        dataset = load_dataset("parquet", data_files="./bagel-filtered-v0.4.parquet", split="train")
        logger.info("Minimize dataset based on similarity threshold...")
        result = minimize_dataset(dataset)
        result.to_parquet(f"bagel-filtered-minimized-v0.4.parquet")
    else:
        logger.info("bagel-filtered-minimized-v0.4.parquet is already present, please delete to regenerate.")
    
    logger.info("Finished processing dataset...")