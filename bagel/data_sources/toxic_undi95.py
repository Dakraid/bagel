from loguru import logger
from datasets import load_dataset, Dataset
from .util import get_uid

PRIORITY = 2


def load_data(known_uids=set([]), **_):
    """Toxic Undi95 DPO pairs."""
    logger.info("Loading Toxic Undi95 DPO dataset...")
    dataset = load_dataset(
        "Undi95/toxic-dpo-v0.1-NoWarning", data_files=["default/train/0000.parquet"], 
        split="train", revision="refs/convert/parquet"
    )
    data = []
    for item in dataset:
        # We don't care about the known UIDs here.
        if item["chosen"] != item["rejected"]:
            data.append(
                {
                    "id": get_uid(item["question"]),
                    "source": "toxic-dpo-undi95",
                    "prompt": item["question"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                    "conversations": [
                        {
                            "from": "human",
                            "value": item["question"],
                        },
                        {
                            "from": "gpt",
                            "value": item["chosen"],
                        },
                    ],
                }
            )
    return Dataset.from_list(data)


if __name__ == "__main__":
    result = load_data()
    result.to_json(f"toxic_undi95.jsonl")
    print(result)
