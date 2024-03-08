from tqdm import tqdm
from loguru import logger
from datasets import load_dataset, Dataset
from .util import get_uid

PRIORITY = 3


def load_data(known_uids=set([]), **_):
    """andrijdavid Roleplay ShareGPT"""
    logger.info("Loading Andrijdavid Roleplay...")
    dataset = load_dataset(
        "Undi95/andrijdavid_roleplay-conversation-sharegpt", data_files=["default/train/0000.parquet"],
        split="train", revision="refs/convert/parquet"
    )
    data = []
    for item in tqdm(dataset):
        uid = get_uid(
            "\n".join(
                [
                    turn["value"]
                    for turn in item["conversations"]
                    if turn["from"] in ("user", "human")
                ]
            )
        )
        if uid in known_uids:
            continue
        known_uids.add(uid)
        data.append(
            {
                "id": uid,
                "conversations": [
                    {
                        key: value
                        for key, value in turn.items()
                        if key in ("from", "value")
                    }
                    for turn in item["conversations"]
                ],
            }
        )
    return Dataset.from_list(data)


if __name__ == "__main__":
    result = load_data()
    result.to_json(f"andrijdavid.jsonl")
    print(result)
