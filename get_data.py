import os
import time
from huggingface_hub import login
from datasets import load_dataset, load_dataset_builder


def get_hf_dataset(data_path):
    login()
    print("Downloading & preparing dataset")
    st = time.time()
    builder = load_dataset_builder("ILSVRC/imagenet-1k")
    # , cache_dir=os.path.join(R"/workspace/extra-data-storage/hf_data/cache"))
    builder.download_and_prepare(os.path.join(data_path))
    print(f"Took {time.time() - st}")
    # return builder.as_dataset(split="train"), builder.as_dataset(split="validation")

if __name__ == "__main__":
    get_hf_dataset()
