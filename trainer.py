import asyncio
import queue
import threading
import time

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer





# ============================================================
# GLOBALS
# ============================================================
BUFFER_SIZE = 10_000  # Number of batches kept ready

# How many examples are tokenized together
TOKENIZE_BATCH_SIZE = 64

# Queue waits briefly before retrying when full/empty
QUEUE_TIMEOUT = 1.0
BUFFER = None
STOP_EVENT = threading.Event()


def token_generator(tokenizer, PRETRAINING_DATASET):
    """
    Streams examples from all datasets and yields token IDs.

    Datasets are consumed sequentially.
    """
    
    for dataset_config in PRETRAINING_DATASET:
        if STOP_EVENT.is_set():
            return

        kwargs = {
            "path": dataset_config["base"],
            "split": dataset_config["split"],
            "streaming": True,
        }

        if dataset_config.get("subset") is not None:
            kwargs["name"] = dataset_config["subset"]
            
        if dataset_config.get("data_dir") is not None:
            kwargs["data_dir"] = dataset_config["data_dir"]

        ds = load_dataset(**kwargs)
        total_tokens = 0

        for example in ds:
            if total_tokens >= dataset_config['tokens'] and not dataset_config['train_total']:
                break
            
            if STOP_EVENT.is_set():
                return
            
            if isinstance(dataset_config['column'], list):
                text = ''
                for col in dataset_config['column']:
                    text = text + " " + example.get(col)
            else:
                text = example.get(dataset_config['column'])
            

            if text is None:
                continue

            # Tokenize one document
            tokens = tokenizer.encode(
                text,
                tokenize=True,
                add_generation_prompt=False
            )
            
            for token in tokens:
                total_tokens += 1
                yield token
        
        print(f"TRAINED: {dataset_config['base']} | TOTAL TOKENS: {total_tokens}")
        # TODO: CREATE MODEL CHECKPOINT

# ============================================================
# BATCH GENERATOR
# ============================================================


def batch_generator(tokenizer, BATCH_SIZE, SEQ_LEN, PRETRAINING_DATASET):
    """
    Converts the streaming token sequence into fixed
    [BATCH_SIZE, SEQ_LEN] batches.
    """

    token_iter = token_generator(tokenizer, PRETRAINING_DATASET)

    tokens_per_batch = BATCH_SIZE * SEQ_LEN

    batch = []

    while not STOP_EVENT.is_set():
        try:
            for _ in range(tokens_per_batch):
                if STOP_EVENT.is_set():
                    return

                batch.append(next(token_iter))

        except StopIteration:
            return

        if len(batch) != tokens_per_batch:
            return

        # uint16 is sufficient for GPT-2's 50,257 vocabulary.
        tensor = torch.tensor(
            batch,
            dtype=torch.uint16,
        ).view(BATCH_SIZE, SEQ_LEN)

        batch.clear()

        yield tensor


# ============================================================
# BACKGROUND PRODUCER
# ============================================================


def producer(tokenizer, buffer, BATCH_SIZE, SEQ_LEN, PRETRAINING_DATASET):
    """
    Runs in a background thread.

    Continuously downloads/tokenizes/prepares batches and
    places them into BUFFER.
    """

    print("\n[PRODUCER] Started")

    batches_produced = 0
    start_time = time.time()

    try:
        for batch in batch_generator(
            tokenizer, BATCH_SIZE, SEQ_LEN, PRETRAINING_DATASET
        ):
            if STOP_EVENT.is_set():
                break

            # Wait until there is space in the buffer
            while not STOP_EVENT.is_set():
                try:
                    buffer.put(
                        batch,
                        timeout=QUEUE_TIMEOUT,
                    )
                    break

                except queue.Full:
                    continue

            batches_produced += 1

            # Print producer throughput occasionally
            if batches_produced % 1000 == 0:
                elapsed = time.time() - start_time

                if elapsed > 0:
                    throughput = batches_produced / elapsed

                    print(
                        f"\n[PRODUCER] "
                        f"{batches_produced:,} batches | "
                        f"{throughput:.2f} batches/s | "
                        f"buffer={buffer.qsize():,}"
                    )

    except Exception as e:
        import traceback
        print(f"\n[PRODUCER ERROR] {type(e).__name__}: {e} {traceback.format_exc()}")

    finally:
        # Sentinel tells consumer that producer finished
        while not STOP_EVENT.is_set():
            try:
                buffer.put(
                    None,
                    timeout=QUEUE_TIMEOUT,
                )
                break

            except queue.Full:
                continue

        print("\n[PRODUCER] Stopped")


# ============================================================
# TRAINER
# ============================================================


async def trainer(batch_size, seq_len, dataset):

    global BUFFER
    BUFFER = queue.Queue(maxsize=BUFFER_SIZE)

    tokenizer = AutoTokenizer.from_pretrained("Se00n00/TinyLM-2")

    # --------------------------------------------------------
    # Start producer
    # --------------------------------------------------------

    producer_thread = threading.Thread(
        target=producer,
        args=(tokenizer, BUFFER, batch_size, seq_len, dataset),
        daemon=True,
    )

    producer_thread.start()

    # --------------------------------------------------------
    # Training / consumer loop
    # --------------------------------------------------------

    consumed = 0
    start_time = time.time()

    progress = tqdm(
        unit="batch",
        desc="Training",
        dynamic_ncols=True,
    )

    try:
        while True:
            # queue.get() is blocking, therefore execute it
            # outside the asyncio event loop.
            batch = await asyncio.to_thread(BUFFER.get)

            # Producer finished
            if batch is None:
                break

            consumed += 1

            # ------------------------------------------------
            # THIS IS WHERE YOUR MODEL TRAINING GOES
            # ------------------------------------------------
            # print(batch.shape)
            # Example:
            #
            # batch = batch.to(device)
            # output = model(batch)
            # loss = ...
            # loss.backward()
            # optimizer.step()

            # Simulate GPU work
            await asyncio.sleep(0.01)

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            progress.update(1)

            elapsed = time.time() - start_time

            if elapsed > 0:
                batches_per_second = consumed / elapsed

                progress.set_postfix(
                    {
                        "buffer": BUFFER.qsize(),
                        "batch/s": f"{batches_per_second:.2f}",
                    }
                )

    except asyncio.CancelledError:
        print("\n[TRAINER] Cancellation requested")

        STOP_EVENT.set()

        raise

    finally:
        progress.close()

        STOP_EVENT.set()

        producer_thread.join(timeout=5)

        print(
            f"\nConsumed: {consumed:,} batches | token count: {consumed * batch_size * seq_len}"
        )

        print("[TRAINER] Finished")


async def main():
    
    PRETRAINING_DATASET = [
        {
            "base": "HuggingFaceFW/fineweb",
            "subset": "sample-10BT",
            "split": "train",
            "column": "text",
            "train_total": True,
            "tokens": 10000000000
        },
        {
            "base": "HuggingFaceFW/fineweb-edu",
            "subset": "sample-10BT",
            "split": "train",
            "column": "text",
            "train_total": True,
            "tokens": 10000000000
        },
        {
            "base": "bigcode/the-stack",
            "data_dir": "data/json",
            "split": "train",
            "column": "content",
            "train_total": False,
            "tokens": 2500000000
        },
        {
            "base": "bigcode/the-stack",
            "data_dir": "data/shell",
            "split": "train",
            "column": "content",
            "train_total": False,
            "tokens": 2500000000
        },
        {
            "base": "HuggingFaceTB/smollm-corpus",
            "subset": "cosmopedia-v2",
            "split": "train",
            "column": ["prompt", "text"],
            "train_total": True,
            "tokens": 10000000000
        },
        {
            "base": "bigcode/the-stack",
            "data_dir": "data/api-blueprint",
            "split": "train",
            "column": "content",
            "train_total": False,
            "tokens": 2500000000
        },
        {
            "base": "bigcode/the-stack",
            "data_dir": "data/python",
            "split": "train",
            "column": "content",
            "train_total": False,
            "tokens": 2500000000
        },
        {
            "base": "bigcode/the-stack",
            "data_dir": "data/markdown",
            "split": "train",
            "column": "content",
            "train_total": False,
            "tokens": 2500000000
        },
        {
            "base": "emozilla/pg19",
            "subset": None,
            "split": "train",
            "column": ["short_book_title", "text"],
            "train_total": True,
            "tokens": 10000000000
        }
    ]
    
    
    
    BATCH_SIZE = 100
    SEQ_LEN = 30000
    try:
        await trainer(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, dataset=PRETRAINING_DATASET)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

        STOP_EVENT.set()


if __name__ == "__main__":
    asyncio.run(main())
