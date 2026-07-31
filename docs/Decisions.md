# Tokenizer [ALL TOKENIZER MODEL]

## I used GPT-2-Tokenizer and extended for reasoning & tool calling.
 - It's already trained over meaningful chunks of text - easy to extrapolate for use
 - other pre-trained tokenizer were of huge size >150M (if considering embedding dim = 512).
 
[STILL A MATHEMATICAL REASON NEEDED]


# Dataset

## Main Goal for training the model
 [0] - Better at following instructions [POST TRAINING-IFT]
 [1] - speaking in clear formatted markdown manner for deepsearch use-case [POST TRAINING-AGENT RL]
 [2] - Better at Tool-calling [PRE-TRAINING + POST-TRAINING-IFT]
 [3] - Better at reasoning [POST-TRAINING: RL]

hence for pretraining, i need dataset that contains base web data (fineweb-10B sample), (Stack-v3: python, json subset) for tool calling pre-training.
    Now coming to the post - training. 




# Evaluation [HOW DOES lm-bench EVALUATE MODEL ON DIFFERENT STAGE?]
Along with existing base benchmarks, there should be also benchmark for tool-use and reasoning.

Tool-use : BFCL, Tau-Bench
