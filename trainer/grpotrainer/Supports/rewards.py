from typing import Dict, List

"""
# REWARD FUNCTIONS

    For Reasoning:
        - Specific thinking format {0,1}: <|THINK|> ... <|/THINK|> ...
        - Thinking token counts [-1, 1]

    For Agent:
        - Ground Truth {0,1}: from GAIA

# FORMATS

    ### Standard Format : prompt

        ["Hello! how are you ?", "What can you do for me?"]

    ### Conversational Format: prompt

        [{
            "role":"user",
            "content": "Hello! how are you?"
        },{
            "role":"user",
            "content": "What can you do for me?"
        }]

    ### Completion_ids

        [
            [6303, 13],
            [304, 279, 12884, 13]
        ]

    ### Rewards: Each completions get a reward !

        [
            0,
            1
        ]

    ### Multi-reward Functions based on category !

        Reward from reward function for Category # 1: [NONE, 1, 0, 1, NONE, NONE, 0]
        Reward from reward function for Category # 2: [0, NONE, NONE, NONE, 1, 1, NONE]

    ### Multi-reward Functions are weighted

        weights for Categories: [Category 1, Category 2]: [0.2, 1]

    ### Async reward functions: Tool calls /

    ### Final Advantage

        Advantage: (Reward(i) - mean(Rewards)) / std(Rewards)
"""


def reward_func(
    prompts: List[str] | List[Dict],
    completions: List[str] | List[Dict],
    completion_ids: List[List],  # Tokenized Completions Ids
) -> List[float]:

    return [0.0]
