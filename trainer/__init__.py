"""
TinyLM's Trainer module

    - Contains classes for :
        SFTTrainer - Used for SFT, Pre-training, Reasoninig FineTunning, Function Calling
        [] PrefrenceTrainer - Used for prefernce training using - DPO, DRPO, RLVR

    - Features:
        - Supports Distributed Training
        - Auto OOM recovery
        - learning-rate scheduler
        -

    - Supported Dataset Template
        - Text Genration
            {
                "messages": {"text": ".."}
            }

        - Instruction Fine Tunning
            {
                "messages":[
                    {
                        "role":"system",
                        "content":"..."
                    },
                    {
                        "role":"user",
                        "content":"..."
                    },
                    {
                        "role":"assistant",
                        "content": "..."
                    }
                ]
            }

        - Reasoning Fine Tunning
            {
                "messages":[
                    {
                        "role":"system",
                        "content":"..."
                    },
                    {
                        "role":"user",
                        "content":"..."
                    },
                    {
                        "role":"assistant",
                        "content": "<|THINK|> ... <|THINK|> ..."
                    }
                ]
            }

            - Tool Calling
                {
                    "messages":[{
                        "available_tools": [
                            {
                                "name": "get_weather",
                                "description": "Get the current weather for a city.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "city": {
                                            "type": "string",
                                            "description": "Name of the city."
                                        }
                                    },
                                    "required": ["city"]
                                }
                            }
                        ],
                        "system": "...",
                        "user":"...",
                        "assisstant": ".. <|TOOL_CALLS|>[
                            {
                                "name": "...",
                                "arguments":{
                                    "expression":"..."
                                }
                            }
                        ]<|/TOOL_CALLS|> ..."
                    }]
                }

            - Tool Calling + Reasoning
                {
                    "messages":[
                        {
                            "role":"available_tools",
                            "content":[
                                {
                                    "name": "get_weather",
                                    "description": "Get the current weather for a city.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "city": {
                                                "type": "string",
                                                "description": "Name of the city."
                                            }
                                        },
                                        "required": ["city"]
                                    }
                                }
                            ]
                        },
                        {
                            "role": "system",
                            "content":"...",
                        },
                        {
                            "role": "user",
                            "content": "..."
                        },
                        {
                            "role":"assistant",
                            "content": "<|THINK|> ... <|/THINK|> ... <|TOOL_CALLS|>[
                                {
                                    "name": "...",
                                    "arguments":{
                                        "expression":"..."
                                    }
                                }
                            ]<|/TOOL_CALLS|> ... "
                        }
                    ]
                }
"""

from .sfttrainer.sftconfig import SFTConfig
from .sfttrainer.sfttrainer import SFTTrainer
from .util import (
    preprocess_rft,
    preprocess_text_generation,
    process_cot,
    process_ift_dataset,
    process_iftc,
    process_tc,
    process_warmup,
    filter_valid,
    process_cot2
)

__all__ = [
    "SFTConfig",
    "SFTTrainer",
    "process_ift_dataset",
    "preprocess_rft",
    "process_tc",
    "process_warmup",
    "preprocess_text_generation",
    "process_iftc",
    "process_cot",
    "filter_valid",
    "process_cot2"
]
