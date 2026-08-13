## Chat Template

```jinja
  {%- if messages.text is defined %}
    <|START|>{{ messages.text }}<|END|>
    
  {%- elif messages is string %}    
    <|START| {{ messages }}<|END|>
    
  {%- else %}
    <|START|>
    
    {%- if tools %}
      <|AVAILABLE_TOOLS|>
      {{ tools | tojson(indent=2) }}
      <|/AVAILABLE_TOOLS|>
    {%- endif %}
    
    {%- for message in messages %}
    
      {%- if message.role == "system" %}
        <|SYSTEM|>
        {{ message.content }}
      {%- elif message.role == "user" %}
        <|USER|>
        {{ message.content }}
      {%- elif message.role == "assistant" %}
        <|ASSISTANT|>
        {{ message.content }}
      {%- elif message.role == "tool" %}
        <|TOOL_RESULTS|>
        {{ message.content }}
        <|/TOOL_RESULTS|>
      {%- endif %}
      
    {%- endfor %}
    {%- if add_generation_prompt %}
      <|ASSISTANT|>
    {%- else %}
      <|END|>
    {%- endif %}
  {%- endif %}
```

## EXAMPLES
```python
  PRE_TOKENIZATION: "Hii, How are you Doing"
  POST_TOKENIZATION: '<|START|>\nHii, How are you Doing\n<|END|>'
```
  ---

```python  
  PRE_TOKENIZATION: {"text": "Hii, How are you Doing"}
  POST_TOKENIZATION: '<|START|>\nHii, How are you Doing\n<|END|>'
```
### 
```python
  PRE_TOKENIZATION:
  [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"}
  ]
  POST_TOKENIZATION: '<|START|><|SYSTEM|>\nYou are helpful.<|USER|>\nHello<|ASSISTANT|>\nHi!<|END|>'
```
### `THINKING`

```python
  PRE_TOKENIZATION:
  [
    {
      "role": "user",
      "content": "What is 17 × 24?"
    },
    {
      "role": "assistant",
      "content": "<|THINK|>\n17 × 24 = 17 × (20 + 4) = 340 + 68 = 408.\n<|/THINK|>\n\n17 × 24 = 408."
    }
  ]
  POST_TOKENIZATION: '<|START|><|USER|>\nWhat is 17 × 24?<|ASSISTANT|>\n<|THINK|>\n17 × 24 = 17 × (20 + 4) = 340 + 68 = 408.\n<|/THINK|>\n\n17 × 24 = 408.<|END|>' 
```

### `TOOL CALLING`

```python
tools = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate."
                }
            },
            "required": ["expression"]
        }
    },
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
  PRE_TOKENIZATION:
      [
        {
          "role": "system",
          "content": "You are an assistant that can use tools."
        },
        {
          "role": "user",
          "content": "What is 125 × 37?"
        },
        {
          "role": "assistant",
          "content": "<|TOOL_CALLS|>\n[\n  {\n    \"name\": \"calculator\",\n    \"arguments\": {\n      \"expression\": \"125 * 37\"\n    }\n  }\n]\n<|/TOOL_CALLS|>"
        },
        {
          "role": "tool",
          "content": "4625"
        },
        {
          "role": "assistant",
          "content": "125 × 37 = 4625."
        }
      ]
  POST_TOKENIZATION: '<|START|><|SYSTEM|>\nYou are an assistant that can use tools.<|USER|>\nWhat is 125 × 37?<|ASSISTANT|>\n<|TOOL_CALLS|>\n[\n  {\n    "name": "calculator",\n    "arguments": {\n      "expression": "125 * 37"\n    }\n  }\n]\n<|/TOOL_CALLS|><|TOOL_RESULTS|>\n4625\n<|/TOOL_RESULTS|><|ASSISTANT|>\n125 × 37 = 4625.<|END|>'
```

### `THINKING + TOOL CALLING`

```python
  PRE_TOKENIZATION:
      [
        {
          "role": "user",
          "content": "What is 135 divided by 2.5?"
        },
        {
          "role": "assistant",
          "content": "<|THINK|>\nI need to calculate 135 / 2.5.\n<|/THINK|>\n<|TOOL_CALLS|>\n[\n  {\n    \"name\": \"calculator\",\n    \"arguments\": {\n      \"expression\": \"135 / 2.5\"\n    }\n  }\n]\n<|/TOOL_CALLS|>"
        },
        {
          "role": "tool",
          "content": "54"
        },
        {
          "role": "assistant",
          "content": "The answer is 54."
        }
      ]
      
  POST_TOKENIZATION: '<|START|><|USER|>\nWhat is 135 divided by 2.5?<|ASSISTANT|>\n<|THINK|>\nI need to calculate 135 / 2.5.\n<|/THINK|>\n<|TOOL_CALLS|>\n[\n  {\n    "name": "calculator",\n    "arguments": {\n      "expression": "135 / 2.5"\n    }\n  }\n]\n<|/TOOL_CALLS|><|TOOL_RESULTS|>\n54\n<|/TOOL_RESULTS|><|ASSISTANT|>\nThe answer is 54.<|END|>'
```

## Special Tokens

```python
special_tokens={'bos_token': '<|START|>', 'eos_token': '<|END|>', 'unk_token': '<|UNK|>'}
added_tokens_decoder={
	50256: "<|endoftext|>",
	50257: "<|START|>",
	50258: "<|END|>",
	50259: "<|UNK|>",
	50260: "<|THINK|>",
	50261: "<|AVAILABLE_TOOLS|>",
	50262: "<|/AVAILABLE_TOOLS|>",
	50263: "<|TOOL_CALLS|>",
	50264: "<|/TOOL_CALLS|>",
	50265: "<|TOOL_RESULTS|>",
	50266: "<|/TOOL_RESULTS|>",
	50267: "<|ASSISTANT|>",
	50268: "<|USER|>",
	50269: "<|SYSTEM|>",
	50270: "<|/THINK|>",
}
```