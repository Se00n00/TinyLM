````
-----------------------------------------------------------------------------------

 [OUTPUT]
    │
    + ──────────┐
    |   ┌──────────────────────────┐  
    │   |        FEEDFORWARD       │
    │   └──────────────────────────┘
    |──[RMS-NORM]───┘
    + ──────────┐
    │   ┌──────────────────────────┐   
    │   |   MULTI-HEAD ATTENTION   + ──[ALiBi]
    │   └──────────────────────────┘
    └──[RMS-NORM]───┘
    │
 [INPUT]

-----------------------------------------------------------------------------------
````

|-----------------------------------------------------------------------|
|                           EMBEDDING LAYER                             |
|-----------------------------------------------------------------------|
````
EmbeddingLayer(X:[B, L]):
    Require:
        E:[VOCAB_SIZE, D]       # Embedding Matrix
    
    Steps:
        X_tok = E[X]            # [B, L, D]
        Return X_tok

    Inference:
        Memory: VOCAB_SIZE * D
        Computation: ?

    Training:
        Memory: ?
        Computation: ?
````

|-----------------------------------------------------------------------|
|                           RMS NORM LAYER                              |
|-----------------------------------------------------------------------|
````
RMSNormLayer(X:[B, L, D])
    Require:
        g:[D]
    
    Steps:
        rms = sqrt(mean(X*X + e))   # [B, L, 1
        x_ = X / rms
        Return x_ * g

    Inference:
        Memory: D
        Computation: ?
    
    Training:
        Memory: ?
        Computation: ?
   
````

|-----------------------------------------------------------------------|
|                       MULTI-HEAD ATTENTION LAYER                      |
|-----------------------------------------------------------------------|
````
MHALayer(X:[B, L, D])
    Require:
        W_q: [D, D]
        W_k: [D, D]
        W_v: [D, D]
        W_o: [D, D]
    
    Combined_Mask:
        Cij = Mij  + Alibi(h)ij
          Mij:
            0: if i>= j
            -inf: Otherwise
          
          Alibi(h)ij:
            -Slope(h) * (i-j): if i>=j
            0: Otherwise
    Steps:
        Q = W_q @ X                                 # [B, L, D]
        K = W_k @ X                                 # [B, L, D]
        V = W_v @ X                                 # [B, L, D]

        Q, K, V: [B, L, D] --> [B, H, L, D_H]
        scores = Q @ K.T                            # [B, H, L, L]
        scores = scores + Combined_Mask(Alibi + Causal)
        attention = softmax(scores) @ V             # [B, H, L, D_H]
         
        Return attention @ W_o                      # [B, L, D]
    
    Inference:
        Memory: ?
        Computation: ?
    
    Training:
        Memory:
        Computation
````

|-----------------------------------------------------------------------|
|                        FEED FORWARD LAYER                             |
|-----------------------------------------------------------------------|
````
FeedForwardLayer(X:[B, L, D]):
    Require:
        W_up: [D, 4*D]
        W_gate: [D, 4*D]
        W_down: [4*D, D]
    
    Steps:
        temp = SiLU(X @ W_up) * (X @ W_gate)        # [B, L, 4*D]
        y = temp @ W_down                           # [B, L, D]
        Return y

    Inference:
        Memory: 3 * 4 * D
        Computation: ?
    
    Training:
        Memory: ?
        Computation: ?
````

|-----------------------------------------------------------------------|
|                               MODEL                                   |
|-----------------------------------------------------------------------|
````
MODEL(X:[B, L]):
    Require:
        BLOCK: [RMS-NORM -> MHA -> RMS-NORM -> FFD]
        W_Head: [D, VOCAB_SIZE]
        EmbeddingLayer

    Steps:
        X = EmbeddingLayer(X)
        
        for num_layers:
            X = BLOCK(X)
        Return X @ W_HEAD                           # Tied weights - [B, L, VOCAB_SIZE]

    Inference:
        Memory: ? 
        Computation: ? 
    
    Training:
        Memory: ? 
        Computaion: ?
````
