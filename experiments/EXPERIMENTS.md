# This Document Serves as Experiments Concluded In this Repository

## Current Best
```
-----------------------------------------------------------------------------------

 [OUTPUT]                                 Architecture: Naive MHA
    │                                      .
    + ──────────┐                           \_ [72M Parmaereters]
    |   ┌──────────────────────────┐       .
    │   |        FEEDFORWARD       │        \_ [528 D_model]
    │   └──────────────────────────┘       .
    |──[RMS-NORM]───┘                       \_ [528 Context Length]
    + ──────────┐
    │   ┌──────────────────────────┐   
    │   |   MULTI-HEAD ATTENTION   │
    │   └──────────────────────────┘
    └──[RMS-NORM]───┘
    │
 [INPUT] + ──[LEARNED ENCODING]

-----------------------------------------------------------------------------------
```