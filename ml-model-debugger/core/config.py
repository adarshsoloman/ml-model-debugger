class Config:
    # Diagnostic Heuristics
    DEFAULT_PERFORMANCE_THRESHOLD = 0.5  # Min R2 or Accuracy
    DEFAULT_OVERFITTING_RATIO = 1.15     # Val_Error / Train_Error
    
    # Loop Settings
    MAX_ITERATIONS = 5
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
