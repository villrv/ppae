# ppae

## Training feedbacks on the dataset of 1000 eventfiles
### current problems
- vanilla LSTM is too slow
- sensitive to batch size and t_scale

- Some data summary:
    - properties_full.csv: 
        - total length: 130232
        - with unique (region_id, obsid): 129527
    - eventfiles_full.csv
        - with unique (region_id, obsid): 189389
        - with those also appearing in properties_full.csv: 95821
        
        
        - Harness ratio
        - Variability
        - 

### What did not work
- Small dataset
    - Include time difference in encoding
    - More layers of LSTM
    - Concatenating last output with global pooled output for LSTM
    - Increase to 48 token size for vanilla transformer
    - Mixed precision training REALLY HURTS!
    - Don't do custom initialization for LSTM! Dense layers should start at 0
- Large dataset
    - lr increased to 5e-3
    - TV loss on mesh rates
        - Stuck on the following nonsense: extremely high rate at a single event (which is not on the mesh)
    - TV loss on log event rates
        - Some event might have infinity rate

### What did work
- Small dataset
    - Longer training time: 800 epochs with a patience of 100
    - Bigger LSTM network
    - LSTM used last non-padded output
    - TV loss should be on mesh rates and lam_TV should be 0.2.
        - Questionable!
- Large dataset
    - random shift at 0
    - Potentially eliminating nan/inf values
        - Skip nan loss gradient update (done)
        - torch anomaly detection
        - Gradient clipping
        - Initialization
        - Batch normalization
    - Balance dataset
    - TV loss on usual (exp) rates

### further tuning ideas:
- Look for faster parallalizable networks like linear transformer and CNN+RNN?
- Learnable positional encoding

### Other small TODOs
- Look for some good visualizations
- Switch to transformers