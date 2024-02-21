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
- Include time difference in encoding
- More layers of LSTM
- Concatenating last output with global pooled output for LSTM
- Increase to 48 token size for vanilla transformer
- Mixed precision training REALLY HURTS!
- Don't do custom initialization for LSTM! Dense layers should start at 0

### What did work
- Longer training time: 800 epochs with a patience of 100
- Bigger LSTM network
- LSTM used last non-padded output
- TV loss for mesh, also added first occurrence
- TV loss should be on mesh rates and lam_TV should be 0.2.

### further tuning ideas:
- Look for faster parallalizable networks like linear transformer and CNN+RNN?
- Learnable positional encoding

### Other small TODOs
- Look for some good visualizations

- Potentially eliminating nan/inf values
    - torch anomaly detection (done)
    - Gradient clipping (done)
    - Initialization (done)
    - Batch normalization
    
    
- Switch to transformers