# ppae

## Training feedbacks on the dataset of 1000 eventfiles
### current problems

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
        
## TODO
- Calculate the hardness ratio beforehand so we don't need to calculate everytime
        
### Good results
- Small dataset
    - TV totalrate, plottotalrate, lamTV between 0.001 (obvious overfitting) to 0.01 (some underfitting)
- Large dataset, lightly filtered
    - lamTV 0.001: 80 epochs: some local overfitting, global not fitted enough
    - lamTV 0.003: 20 epochs: pretty flat
- Large dataset, heavily filtered
    - lamTV 0.0003: ovbious overfitting
    - lamTV 0.001: some global fitting happening

### What did not work
- Small dataset
    - Include time difference in encoding
    - More layers of LSTM
    - Concatenating last output with global pooled output for LSTM
    - Increase to 48 token size for vanilla transformer
    - Mixed precision training REALLY HURTS!
    - Don't do custom initialization for LSTM! Dense layers should start at 0
    - Should multiply by delta t in order to make things scale-less
        - Not really needed, just tune lambda

- Large dataset
    - lr increased to 5e-3
    - accumulate gradient doesn't help a huge
    - TV loss on log rates
        - Some event might have infinity rate
    - TV loss on event rates only
        - Very wiggling
    

### What did work
- Small dataset
    - Longer training time: 800 epochs with a patience of 100
    - Bigger LSTM network
    - LSTM used last non-padded output
    - TV loss should be on mesh rates and lam_TV should be 0.2.
        - Questionable!
    - Should include the t input for positional encoding
    - TV on total grid.
        - A big more smooth than just use meshrate
    - Plot also on total grid
    - t_scale is just T=43200
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

    
    
- Gradient accumulation
- Stochastic Weight Averaging
- Automatic learning rate finder
- Automatic batch size finder
- Look for faster parallalizable networks like linear transformer and CNN+RNN?
