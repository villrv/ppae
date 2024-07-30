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
    - Different datasets:
        - large_filtered: 14891
        - large: 109656
        


### What did not work
- Small dataset
    - Include time difference in encoding
    - Encoder
    - Mixed precision training REALLY HURTS!

- Large dataset
    - accumulate gradient doesn't help a huge

    

### What did work
- General:
    - TV loss on usual (exp) rates
    - TV should be separate: event rate + mesh rate, this is a lot faster
    - Plotting should use total rate
    - Train longer
    - For positional encoding, should normalize T to be 1, and include t
    - include random shift at 0 to prevent overfitting
    
- Large dataset

    - Potentially eliminating nan/inf values
        - Skip nan loss gradient update (done)
        - torch anomaly detection
        - Gradient clipping
        - Initialization
        - Batch normalization
    - Balance dataset
    
