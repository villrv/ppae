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
        
- Interesting sources to look at:
    - 2CXO J011908.6-341130 (obsid 22096 region ID 9).
        - No info?
    - 2CXO J005440.5-374320        16029                    162   
        - [64709, 64710]
    - 2CXO J175046.8-311629        21218                    105 
       - No info
    - 2CXO J140314.3+541806        4732                     10 
        - [1792, 1793]
    - 2CXO J132943.3+471134        13814                    567   
        - [90436, 90437, 90438, 90439, 90440, 90446]
        
- Questions:
    - What other quantities to look at
    - What I already wrote
    - What else plot to put there

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
    
## ToDO list
- visualize classes in tsne
- interesting event files
- change reference chandra
- YSO vs AGN classification