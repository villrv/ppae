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
    - 2CXO J011908.6-341130       22096                     9
        - No info?
    - 2CXO J005440.5-374320        16029                    162   
        - [64709, 64710]
        - 64709 is a small dip
    - 2CXO J175046.8-311629        21218                    105 
       - No info
    - 2CXO J140314.3+541806        4732                     10 
        - [1792, 1793]
        - nothing interesting
    - 2CXO J132943.3+471134        13814                    567   
        - [90436, 90437, 90438, 90439, 90440, 90446]
        - 90438 is a dip
        
- Dips from Steven
    - "2E 1402.4+5440 (ULX)": [45248, 45250, 42824, 49539, 74752, 74753, 45249]
    - "V* UY Vol (LMXB)": [1920]
    - "Other Dips": [4420, 4421, 90758, 90760, 90761, 90766, 90767, 73545, 73546, 73547, 73550, 73552, 7552, 7553, 51073, 44675, 90753, 90754, 90755, 90757, 73540, 73543, 73548, 73553, 73565, 73598, 56640]
    
- Flares from Steven
    - "Lin et al. 2022": [17217, 34890]
    - "Other Flares": [88128, 88133, 88584, 88585, 68512, 68599, 54209, 54217, 54221, 12056, 12093, 99970, 2512, 107776, 107778, 107779, 107782, 107794]
    - "Yang et al. 2019": [78722, 78740, 78754] 
    - "Glennie et al. 2015": [2048, 2064]

- Final list: 
    - "2CXO J005440.5-374320": [64709]
    - "2CXO J132943.3+471134": [90438]
    - "2E 1402.4+5440 (ULX)": [45248, 45250, 42824, 49539, 74752, 74753, 45249]
    - "V* UY Vol (LMXB)": [1920]
    - "Other Dips": [4421, 90767, 7552, 7553, 51073, 90753, 90757, 73565, 56640]
    - "Lin et al. 2022": [17217, 34890]
    - "Glennie et al. 2015": [2048]
    - "Other Flares": [88133, 88585, 54209, 99970, 2512, 107776, 107778, 107779]




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
    
##### Interesting events to record:
- A very strong cluster in filtered dataset, tSNE space:
554
2445
3407
4219
4332
4711
4971
6692
6933
7102
8247
8669
10015
10041
10575
10859
11214
11228
12009
12324
12325
12494
12645
12704
12712
13030
13039
13473
13485
13614
13991
14448
14555
14652
14674
14741
14767