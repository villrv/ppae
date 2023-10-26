# ppae

## Current problem
- Need to deal with 

## TODO list
- Think about MLP structure. Might need convolutional and recurrent layers
- Some tricks heard from Pavlos' talk on Oct 18 that might worth integrating:
    - Random sample t when computing the integral (although currently it's already pseudo-random since T is different for different event list)
    - Include smoothness contraint of the rate function into the loss (derivative available from back propagation)