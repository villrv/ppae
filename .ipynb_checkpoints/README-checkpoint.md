# ppae

## Current problem
- MLP is too simple.
- Regularization parameter very sensitive. The latent codes are either degenerate (all 0) or has very large variance
- TSNE somehow always kills the kernel on my PC. Not sure why (but it should work conceptually)

## TODO list
- Write more helpful comments for better collaboration
- Think about MLP structure. Might need convolutional and recurrent layers
- Some tricks heard from Pavlos' talk on Oct 18 that might worth integrating:
    - Random sample t when computing the integral (although currently it's already pseudo-random since T is different for different event list)
    - Include smoothness contraint of the rate function into the loss (derivative available from back propagation)