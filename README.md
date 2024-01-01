# ppae

## Training feedbacks on the dataset of 1000 eventfiles
### What worked
- TV loss should be on mesh rates and lam_TV should be 0.2.

### What did not work
- Include time difference in encoding
- More layers of LSTM
- Concatenating last output with global pooled output for LSTM
- Increase to 48 token size for vanilla transformer

## What did work
- Longer training time: 800 epochs with a patience of 100
- Bigger LSTM network
- LSTM used last non-padded output
- TV loss for mesh, also added first occurrence

## further tuning ideas:
- Learnable positional encoding