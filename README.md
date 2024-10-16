# PPAD

This is the official codebase for the paper [A Poisson-process AutoDecoder for X-ray Sources](). The code and README is relatively poorly documented and some features are obsolete. The code only supports single GPU at the moment.
Feel free to contact Yanke Song (ysong@g.harvard.edu) if you have any questions.

```
pip install requirements.txt
```



## Inference
For inference on new event files, you would first want to create a data list, which is a python list of dictionaries. Each dictionary contains the key "event_list", which stores a two-dimensional numpy array recording the arrival time and energy of photons (an event list), and other variables for your convenience, like "obsreg_id".

Then run the following code, with your options 
```
python inference.py --checkpoint ${checkpoint} --data ${data} --save_location ${save_location} root_dir ${root_dir}
```


## Training
For training with the data we prepared, refer to the training code below. You would need to place the relevant data pkl files at corresponding locations.

```
B=16
starting_epoch=0
num_epochs=100
checkpoint_every=50
latent_size=8
hidden_size=512
hidden_blocks=5
data_type="large_filtered"
model_type="decoder"
TV_type="separate"
lam_latent=1
lam_TV=10
lr=0.00001

# First stage training, on a filtered subset
python main.py --data_type "${data_type}" --model_name "${model_name}" --model_type "${model_type}" --TV_type "${TV_type}" --random_shift --starting_epoch "${starting_epoch}" --num_epochs "${num_epochs}" --latent_size ${latent_size} --hidden_size ${hidden_size} --hidden_blocks ${hidden_blocks} --checkpoint_every ${checkpoint_every} --lam_TV ${lam_TV} --B "${B}" --lr ${lr} --lam_latent ${lam_latent} --thin_resnet 

# For second stage training, change data_type to "large", and add the following checkpoint info
--finetune_checkpoint "${finetune_checkpoint}" --latent_only

# For third stage training, add the following checkpoint info
--finetune_checkpoint2 "${finetune_checkpoint2}"
```

If you want to train the model yourself, please refer to `main.py` for the training script and relevant options.