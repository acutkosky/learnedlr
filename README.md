# testing optimizers on C4

To setup on SCC, first load modules:

```
module load python3 pytorch tensorflow
```

Now, setup virtual environment:

```
[ ! -d "env" ] && virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

finally, initialize wandb:

```
wandb init
```

To configure runs, you should create a config yaml file (e.g. see `config/default.yaml`). The default values for the model architecture are stored in `config/model/base.yaml`.
The default values for the training parameters are stored in `config/train/base.yaml`. Your config file can override any of these. Model parameters
should be nested under a `model_config` heading and training parameters under a `train_config` heading, as in:
```
train_config:
    lr: 0.001
    optimizer: adamw

model_config:
    n_layers: 7
```

To submit a job to the scheduler, use the submit script `submit.sh`. This is basically a small wrapper around the command `python trainer_c4.py`, so you can provide extra arguments like so:
```
qsub submit.sh --config config/default.yaml # will launch the command python trainer_c4.py --config config/default.yaml
```


Configuration for wandb logging is left under the training config (probably the train config could be broken down a bit more, but this whole thing is
already overengineered...). The relevant values are: `logging`, which can be set to `false` to turn off logging, `wandb_project`, which sets the project
name in wandb, and `log_interval` which sets how many iterations elapse between sending training statistics to wandb.

