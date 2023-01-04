
# from torchtext.datasets import WikiText2
import transformers
# from transformers import DataCollatorForLanguageModeling
import torch
# from torch.utils.data import DataLoader
import argparse
import time
from tqdm import tqdm
# from matplotlib import pyplot as plt
from model_jax import StackedAttention
import model_jax_pt
import onlineopt
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
from optional_module import optional_module
import c4_loader
import wandb
import numpy as np
import jax
import optax

from functools import partial

import opt_jax

from typing import Any
from jax import numpy as jnp
from dataclasses import dataclass

import gc

parser = argparse.ArgumentParser(description='Simple pretraining thing')

parser.add_argument('--config', type=str, default='config/default.yaml')
parser.add_argument('--model_config', type=str, default='config/model/base.yaml')
parser.add_argument('--train_config', type=str, default='config/train/base.yaml')

#@jax.jit
def optax_init(optimizer, model_state):
    return optimizer.init(model_state)


def optax_state_and_step(optimizer, model_state, *args, **kwargs):
    optimizer = optimizer(*args,  **kwargs)



    opt_state = optimizer.init(model_state['params'])

    #@jax.jit
    def update_step(loss_and_grad_fn, rng, model_and_example_state, opt_state, scale, do_logging=False):
        '''
        update_step: computes the update.
        An optimizer should implement a function with this signature:
        
            rng: jax rng
            loss_and_grad_fn: a function that will return a tuple loss, grad.
                this function shouldtake as input a single jax tree argument.
            model_and_example_state: input to loss_and_grad_fn (probably a
                combination of the model state and the example/minibatch)
            opt_state: state of optimizer
            scale: scaling value (e.g. a learning rate).
            do_logging: boolean indicating whether to return a dict of values to 
                log (ignored here).
        
        returns:
            rng, value, grad, model_state, opt_state, log_dict

            rng: next value for jax rng
            value: value returned by loss_and_grad_fn
            grad: gradient returned by loss_and_grad_fn
            model_state: next value of the model_state
            opt_state: next state of optimizer
            log_dict: dictionary of keys to be logged in wandb (can be set to
                None to log nothing)
        '''
        print("tracing update step")
        value, grad = loss_and_grad_fn(model_and_example_state)
        # grads = loss_and_grads['grads']

        model_state = model_and_example_state['model_state']
        constants = model_state['constants']
        params = model_state['params']
        updates, opt_state = optimizer.update(grad,  opt_state, model_state['params'])

        # The learning rate scheduler thing in optax seems more complicated than
        # just doing this:
        updates = jax.tree_util.tree_map(lambda p: scale * p, updates)


        params = optax.apply_updates(params, updates)
        model_state = {'constants': constants, 'params': params}
        return rng, value, grad, model_state, opt_state, None
    
    return opt_state, update_step

class Trainer:
    def __init__(self, rng, model_state, model_apply, config, tokenizer):
        print(config)

        self.rng = rng
        self.config = config
        self.model_state = model_state
        self.model_apply = model_apply
        self.tokenizer = tokenizer

        self.current_lr = config.lr

        self.config.warmup_steps = self.config.warmup_examples // self.config.batch_size
        self.config.total_steps = self.config.epochs *  self.config.valid_frequency_examples // self.config.batch_size

        if self.config.optimizer == 'adamw':
            # self.optimizer_state = opt_jax.adamw_init(model_state['params'], lr=1.0, beta1=config.beta1, beta2=config.beta2, wd=config.wd)
            # self.optimizer_step = opt_jax.adamw
            self.optimizer_state, self.optimizer_step = optax_state_and_step(optax.adamw, model_state, learning_rate=1.0, weight_decay=config.wd)
            # self.optimizer_step = jax.jit(self.optimizer_step)
        if self.config.optimizer == 'custom':
            if self.config.custom_opt.name == 'adamw':
                self.optimizer_state = opt_jax.adamw_init(model_state['params'], lr=1.0, beta1=config.beta1, beta2=config.beta2, wd=config.wd)
                self.optimizer_step = opt_jax.adamw

            if self.config.custom_opt.name == 'ol_expmd':
                self.optimizer_state = opt_jax.OL_momentum_expmd_init(model_state['params'], grid_size=2)
                self.optimizer_step = opt_jax.OL_momentum_expmd_update

            if self.config.custom_opt.name == 'ol_ogd':
                self.optimizer_state = opt_jax.OL_momentum_ogd_init(model_state['params'], ol_lr=config.custom_opt.ogd.lr)
                self.optimizer_step = opt_jax.OL_momentum_ogd_update

            if self.config.custom_opt.name == 'ol_cb':
                self.optimizer_state = opt_jax.OL_momentum_init(
                    model_state['params'],
                    ol_init=opt_jax.cb_init,
                    eps=config.custom_opt.cb.eps,
                    eta=config.custom_opt.cb.eta,
                    decay=config.custom_opt.cb.decay
                )
                self.optimizer_step = partial(opt_jax.OL_momentum_update, opt_jax.cb_update)

            if self.config.custom_opt.name == 'adamw_learned_lr':
                opt_conf = config.custom_opt.adamw_learned_lr
                self.optimizer_state = opt_jax.adamw_learned_lr_init(
                    model_state['params'],
                    ol_init=lambda x: opt_jax.cb_init(x, opt_conf.cb.eps, opt_conf.cb.eta, opt_conf.cb.decay),
                    lr=1.0,
                    beta1=opt_conf.beta1,
                    beta2=opt_conf.beta2,
                    wd=opt_conf.wd,
                    lower_bound=opt_conf.lower_bound,
                    upper_bound=opt_conf.upper_bound,
                )
                
                self.optimizer_step = partial(opt_jax.adamw_learned_lr_update, opt_jax.cb_update)

        # self.losses = []


        print("loaded optimizer...")

        self.iterations = 0
        self.examples = 0
        self.tokens = 0

        # load train data
        self.train_loader = load_c4_data(self.config, self.tokenizer, split='train')
        self.train_iter = enumerate(self.train_loader)
        # load valid data
        # self.valid_loader = load_valid_data(self.config, self.tokenizer)
        # self.valid_iter = enumerate(self.valid_loader)


        def apply_from_params(params, constants, idx, mask, labels):
            return self.model_apply({
                                        'constants': constants,
                                        'params': params
                                        },
                                    idx, mask, labels)[1]
        
        grad_func = jax.grad(apply_from_params)

        def grad_from_state(model_state, idx, mask, labels):
            return grad_func(model_state['params'], model_state['constants'], idx, mask, labels)


        # grad_func = lambda model_state, *args, **kwargs: grad_func(model_state['params'], model_state['constants'], *args, **kwargs)

        # @jax.jit

        def get_loss(model_state, idx, mask, labels):
            print("tracing get_loss")
            features, loss, accuracy = self.model_apply(model_state, idx, mask, labels)

            # print(model_state.keys())
            # print(model_state['params'])
            grads = grad_from_state(model_state, idx, mask, labels)

            # print("finished")
            return features, loss, accuracy, grads

        def loss_and_grad_fn(model_and_example_state):
            model_state = model_and_example_state['model_state']
            idx = model_and_example_state['idx']
            mask = model_and_example_state['mask']
            labels = model_and_example_state['labels']
            features, loss, accuracy, grads = get_loss(model_state, idx, mask, labels)
            return {
                'features': features,
                'loss': loss,
                'accuracy': accuracy
                }, grads




        def loss_and_step(rng, model_state, optimizer_state, idx, mask, labels, lr, do_logging=False):
            print("tracing loss and step")
            # print("jitting...")
            # features, loss, accuracy, grads = get_loss(model_state, idx, mask, labels)
            model_and_example_state = {
                'model_state': model_state,
                'idx': idx,
                'mask': mask,
                'labels': labels,
            }
            rng, loss, grad, model_state, opt_state, log_data = self.optimizer_step(loss_and_grad_fn, rng, model_and_example_state, optimizer_state, lr, do_logging)
            return rng, loss, model_state, opt_state, log_data

        self.loss_and_step = jax.jit(loss_and_step, static_argnames='do_logging')

        # self.loss_and_step = loss_and_step


    def run_epoch(self, epoch_num):
        iterations_per_epoch = self.config.valid_frequency_examples // self.config.batch_size
 
        pbar = tqdm(self.train_iter, total=iterations_per_epoch)#, total=len(loader))
        running_loss = 0.0
        running_accuracy = 0.0
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        last_time = time.monotonic()
        cur_run_it = 0
        
        # abbreviate so we don't need to write self.config and self.model everywhere.
        config = self.config



    
        for t, strings in pbar:
            cur_run_it += 1
            # print(strings)
            # encoded = tokenizer(strings, padding=True, truncation=True, return_tensors='pt', max_length=model.config.context_length)
            
            # num_tokens = strings['input_ids'].numel()

            # self.total_tokens += num_tokens

            idx = jax.lax.stop_gradient(jnp.array(strings['input_ids']))
            mask = jax.lax.stop_gradient(jnp.array(strings['attention_mask']))
            labels = jax.lax.stop_gradient(jnp.array(strings['labels']))

            

            log_interval = config.get('log_interval', 100)
            log_dict = {}

            log_iteration = (self.iterations % log_interval == 0) and config.logging

            # loss_and_step = jax



            # if self.do_warmup:

            # cosine decay
            lr = config.lr
            if config.decay_type == 'half_cosine':
                lr = lr * np.cos(0.5 * np.pi * self.iterations/config.total_steps)
            if config.decay_type == 'true_cosine':
                lr = lr *  0.5 * (1+ np.cos(np.pi * self.iterations/config.total_steps))
            if config.decay_type == 'linear':
                lr = lr * (1.0 - self.iterations/config.total_steps)
            # linear warmup
            lr = lr * min(1, float(self.iterations) / float(max(1, config.warmup_steps)))
            self.current_lr = lr
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = lr
            if log_iteration:
                log_dict.update({
                        'optimizer/lr_schedule': lr
                    })



            self.rng, loss_and_accuracy, self.model_state, self.optimizer_state, log_data  = self.loss_and_step(
                self.rng,
                self.model_state,
                self.optimizer_state,
                idx,
                mask,
                labels,
                jnp.array(self.current_lr),
                log_iteration)

            loss = loss_and_accuracy['loss']
            accuracy = loss_and_accuracy['accuracy']
            if log_data is not None and log_iteration:
                for key, value in log_data.items():
                    log_dict["optimizer/" + key] = value
    
            self.tokens += (mask >= 0).sum()


            current_time = time.monotonic()
            delta_time = current_time - last_time
            last_time = current_time
            running_loss += (loss - running_loss)/min(cur_run_it, 1000.0)
            epoch_train_loss += (loss - epoch_train_loss)/(cur_run_it)
            running_accuracy += (accuracy - running_accuracy)/min(cur_run_it, 1000.0)
            epoch_train_accuracy += (accuracy - epoch_train_accuracy)/(cur_run_it)

            # losses.append(loss.item())
            self.iterations += 1
            self.examples += idx.shape[0]


            if log_iteration:
                log_dict.update({
                        "epoch": epoch_num,
                        "train/loss": loss,
                        "train/accuracy": accuracy,
                        "it_per_second": 1.0/delta_time,
                        "examples": self.examples,
                        "tokens": self.tokens,
                    })
                # if config.log_differences and 'randomol' in config.optimizer:
                #     log_dict["optimizer/measured_minus_estimate"] = self.total_difference - total_reward

                wandb.log(log_dict, step = self.iterations)


            pbar.set_description(f"train epoch {epoch_num+1} iter {t}: current loss {loss:.5f}, running loss {running_loss:0.5f}, running accuracy {running_accuracy:0.5f} speed {1.0/delta_time:0.5f}")

            for _ in range(config.ministeps-1):
                # take ministeps steps on the current loss...
                features, loss, accuracy = get_loss()
                self.optimizer.step()

            if (t+1) % iterations_per_epoch == 0 or self.examples >= self.config.total_examples:
                break

        wandb.log({
                "train/epoch_loss": epoch_train_loss,
                "train/epoch_accuracy": epoch_train_accuracy
            },
            step = self.iterations)     
        return self.iterations
 
def load_c4_data(config, tokenizer, split):
    if config.dataset == 'c4':
        loader = c4_loader.get_c4_loader_next_token(tokenizer,
                    split=split,
                    batch_size=config.batch_size,
                    max_length=config.context_length,
                    pad_to_multiple_of=config.context_length,
                    num_workers=config.num_workers)
    else:
        raise ValueError(f'We only support C4 dataset right now, not {config.dataset}')
    return loader

# def load_train_data(config, tokenizer, workers=2):
#     if config.dataset == 'c4':
#         loader = c4_loader.get_c4_loader_next_token(tokenizer,
#                     split='train',
#                     batch_size=config.batch_size,
#                     max_length=config.context_length,
#                     pad_to_multiple_of=config.context_length,
#                     num_workers=8)
#     else:
#         raise ValueError(f'We only support C4 dataset right now, not {config.dataset}')
#     return loader


class Validator:
    def __init__(self, rng, model_apply, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.model_apply = model_apply
        self.rng = rng

        self.model_apply = jax.jit(model_apply)
        
        self.load_data()
        # losses = []


    def load_data(self):
        self.valid_loader = load_c4_data(self.config, self.tokenizer, split='validation')
        self.valid_iter = enumerate(self.valid_loader)


    def valid(self,  model_state, epoch, iterations):


        config = self.config

        iterations_in_valid_phase = config.valid_examples // config.batch_size
        self.running_loss = 0.0
        self.running_accuracy = 0.0

        # train_iter = WikiText2(root='data/', split='train')
        # loader = DataLoader(train_iter, batch_size=config.batch_size, shuffle=True)

        model_state = jax.tree_util.tree_map(lambda x: jax.lax.stop_gradient(x), model_state)

        def run_valid_iterations(iterations_completed, valid_iterations_left):
            pbar = tqdm(self.valid_iter, total=valid_iterations_left)#, total=len(loader))
            last_time = time.monotonic()
            cur_run_it = 0
            for t, strings in pbar:
                # encoded = tokenizer(strings, padding=True, truncation=True, return_tensors='pt', max_length=model.config.context_length)
                idx = jax.lax.stop_gradient(jnp.array(strings['input_ids']))
                mask = jax.lax.stop_gradient(jnp.array(strings['attention_mask']))
                labels = jax.lax.stop_gradient(jnp.array(strings['labels']))
                cur_run_it += 1
                iterations_completed += 1
                

                features, loss, accuracy = self.model_apply(model_state, idx, mask, labels)


                current_time = time.monotonic()
                delta_time = current_time - last_time
                last_time = current_time
                self.running_loss += (loss - self.running_loss)/(iterations_completed)
                self.running_accuracy += (accuracy - self.running_accuracy)/(iterations_completed)

                # self.losses.append(loss.item())
                pbar.set_description(f"valid epoch {epoch+1} iter {t}: current loss {loss:.5f}, running loss {self.running_loss:0.5f}, running accuracy {self.running_accuracy:0.5f} speed {1.0/delta_time:0.5f}")
                if cur_run_it >= valid_iterations_left:
                    break
            return cur_run_it


        iterations_completed = 0
        while iterations_in_valid_phase > iterations_completed:
            iterations_completed += run_valid_iterations(iterations_completed,
                                                                    iterations_in_valid_phase - iterations_completed)
            if iterations_in_valid_phase > iterations_completed:
                # we exhausted the valid set! let's finish out by starting from the beginning again
                print("exhausted validation set! reloading...")
                self.load_data()

        
        wandb.log({
                "valid/loss": self.running_loss,
                "valid/accuracy": self.running_accuracy
            },
            step=iterations)





def initialize_and_train_model(config):

    model_config = config.model
    train_config = config.train

    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

    model_config.vocab_size = tokenizer.vocab_size
    # model_config = ModelConfig(
    #     vocab_size=tokenizer.vocab_size,
    #     context_length=10,
    #     embedding_dim=args.dim,
    #     n_heads=args.n_heads,
    #     n_layers=args.n_layers,
    #     args=args,
    #     use_diag=args.use_diag)
    # print(config.use_diag)

    # train_config = TrainConfig(
    #     batch_size=args.batch_size,
    #     learning_rate=args.lr,#*args.batch_size/args.ministeps,
    #     epochs=args.epochs,
    #     wd=args.wd,
    #     opt=args.opt,
    #     ministeps=args.ministeps,
    #     eps=args.eps,
    #     recenter=(args.recenter == 'true') or (args.recenter == True),
    #     beta=args.beta,
    #     implicit=(args.implicit == 'true') or (args.implicit == True),
    #     adaptive=(args.adaptive == 'true') or (args.adaptive == True),
    #     scale_type=args.scale_type,
    #     beta2=args.beta2,
    #     ol=args.ol,
    #     args=args,
    # )

    # device = 'cpu'
    # if torch.cuda.is_available(): 
    #     print("setting up model on gpu")           
    #     device = torch.cuda.current_device()
    # else:
    #     print("there is no gpu!")

    # print("ready to configure model...")

    # attention_model = StackedAttention(model_config)

    model_state, model_apply, *_ = model_jax_pt.StackedAttention.init(model_config)




    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    rng, train_rng = jax.random.split(rng)
    rng, valid_rng = jax.random.split(rng)


    # model_state = attention_model.init(init_rng, jnp.ones([2,4000],dtype=int), jnp.full([2,4000],False), jnp.ones([2,4000], dtype=int))

    print("about to train...")

    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # trainer = Trainer(train_rng, model_state, attention_model.apply, train_config, tokenizer)
    
    # validator = Validator(valid_rng, jax.jit(attention_model.apply), train_config, tokenizer)


    trainer = Trainer(train_rng, model_state, model_apply, train_config, tokenizer)
    
    validator = Validator(valid_rng, jax.jit(model_apply), train_config, tokenizer)

    for e in range(train_config.epochs):
        print(f"starting epoch {e+1}")
        iterations = trainer.run_epoch(e)
        validator.valid(trainer.model_state, e, iterations)

    # losses = train(attention_model, train_config, device)#args, device)

    # plt.plot(losses)
    # plt.xlabel('iterations')
    # plt.ylabel('loss')
    # plt.title('Loss of awesome model!')

    # plt.savefig(args.save_path)



if __name__=='__main__':
    args = parser.parse_args()


    config = OmegaConf.load(args.config)
    print(args)
    print(config)

    # fill in default values from base model and trainer configs
    if args.model_config is not None:
        model_config = OmegaConf.load(args.model_config)
        try:
            config.model = OmegaConf.merge(model_config, config.model)
        except ConfigAttributeError:
            # if there is no model_config specified in the file args.config, then the above merge might fail.
            # in this case, we just use the defaults.
            config.model = model_config
    if args.train_config is not None:
        train_config = OmegaConf.load(args.train_config)
        try:
            config.train = OmegaConf.merge(train_config, config.train)
        except ConfigAttributeError:
            config.train = train_config



    if config.train.total_examples is not None:
        # when total_examples is specified, then we use this to override epochs.
        config.train.epochs = config.train.total_examples // config.train.valid_frequency_examples
  
    

    # args = OmegaConf.create(vars(args))
    # if args.config is not None:
    #     conf = OmegaConf.load(args.config)
    #     conf.debias = conf.get('debias', True)

    #     args = OmegaConf.merge(args, conf)

    wandb = optional_module(wandb, config.train.logging)
    wandb.init(project=config.train.wandb_project)
    wandb.config.update({
        'train': config.train._content,
        'model': config.model._content,
        })
    initialize_and_train_model(config)



    

    


    
