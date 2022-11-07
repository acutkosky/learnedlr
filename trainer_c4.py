
# from torchtext.datasets import WikiText2
import transformers
# from transformers import DataCollatorForLanguageModeling
import torch
# from torch.utils.data import DataLoader
import argparse
import time
from tqdm import tqdm
# from matplotlib import pyplot as plt
from model import StackedAttention
import onlineopt
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
from optional_module import optional_module
import c4_loader
import wandb

parser = argparse.ArgumentParser(description='Simple pretraining thing')

parser.add_argument('--config', type=str, default='config/default.yaml')
parser.add_argument('--model_config', type=str, default='config/model/base.yaml')
parser.add_argument('--train_config', type=str, default='config/train/base.yaml')



# context manager for swapping variables in a model:


def get_param_copy(model):
    return [x.detach().clone() for x in model.parameters()]

def update_param_copy_(model, to_store):
    for store, p in zip(to_store, model.parameters()):
        store.copy_(p)

@torch.no_grad()
def set_params(model, params):
    for model_p, toset_p in zip(model.parameters(), params):
        model_p.copy_(toset_p)

class SwapParameters:
    '''
    warning: this contex manager will wipe out any gradient info!
    '''
    def __init__(self, model, parameters, saved_storage):
        self.model = model
        self.parameters = parameters
        self.saved_storage = saved_storage

    def __enter__(self):
        # self.saved_parameters = get_param_copy(self.model)
        update_param_copy_(self.model, self.saved_storage)
        set_params(self.model, self.parameters)
    
    def __exit__(self, exc_type, exc_value, traceback):
        set_params(self.model, self.saved_storage)
            

class SwapManager:
    '''
    wrapper around SwapParameters so that if we use it many times
    with the same model, the temporary storage for the model parameters
    is always the same memory, saving a lot of allocation/delocations.
    '''
    def __init__(self, model):
        self.model = model
        self.saved_storage = get_param_copy(model)
    
    def swap(self, parameters):
        return SwapParameters(self.model, parameters, self.saved_storage)

# class TrainConfig:
#     def __init__(self, **kwargs):
#         self.batch_size = 64
#         self.learning_rate = 0.1
#         self.beta1 = 0.9
#         self.beta2 = 0.99
#         self.epochs = 1
#         self.wd = 0.0
#         opt = 'adamw'


#         for k,v in kwargs.items():
#             setattr(self, k, v)


class Trainer:
    def __init__(self, model, config, device, tokenizer):
        print(config)

        self.config = config
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

        self.config.warmup_steps = self.config.warmup_examples // self.config.batch_size

        if self.config.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd, betas=(config.beta1, config.beta2))
        elif self.config.optimizer == 'pertensor_randomol':
            self.optimizer = onlineopt.PerTensorRandomOL(model.parameters(), config=config, logger=wandb)
        elif self.config.optimizer == 'global_randomol':
            self.optimizer = onlineopt.GlobalRandomOL(model.parameters(), config=config, logger=wandb)
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

        self.total_difference = 0

        self.swap_manager = SwapManager(self.model)

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
        model = self.model

        previous_parameters = get_param_copy(self.model)

        for t, strings in pbar:
            cur_run_it += 1
            # print(strings)
            # encoded = tokenizer(strings, padding=True, truncation=True, return_tensors='pt', max_length=model.config.context_length)
            idx = strings['input_ids'].to(self.device)
            mask = strings['attention_mask'].to(self.device)
            labels = strings['labels'].to(self.device)

            log_interval = config.get('log_interval', 100)
            log_dict = {}

            def get_loss():
                # this function computes gradients
                model.zero_grad()
                features, loss, accuracy = model(idx, mask, labels)
                loss.backward()
                return features, loss, accuracy

            def inference():
                # this function is faster than get_loss() but does not allow computing gradients
                with torch.no_grad():
                    features, loss, accuracy = model(idx, mask, labels)
                return features, loss, accuracy

            if config.log_differences or config.correct_inner_products:
                with self.swap_manager.swap(previous_parameters):
                    features, prev_loss, accuracy = inference()
                
                features, cur_loss, accuracy = inference()

                loss_difference = prev_loss - cur_loss
                self.total_difference += loss_difference

                log_dict.update({
                        'optimizer/loss_difference': loss_difference,
                        'optimizer/total_loss_difference': self.total_difference,
                    })     

            update_param_copy_(model, previous_parameters)

            features, loss, accuracy = get_loss()

            if 'randomol' in config.optimizer and config.correct_inner_products:
                step_data = loss_difference
            else:
                step_data = None

            log_data = self.optimizer.step(step_data)

            if log_data is not None and 'randomol' in config.optimizer:
                log_dict.update(log_data)

            self.tokens += (mask >= 0).sum()


            current_time = time.monotonic()
            delta_time = current_time - last_time
            last_time = current_time
            running_loss += (loss.item() - running_loss)/min(cur_run_it, 1000.0)
            epoch_train_loss += (loss.item() - epoch_train_loss)/(cur_run_it)
            running_accuracy += (accuracy.item() - running_accuracy)/min(cur_run_it, 1000.0)
            epoch_train_accuracy += (accuracy.item() - epoch_train_accuracy)/(cur_run_it)

            # losses.append(loss.item())
            self.iterations += 1
            self.examples += idx.size()[0]


            if config.optimizer == 'adamw':
                # linear warmup
                lr = config.lr * min(1, float(self.iterations) / float(max(1, config.warmup_steps)))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                if self.iterations % log_interval == 0:
                    log_dict.update({
                            'optimizer/learned_lr': lr
                        })

            
            if self.iterations % log_interval == 0:
                log_dict.update({
                        "epoch": epoch_num,
                        "train/loss": loss.item(),
                        "train/accuracy": accuracy.item(),
                        "it_per_second": 1.0/delta_time,
                        "examples": self.examples,
                    })
                # if config.log_differences and 'randomol' in config.optimizer:
                #     log_dict["optimizer/measured_minus_estimate"] = self.total_difference - total_reward

                wandb.log(log_dict, step = self.iterations)


            pbar.set_description(f"train epoch {epoch_num+1} iter {t}: current loss {loss.item():.5f}, running loss {running_loss:0.5f}, running accuracy {running_accuracy:0.5f} speed {1.0/delta_time:0.5f}")

            for _ in range(config.ministeps-1):
                # take ministeps steps on the current loss...
                features, loss, accuracy = get_loss()
                self.optimizer.step()

            if (t+1) % iterations_per_epoch == 0:
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
    def __init__(self, model, config, device, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        
        self.load_data()
        # losses = []


    def load_data(self):
        self.valid_loader = load_c4_data(self.config, self.tokenizer, split='validation')
        self.valid_iter = enumerate(self.valid_loader)

    @torch.no_grad()
    def valid(self, epoch, iterations):


        config = self.config
        model = self.model
        device = self.device

        iterations_in_valid_phase = config.valid_examples // config.batch_size
        self.running_loss = 0.0
        self.running_accuracy = 0.0

        # train_iter = WikiText2(root='data/', split='train')
        # loader = DataLoader(train_iter, batch_size=config.batch_size, shuffle=True)
        def run_valid_iterations(iterations_completed, valid_iterations_left):
            pbar = tqdm(self.valid_iter, total=valid_iterations_left)#, total=len(loader))
            last_time = time.monotonic()
            cur_run_it = 0
            for t, strings in pbar:
                # encoded = tokenizer(strings, padding=True, truncation=True, return_tensors='pt', max_length=model.config.context_length)
                idx = strings['input_ids'].to(device)
                mask = strings['attention_mask'].to(device)
                labels = strings['labels'].to(device)
                cur_run_it += 1
                iterations_completed += 1
                

                features, loss, accuracy = model(idx, mask, labels)


                current_time = time.monotonic()
                delta_time = current_time - last_time
                last_time = current_time
                self.running_loss += (loss.item() - self.running_loss)/(iterations_completed)
                self.running_accuracy += (accuracy.item() - self.running_accuracy)/(iterations_completed)

                # self.losses.append(loss.item())
                pbar.set_description(f"valid epoch {epoch+1} iter {t}: current loss {loss.item():.5f}, running loss {self.running_loss:0.5f}, running accuracy {self.running_accuracy:0.5f} speed {1.0/delta_time:0.5f}")
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

    device = 'cpu'
    if torch.cuda.is_available(): 
        print("setting up model on gpu")           
        device = torch.cuda.current_device()
    else:
        print("there is no gpu!")

    print("ready to configure model...")

    attention_model = StackedAttention(model_config).to(device)

    print("about to train...")

    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    trainer = Trainer(attention_model, train_config, device, tokenizer)
    validator = Validator(attention_model, train_config, device, tokenizer)

    for e in range(train_config.epochs):
        print(f"starting epoch {e+1}")
        iterations = trainer.run_epoch(e)
        validator.valid(e, iterations)

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



    

    


    
