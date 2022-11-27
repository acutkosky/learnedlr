
weight_decay_values = [1e-3, 1e-4, 0.0]
learning_rate_values = [3e-5, 1e-4, 3e-4, 1e-3, 2e-3]
decay_values = ['linear', 'true_cosine']
warmup_values = [40960, 204800, 1024000]


base_file = open('tuning_base.template', 'r')
base_config = base_file.read()
base_file.close()


base_filename = 'lr+LR_wd+WEIGHT_DECAY_decay+DECAY_TYPE_warmup+WARMUP_EXAMPLES.yaml'

def make_tuning_string(base_string, wd, lr, decay_type, warmup_examples):
    return base_string.replace('+WEIGHT_DECAY', str(wd)) \
                      .replace('+LR', str(lr)) \
                      .replace('+DECAY_TYPE', str(decay_type)) \
                      .replace('+WARMUP_EXAMPLES', str(warmup_examples))




for wd in weight_decay_values:
    for lr in learning_rate_values:
        for decay in decay_values:
            for warmup_examples in warmup_values:

                tuning_config = make_tuning_string(base_config, wd, lr, decay, warmup_examples)
                tuning_filename = make_tuning_string(base_filename, wd, lr,  decay, warmup_examples)

                tuning_file = open(tuning_filename, 'w')
                tuning_file.write(tuning_config)
                tuning_file.close()





            



