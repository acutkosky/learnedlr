
weight_decay_values = [1e-3, 1e-4, 0.0]
learning_rate_values = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]


base_file = open('tuning_base.template', 'r')
base_config = base_file.read()
base_file.close()


base_filename = 'lr+LR_wd+WEIGHT_DECAY_cos+COSINE_DECAY_lin+LINEAR_DECAY_truecos+TRUE_COSINE_DECAY.yaml'

def make_tuning_string(base_string, wd, lr, cosine, true_cosine, linear):
    return base_string.replace('+WEIGHT_DECAY', str(wd)) \
                      .replace('+LR', str(lr)) \
                      .replace('+COSINE_DECAY', str(cosine)) \
                      .replace('+LINEAR_DECAY', str(linear)) \
                      .replace('+TRUE_COSINE_DECAY', str(true_cosine))




for wd in weight_decay_values:
    for lr in learning_rate_values:
        for decay in ['cos', 'linear', 'true_cos']:
            linear = (decay == 'linear')
            cosine = (decay == 'cos')
            true_cosine = (decay == 'true_cos')

            tuning_config = make_tuning_string(base_config, wd, lr, cosine, true_cosine, linear)
            tuning_filename = make_tuning_string(base_filename, wd, lr, cosine, true_cosine,linear)

            tuning_file = open(tuning_filename, 'w')
            tuning_file.write(tuning_config)
            tuning_file.close()





            



