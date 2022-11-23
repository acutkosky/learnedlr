
weight_decay_values = [1e-3, 1e-4, 0.0]
learning_rate_values = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]


base_file = open('tuning_base.yaml', 'r')
base_config = base_file.read()
base_file.close()


base_filename = 'lrLR_wdWEIGHT_DECAY_cosCOSINE_DECAY_linLINEARDECAY.yaml'

def make_tuning_string(base_string, wd, lr, cosine, linear):
    return base_string.replace('WEIGHT_DECAY', str(wd)) \
                      .replace('LR', str(lr)) \
                      .replace('COSINE_DECAY', str(cosine)) \
                      .replace('LINEAR_DECAY', str(linear))




for wd in weight_decay_values:
    for lr in learning_rate_values:
        for cosine in [True, False]:
            linear = not cosine

            tuning_config = make_tuning_string(base_config, wd, lr, cosine, linear)
            tuning_filename = make_tuning_string(base_filename, wd, lr, cosine, linear)

            tuning_file = open(tuning_filename, 'w')
            tuning_file.write(tuning_config)
            tuning_file.close()





            



