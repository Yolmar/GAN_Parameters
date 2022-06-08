from train import train_model

data_names = ['Data_PNG', 'Data_Augmented_3k', 'Data_Augmented_30k', 'Data_Augmented_30k'
              'HSIL_Original', 'HSIL_1k', 'HSIL_10k', 'HSIL_30k', 'LSIL_Original', 
              'LSIL_1k', 'LSIL_10k', 'LSIL_30k', 'NSIL_Original', 'NSIL_1k', 'NSIL_10k',
              'NSIL_30k']

batch_sizes = 128 #[8, 16, 32, 64, 128, 256]
lrs = 2*1e-4# [2*1e-4, 1e-4, 2*1e-5, 1e-5]
epoch_snapshot = [1, 5, 25, 50, 75, 100]

# batch_size experiment
for lr in lrs:
    for batch in batch_sizes:
        train_model(batch_size=batch, data_name='Data_Augmented_3k', lr=lr)

# learing rate experiment


