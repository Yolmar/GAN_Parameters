from train import train_model

data_names = ['Data_PNG', 'Data_Augmented_3k', 'Data_Augmented_30k', 'Data_Augmented_30k'
              'HSIL_Original', 'HSIL_1k', 'HSIL_10k', 'HSIL_30k', 'LSIL_Original', 
              'LSIL_1k', 'LSIL_10k', 'LSIL_30k', 'NSIL_Original', 'NSIL_1k', 'NSIL_10k',
              'NSIL_30k']

# batch_size experiment
train_model(manualSeed=28180, 
            batch_size=8, 
            data_name='Data_PNG', 
            nz=256,
            experiment_name='Test')

# learing rate experiment


