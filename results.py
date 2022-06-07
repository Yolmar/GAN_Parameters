import os
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt


from dcgan import Generator

# Root directory for dataset
data_name = "Data_PNG"
epoch_snapshot = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

for epoch in epoch_snapshot:
    modelroot = f"Models/{data_name}/e{epoch}.pth"
    isModelExist = os.path.exists(modelroot)
    if not isModelExist:
        print(f'File {modelroot} doesnt exist...')
    else:
        # Load the checkpoint file.
        state_dict = torch.load(modelroot)

        # Set the device to run on: GPU or CPU.
        device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

        params = state_dict['params']
        G_losses = state_dict['GLosses']
        D_losses = state_dict['DLosses']

        # Create the generator network.
        netG = Generator(params).to(device)
        # Load the trained generator weights.
        netG.load_state_dict(state_dict['generator'])

        noise = torch.randn(64, params['nz'], 1, 1, device=device)

        # Turn off gradient calculation to speed up the process.
        with torch.no_grad():
    	    # Get generated image from the noise vector using
    	    # the trained generator.
            generated_img = netG(noise).detach().cpu()

        # Display the generated image.
        plt.figure(figsize=(20,10))
        plt.axis("off")
        plt.title("Generated Images")
        plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)))

        save_path = f'Plots&Images/{data_name}'
        isExist = os.path.exists(save_path)
        if not isExist:
            os.makedirs(save_path)
        plt.savefig(f'{save_path}/e{epoch}_Generated_Images.png')

        plt.clf()

        plt.figure(figsize=(20,10))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'{save_path}/e{epoch}_G&D_Loss.png')
