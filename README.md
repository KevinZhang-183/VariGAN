# The offical code of VariGAN: Enhancing Image Style Transfer by UNet Generator & Auto-encoding Discriminator via GAN


### environmental requirement
- Windows
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

---
# Configuration procedure：

## Clone project
1. Open PyCharm, Tools -> Space -> Clone Repository
2. Repository URL Enter the GitHub project URL

## Install necessary packages
Create an environment using Anaconda and pip the installation library
   1. Open the Anaconda Prompt
   2. Create a virtual environment:`conda create -n pytorch-CycleGAN-and-pix2pix python=3.8`
   3.Enter the virtual environment:`conda activate pytorch-CycleGAN-and-pix2pix`
   4. Enter the 'cd' command in the command window to enter the project directory and use the following command to download the required dependency library:

      ```
      pip Pip install -r requirements.txt
      ```

## Download Datsets

Download and extract the official dataset into the project's 'datasets' directory. https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

## Train the backbone

1. Set the debugging configuration before running train.py.
2. Enter parameters in the file configuration interface in the format of: '--dataroot [path to your training set tastA] --name [custom weight save file name] --model cycle_gan'
3. Modify training parameters in options/train_options.py as needed, such as training times, batch size, etc.
4. Enter the command in the terminal to enable the visdom service (visual interface) : 'python -m visdom.server'
5. After starting the service, directly run the 'train.py' file to start training the model.
## Custom training, using their own training set training

1. Naming:
- The training sets are named 'trainA' and 'trainB'

- The test sets are named 'tastA' and 'tastB'

2. Set the training configuration file directly and set the training set path to the path of 'train'.

3. Train directly, the same way as the previous step.

## CycleGAN weight file obtained after training, we can use it

1. Find the path to the 'testA' or 'testB' image that you want to convert.

2. Rename the weight file you want to use to 'latest_net_G.pth'

3. Before running 'test.py', set the debugging configuration in the format of '--dataroot [path to test set A] --name [weight file name used] --model test --no_dropout'

4. Run the 'test.py' file for testing, and the generated converted image will be saved in the 'results' directory.
---
