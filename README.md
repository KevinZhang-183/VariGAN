# The offical code of VariGAN: Enhancing Image Style Transfer by UNet Generator & Auto-encoding Discriminator via GAN


### 环境要求

- Windows
- Python 3
- CPU或NVIDIA GPU + CUDA CuDNN

### 作者环境如下：

- Windows11
- python3.8
- PyTorch1.9.0
- CUDA11.1
- GeForce GTX 3060 Laptop GPU

---
# 配置步骤：
## 克隆项目

1. 打开PyCharm，Tools（工具）-> Space(空间）-> Clone Repository（克隆存储库）
2. Repository URL输入GitHub项目URL
   <br>1：官方URL：https://gitcode.net/mirrors/junyanz/pytorch-cyclegan-and-pix2pix.git
   <br>2：我的URL：https://github.com/qianqianlwg/cycleGAN-.git

## 安装必要库

本文提供两种安装方式。

1. 通过PyCharm自动创建conda环境，并根据项目的environment.yml自动更新需要的库。

   - 进入Conda配置文件中输入conda.exe路径，下面一栏找到克隆项目的environment.yml路径
      

2. 通过Anaconda创建环境，自行pip安装库

   1. 打开Anaconda Prompt
   2. 创建虚拟环境：`conda create -n pytorch-CycleGAN-and-pix2pix python=3.8`
   3. 进入虚拟环境：`conda activate pytorch-CycleGAN-and-pix2pix`
   4. 在命令窗口输入`cd`命令进入项目目录，使用以下命令下载需要的依赖库：

      ```
      pip Pip install -r requirements.txt
      ```

## 下载数据集

下载并解压官方给出的数据集到项目的`datasets`目录下。https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

## 训练模型

1. 运行train.py前先设置调试配置。
2. 在文件配置界面中输入形参，格式为：`--dataroot [你训练集tastA的路径] --name [自定义权值保存文件名] --model cycle_gan`
3. 根据需要在`options/train_options.py`修改训练参数，如：训练次数、批量大小等。
4. 在终端中输入命令，开启visdom服务（可视化界面）：`python -m visdom.server`
5. 开启服务后直接运行`train.py`文件开始训练模型。
## 自定义训练，使用自己的训练集训练

1. 命名：
   - 训练集命名为`trainA`和`trainB`

   - 测试集命名为`tastA`和`tastB`

2. 直接设置训练配置文件，训练集路径为`train`的路径。

3. 直接训练即可，和上一步的训练方式相同。

## CycleGAN通过训练后得到的权值文件，我们可以拿来使用了

1. 找到需要转换的`testA`或者`testB`图片的路径。

2. 将需要使用的权值文件重命名为`latest_net_G.pth`

3. 运行`test.py`前先设置调试配置，格式：`--dataroot [测试集A的路径] --name [使用的权值文件名] --model test --no_dropout`

4. 运行`test.py`文件进行测试，生成的转换图像将保存在`results`目录下。
---
## 总结
&nbsp;&nbsp;&nbsp;&nbsp;本文介绍了CycleGAN的基本概念和应用领域，并详细
介绍了如何进行配置、测试和自定义训练。在配置部分，我们通过anaconda环境创建、安装必要库和下载数据集。在训练模型部分，我们介绍了train.py文件配置、修改训练参数以及如何开启visdom服务。在自定义训练部分，我们介绍了如何给自己的训练集命名、设置训练配置文件并直接训练。在测试CycleGAN部分，我们介绍了如何使用已训练好的权值文件和测试配置文件，并生成图片。
