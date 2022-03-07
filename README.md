# Goal-GAN for DDPG Agent

## TODO
- [ ] LSGAN training?
- [ ] What are the observations from the enviroment?
- [ ] Saving the model during the training
- [ ] Choose optimizer for the DDPG agent
- [ ] Update the CLI

## Installation
- First, install mujoco, the installation instructions can be found on their [Github-page](https://github.com/openai/mujoco-py) (NOTE: as of writing, only version 2.10 of mujoco is supported)
- Then, follow the installation instruction from [here](https://github.com/geyang/jaynes-starter-kit/tree/master/07_supercloud_setup). This tutorial shows step by step how the correct enviroment variables are set (for linux)
- Pip installation:
  - Create a virtual enviroment using _venv_
  ```shell
    python -m venv path/to/virtual/enviroment
  ```
  - Activate the enviroment, with bash/zsh
  ```shell
  source <venv>/bin/activate
  ```
  where the <venv> is path to the just created virtual enviroment
   (see [this](https://docs.python.org/3/library/venv.html) for the command for e.g. csh or windows)
  - Install the needed packages
  ```shell
  pip install -r requirements.txt
  ```
- Conda installation
  - Create virtual environment from the _environment.yml_ file
  ```shell
  conda env create -f environment.yml
  ```
  - Activate the virtual environment
  ```shell
  conda activate <name-of-the-env>
  ```

## Pre-producing the results
