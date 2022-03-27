import torch

from argparse import ArgumentParser
import argparse

import utils
utils.set_logging_config() #Setup logger


from models.gan import LSGAN, GANConfig
from models.agent import DDPGAgent, DDPGConfig
from environment.mujoco_env import MazeEnv
from train import train
from eval import eval_policy

from typing import Tuple


_ENV_NAME: str = "AntUMaze-v1"
_GOAL_SIZE: int = 2

_logger = utils.get_logger(__name__)
_writer = utils.get_writer()

def _create(env: str, generator_config: GANConfig, discriminator_config: GANConfig, ddpg_config: DDPGConfig, device: torch.device) -> Tuple[MazeEnv, DDPGAgent, LSGAN]:
    '''
    Creates the environment, agent and GAN model.

    Parameters
    ----------
    env: str
        Identifier of the used environment
    generator_config: GANConfig
        The generator configuration. The input and output size will be modified,
        but not other parameter is changed.
    discriminator_config: GANConfig
        The configuration for the GAN's discriminator. The input and ouput size will be
        modified, but no other parameter is changed.
    ddpg_config: DDPGConfig
        The configuration for the DDPG Agent. The state and action size will be 
        set depending of the used environment, but other (hyper)parameters
        are not modified.
    device: torch.device
        The device to be used to train/evaluate the models.
    Returns
    -------
    Tuple[MazeEnv, DDPGAgent, LSGAN]
        Returns the created env, agent and GAN model.
    '''
    #Create the environment
    env = MazeEnv(args.env, _GOAL_SIZE)

    #Define the state and action sizes.
    ddpg_config.state_size = env.observation_space.shape[0]
    ddpg_config.action_size = env.action_space.shape[0]

    #Create agent for the enviroment
    agent = DDPGAgent(ddpg_config, device)

    #Generators input size is the "noise size".
    generator_config.input_size = 4
    generator_config.output_size = env.goal_size
    
    #Discriminator takes in items with same size as generator's output, and produces only 1 value.
    discriminator_config.input_size = env.goal_size
    discriminator_config.output_size = 1

    lsgan = LSGAN(generator_config, discriminator_config, device)
    return env, agent, lsgan


def _parse_and_train(args: argparse.Namespace) -> None:
    '''
    Parses the given arguments and then trains the model's using the specified parameters

    Parameters
    ----------
    args: argparse.Namespace
        A namespace object containing all the options passed in the cli.

    '''
    if args.use_checkpoint and (args.gan_checkpoint is None or args.agent_checkpoint is None):
        raise ValueError("If --use-checkpoint flag is set, then both --gan-checkpoint and --agent-checkpoint must be specified!")
    
    if args.save_after is not None and (args.gan_save_path is None or args.agent_save_path is None):
        raise ValueError("If --save-after is specified, then both --gan-save-path and --agent-save--path must also be specified!")

    device = utils.get_device()
    _logger.info(f"Using device: {utils.get_device_repr(device)}")

    #Create configurations for generator and discriminator, with the specified hyperparameters.
    generator_config = GANConfig(
        hidden_size=args.gen_hidden_size, layer_count=args.gen_nlayers,
        opt_lr=args.gen_lr, opt_alpha=args.gen_alpha, opt_momentum=args.gen_momentum
    )
    _writer.add_text("GAN/generator", f"{generator_config}")

    discriminator_config = GANConfig(
        hidden_size=args.disc_hidden_size, layer_count=args.disc_nlayers,
        opt_lr=args.disc_lr, opt_alpha=args.disc_alpha, opt_momentum=args.disc_momentum
    )
    _writer.add_text("GAN/discriminator", f"{discriminator_config}")

    #Create configuration for the DDPG Agent with specified hyperparameters.
    ddpg_config = DDPGConfig(
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, weight_decay=args.weight_decay,
        tau=args.tau, gamma=args.gamma, buffer_size=args.buffer_size, batch_size=args.batch_size
    )
   
    env, agent, lsgan = _create(args.env, generator_config, discriminator_config, ddpg_config, device)
    _writer.add_text("Agent/DDPG", f"{ddpg_config}")

    #If the checkpoints where specified, load the models.
    if args.use_checkpoint:
        lsgan.load_model(args.gan_checkpoint)
        _logger.info("Loaded GAN")
        agent.load_model(args.agent_checkpoint)
        _logger.info("Loaded DDPG agent")
    

    header = f"{'iter-count':^10s}|{'gan-iter-count':^14s}|{'policy-iter-count':^17s}|{'timestep-count':^14s}|{'goal-count':10s}|{'rmin':^8s}|{'rmax':^8s}"
    delim = f"{10*'-'}|{14*'-'}|{17*'-'}|{14*'-'}|{10*'-'}|{8*'-'}|{8*'-'}"
    values = f"{args.train_iter_count:^10d}|{args.gan_iter_count:^14d}|{args.policy_iter_count:^17d}|{args.timestep_count:^14d}|{args.goal_count:^10d}|{args.rmin:^8.4f}|{args.rmax:^8.4f}"
    _writer.add_text("train/hyperparams", f"{header}\n{delim}\n{values}")

    if args.save_after is None:
        train(
            lsgan, agent, env, args.gan_iter_count, args.policy_iter_count,
            args.train_iter_count, args.goal_count, args.episode_count, args.timestep_count, 
            args.rmin, args.rmax
        )
    else:
        train(
            lsgan, agent, env, args.gan_iter_count, args.policy_iter_count,
            args.train_iter_count, args.goal_count, args.episode_count, 
            args.timestep_count, args.rmin, args.rmax, args.save_after, 
            args.gan_save_path, args.agent_save_path
        )
    _logger.info("Exiting...")
    env.close()

def _parse_and_eval(args: argparse.Namespace) -> None:
    '''
    Parses the given arguments and then evaluates the trained model using the
    specified parameters

    Parameters
    ----------
    args: argparse.Namespace
        A namespace object containing all the options passed to the CLI.
    '''

    device = utils.get_device()
    _logger.info(f"Using device: {utils.get_device_repr(device)}")

    #Create configuration for the DDPG Agent with specified hyperparameters.
    ddpg_config = DDPGConfig(
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, weight_decay=args.weight_decay,
        tau=args.tau, gamma=args.gamma, buffer_size=args.buffer_size, batch_size=args.batch_size
    )

    #Create the environment
    env = MazeEnv(args.env, _GOAL_SIZE)

    #Define the state and action sizes.
    ddpg_config.state_size = env.observation_space.shape[0]
    ddpg_config.action_size = env.action_space.shape[0]

    #Create agent for the enviroment
    agent = DDPGAgent(ddpg_config, device)

    #Load the saved model
    agent.load_model(args.agent_model)

    _logger.info("Loaded Agent")
    eval_policy(agent, env, args.eval_iter_count, args.episode_count, args.timestep_count, render=args.render)
    _logger.info("Exiting")
    env.close()

def _add_ddpg_hyperparameters(group) -> None:
    '''Adds the DDPG agent's hyperparameters to a given argument group'''
    group.add_argument("--actor-learning-rate",  type=float, default=1e-4, dest="actor_lr",  help=("Defines the learning rate used with the "
                                                                                                   "Actor's optimizer. Default %(default)s"))
    group.add_argument("--critic-learning-rate", type=float, default=1e-4, dest="critic_lr", help=("Defines the learning rate used with the "
                                                                                                   "Critic's optimizer. Default %(default)s"))
    group.add_argument("--weight-decay",         type=float, default=0.0,                    help=("Defines the weight decay used with the "
                                                                                                   "Actor's and Critic's optimizer. Default %(default)s"))
    group.add_argument("--tau",                  type=float, default=1e-3,                   help=("Define interpolation parameter used when"
                                                                                                   " doing a soft update with DDPG agent. Default %(default)s"))
    group.add_argument("--gamma",                type=float, default=1e-3,                   help=("Defines the discount factor used value"
                                                                                                   " function of the Agent. Default %(default)s"))
    group.add_argument("--buffer-size",          type=int,   default=1000,                   help=("Defines the maximum buffer size for the"
                                                                                                   " replay memory of the Agent. Default %(default)s"))
    group.add_argument("--batch-size",           type=int,   default=128 ,                   help=("Defines the batch size of the replay buffer,"
                                                                                                   " i.e. the size of sampling. Default %(default)s"))

def get_parser() -> ArgumentParser: 
    '''
    Creates an argument parser that defines the CLI for training and evaluating the Goal GAN

    Returns
    -------
    ArgumentParser:
        The CLI for the program.
    '''
    parser = ArgumentParser(prog="GoalGAN", description=("Interface for training and evaluating the GoalGAN."
                                                         " See help for train and eval commands for more information"))
    sub_parsers = parser.add_subparsers()
    
    # ---------------- CLI for training the network ------------------------------------
    train_parser = sub_parsers.add_parser("train", description="Train the Goal Gan")
    train_parser.add_argument("--env",                 type=str,   default=_ENV_NAME, help=("The identifier of the used environment."
                                                                                            " Default %(default)s"))
    train_parser.add_argument("--gan-iter-count",      type=int,   default=200,       help=("The amount of iterations the gan is trained "
                                                                                           "for during each outer iteration. Default %(default)s"))
    train_parser.add_argument("--train-iter-count",    type=int,   default=100,       help=("The amount training iterations done with "
                                                                                           "the model. Default %(default)s"))
    train_parser.add_argument("--policy-iter-count",   type=int,   default=5,         help=("The amount of iterations the policy is updated "
                                                                                           "for during each outer iteration. Default %(default)s"))
    train_parser.add_argument("--goal-count",          type=int,   default=10,        help=("The amount of goals produced by the Goal"
                                                                                           " GAN during each iteration. Default %(default)s") )
    train_parser.add_argument("--timestep-count",      type=int,   default=500,       help=("The amount of timesteps allowed in each"
                                                                                           " episode. Default %(default)s "))
    train_parser.add_argument("--episode-count",       type=int,   default=10,        help=("The amount of episodes evaluated on each"
                                                                                           " set of goals. Default %(default)s"))
    train_parser.add_argument("--rmax",                type=float, default=0.9,       help=("The highest evalution score that is considered to"
                                                                                           " be feasible. Default %(default)s "))
    train_parser.add_argument("--rmin",                type=float, default=0.1,       help=("The lowest evaluation score that is considered to"
                                                                                           " be feasible. Default %(default)s"))
    # <<<<< Continue from previously trained model >>>>>
    continue_group = train_parser.add_argument_group("Continue training from previously saved model") 
    continue_group.add_argument("--use-checkpoint",  action="store_true", help=("If this flag is set, "
                                                                                "the training continues from previous checkpoint."
                                                                                " See also gan-checkpoint and agent-checkpoint flags"))
    continue_group.add_argument("--gan-checkpoint", type=str,             help=("path to the file containing the GAN model to"
                                                                                " continue from. Must be specified if"
                                                                                " --use-checkpoint flag is set"))
    continue_group.add_argument("--agent-checkpoint", type=str,           help=("path to the file containing the Agent model"
                                                                                " to continue from. Must be specified if "
                                                                                "--use-checkpoint flag is set"))

    # <<<<< Saving the model's during training >>>>>
    saving_group = train_parser.add_argument_group("Saving model during training") 
    saving_group.add_argument("--save-after",      type=int, default=None, help=( "The amount of iterations after which the"
                                                                              " models are saved in. Default %(default)s"))
    saving_group.add_argument("--gan-save-path",   type=str,             help=("Path to file, where the gan-model files "
                                                                              "should be saved to. Must be specified if "
                                                                              "save-after is defined"))
    saving_group.add_argument("--agent-save-path", type=str,             help=("Path to file, where the agent-model "
                                                                               "files should be saved to. Must be specified "
                                                                               "if save-after is defined"))

    # <<<<< GAN Hyperparameters >>>>> 
    gan_hp_group = train_parser.add_argument_group("GAN hyperparameters") 
    gan_hp_group.add_argument("--generator-layer-count",       type=int,   default=2,    dest="gen_nlayers",      help=("Defines the amount of linear layers used"
                                                                                                                        " in the Generator. Default %(default)s"))
    gan_hp_group.add_argument("--generator-hidden-size",       type=int,   default=128,  dest="gen_hidden_size",  help=("Defines the size of the hidden layers used"
                                                                                                                        " in the Generator. Default %(default)s"))
    gan_hp_group.add_argument("--generator-learning_rate",     type=float, default=0.01, dest="gen_lr",           help=("Defines learning rate used "
                                                                                                                        "with the Generator's optimizer. Default %(default)s"))
    gan_hp_group.add_argument("--generator-alpha",             type=float, default=0.99, dest="gen_alpha",        help=("Defines smoothing constant used with "
                                                                                                                        "the Generator's optimizer. Default %(default)s"))
    gan_hp_group.add_argument("--generator-momentum",          type=float, default=1e-3, dest="gen_momentum",     help=("Defines the momentum used with"
                                                                                                                        " the Generator's optimizer. Default %(default)s"))
    gan_hp_group.add_argument("--discriminator-layer-count",   type=int,   default=2,    dest="disc_nlayers",     help=("Defines the amount of linear layers used"
                                                                                                                        " in the Discriminator. Default %(default)s"))
    gan_hp_group.add_argument("--discriminator-hidden-size",   type=int,   default=256,  dest="disc_hidden_size", help=("Defines the size of the hidden layers used"
                                                                                                                        " in the Discriminator. Default %(default)s"))
    gan_hp_group.add_argument("--discriminator-learning-rate", type=float, default=0.01, dest="disc_lr",          help=("Defines the used learning rate "
                                                                                                                        "with the Discriminator. Default %(default)s"))
    gan_hp_group.add_argument("--discriminator-alpha",         type=float, default=0.99, dest="disc_alpha",       help=("Defines the smoothing constant used"
                                                                                                                        " with the Discriminators optimizer. Default %(default)s"))
    gan_hp_group.add_argument("--discriminator-momentum",      type=float, default=1e-3, dest="disc_momentum",    help=("Defines the momentum used with the "
                                                                                                                        "Discriminator's optimizer. Default %(default)s"))
    # <<<<< DDPG Hyperparameters >>>>>
    train_ddpg_hp_group = train_parser.add_argument_group("DDPG hyperparameters")
    _add_ddpg_hyperparameters(train_ddpg_hp_group)

    train_parser.set_defaults(func=_parse_and_train)
    

    # ----------------- CLI for evaluating the network ---------------------------------
    eval_parser = sub_parsers.add_parser("eval", description=("Evaluate the trained Agent. NOTE: The same hyperparameters that"
                                                             " were used during the training of the model must be specified also here."))
    
    eval_parser.add_argument("model_path",        type=str,  metavar="model-path", help=("Path to file containing the"
                                                                                        " agent model to be evaluated"))
    eval_parser.add_argument("--env",             type=str,  default=_ENV_NAME,    help=("The identifier of the used"
                                                                                        " environment. Default %(default)s"))
    eval_parser.add_argument("--eval-iter-count", type=int,  default=5,            help=("The amount of iterations the"
                                                                                        " agent is evaluated for. Default %(default)s"))
    eval_parser.add_argument("--timestep-count",  type=int,  default=500,          help=("The maximum amount timesteps the agent has to"
                                                                                        " find reach the goal during each episode. Default %(default)s"))
    eval_parser.add_argument("--episode-count",   type=int,  default=10,           help=("The amount of episodes each evaluation iteration"
                                                                                        " contains. Default %(default)s"))
    eval_parser.add_argument("--render",          type=bool, default=False,        help=("If set to true, the environment will be rendered on-screen"
                                                                                        " during each iteration. Default %(default)s"))

    # <<<<< DDPG Hyperparameters >>>>>
    eval_ddpg_hp_group = eval_parser.add_argument_group("DDPG hyperparameters")
    _add_ddpg_hyperparameters(eval_ddpg_hp_group)
    
    eval_parser.set_defaults(func=_parse_and_eval)

    # <<<<< If no command is given, print the help of the parser >>>>>>>
    parser.set_defaults(func=lambda args: args.parser.print_help(), parser=parser)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()    
    args.func(args)
