from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from Energy_District_Gym_Environment import EnergyDistrictEnvironment


def make_env(rank: int = 0):
    def _init():
        env = EnergyDistrictEnvironment()
        env = Monitor(env, filename=f"./logs/monitor/monitor_{rank}")
        return env
    return _init


if __name__ == "__main__":
    N_ENVS = 16 # parallele Environments
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    #env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    eval_env = SubprocVecEnv([make_env(999)])
    #eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    eval_env.training = False
    eval_env.norm_reward = False

    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     device="cpu",
    #     n_steps=96,
    #     batch_size=256,
    #     learning_rate=3e-4,
    #     ent_coef=0.1,
    #     gamma=0.999,
    #     gae_lambda=0.95,
    #     n_epochs=10,
    #     vf_coef=0.5,
    #     clip_range=0.2,
    #     verbose=1,
    #     tensorboard_log="./logs/tb/"
    # )
    model = SAC(
        "MlpPolicy",
        env,
        device="auto",
        learning_rate=3e-4,
        buffer_size=1000000,   # Replay Buffer
        learning_starts=10000, # Schritte bevor Training startet
        batch_size=256,
        tau=0.005,
        gamma=0.999,           
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",       
        verbose=1,
        tensorboard_log="./logs/tb/"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/eval/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=1000000, callback=eval_callback)
    model.save("./output/models/sac_energy")

    # Normalisierungs-Statistiken sichern
    #env.save("./models/vecnormalize.pkl")

