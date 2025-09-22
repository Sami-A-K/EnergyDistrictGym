import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from Energy_District_Gym_Environment import EnergyDistrictEnvironment

# Modell laden (ohne Env)
model = SAC.load("./output/models/sac_policy_dyn_cost", env=None)

# Test-Environment
env = EnergyDistrictEnvironment()
obs, info = env.reset(timestep=pd.to_datetime("2015-08-06 00:00:00"))

# Episode aufzeichnen
records = []
terminated, truncated = False, False
while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    row = {"timestep": info["timestep"]}

    # Net Load je Actor
    for actor in env.config["actors"]:
        name = actor["name"]

        el_load = env.network.loads_t.p_set.get(f"{name}_electrical_load", pd.Series(0)).get(info["timestep"], 0.0)
        hp_el = env.network.generators_t.p_set.get(f"{name}_heatpump_el", pd.Series(0)).get(info["timestep"], 0.0)
        pv = env.network.generators_t.p_set.get(f"{name}_pv", pd.Series(0)).get(info["timestep"], 0.0)

        row[f"{name}_net_load"] = float(pv - el_load - hp_el)
        row[f"{name}_hp_P"] = float(hp_el)

    # Batterie-Leistungen & SOCs
    for c in env.controllables:
        if c["type"] == "battery":
            row[f"{c['name']}_P"] = float(env.network.storage_units_t.p_set.loc[info["timestep"], c["name"]])
            soc = float(env.network.storage_units_t.state_of_charge_set.loc[info["timestep"], c["name"]])
            row[f"{c['name']}_SOC"] = soc / c["e_bat_max"]
        elif c["type"] == "heatpump":
            soc = float(env.network.storage_units_t.state_of_charge_set.loc[info["timestep"], c["con_thermal_storage"]])
            row[f"{c['con_thermal_storage']}_SOC"] = soc / c["e_ths_max"]

    # Faktoren
    row["supply_factor"] = env.weighting_factor_supply.loc[info["timestep"], "value"]
    row["feed_in_factor"] = env.weighting_factor_feed_in.loc[info["timestep"], "value"]

    records.append(row)

# DataFrame
df = pd.DataFrame(records)
df["timestep"] = pd.to_datetime(df["timestep"])
df = df.set_index("timestep").sort_index()

# Tagesbereich
day_start = df.index[0].normalize()
day_end = day_start + pd.Timedelta(hours=24)

plt.rcParams.update({
    "figure.figsize": (5.5, 7), 
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

fig, axes = plt.subplots(3, 1, sharex=True)


for col in df.filter(like="_net_load").columns:
    axes[0].plot(df.index, df[col], label=col)
for col in df.filter(like="_hp_P").columns:
    axes[0].plot(df.index, -df[col], label=col)
for col in df.filter(like="_battery_P").columns:
    axes[0].plot(df.index, df[col], label=col)
axes[0].set_ylabel("Power [kW]")
axes[0].grid(True)
axes[0].set_xlim([day_start, day_end])
# Legende oben drüber
axes[0].legend(frameon=True)

# 2) SOCs
for col in df.filter(like="_SOC").columns:
    axes[1].plot(df.index, df[col], label=col)
axes[1].set_ylabel("SOC [-]")
axes[1].grid(True)
axes[1].set_xlim([day_start, day_end])
# Legende unten drunter
axes[1].legend(frameon=True)

# 3) Faktoren
axes[2].plot(df.index, df["supply_factor"], label="Electricity Tariff", color="tab:blue")
axes[2].plot(df.index, df["feed_in_factor"], label="Feed-in Tariff", color="tab:orange")
axes[2].set_ylabel("Costs [ct/kWh]")
axes[2].grid(True)
axes[2].set_xlim([day_start, day_end])
# Legende rechts oben (außerhalb)
axes[2].legend(frameon=True)

fig.align_ylabels(axes)
plt.tight_layout()
plt.show()
