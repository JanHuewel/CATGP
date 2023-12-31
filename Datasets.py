from Data import Data
import pandas as pd
from scipy import io
import numpy as np
import torch

dataset_d1_solar = Data(file="data/lloyd_et_al/d1_solar_irradiance.csv", X_key="year", Y_key="cycle") # len: 391
dataset_d2_mauna = Data(file="data/lloyd_et_al/d2_maunaloa.csv", X_key="Decimal Date", Y_key="Carbon Dioxide (ppm)") # len: 702
#dataset_d3_powerplant = Data(file="data/lloyd_et_al/d3_powerplant.csv", X_key="__index__", Y_key="EP") # len: 9568
dataset_d4_gefcom = Data(file="data/lloyd_et_al/d4_gefcom.csv", X_key="timestamp", Y_key="load") # len: 38064
#dataset_d5_jena = Data(file="data/lloyd_et_al/d5_jena.csv", X_key="timestamp_long", Y_key="T (degC)") # len: 420551
dataset_d6_airline = Data(file="data/lloyd_et_al/d6_airline.csv", X_key="X", Y_key="y") # len: 144
dataset_d7_wheat = Data(file="data/lloyd_et_al/d7_wheat.csv", X_key="X", Y_key="y") # len: 370
dataset_d8_temperature = Data(file="data/lloyd_et_al/d8_temperature.csv", X_key="X", Y_key="y") # len: 1000
dataset_d9_internet = Data(file="data/lloyd_et_al/d9_internet.csv", X_key="X", Y_key="y") # len: 1000
dataset_d10_call_centre = Data(file="data/lloyd_et_al/d10_call-centre.csv", X_key="X", Y_key="y") # len: 180
dataset_d11_radio = Data(file="data/lloyd_et_al/d11_radio.csv", X_key="X", Y_key="y") # len: 240
dataset_d12_gas_production = Data(file="data/lloyd_et_al/d12_gas-production.csv", X_key="X", Y_key="y") # len: 476
dataset_d13_sulphuric = Data(file="data/lloyd_et_al/d13_sulphuric.csv", X_key="X", Y_key="y") # len: 462
dataset_d14_unemployment = Data(file="data/lloyd_et_al/d14_unemployment.csv", X_key="X", Y_key="y") # len: 408
dataset_d15_births = Data(file="data/lloyd_et_al/d15_births.csv", X_key="X", Y_key="y") # len: 1000
dataset_d16_wages = Data(file="data/lloyd_et_al/d16_wages.csv", X_key="X", Y_key="y") # len: 735
dataset_d17_airquality = Data(file="data/lloyd_et_al/d17_airquality.csv", X_key="timestamp_long", Y_key="COx") # len: 7718


dataset_temperature = Data(file="data/d8_temperature.csv", X_key="X", Y_key="y") # len: 1000
dataset_cashier = Data(file="data/CashierData.csv", X_key="timestamp", Y_key="mean_temp") # len: 1354
dataset_delhi = Data(file="data/DailyDelhiClimate_2.csv", X_key="index", Y_key="humidity") # len: 1462
dataset_methane = Data(file="data/04-methane.mat", X_key="X", Y_key="y") # len: 4915
dataset_powerplant = Data(file="data/d3_powerplant.csv", X=torch.linspace(-1,1,9568), Y_key="EP") #len: 9568



# -------------------------------------------------------------------------
# GENERATED DATA
# -------------------------------------------------------------------------
dataset_simul_noiseless_1cp_1 = Data(file="data/generated_data/Small_noise/single_changes/short-season_low-ampl_wn-small_cp500_small-trend.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_noiseless_1cp_2 = Data(file="data/generated_data/Small_noise/single_changes/short-season_low-ampl_wn-small_cp600-650_long-season_low-ampl.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_noiseless_1cp_3 = Data(file="data/generated_data/Small_noise/single_changes/short-season_low-ampl_wn-small_cp1000-1050_short-season_high-ampl.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_noiseless_1cp_4 = Data(file="data/generated_data/Small_noise/single_changes/short-season_low-ampl_wn-small_trend-small_cp500_trend-big.csv",
                       X_key="__index__", Y_key="value")

dataset_simul_noiseless_2cp_1 = Data(file="data/generated_data/Small_noise/multi_changes/multi-trend-period-amplitude-changes.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_noiseless_2cp_2 = Data(file="data/generated_data/Small_noise/multi_changes/short-season_low-ampl__cp500_short-season_high-ampl_cp1250_long-season_high-ampl.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_noiseless_2cp_3 = Data(file="data/generated_data/Small_noise/multi_changes/short-season_low-ampl_big-trend_cp500_small-trend_cp1250_neg-trend_cp1500_small-trend.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_noiseless_2cp_4 = Data(file="data/generated_data/Small_noise/multi_changes/short-season_low-ampl_trend_small_cp500_big-trend_long-season_cp1250_no-trend.csv",
                       X_key="__index__", Y_key="value")

dataset_simul_noisy_1cp_1 = Data(file="data/generated_data/Big_noise/single_changes/short-season_low-ampl_wn-small_cp500_small-trend.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_noisy_1cp_2 = Data(file="data/generated_data/Big_noise/single_changes/short-season_low-ampl_wn-small_cp600-650_long-season_low-ampl.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_noisy_1cp_3 = Data(file="data/generated_data/Big_noise/single_changes/short-season_low-ampl_wn-small_cp1000-1050_short-season_high-ampl.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_noisy_1cp_4 = Data(file="data/generated_data/Big_noise/single_changes/short-season_low-ampl_wn-small_trend-small_cp500_trend-big.csv",
                       X_key="__index__", Y_key="value")

dataset_simul_noisy_2cp_1 = Data(file="data/generated_data/Big_noise/multi_changes/multi-trend-period-amplitude-changes.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_noisy_2cp_2 = Data(file="data/generated_data/Big_noise/multi_changes/short-season_low-ampl__cp500_short-season_high-ampl_cp1250_long-season_high-ampl.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_noisy_2cp_3 = Data(file="data/generated_data/Big_noise/multi_changes/short-season_low-ampl_big-trend_cp500_small-trend_cp1250_neg-trend_cp1500_small-trend.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_noisy_2cp_4 = Data(file="data/generated_data/Big_noise/multi_changes/short-season_low-ampl_trend_small_cp500_big-trend_long-season_cp1250_no-trend.csv",
                       X_key="__index__", Y_key="value")

dataset_simul_varnoise_1cp_1 = Data(file="data/generated_data/Changing_noise/single_changes/short-season_low-ampl_cp500_small-trend.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_varnoise_1cp_2 = Data(file="data/generated_data/Changing_noise/single_changes/short-season_low-ampl_cp600-650_long-season_low-ampl.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_varnoise_1cp_3 = Data(file="data/generated_data/Changing_noise/single_changes/short-season_low-ampl_cp1000-1050_short-season_high-ampl.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_varnoise_1cp_4 = Data(file="data/generated_data/Changing_noise/single_changes/short-season_low-ampl_trend-small_cp500_trend-big.csv",
                       X_key="__index__", Y_key="value")

dataset_simul_varnoise_2cp_1 = Data(file="data/generated_data/Changing_noise/multi_changes/multi-trend-period-amplitude-changes.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_varnoise_2cp_2 = Data(file="data/generated_data/Changing_noise/multi_changes/short-season_low-ampl__cp500_short-season_high-ampl_cp1250_long-season_high-ampl.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_varnoise_2cp_3 = Data(file="data/generated_data/Changing_noise/multi_changes/short-season_low-ampl_big-trend_cp500_small-trend_cp1250_neg-trend_cp1500_small-trend.csv",
                       X_key="__index__", Y_key="value")
dataset_simul_varnoise_2cp_4 = Data(file="data/generated_data/Changing_noise/multi_changes/short-season_low-ampl_trend_small_cp500_big-trend_long-season_cp1250_no-trend.csv",
                       X_key="__index__", Y_key="value")