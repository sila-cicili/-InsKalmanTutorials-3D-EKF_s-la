# create_sample_data.py
import numpy as np
import pandas as pd
import json
import os

print("="*60)
print("📊 3D ÖRNEK VERİ SETİ OLUŞTURUCU")
print("="*60)

os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("\n✅ Klasörler oluşturuldu")

# Zaman parametreleri
imu_frequency = 100  # Hz
gps_frequency = 2    # Hz
total_duration = 30  # saniye

imu_dt = 1 / imu_frequency
gps_dt = 1 / gps_frequency

imu_time = np.arange(0, total_duration, imu_dt)
gps_time = np.arange(gps_dt, total_duration + gps_dt, gps_dt)

print(f"\n📱 İMU Verisi: {len(imu_time)} ölçüm @ {imu_frequency}Hz")
print(f"📡 GPS Verisi: {len(gps_time)} ölçüm @ {gps_frequency}Hz")

# ============================================
# 3D HAREKET MODELİ
# ============================================

print("\n🚗 3D Hareket Modeli Oluşturuluyor...")

positions_x = []
positions_y = []
positions_z = []
velocities_x = []
velocities_y = []
velocities_z = []
accelerations_x = []
accelerations_y = []
accelerations_z = []
gyro_x_list = []
gyro_y_list = []
gyro_z_list = []

pos_x, pos_y, pos_z = 0.0, 0.0, 0.0
vel_x, vel_y, vel_z = 0.0, 0.0, 0.0
roll, pitch, yaw = 0.0, 0.0, 0.0

for t in imu_time:
    # İVME
    if t < 5:
        acc_x = 0.5 * np.sin(t)
        acc_y = 0.1 * np.cos(t * 0.5)
        acc_z = 0.0
    elif t < 10:
        acc_x = 0.3
        acc_y = 0.4 * np.sin((t - 5) * np.pi / 5)
        acc_z = 0.05
    elif t < 15:
        acc_x = 0.0
        acc_y = 0.0
        acc_z = 0.0
    elif t < 20:
        acc_x = -0.3
        acc_y = -0.3 * np.sin((t - 15) * np.pi / 5)
        acc_z = -0.05
    else:
        acc_x = -0.2 * np.cos(t * 0.2)
        acc_y = -0.1 * np.sin(t * 0.3)
        acc_z = 0.0
    
    # JIROSKOP (açısal hız)
    gyro_x = 0.02 * np.sin(t * 0.3)
    gyro_y = 0.01 * np.cos(t * 0.2)
    gyro_z = 0.05 * np.sin(t * 0.1)
    
    # Kinematik güncelleme
    vel_x += acc_x * imu_dt
    vel_y += acc_y * imu_dt
    vel_z += acc_z * imu_dt
    pos_x += vel_x * imu_dt
    pos_y += vel_y * imu_dt
    pos_z += vel_z * imu_dt
    
    roll += gyro_x * imu_dt
    pitch += gyro_y * imu_dt
    yaw += gyro_z * imu_dt
    
    accelerations_x.append(acc_x)
    accelerations_y.append(acc_y)
    accelerations_z.append(acc_z)
    velocities_x.append(vel_x)
    velocities_y.append(vel_y)
    velocities_z.append(vel_z)
    positions_x.append(pos_x)
    positions_y.append(pos_y)
    positions_z.append(pos_z)
    gyro_x_list.append(gyro_x)
    gyro_y_list.append(gyro_y)
    gyro_z_list.append(gyro_z)

print("✅ Hareket modeli oluşturuldu")

# ============================================
# SENSOR GÜRÜLTÜSÜ VE BİAS
# ============================================

print("\n🔧 Sensor Gürültüsü Ekleniyor...")

accel_bias_x = 0.15
accel_bias_y = -0.08
accel_bias_z = 0.05
accel_noise_std = 0.03

gyro_bias_x = 0.01
gyro_bias_y = -0.005
gyro_bias_z = 0.01
gyro_noise_std = 0.005

imu_accel_x_noisy = np.array(accelerations_x) + accel_bias_x + np.random.normal(0, accel_noise_std, len(imu_time))
imu_accel_y_noisy = np.array(accelerations_y) + accel_bias_y + np.random.normal(0, accel_noise_std, len(imu_time))
imu_accel_z_noisy = np.array(accelerations_z) + accel_bias_z + np.random.normal(0, accel_noise_std, len(imu_time))

imu_gyro_x_noisy = np.array(gyro_x_list) + gyro_bias_x + np.random.normal(0, gyro_noise_std, len(imu_time))
imu_gyro_y_noisy = np.array(gyro_y_list) + gyro_bias_y + np.random.normal(0, gyro_noise_std, len(imu_time))
imu_gyro_z_noisy = np.array(gyro_z_list) + gyro_bias_z + np.random.normal(0, gyro_noise_std, len(imu_time))

gps_position_noise_std = 3.0
gps_speed_noise_std = 0.3

# GPS ölçümleri (zaman aralığında)
gps_indices = (np.searchsorted(imu_time, gps_time)).clip(0, len(imu_time)-1)
gps_pos_x_clean = np.array(positions_x)[gps_indices]
gps_pos_y_clean = np.array(positions_y)[gps_indices]
gps_pos_z_clean = np.array(positions_z)[gps_indices]

gps_pos_x_noisy = gps_pos_x_clean + np.random.normal(0, gps_position_noise_std, len(gps_time))
gps_pos_y_noisy = gps_pos_y_clean + np.random.normal(0, gps_position_noise_std, len(gps_time))
gps_pos_z_noisy = gps_pos_z_clean + np.random.normal(0, gps_position_noise_std, len(gps_time))

gps_vel_x_clean = np.array(velocities_x)[gps_indices]
gps_vel_y_clean = np.array(velocities_y)[gps_indices]
gps_vel_z_clean = np.array(velocities_z)[gps_indices]

gps_vel_x_noisy = gps_vel_x_clean + np.random.normal(0, gps_speed_noise_std, len(gps_time))
gps_vel_y_noisy = gps_vel_y_clean + np.random.normal(0, gps_speed_noise_std, len(gps_time))
gps_vel_z_noisy = gps_vel_z_clean + np.random.normal(0, gps_speed_noise_std, len(gps_time))

print(f"   ✓ İvmeölçer: Bias=[{accel_bias_x:.2f}, {accel_bias_y:.2f}, {accel_bias_z:.2f}]")
print(f"   ✓ Jiroskop: Bias=[{gyro_bias_x:.3f}, {gyro_bias_y:.3f}, {gyro_bias_z:.3f}]")

# ============================================
# İMU VERİ DOSYASI
# ============================================

print("\n💾 İMU Veri Dosyası Oluşturuluyor...")

imu_data = pd.DataFrame({
    'time': imu_time,
    'accel_x': imu_accel_x_noisy,
    'accel_y': imu_accel_y_noisy,
    'accel_z': imu_accel_z_noisy,
    'gyro_x': imu_gyro_x_noisy,
    'gyro_y': imu_gyro_y_noisy,
    'gyro_z': imu_gyro_z_noisy
})

imu_data.to_csv('data/imu_data.csv', index=False)
print(f"✅ Kaydedildi: data/imu_data.csv ({len(imu_data)} satır)")

# ============================================
# GPS VERİ DOSYASI
# ============================================

print("\n💾 GPS Veri Dosyası Oluşturuluyor...")

gps_data = pd.DataFrame({
    'time': gps_time,
    'position_x': gps_pos_x_noisy,
    'position_y': gps_pos_y_noisy,
    'position_z': gps_pos_z_noisy,
    'velocity_x': gps_vel_x_noisy,
    'velocity_y': gps_vel_y_noisy,
    'velocity_z': gps_vel_z_noisy
})

gps_data.to_csv('data/gps_data.csv', index=False)
print(f"✅ Kaydedildi: data/gps_data.csv ({len(gps_data)} satır)")

# ============================================
# PARAMETRELER DOSYASI
# ============================================

print("\n⚙️ Parametreler Dosyası Oluşturuluyor...")

parameters = {
    "experiment_name": "3D Araç Navigasyonu",
    "description": "3D IMU + GPS fusion testi",
    "data_files": {
        "imu_file": "data/imu_data.csv",
        "gps_file": "data/gps_data.csv"
    },
    "filter_config": {
        "filter_type": "ekf_3d",
        "state_dimension": 15,
        "measurement_imu_dimension": 6,
        "measurement_gps_dimension": 4
    },
    "sensor_parameters": {
        "accel_bias_std": 0.2,
        "accel_noise_std": accel_noise_std,
        "gyro_bias_std": 0.01,
        "gyro_noise_std": gyro_noise_std,
        "gps_position_noise_std": gps_position_noise_std,
        "gps_speed_noise_std": gps_speed_noise_std
    },
    "initial_conditions": {
        "initial_attitude": [0.0, 0.0, 0.0],
        "initial_attitude_std": 0.1,
        "initial_gyro_bias": [0.0, 0.0, 0.0]
    },
    "kalman_tuning": {
        "process_noise_q": 0.0,
        "measurement_noise_r_position": gps_position_noise_std**2,
        "measurement_noise_r_velocity": gps_speed_noise_std**2
    },
    "time_parameters": {
        "imu_sampling_period": imu_dt,
        "gps_sampling_period": gps_dt,
        "simulation_duration": total_duration
    }
}

with open('data/parameters.json', 'w') as f:
    json.dump(parameters, f, indent=2)

print(f"✅ Kaydedildi: data/parameters.json")

# ============================================
# ÖZET
# ============================================

print("\n" + "="*60)
print("📊 3D VERİ SETİ ÖZETI")
print("="*60)

print(f"\n📱 İMU (6 kanallı):")
print(f"   {len(imu_time)} ölçüm, Accel + Gyro")
print(f"   Accel X: [{imu_accel_x_noisy.min():.2f}, {imu_accel_x_noisy.max():.2f}] m/s²")
print(f"   Gyro X:  [{imu_gyro_x_noisy.min():.3f}, {imu_gyro_x_noisy.max():.3f}] rad/s")

print(f"\n📡 GPS (6 kanallı):")
print(f"   {len(gps_time)} ölçüm, Position + Velocity")
print(f"   Pos X: [{gps_pos_x_noisy.min():.1f}, {gps_pos_x_noisy.max():.1f}] m")
print(f"   Vel X: [{gps_vel_x_noisy.min():.2f}, {gps_vel_x_noisy.max():.2f}] m/s")

print("\n" + "="*60)
print("✅ 3D VERİ SETİ HAZIR!")
print("✅ Şimdi ins_em_MODIFIED.py çalıştır!")
print("="*60)