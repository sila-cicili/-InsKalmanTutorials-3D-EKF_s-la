# ins_em_MODIFIED.py
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from ins_ekf import ins_ext_kfilter

print("="*70)
print("🚀 3D KALMAN FİLTRESİ - ÖRNEK VERİ SETİ İLE TEST")
print("="*70)

# ============================================
# 1. PARAMETRELER DOSYASINI YÜKLEMESİ
# ============================================

print("\n📋 Parametreler Yükleniyor...")

with open('data/parameters_tuned_v1.json', 'r') as f:
    params = json.load(f)

print(f"✅ {params['experiment_name']} yüklendi")

# ============================================
# 2. VERİ DOSYALARINI YÜKLEMESİ
# ============================================

print("\n📊 Veriler Yükleniyor...")

imu_df = pd.read_csv(params['data_files']['imu_file'])
gps_df = pd.read_csv(params['data_files']['gps_file'])

print(f"✅ İMU verisi yüklendi: {len(imu_df)} ölçüm")
print(f"✅ GPS verisi yüklendi: {len(gps_df)} ölçüm")

# ============================================
# 3. VERİLERİ HAZIRLA
# ============================================

print("\n🔧 Veriler Hazırlanıyor...")

# İMU zamanı, ivmeler ve jiroskop
imu_time = imu_df['time'].values

imu_accel = []
imu_gyro = []
for idx, row in imu_df.iterrows():
    accel_matrix = np.matrix([
        [row['accel_x']],
        [row['accel_y']],
        [row['accel_z']]
    ])
    gyro_matrix = np.matrix([
        [row['gyro_x']],
        [row['gyro_y']],
        [row['gyro_z']]
    ])
    imu_accel.append(accel_matrix)
    imu_gyro.append(gyro_matrix)

# GPS zamanı, konumları ve hızları
gnss_time = gps_df['time'].values

gnss_dist = []
gnss_speed = []
for idx, row in gps_df.iterrows():
    dist_matrix = np.matrix([
        [row['position_x']],
        [row['position_y']],
        [row['position_z']]
    ])
    # Hız normu (magnitude)
    speed_norm = np.sqrt(
        row['velocity_x']**2 + 
        row['velocity_y']**2 + 
        row['velocity_z']**2
    )
    speed_matrix = np.matrix([[speed_norm]])
    
    gnss_dist.append(dist_matrix)
    gnss_speed.append(speed_matrix)

print(f"✅ İMU: {len(imu_accel)} accel + {len(imu_gyro)} gyro matris")
print(f"✅ GPS: {len(gnss_dist)} pos + {len(gnss_speed)} speed matris")

# ============================================
# 4. KALMAN FİLTRESİNİ ÇALIŞTIR
# ============================================

print("\n🔄 Kalman Filtresi Çalıştırılıyor...")

sensor_params = params['sensor_parameters']
initial_cond = params['initial_conditions']

try:
    # Başlangıç yönelim ve bias
    attitude0 = np.matrix([
        [initial_cond['initial_attitude'][0]],
        [initial_cond['initial_attitude'][1]],
        [initial_cond['initial_attitude'][2]]
    ])
    
    gyro_bias0 = np.matrix([
        [initial_cond['initial_gyro_bias'][0]],
        [initial_cond['initial_gyro_bias'][1]],
        [initial_cond['initial_gyro_bias'][2]]
    ])
    
    # Filtreyi çalıştır
    [state_list, var_list] = ins_ext_kfilter(
        imu_time,
        imu_accel,
        imu_gyro,
        sensor_params['accel_bias_std'],
        sensor_params['accel_noise_std'],
        sensor_params['gyro_bias_std'],
        sensor_params['gyro_noise_std'],
        attitude0,
        initial_cond['initial_attitude_std'],
        gyro_bias0,
        gnss_time,
        gnss_speed,
        gnss_dist,
        sensor_params['gps_speed_noise_std'],
        sensor_params['gps_position_noise_std']
    )
    
    print(f"✅ Filtre tamamlandı!")
    print(f"   {len(state_list)} durum tahmini yapıldı")
    
except Exception as e:
    print(f"❌ HATA: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================
# 5. SONUÇLARI ÇIKAR
# ============================================

print("\n📈 Sonuçlar İşleniyor...")

# State'ten değerleri çıkar (15 durum değişkeni)
ekf_pos_x = np.array([v.item((0, 0)) for v in state_list])
ekf_pos_y = np.array([v.item((1, 0)) for v in state_list])
ekf_pos_z = np.array([v.item((2, 0)) for v in state_list])

ekf_vel_x = np.array([v.item((3, 0)) for v in state_list])
ekf_vel_y = np.array([v.item((4, 0)) for v in state_list])
ekf_vel_z = np.array([v.item((5, 0)) for v in state_list])

ekf_bias_accel_x = np.array([v.item((6, 0)) for v in state_list])
ekf_bias_accel_y = np.array([v.item((7, 0)) for v in state_list])
ekf_bias_accel_z = np.array([v.item((8, 0)) for v in state_list])

ekf_bias_gyro_x = np.array([v.item((9, 0)) for v in state_list])
ekf_bias_gyro_y = np.array([v.item((10, 0)) for v in state_list])
ekf_bias_gyro_z = np.array([v.item((11, 0)) for v in state_list])

ekf_psi = np.array([v.item((12, 0)) for v in state_list])  # Yaw
ekf_theta = np.array([v.item((13, 0)) for v in state_list])  # Pitch
ekf_gamma = np.array([v.item((14, 0)) for v in state_list])  # Roll

# GPS verilerini al
gps_pos_x = gps_df['position_x'].values
gps_pos_y = gps_df['position_y'].values
gps_pos_z = gps_df['position_z'].values

gps_vel_x = gps_df['velocity_x'].values
gps_vel_y = gps_df['velocity_y'].values
gps_vel_z = gps_df['velocity_z'].values

print(f"✅ Veriler çıkarıldı (15 durum değişkeni)")

# ============================================
# 6. SONUÇLARI GÖRSELLEŞTİR
# ============================================

print("\n📊 Grafikler Çiziliyor...")

plt.rcParams['figure.figsize'] = (18, 14)
plt.rcParams['font.size'] = 9

fig = plt.figure(figsize=(18, 14))

# ─── ROW 1: KONUMlar ───
# Konum X
ax1 = plt.subplot(4, 4, 1)
ax1.plot(imu_time, ekf_pos_x, 'b-', linewidth=2, label='Tahmin')
ax1.scatter(gnss_time, gps_pos_x, color='red', s=20, label='GPS', zorder=5)
ax1.set_xlabel('Zaman (s)')
ax1.set_ylabel('X (m)')
ax1.set_title('X Konumu')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)

# Konum Y
ax2 = plt.subplot(4, 4, 2)
ax2.plot(imu_time, ekf_pos_y, 'b-', linewidth=2, label='Tahmin')
ax2.scatter(gnss_time, gps_pos_y, color='red', s=20, label='GPS', zorder=5)
ax2.set_xlabel('Zaman (s)')
ax2.set_ylabel('Y (m)')
ax2.set_title('Y Konumu')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)

# Konum Z
ax3 = plt.subplot(4, 4, 3)
ax3.plot(imu_time, ekf_pos_z, 'b-', linewidth=2, label='Tahmin')
ax3.scatter(gnss_time, gps_pos_z, color='red', s=20, label='GPS', zorder=5)
ax3.set_xlabel('Zaman (s)')
ax3.set_ylabel('Z (m)')
ax3.set_title('Z Konumu')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=8)

# 3D İz
ax4 = plt.subplot(4, 4, 4, projection='3d')
ax4.plot(ekf_pos_x, ekf_pos_y, ekf_pos_z, 'b-', linewidth=2, label='Tahmin')
ax4.scatter(gps_pos_x, gps_pos_y, gps_pos_z, color='red', s=20, label='GPS', zorder=5)
ax4.set_xlabel('X (m)')
ax4.set_ylabel('Y (m)')
ax4.set_zlabel('Z (m)')
ax4.set_title('3D Yol')
ax4.legend(fontsize=8)

# ─── ROW 2: HIZLAR ───
# Hız X
ax5 = plt.subplot(4, 4, 5)
ax5.plot(imu_time, ekf_vel_x, 'g-', linewidth=2, label='Tahmin')
ax5.scatter(gnss_time, gps_vel_x, color='orange', s=20, label='GPS', zorder=5)
ax5.set_xlabel('Zaman (s)')
ax5.set_ylabel('Vx (m/s)')
ax5.set_title('X Hızı')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=8)

# Hız Y
ax6 = plt.subplot(4, 4, 6)
ax6.plot(imu_time, ekf_vel_y, 'g-', linewidth=2, label='Tahmin')
ax6.scatter(gnss_time, gps_vel_y, color='orange', s=20, label='GPS', zorder=5)
ax6.set_xlabel('Zaman (s)')
ax6.set_ylabel('Vy (m/s)')
ax6.set_title('Y Hızı')
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=8)

# Hız Z
ax7 = plt.subplot(4, 4, 7)
ax7.plot(imu_time, ekf_vel_z, 'g-', linewidth=2, label='Tahmin')
ax7.scatter(gnss_time, gps_vel_z, color='orange', s=20, label='GPS', zorder=5)
ax7.set_xlabel('Zaman (s)')
ax7.set_ylabel('Vz (m/s)')
ax7.set_title('Z Hızı')
ax7.grid(True, alpha=0.3)
ax7.legend(fontsize=8)

# Hız Normu
ax8 = plt.subplot(4, 4, 8)
speed_total = np.sqrt(ekf_vel_x**2 + ekf_vel_y**2 + ekf_vel_z**2)
gps_speed_total = np.sqrt(gps_vel_x**2 + gps_vel_y**2 + gps_vel_z**2)
ax8.plot(imu_time, speed_total, 'g-', linewidth=2, label='Tahmin')
ax8.scatter(gnss_time, gps_speed_total, color='orange', s=20, label='GPS', zorder=5)
ax8.set_xlabel('Zaman (s)')
ax8.set_ylabel('Hız (m/s)')
ax8.set_title('Toplam Hız')
ax8.grid(True, alpha=0.3)
ax8.legend(fontsize=8)

# ─── ROW 3: SENSOR BİAS ───
# Accel Bias
ax9 = plt.subplot(4, 4, 9)
ax9.plot(imu_time, ekf_bias_accel_x, 'purple', linewidth=2, label='Bias X')
ax9.plot(imu_time, ekf_bias_accel_y, 'brown', linewidth=2, label='Bias Y')
ax9.plot(imu_time, ekf_bias_accel_z, 'pink', linewidth=2, label='Bias Z')
ax9.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax9.set_xlabel('Zaman (s)')
ax9.set_ylabel('Bias (m/s²)')
ax9.set_title('İvmeölçer Bias')
ax9.grid(True, alpha=0.3)
ax9.legend(fontsize=8)

# Gyro Bias
ax10 = plt.subplot(4, 4, 10)
ax10.plot(imu_time, ekf_bias_gyro_x, 'red', linewidth=2, label='Bias X')
ax10.plot(imu_time, ekf_bias_gyro_y, 'blue', linewidth=2, label='Bias Y')
ax10.plot(imu_time, ekf_bias_gyro_z, 'green', linewidth=2, label='Bias Z')
ax10.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax10.set_xlabel('Zaman (s)')
ax10.set_ylabel('Bias (rad/s)')
ax10.set_title('Jiroskop Bias')
ax10.grid(True, alpha=0.3)
ax10.legend(fontsize=8)

# ─── ROW 4: YÖNELIM (EULER AÇILARI) ───
# Psi (Yaw)
ax11 = plt.subplot(4, 4, 11)
ax11.plot(imu_time, ekf_psi, 'cyan', linewidth=2)
ax11.set_xlabel('Zaman (s)')
ax11.set_ylabel('Psi (rad)')
ax11.set_title('Yaw (Psi)')
ax11.grid(True, alpha=0.3)

# Theta (Pitch)
ax12 = plt.subplot(4, 4, 12)
ax12.plot(imu_time, ekf_theta, 'magenta', linewidth=2)
ax12.set_xlabel('Zaman (s)')
ax12.set_ylabel('Theta (rad)')
ax12.set_title('Pitch (Theta)')
ax12.grid(True, alpha=0.3)

# Gamma (Roll)
ax13 = plt.subplot(4, 4, 13)
ax13.plot(imu_time, ekf_gamma, 'lime', linewidth=2)
ax13.set_xlabel('Zaman (s)')
ax13.set_ylabel('Gamma (rad)')
ax13.set_title('Roll (Gamma)')
ax13.grid(True, alpha=0.3)

# İMU Raw Data
ax14 = plt.subplot(4, 4, 14)
ax14.plot(imu_time, imu_df['accel_x'], 'r-', alpha=0.7, label='Accel X')
ax14.plot(imu_time, imu_df['accel_y'], 'g-', alpha=0.7, label='Accel Y')
ax14.plot(imu_time, imu_df['accel_z'], 'b-', alpha=0.7, label='Accel Z')
ax14.set_xlabel('Zaman (s)')
ax14.set_ylabel('İvme (m/s²)')
ax14.set_title('Raw İvme')
ax14.grid(True, alpha=0.3)
ax14.legend(fontsize=8)

# ─── ROW 5: HATALAR ───
# Konum Hatası
ax15 = plt.subplot(4, 4, 15)
gps_indices = np.searchsorted(imu_time, gnss_time)
gps_indices = np.clip(gps_indices, 0, len(imu_time)-1)
error_x = gps_pos_x - ekf_pos_x[gps_indices]
error_y = gps_pos_y - ekf_pos_y[gps_indices]
error_z = gps_pos_z - ekf_pos_z[gps_indices]
error_mag = np.sqrt(error_x**2 + error_y**2 + error_z**2)

ax15.plot(gnss_time, error_x, 'r-', linewidth=2, label='X')
ax15.plot(gnss_time, error_y, 'g-', linewidth=2, label='Y')
ax15.plot(gnss_time, error_z, 'b-', linewidth=2, label='Z')
ax15.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax15.set_xlabel('Zaman (s)')
ax15.set_ylabel('Hata (m)')
ax15.set_title('Konum Hatası')
ax15.grid(True, alpha=0.3)
ax15.legend(fontsize=8)

# Konum Hatası Normu
ax16 = plt.subplot(4, 4, 16)
ax16.plot(gnss_time, error_mag, 'purple', linewidth=2)
ax16.fill_between(gnss_time, error_mag, alpha=0.3, color='purple')
ax16.set_xlabel('Zaman (s)')
ax16.set_ylabel('Hata (m)')
ax16.set_title('Konum Hatası Normu')
ax16.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/kalman_filter_results_3d.png', dpi=150, bbox_inches='tight')
print("✅ Grafik kaydedildi: results/kalman_filter_results_3d.png")

plt.show()

# ============================================
# 7. İSTATİSTİKSEL SONUÇLAR
# ============================================

print("\n" + "="*70)
print("📊 İSTATİSTİKSEL SONUÇLAR")
print("="*70)

print(f"\n🎯 3D Konum Hatası (GPS vs Filtre):")
print(f"   - X: ortalama={np.mean(error_x):.3f}m, std={np.std(error_x):.3f}m")
print(f"   - Y: ortalama={np.mean(error_y):.3f}m, std={np.std(error_y):.3f}m")
print(f"   - Z: ortalama={np.mean(error_z):.3f}m, std={np.std(error_z):.3f}m")
print(f"   - TOPLAM: ortalama={np.mean(error_mag):.3f}m, max={np.max(error_mag):.3f}m")

print(f"\n📈 Hız Özeti:")
print(f"   - Max Vx: {np.max(ekf_vel_x):.3f} m/s")
print(f"   - Max Vy: {np.max(ekf_vel_y):.3f} m/s")
print(f"   - Max Vz: {np.max(ekf_vel_z):.3f} m/s")
print(f"   - Max Toplam Hız: {np.max(speed_total):.3f} m/s")

print(f"\n⚙️ Sensor Bias Tahmini:")
print(f"   İvmeölçer:")
print(f"   - X: başlangıç={ekf_bias_accel_x[0]:.4f} → son={ekf_bias_accel_x[-1]:.4f} m/s²")
print(f"   - Y: başlangıç={ekf_bias_accel_y[0]:.4f} → son={ekf_bias_accel_y[-1]:.4f} m/s²")
print(f"   - Z: başlangıç={ekf_bias_accel_z[0]:.4f} → son={ekf_bias_accel_z[-1]:.4f} m/s²")
print(f"   Jiroskop:")
print(f"   - X: başlangıç={ekf_bias_gyro_x[0]:.4f} → son={ekf_bias_gyro_x[-1]:.4f} rad/s")
print(f"   - Y: başlangıç={ekf_bias_gyro_y[0]:.4f} → son={ekf_bias_gyro_y[-1]:.4f} rad/s")
print(f"   - Z: başlangıç={ekf_bias_gyro_z[0]:.4f} → son={ekf_bias_gyro_z[-1]:.4f} rad/s")

print(f"\n🎯 Yönelim Açıları (Son Değerler):")
print(f"   - Psi (Yaw):   {ekf_psi[-1]:.4f} rad ({ekf_psi[-1]*180/np.pi:.2f}°)")
print(f"   - Theta (Pitch): {ekf_theta[-1]:.4f} rad ({ekf_theta[-1]*180/np.pi:.2f}°)")
print(f"   - Gamma (Roll):  {ekf_gamma[-1]:.4f} rad ({ekf_gamma[-1]*180/np.pi:.2f}°)")

print(f"\n🔄 Zaman Senkronizasyonu:")
print(f"   - İMU periyotu: {imu_time[1]-imu_time[0]:.4f} s ({1/(imu_time[1]-imu_time[0]):.0f} Hz)")
print(f"   - GPS periyotu: {gnss_time[1]-gnss_time[0]:.2f} s ({1/(gnss_time[1]-gnss_time[0]):.0f} Hz)")
print(f"   - Senkronizasyon oranı: {len(gnss_time)}/{len(imu_time)} = %{100*len(gnss_time)/len(imu_time):.1f}")

print("\n" + "="*70)
print("✅ 3D KALMAN FİLTRE TEST TAMAMLANDI!")
print("="*70)