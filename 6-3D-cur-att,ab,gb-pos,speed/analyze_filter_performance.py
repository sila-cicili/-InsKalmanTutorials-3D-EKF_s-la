# analyze_filter_performance.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ins_ekf import ins_ext_kfilter
import json

print("="*70)
print("📊 KALMAN FİLTRE PERFORMANS ANALİZİ")
print("="*70)

# ============================================
# 1. VERİLERİ YÜKLEMESİ
# ============================================

with open('data/parameters.json', 'r') as f:
    params = json.load(f)

imu_df = pd.read_csv(params['data_files']['imu_file'])
gps_df = pd.read_csv(params['data_files']['gps_file'])

imu_time = imu_df['time'].values
gnss_time = gps_df['time'].values

# ============================================
# 2. HAZIRLIK VE FİLTRE ÇALIŞTIRMA
# ============================================

imu_accel = []
imu_gyro = []
for idx, row in imu_df.iterrows():
    accel_matrix = np.matrix([[row['accel_x']], [row['accel_y']], [row['accel_z']]])
    gyro_matrix = np.matrix([[row['gyro_x']], [row['gyro_y']], [row['gyro_z']]])
    imu_accel.append(accel_matrix)
    imu_gyro.append(gyro_matrix)

gnss_dist = []
gnss_speed = []
for idx, row in gps_df.iterrows():
    dist_matrix = np.matrix([[row['position_x']], [row['position_y']], [row['position_z']]])
    speed_norm = np.sqrt(row['velocity_x']**2 + row['velocity_y']**2 + row['velocity_z']**2)
    speed_matrix = np.matrix([[speed_norm]])
    gnss_dist.append(dist_matrix)
    gnss_speed.append(speed_matrix)

sensor_params = params['sensor_parameters']
initial_cond = params['initial_conditions']

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

print("\n🔄 Filtre çalıştırılıyor...")
[state_list, var_list] = ins_ext_kfilter(
    imu_time, imu_accel, imu_gyro,
    sensor_params['accel_bias_std'],
    sensor_params['accel_noise_std'],
    sensor_params['gyro_bias_std'],
    sensor_params['gyro_noise_std'],
    attitude0,
    initial_cond['initial_attitude_std'],
    gyro_bias0,
    gnss_time, gnss_speed, gnss_dist,
    sensor_params['gps_speed_noise_std'],
    sensor_params['gps_position_noise_std']
)

print(f"✅ Filtre çalıştırıldı: {len(state_list)} durum")

# ============================================
# 3. SONUÇLARI ÇIKAR
# ============================================

ekf_pos_x = np.array([v.item((0, 0)) for v in state_list])
ekf_pos_y = np.array([v.item((1, 0)) for v in state_list])
ekf_pos_z = np.array([v.item((2, 0)) for v in state_list])

ekf_vel_x = np.array([v.item((3, 0)) for v in state_list])
ekf_vel_y = np.array([v.item((4, 0)) for v in state_list])
ekf_vel_z = np.array([v.item((5, 0)) for v in state_list])

gps_pos_x = gps_df['position_x'].values
gps_pos_y = gps_df['position_y'].values
gps_pos_z = gps_df['position_z'].values

gps_vel_x = gps_df['velocity_x'].values
gps_vel_y = gps_df['velocity_y'].values
gps_vel_z = gps_df['velocity_z'].values

# ============================================
# 4. HATA HESAPLAMALARI
# ============================================

print("\n" + "="*70)
print("📈 PERFORMANS METRİKLERİ")
print("="*70)

# GPS zamanlarında filter tahmini al
gps_indices = np.searchsorted(imu_time, gnss_time)
gps_indices = np.clip(gps_indices, 0, len(imu_time)-1)

ekf_at_gps_x = ekf_pos_x[gps_indices]
ekf_at_gps_y = ekf_pos_y[gps_indices]
ekf_at_gps_z = ekf_pos_z[gps_indices]

ekf_vel_at_gps_x = ekf_vel_x[gps_indices]
ekf_vel_at_gps_y = ekf_vel_y[gps_indices]
ekf_vel_at_gps_z = ekf_vel_z[gps_indices]

# Hata vektörleri
error_pos_x = gps_pos_x - ekf_at_gps_x
error_pos_y = gps_pos_y - ekf_at_gps_y
error_pos_z = gps_pos_z - ekf_at_gps_z

error_vel_x = gps_vel_x - ekf_vel_at_gps_x
error_vel_y = gps_vel_y - ekf_vel_at_gps_y
error_vel_z = gps_vel_z - ekf_vel_at_gps_z

# Hata büyüklükleri
error_pos_mag = np.sqrt(error_pos_x**2 + error_pos_y**2 + error_pos_z**2)
error_vel_mag = np.sqrt(error_vel_x**2 + error_vel_y**2 + error_vel_z**2)

# ============================================
# 1. KONUM HATASI (RMSE - Root Mean Square Error)
# ============================================

print("\n🎯 1. KONUM HATASI (Position Error)")
print("-" * 70)

rmse_pos_x = np.sqrt(np.mean(error_pos_x**2))
rmse_pos_y = np.sqrt(np.mean(error_pos_y**2))
rmse_pos_z = np.sqrt(np.mean(error_pos_z**2))
rmse_pos_total = np.sqrt(np.mean(error_pos_mag**2))

mae_pos_x = np.mean(np.abs(error_pos_x))
mae_pos_y = np.mean(np.abs(error_pos_y))
mae_pos_z = np.mean(np.abs(error_pos_z))
mae_pos_total = np.mean(error_pos_mag)

max_error_pos = np.max(error_pos_mag)
min_error_pos = np.min(error_pos_mag)

print(f"📊 RMSE (Root Mean Square Error):")
print(f"   X: {rmse_pos_x:.3f} m")
print(f"   Y: {rmse_pos_y:.3f} m")
print(f"   Z: {rmse_pos_z:.3f} m")
print(f"   TOPLAM 3D: {rmse_pos_total:.3f} m ⭐")

print(f"\n📊 MAE (Mean Absolute Error):")
print(f"   X: {mae_pos_x:.3f} m")
print(f"   Y: {mae_pos_y:.3f} m")
print(f"   Z: {mae_pos_z:.3f} m")
print(f"   TOPLAM 3D: {mae_pos_total:.3f} m")

print(f"\n📊 HATA ARALIGI:")
print(f"   Min: {min_error_pos:.3f} m")
print(f"   Max: {max_error_pos:.3f} m")
print(f"   Fark: {max_error_pos - min_error_pos:.3f} m")

# ============================================
# 2. HIZ HATASI
# ============================================

print("\n" + "="*70)
print("🚀 2. HIZ HATASI (Velocity Error)")
print("-" * 70)

rmse_vel_x = np.sqrt(np.mean(error_vel_x**2))
rmse_vel_y = np.sqrt(np.mean(error_vel_y**2))
rmse_vel_z = np.sqrt(np.mean(error_vel_z**2))
rmse_vel_total = np.sqrt(np.mean(error_vel_mag**2))

mae_vel_x = np.mean(np.abs(error_vel_x))
mae_vel_y = np.mean(np.abs(error_vel_y))
mae_vel_z = np.mean(np.abs(error_vel_z))
mae_vel_total = np.mean(error_vel_mag)

print(f"📊 RMSE (Hız Hatası):")
print(f"   Vx: {rmse_vel_x:.3f} m/s")
print(f"   Vy: {rmse_vel_y:.3f} m/s")
print(f"   Vz: {rmse_vel_z:.3f} m/s")
print(f"   TOPLAM: {rmse_vel_total:.3f} m/s ⭐")

print(f"\n📊 MAE (Hız Hatası):")
print(f"   Vx: {mae_vel_x:.3f} m/s")
print(f"   Vy: {mae_vel_y:.3f} m/s")
print(f"   Vz: {mae_vel_z:.3f} m/s")
print(f"   TOPLAM: {mae_vel_total:.3f} m/s")

# ============================================
# 3. KALMAN FİLTRESİ KALİTESİ
# ============================================

print("\n" + "="*70)
print("⚙️ 3. KALMAN FİLTRESİ KALİTESİ")
print("-" * 70)

# Covariance analizi
P_final = var_list[-1]

# Köşegen elemanları al (matrix veya array olabilir)
if isinstance(P_final, np.matrix):
    p_diag = np.asarray(P_final.diagonal()).flatten()
else:
    p_diag = np.diag(P_final).flatten()

print(f"\n📊 Durum Kovaryansı (Son Değer):")
print(f"   Pos X: {np.sqrt(p_diag[0]):.6f} m")
print(f"   Pos Y: {np.sqrt(p_diag[1]):.6f} m")
print(f"   Pos Z: {np.sqrt(p_diag[2]):.6f} m")
print(f"   Vel X: {np.sqrt(p_diag[3]):.6f} m/s")
print(f"   Vel Y: {np.sqrt(p_diag[4]):.6f} m/s")
print(f"   Vel Z: {np.sqrt(p_diag[5]):.6f} m/s")

print(f"\n📊 Sensor Bias Belirsizliği:")
print(f"   Accel X: {np.sqrt(p_diag[6]):.6f} m/s²")
print(f"   Accel Y: {np.sqrt(p_diag[7]):.6f} m/s²")
print(f"   Accel Z: {np.sqrt(p_diag[8]):.6f} m/s²")
print(f"   Gyro X: {np.sqrt(p_diag[9]):.6f} rad/s")
print(f"   Gyro Y: {np.sqrt(p_diag[10]):.6f} rad/s")
print(f"   Gyro Z: {np.sqrt(p_diag[11]):.6f} rad/s")

# ============================================
# 4. OPTİMALLİK KONTROL (NIS - Normalized Innovation Squared)
# ============================================

print("\n" + "="*70)
print("✅ 4. FİLTRE OPTİMALLİĞİ KONTROL (NIS)")
print("-" * 70)

# Innovation (ölçüm - tahmin)
innovations = error_pos_mag  # GPS - Filtre farkı
measurement_std = params['sensor_parameters']['gps_position_noise_std']

# NIS = innovation² / measurement_variance
nis = innovations**2 / (measurement_std**2)
mean_nis = np.mean(nis)

print(f"\n📊 NIS (Normalized Innovation Squared):")
print(f"   Ortalama NIS: {mean_nis:.3f}")
print(f"   Min NIS: {np.min(nis):.3f}")
print(f"   Max NIS: {np.max(nis):.3f}")

# Optimal NIS değeri 1 civarında olmalı
if 0.5 < mean_nis < 2.0:
    print(f"   ✅ OPTIMAL: NIS kabul edilebilir aralıkta (0.5-2.0)")
elif mean_nis < 0.5:
    print(f"   ⚠️ UYARI: NIS çok düşük (filtre çok güveni yüksek)")
else:
    print(f"   ⚠️ UYARI: NIS çok yüksek (filtre iyi çalışmıyor)")

# ============================================
# 5. HATA TRENDİ ANALİZİ
# ============================================

print("\n" + "="*70)
print("📈 5. HATA TRENDİ ANALİZİ")
print("-" * 70)

# İlk 1/4, 2/4, 3/4, 4/4 periyotlarda hata
n_quarters = 4
quarter_len = len(error_pos_mag) // n_quarters

print(f"\n📊 Hata Zaman İçinde (4 eşit periyot):")
for i in range(n_quarters):
    start_idx = i * quarter_len
    end_idx = (i + 1) * quarter_len if i < n_quarters - 1 else len(error_pos_mag)
    
    quarter_error = error_pos_mag[start_idx:end_idx]
    quarter_mean = np.mean(quarter_error)
    quarter_std = np.std(quarter_error)
    
    print(f"   Periyot {i+1}: ortalama={quarter_mean:.3f}m, std={quarter_std:.3f}m")

# ============================================
# 6. KARŞILAŞTIRMA STANDARTLARI
# ============================================

print("\n" + "="*70)
print("🎯 6. PERFORMANS STANDARTLARI")
print("-" * 70)

# Orman harita oluştur
standards = {
    "Çok İyi": {"pos": 1.0, "vel": 0.1},
    "İyi": {"pos": 3.0, "vel": 0.3},
    "Kabul Edilebilir": {"pos": 5.0, "vel": 0.5},
    "Zayıf": {"pos": 10.0, "vel": 1.0},
}

print(f"\n📊 Standartlara Göre Değerlendirme:")
print(f"   KONUM RMSE: {rmse_pos_total:.3f} m")

rating = "Bilinmiyor"
if rmse_pos_total < standards["Çok İyi"]["pos"]:
    rating = "🟢 ÇOK İYİ"
elif rmse_pos_total < standards["İyi"]["pos"]:
    rating = "🟢 İYİ"
elif rmse_pos_total < standards["Kabul Edilebilir"]["pos"]:
    rating = "🟡 KABUL EDİLEBİLİR"
elif rmse_pos_total < standards["Zayıf"]["pos"]:
    rating = "🔴 ZAYIF"
else:
    rating = "❌ ÇOK ZAYIF"

print(f"   ⭐ SONUÇ: {rating}")

print(f"\n   HIZ RMSE: {rmse_vel_total:.3f} m/s")
rating_vel = "Bilinmiyor"
if rmse_vel_total < standards["Çok İyi"]["vel"]:
    rating_vel = "🟢 ÇOK İYİ"
elif rmse_vel_total < standards["İyi"]["vel"]:
    rating_vel = "🟢 İYİ"
elif rmse_vel_total < standards["Kabul Edilebilir"]["vel"]:
    rating_vel = "🟡 KABUL EDİLEBİLİR"
else:
    rating_vel = "🔴 ZAYIF"

print(f"   ⭐ SONUÇ: {rating_vel}")

# ============================================
# 7. DETAYLI RAPOR
# ============================================

print("\n" + "="*70)
print("📋 7. DETAYLI PERFORMANS RAPORU")
print("="*70)

report = f"""
🎯 FİLTRE PERFORMANS ÖZETI
{'='*70}

1️⃣  KONUM TAKİBİ
   - RMSE: {rmse_pos_total:.3f} m (Ortalama Hata)
   - MAE: {mae_pos_total:.3f} m
   - Max Hata: {max_error_pos:.3f} m
   - Değerlendirme: {rating}

2️⃣  HIZ TAKİBİ
   - RMSE: {rmse_vel_total:.3f} m/s
   - MAE: {mae_vel_total:.3f} m/s
   - Değerlendirme: {rating_vel}

3️⃣  FİLTRE YAKINSAMA
   - Başlangıç Hata: {error_pos_mag[0]:.3f} m
   - Son Hata: {error_pos_mag[-1]:.3f} m
   - İyileşme: {((error_pos_mag[0] - error_pos_mag[-1]) / error_pos_mag[0] * 100):.1f}%
   
4️⃣  OPTİMALLİK
   - Mean NIS: {mean_nis:.3f} (Optimal: 1.0)
   - Durum: {'✅ OPTIMAL' if 0.5 < mean_nis < 2.0 else '⚠️ UYARI'}

5️⃣  KALİTE GÖSTERGESI
   - Belirsizlik (σ_pos): {np.sqrt(p_diag[0]):.6f} m
   - Covaryans Konverjansı: {'✅ YETERLİ' if np.sqrt(p_diag[0]) < 0.1 else '⚠️ ZAYIF'}

{'='*70}
GENEL SONUÇ: Filtre {'✅ İYİ ÇALIŞIYOR' if rmse_pos_total < 5.0 else '⚠️ İYİLEŞTİRİLMESİ GEREKİYOR'}
{'='*70}
"""

print(report)

# ============================================
# 8. GRAFİKLERİ ÇİZ
# ============================================

print("\n📊 Grafikler Çiziliyor...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Konum Hatası
axes[0, 0].plot(gnss_time, error_pos_mag, 'b-', linewidth=2)
axes[0, 0].axhline(y=rmse_pos_total, color='r', linestyle='--', label=f'RMSE={rmse_pos_total:.3f}m')
axes[0, 0].fill_between(gnss_time, 0, error_pos_mag, alpha=0.3)
axes[0, 0].set_xlabel('Zaman (s)')
axes[0, 0].set_ylabel('Hata (m)')
axes[0, 0].set_title('Konum Hatası Zaman Serileri')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Hız Hatası
axes[0, 1].plot(gnss_time, error_vel_mag, 'g-', linewidth=2)
axes[0, 1].axhline(y=rmse_vel_total, color='r', linestyle='--', label=f'RMSE={rmse_vel_total:.3f}m/s')
axes[0, 1].fill_between(gnss_time, 0, error_vel_mag, alpha=0.3, color='green')
axes[0, 1].set_xlabel('Zaman (s)')
axes[0, 1].set_ylabel('Hata (m/s)')
axes[0, 1].set_title('Hız Hatası Zaman Serileri')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# NIS
axes[0, 2].plot(gnss_time, nis, 'purple', linewidth=2)
axes[0, 2].axhline(y=1.0, color='g', linestyle='--', label='Optimal (1.0)')
axes[0, 2].axhline(y=0.5, color='orange', linestyle=':', label='Alt Limit (0.5)')
axes[0, 2].axhline(y=2.0, color='orange', linestyle=':', label='Üst Limit (2.0)')
axes[0, 2].fill_between(gnss_time, 0.5, 2.0, alpha=0.2, color='green')
axes[0, 2].set_xlabel('Zaman (s)')
axes[0, 2].set_ylabel('NIS')
axes[0, 2].set_title('Normalized Innovation Squared')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Konum Hata Dağılımı (X, Y, Z)
axes[1, 0].plot(gnss_time, error_pos_x, 'r-', label='X', linewidth=2)
axes[1, 0].plot(gnss_time, error_pos_y, 'g-', label='Y', linewidth=2)
axes[1, 0].plot(gnss_time, error_pos_z, 'b-', label='Z', linewidth=2)
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].set_xlabel('Zaman (s)')
axes[1, 0].set_ylabel('Hata (m)')
axes[1, 0].set_title('Konum Hatası (X, Y, Z Bileşenleri)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Histogram - Konum Hatası
axes[1, 1].hist(error_pos_mag, bins=20, color='blue', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(x=rmse_pos_total, color='r', linestyle='--', linewidth=2, label=f'RMSE={rmse_pos_total:.3f}m')
axes[1, 1].axvline(x=mae_pos_total, color='g', linestyle='--', linewidth=2, label=f'MAE={mae_pos_total:.3f}m')
axes[1, 1].set_xlabel('Hata (m)')
axes[1, 1].set_ylabel('Frekans')
axes[1, 1].set_title('Konum Hatası Dağılımı')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Hata Trendi (4 periyot)
quarter_means = []
quarter_labels = []
for i in range(n_quarters):
    start_idx = i * quarter_len
    end_idx = (i + 1) * quarter_len if i < n_quarters - 1 else len(error_pos_mag)
    quarter_error = error_pos_mag[start_idx:end_idx]
    quarter_means.append(np.mean(quarter_error))
    quarter_labels.append(f'P{i+1}')

axes[1, 2].bar(quarter_labels, quarter_means, color='skyblue', edgecolor='black')
axes[1, 2].set_ylabel('Ortalama Hata (m)')
axes[1, 2].set_title('Hata Trendi (Zaman Periyotları)')
axes[1, 2].grid(True, alpha=0.3, axis='y')

# Değerleri bar üzerinde göster
for i, v in enumerate(quarter_means):
    axes[1, 2].text(i, v + 0.1, f'{v:.2f}m', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/filter_performance_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Grafik kaydedildi: results/filter_performance_analysis.png")

plt.show()

# ============================================
# 9. SONUÇ VE ÖNERİLER
# ============================================

print("\n" + "="*70)
print("💡 ÖNERİLER VE İYİLEŞTİRME TAVSIYELERI")
print("="*70)

recommendations = []

if rmse_pos_total > 5.0:
    recommendations.append("❌ Konum hatası çok yüksek. Parametreleri gözden geçir.")
elif rmse_pos_total > 3.0:
    recommendations.append("⚠️ Konum hatası orta düzeyde. GPS gürültüsünü azaltmayı dene.")
else:
    recommendations.append("✅ Konum hatası iyi seviyelerde.")

if mean_nis > 2.0:
    recommendations.append("⚠️ NIS çok yüksek: Ölçüm gürültüsü tahminini artır.")
elif mean_nis < 0.5:
    recommendations.append("⚠️ NIS çok düşük: Ölçüm gürültüsü tahminini azalt.")
else:
    recommendations.append("✅ NIS optimal aralıkta.")

if np.sqrt(p_diag[0]) > 0.1:
    recommendations.append("⚠️ Durum belirsizliği yüksek: Başlangıç koşullarını iyileştir.")

if rmse_vel_total > 0.5:
    recommendations.append("⚠️ Hız hatası yüksek: Jiroskop bias tahminini kontrol et.")

for rec in recommendations:
    print(f"\n{rec}")

print("\n" + "="*70)
print("✅ PERFORMANS ANALİZİ TAMAMLANDI!")
print("="*70)