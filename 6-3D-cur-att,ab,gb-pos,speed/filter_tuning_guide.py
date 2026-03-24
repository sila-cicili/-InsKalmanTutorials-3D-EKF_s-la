# filter_tuning_guide.py
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from ins_ekf import ins_ext_kfilter

print("="*80)
print("🎯 KALMAN FİLTRE İYİLEŞTİRME REHBERİ - ADIM ADIM")
print("="*80)

# ============================================
# ADIM 1: TEMEL PARAMETRELER
# ============================================

print("\n" + "="*80)
print("ADIM 1: SENSOR GÜRÜLTÜLERİNİ ÖLÇMEK")
print("="*80)

guide_step1 = """
🔬 SENSOR GÜRÜLTÜLERİ NASIL BELİRLENİR?

1. GPS HATASI (gps_position_noise_std):
   ├─ GPS spesifikasyonunda bak (datasheet)
   ├─ Örnek: Smartphone GPS: ±5-10m
   ├─ Profesyonel GPS: ±1-2m
   └─ AYARLAMA: GPS 5-10m hata alıyorsa → 5.0-10.0 gir

2. İMU GÜRÜLTÜSÜ (accel_noise_std):
   ├─ İvmeölçer specs'inde "Noise Density" bul
   ├─ Örnek: 400 µg/√Hz = 0.0004 m/s²
   ├─ Test: Sensörü hareketsiz tut, standart sapmasını ölç
   └─ AYARLAMA: 0.01-0.05 m/s² arası

3. JİROSKOP GÜRÜLTÜSÜ (gyro_noise_std):
   ├─ Jiroskop specs'inde "Angle Random Walk" bul
   ├─ Test: 1 saat hareketsiz tut, drift'i ölç
   └─ AYARLAMA: 0.001-0.01 rad/s arası

✅ SONUÇ: Gerçek sensor verilerinden ayar yap!
"""

print(guide_step1)

# ============================================
# ADIM 2: BIAS STANDART SAPMALARI
# ============================================

print("\n" + "="*80)
print("ADIM 2: SENSOR BİAS STANDART SAPMALARI")
print("="*80)

guide_step2 = """
🔍 BİAS NEDİR VE NASIL AYARLANIR?

BİAS = Sensörün SABİT SAPMA (offset)
      Örnek: Teoride 0 ama her zaman +0.1 ölçüyor

1. IVMEÖLÇER BİAS (accel_bias_std):
   ├─ Sensörü kalibre etmeden önce sapması
   ├─ Tipik: 0.1-0.5 m/s²
   ├─ AYARLAMA KURALARI:
   │  └─ Kalibre edilmiş sensör → 0.1-0.2
   │  └─ Ucuz sensör → 0.3-0.5
   │  └─ Profesyonel → 0.01-0.05
   └─ ARACA ÖZEL: Sıcaklık değişimi nedeniyle bias değişir!

2. JİROSKOP BİAS (gyro_bias_std):
   ├─ Jiroskopun açısal hız sabitlemesi
   ├─ Tipik: 0.01-0.1 rad/s
   ├─ AYARLAMA:
   │  └─ Başlangıçta 0.05-0.1 kullan
   │  └─ Filtre yakinsarsa değer azalabilir
   └─ ÖNEMLİ: Jiroskop drift'i (zaman içinde hata birikimi)

✅ EN ÖNEMLİ: Sensörünün gerçek bias'ını ölç!
   → Testlerin sonunda bias değeri frame'de gösteriliyor
   → Bunu next parametresine gir!
"""

print(guide_step2)

# ============================================
# ADIM 3: BAŞARILI AYAR STRATEJİSİ
# ============================================

print("\n" + "="*80)
print("ADIM 3: SİSTEMATİK AYARLAMA (TUNING) STRATEJİSİ")
print("="*80)

tuning_strategy = """
🎯 5-ADIMLI TUNİNG PROSESI

ADIM A: GPS HATASINI AYARLA (En kritik!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Eğer RMSE(konum) > 5m ise:

1. GPS hatası çok yüksek mi kontrol et:
   
   TEST 1: GPS verilerini kendi aralarında karşılaştır
   - max(GPS_X) - min(GPS_X) = ?
   - Eğer range > 50m ise GPS çok kötü

2. GPS_NOISE parametresini azalt:
   
   Iteratif Azaltma:
   ┌─────────────┬──────────────────┐
   │ GPS hatası  │ gps_noise_std    │
   ├─────────────┼──────────────────┤
   │ >10m        │ 10.0             │
   │ 5-10m       │ 5.0              │
   │ 2-5m        │ 2.0              │
   │ 1-2m        │ 1.0              │
   │ <1m         │ 0.5              │
   └─────────────┴──────────────────┘

ADIM B: İMU GÜRÜLTÜSÜNÜ AYARLA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hız hatası hala > 0.5 m/s ise:

1. accel_noise_std azalt:
   
   Current > 0.05 m/s² ise:
   ├─ 0.05 → 0.03 (1/3 azalt)
   ├─ 0.03 → 0.01 (3x azalt)
   └─ Test et

2. gyro_noise_std azalt:
   
   Current > 0.01 rad/s ise:
   ├─ 0.01 → 0.005 (2x azalt)
   ├─ 0.005 → 0.002 (2.5x azalt)
   └─ Test et

ADIM C: BİAS STANDART SAPMASINI AYARLA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sensor bias tahmini zayıfsa:

1. accel_bias_std artır:
   
   Eğer tahmini bias salınıyorsa:
   ���─ Başlangıç: 0.2
   ├─ Artır: 0.3 → 0.5 → 1.0
   └─ Test et

2. gyro_bias_std artır:
   
   Yönelim kararsızsa:
   ├─ Başlangıç: 0.01
   ├─ Artır: 0.05 → 0.1 → 0.2
   └─ Test et

ADIM D: BAŞLANGIÇ BELIRSIZLIKLERI AYARLA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Başlangıçta hata çok büyükse:

1. initial_attitude_std artır:
   
   Yönelim belirsizliği:
   ├─ Başlangıç: 0.1 rad
   ├─ Artır: 0.2 → 0.5 → 1.0
   └─ Test et

2. İlk 5-10 saniyeyi izle:
   ├─ Hata hızla düşüyor mu? → İyi
   ├─ Hata sabit mi? → Zayıf
   └─ Hata artıyor mu? → Çok zayıf

ADIM E: PROSES GÜRÜLTÜSÜ (Q matrisi)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sistem modeli uyumsuzsa:

Eğer Filtre modele çok güveniyorsa:
├─ accel_w_std artır (0.1 → 0.5)
├─ gyro_w_std artır (0.01 → 0.05)
└─ Test et
"""

print(tuning_strategy)

# ============================================
# ADIM 4: AYARLAMA TABLOSU
# ============================================

print("\n" + "="*80)
print("ADIM 4: HIZLI AYARLAMA REFERANS TABLOSU")
print("="*80)

tuning_table = """
📊 PROBLEM VE ÇÖZÜM TABLOSU

┌─────────────────────────────────────────────────────────────────────────┐
│ SORUN                          │ NEDENİ                    │ ÇÖZÜM       │
├─────────────────────────────────────────────────────────────────────────┤
│ Konum hatası > 10m              │ GPS noise çok yüksek      │ ⬇️ GPS std  │
│ Konum hatası salınıyor          │ Filtre GPS'e çok güvenip  │ ⬆️ GPS std  │
│                                 │ GPS'e güvenmiyor          │             │
├─────────────────────────────────────────────────────────────────────────┤
│ Hız hatası > 1 m/s              │ İMU noise çok yüksek      │ ⬇️ Accel st │
│ Hız salınıyor                   │ Jiroskop drift            │ ⬆️ Gyro std │
├─────────────────────────────────────────────────────────────────────────┤
│ Yönelim kararsız (spiky)        │ Gyro bias zayıf tahmini   │ ⬆️ Gyro_b   │
│ Yönelim gerililiyor             │ Gyro bias çok güvenilip   │ ⬇️ Gyro_b   │
├─────────────────────────────────────────────────────────────────────────┤
│ Başlangıçta çok kötü            │ İlk belirsizlik düşük     │ ⬆️ Att_std  │
│ Yakinsama çok yavaş             │ Sensor bias belirsizliği  │ ⬆️ Bias_st  │
├─────────────────────────────────────────────────────────────────────────┤
│ NIS > 2.0                       │ Ölçüm modeli yanlış       │ ⬆️ Noise st │
│ NIS < 0.5                       │ Filtre çok güvenli        │ ⬇️ Noise st │
├─────────────────────────────────────────────────────────────────────────┤
│ Divergence (hata artıyor)       │ Model uyumsuzluğu         │ ⬆️ Q matrix │
│ Filtre uyum kaybı               │ İlk tahmin çok kötü       │ ⬆️ P0 matrx │
└─────────────────────────────────────────────────────────────────────────┘

Simgeler:
⬆️ = ARTTIR    ⬇️ = AZALT    std = standard deviation
"""

print(tuning_table)

# ============================================
# ADIM 5: ÖNERILEN BAŞLANGIÇ DEĞERLERİ
# ============================================

print("\n" + "="*80)
print("ADIM 5: FARKLI SENARYOLAR İÇİN ÖNERİLEN PARAMETRELER")
print("="*80)

scenarios = {
    "Düşük Maliyetli Sensor (Smartphone)": {
        "accel_bias_std": 0.5,
        "accel_noise_std": 0.05,
        "gyro_bias_std": 0.1,
        "gyro_noise_std": 0.01,
        "gps_position_noise_std": 10.0,
        "gps_speed_noise_std": 1.0,
        "initial_attitude_std": 0.5,
    },
    
    "Orta Seviye Sensor (Drone)": {
        "accel_bias_std": 0.3,
        "accel_noise_std": 0.02,
        "gyro_bias_std": 0.05,
        "gyro_noise_std": 0.005,
        "gps_position_noise_std": 5.0,
        "gps_speed_noise_std": 0.5,
        "initial_attitude_std": 0.3,
    },
    
    "Profesyonel Sensor (RTK-GPS)": {
        "accel_bias_std": 0.1,
        "accel_noise_std": 0.005,
        "gyro_bias_std": 0.01,
        "gyro_noise_std": 0.001,
        "gps_position_noise_std": 1.0,
        "gps_speed_noise_std": 0.1,
        "initial_attitude_std": 0.1,
    },
}

for scenario_name, params in scenarios.items():
    print(f"\n📱 {scenario_name}")
    print("-" * 80)
    for param, value in params.items():
        print(f"   {param}: {value}")

# ============================================
# ADIM 6: İTERATİF AYARLAMA PROTOKOLÜ
# ============================================

print("\n" + "="*80)
print("ADIM 6: İTERATİF AYARLAMA PROTOKOLÜ")
print("="*80)

protocol = """
🔄 AYARLAMA DÖNGÜSÜ

Iteration 0 (Başlangıç):
├─ Senaryo seç (üstteki tablodan)
└─ parameters.json'u güncelle

Iteration 1:
├─ Çalıştır: python ins_em_MODIFIED.py
├─ Analiz: python analyze_filter_performance.py
├─ Sonuç:
│  ├─ RMSE(konum) ölç
│  ├─ RMSE(hız) ölç
│  └─ NIS ortalaması ölç
└─ Karar ver: Hangi parametreyi değiştir?

Iteration 2:
├─ En kötü metriği belirle:
│  ├─ RMSE(konum) > 5m? → GPS noise azalt
│  ├─ RMSE(hız) > 0.5? → Accel/Gyro noise azalt
│  └─ NIS > 2.0? → Sensor noise artır
├─ 1-2 parametre değiştir (aynı anda max 2)
└─ GOTO Iteration 1

Bu işlemi 5-10 kez tekrar et
"""

print(protocol)

# ============================================
# ADIM 7: AYARLANMIŞ PARAMETRELER
# ============================================

print("\n" + "="*80)
print("ADIM 7: SENIN DÜŞÜNCELİ PARAMETRELER")
print("="*80)

# Senin test sonuçlarına göre önerilen
recommended_params = {
    "experiment_name": "3D Araç Navigasyonu - FİLTRE İYİLEŞTİRME",
    "description": "Systematik tuning sonrası optimized parametreler",
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
        # ← BUNLARI DEĞİŞTİR ←
        "accel_bias_std": 0.5,           # ← ARTTI
        "accel_noise_std": 0.01,         # ← AZALDI
        "gyro_bias_std": 0.1,            # ← ARTTI
        "gyro_noise_std": 0.002,         # ← AZALDI
        "gps_position_noise_std": 3.0,   # ← AZALDI (5-10 dene)
        "gps_speed_noise_std": 0.3       # ← AZALDI
    },
    "initial_conditions": {
        "initial_attitude": [0.0, 0.0, 0.0],
        "initial_attitude_std": 0.5,     # ← ARTTI
        "initial_gyro_bias": [0.0, 0.0, 0.0]
    },
    "kalman_tuning": {
        "process_noise_q": 0.0,
        "measurement_noise_r_position": 3.0**2,
        "measurement_noise_r_velocity": 0.3**2
    },
    "time_parameters": {
        "imu_sampling_period": 0.01,
        "gps_sampling_period": 0.5,
        "simulation_duration": 30
    }
}

# Kaydet
with open('data/parameters_tuned_v1.json', 'w') as f:
    json.dump(recommended_params, f, indent=2)

print("\n✅ İlk tuning parametreleri kaydedildi: data/parameters_tuned_v1.json")

print("""

🎯 SONRAKI ADIMLAR:

1. parameters_tuned_v1.json'u kontrol et
2. Senin sensörüne göre değiştir:
   ├─ GPS hatası fazlaysa: gps_position_noise_std = 2.0 dene
   ├─ Hız salınıyorsa: accel_noise_std = 0.005 dene
   └─ Yaw kararsızsa: gyro_bias_std = 0.2 dene

3. ins_em_MODIFIED.py'yi kopyala:
   ins_em_TUNED.py

4. Satır 18'i değiştir:
   with open('data/parameters_tuned_v1.json', 'r') as f:

5. Çalıştır:
   python ins_em_TUNED.py
   python analyze_filter_performance.py

6. Sonuçları kontrol et:
   ├─ RMSE(konum) < 3m oldu mu?
   ├─ RMSE(hız) < 0.3 m/s oldu mu?
   └─ NIS 0.5-2.0 aralığında mı?

7. Eğer iyileşmediyse ADIM 6'ya geri dön (iteratif)
""")

print("\n" + "="*80)
print("✅ FİLTRE İYİLEŞTİRME REHBERİ TAMAMLANDI!")
print("="*80)