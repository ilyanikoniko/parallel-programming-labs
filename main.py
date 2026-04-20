import matplotlib.pyplot as plt
import numpy as np

# =============================================
# ТВОИ РЕАЛЬНЫЕ ДАННЫЕ
# =============================================
sizes = [200, 400, 800, 1200, 1600, 2000]

# CPU время (минимальное из трёх запусков из лабы 1)
cpu_times = [0.037514, 0.303019, 2.690870, 12.927400, 21.244100, 50.331300]

# GPU время для разных блоков (ТВОИ данные)
gpu_block8  = [0.000303883, 0.000861341, 0.0060719, 0.0202066, 0.0476728, 0.141791]
gpu_block16 = [0.000696491, 0.000907104, 0.00659965, 0.021984, 0.0518783, 0.154752]
gpu_block32 = [0.000401804, 0.00152203, 0.00624153, 0.0207606, 0.0489896, 0.117228]

# =============================================
# РАСЧЁТ УСКОРЕНИЯ
# =============================================
speedup_block8  = [cpu_times[i] / gpu_block8[i] for i in range(len(sizes))]
speedup_block16 = [cpu_times[i] / gpu_block16[i] for i in range(len(sizes))]
speedup_block32 = [cpu_times[i] / gpu_block32[i] for i in range(len(sizes))]

# =============================================
# ФУНКЦИЯ ДЛЯ РАСЧЁТА GFLOPS
# =============================================
def calculate_gflops(n, time):
    ops = 2 * n * n * n  # операций для умножения матриц
    return ops / (time * 1e9) if time > 0 else 0

gflops_block8  = [calculate_gflops(sizes[i], gpu_block8[i]) for i in range(len(sizes))]
gflops_block16 = [calculate_gflops(sizes[i], gpu_block16[i]) for i in range(len(sizes))]
gflops_block32 = [calculate_gflops(sizes[i], gpu_block32[i]) for i in range(len(sizes))]

# Настройки для графиков
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# =============================================
# ГРАФИК 1: Сравнение времени выполнения (логарифмическая шкала)
# =============================================
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(sizes, cpu_times, 'ro-', linewidth=2, markersize=8, label='CPU (AMD Ryzen 5 4600H)')
ax.plot(sizes, gpu_block8, 'bs-', linewidth=2, markersize=8, label='GPU Block 8×8')
ax.plot(sizes, gpu_block16, 'g^-', linewidth=2, markersize=8, label='GPU Block 16×16')
ax.plot(sizes, gpu_block32, 'md-', linewidth=2, markersize=8, label='GPU Block 32×32')

ax.set_xlabel('Размер матрицы N', fontsize=14)
ax.set_ylabel('Время выполнения (секунды)', fontsize=14)
ax.set_title('Сравнение времени умножения матриц: CPU vs CUDA (Tesla T4)', fontsize=16)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig('time_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================
# ГРАФИК 2: Ускорение GPU относительно CPU
# =============================================
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(sizes, speedup_block8, 'bs-', linewidth=2, markersize=8, label='Block 8×8')
ax.plot(sizes, speedup_block16, 'g^-', linewidth=2, markersize=8, label='Block 16×16')
ax.plot(sizes, speedup_block32, 'md-', linewidth=2, markersize=8, label='Block 32×32')

ax.set_xlabel('Размер матрицы N', fontsize=14)
ax.set_ylabel('Ускорение (раз)', fontsize=14)
ax.set_title('Ускорение GPU (Tesla T4) относительно CPU (AMD Ryzen 5)', fontsize=16)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', fontsize=12)

# Добавляем значения максимального ускорения
for i, (x, y) in enumerate(zip(sizes, speedup_block8)):
    ax.annotate(f'{int(y)}×', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('speedup.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================
# ГРАФИК 3: Производительность (GFLOPS)
# =============================================
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(sizes, gflops_block8, 'bs-', linewidth=2, markersize=8, label='Block 8×8')
ax.plot(sizes, gflops_block16, 'g^-', linewidth=2, markersize=8, label='Block 16×16')
ax.plot(sizes, gflops_block32, 'md-', linewidth=2, markersize=8, label='Block 32×32')

ax.set_xlabel('Размер матрицы N', fontsize=14)
ax.set_ylabel('Производительность (GFLOPS)', fontsize=14)
ax.set_title('Производительность GPU при умножении матриц', fontsize=16)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='lower right', fontsize=12)

# Отмечаем пиковую производительность
max_gflops = max(max(gflops_block8), max(gflops_block16), max(gflops_block32))
max_idx = np.argmax([max(gflops_block8), max(gflops_block16), max(gflops_block32)])
block_names = ['8×8', '16×16', '32×32']
ax.annotate(f'Пик: {max_gflops:.0f} GFLOPS\n({block_names[max_idx]})',
            xy=(sizes[np.argmax(gflops_block8)], max_gflops),
            xytext=(sizes[np.argmax(gflops_block8)] + 100, max_gflops - 100),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=11, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('gflops.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================
# ВЫВОД СТАТИСТИКИ
# =============================================
print("\n" + "="*60)
print("СТАТИСТИКА ПО РЕЗУЛЬТАТАМ (ТВОИ ДАННЫЕ)")
print("="*60)

print("\n📊 Максимальное ускорение:")
print(f"   Block 8×8:  {max(speedup_block8):.0f}× (при N={sizes[np.argmax(speedup_block8)]})")
print(f"   Block 16×16: {max(speedup_block16):.0f}× (при N={sizes[np.argmax(speedup_block16)]})")
print(f"   Block 32×32: {max(speedup_block32):.0f}× (при N={sizes[np.argmax(speedup_block32)]})")

print("\n🚀 Пиковая производительность:")
print(f"   {max_gflops:.0f} GFLOPS ({max_gflops/1000:.2f} TFLOPS)")

print("\n🏆 Лучшее время на GPU:")
best_time = min(min(gpu_block8), min(gpu_block16), min(gpu_block32))
print(f"   {best_time*1000:.3f} мс")

print("\n📈 Сравнение CPU vs GPU (лучший результат по каждому размеру):")
best_gpu_times = [min(gpu_block8[i], gpu_block16[i], gpu_block32[i]) for i in range(len(sizes))]
for i, n in enumerate(sizes):
    print(f"   N={n:4d}: CPU {cpu_times[i]:8.6f}с → GPU {best_gpu_times[i]:8.6f}с → Ускорение {int(cpu_times[i]/best_gpu_times[i]):4d}×")