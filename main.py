import matplotlib.pyplot as plt
import numpy as np

# Данные измерений
N = np.array([200, 400, 800, 1200, 1600, 2000])
time = np.array([0.00118042, 0.00577363, 0.0412052, 0.256761, 0.602928, 1.1667])

# Построение графика
plt.figure(figsize=(8, 5))
plt.plot(N, time, marker='o', linestyle='-', color='b', label='Время выполнения')

# Подписи осей и заголовок
plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('Время выполнения (с)', fontsize=12)
plt.title('Зависимость времени выполнения от размера матрицы\n(8 процессов MPI)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Опционально: логарифмический масштаб для наглядности (раскомментировать при необходимости)
# plt.xscale('log')
# plt.yscale('log')

# Показать график
plt.tight_layout()
plt.savefig('graph_time.png', dpi=300)  # Сохранить в файл
plt.show()