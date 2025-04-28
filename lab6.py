import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Настройки для графиков
plt.style.use('seaborn-v0_8')  # Используем актуальный стиль seaborn
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 200
np.random.seed(42)

# Функция для расчета доверительных интервалов
def calculate_intervals(data, alpha=0.05):
    n = len(data)
    x_bar = np.mean(data)
    s = np.std(data, ddof=1)
    
    # Для нормального распределения
    t = stats.t.ppf(1 - alpha/2, n-1)
    ci_m_normal = (x_bar - s*t/np.sqrt(n), x_bar + s*t/np.sqrt(n))
    
    chi2_low = stats.chi2.ppf(alpha/2, n-1)
    chi2_high = stats.chi2.ppf(1 - alpha/2, n-1)
    ci_sigma_normal = (s*np.sqrt(n-1)/np.sqrt(chi2_high), s*np.sqrt(n-1)/np.sqrt(chi2_low))
    
    # Асимптотический подход
    u = stats.norm.ppf(1 - alpha/2)
    ci_m_asympt = (x_bar - s*u/np.sqrt(n), x_bar + s*u/np.sqrt(n))
    
    m4 = np.mean((data - x_bar)**4)
    e = m4 / s**4 - 3
    U = u * np.sqrt((e + 2)/n)
    ci_sigma_asympt = (s*(1 - U/2), s*(1 + U/2))  # Упрощенная формула
    
    return {
        'mean': x_bar,
        'std': s,
        'normal': {'mean': ci_m_normal, 'sigma': ci_sigma_normal},
        'asymptotic': {'mean': ci_m_asympt, 'sigma': ci_sigma_asympt}
    }

# Генерация данных
data_20 = np.random.normal(0, 1, 20)
data_100 = np.random.normal(0, 1, 100)

# Расчет интервалов
results_20 = calculate_intervals(data_20)
results_100 = calculate_intervals(data_100)

# 1. Вывод информации о доверительных интервалах для математического ожидания
print("Доверительные интервалы для математического ожидания:")
print(f"n=20 Точный: [{results_20['normal']['mean'][0]:.3f}, {results_20['normal']['mean'][1]:.3f}]")
print(f"n=20 Асимпт: [{results_20['asymptotic']['mean'][0]:.3f}, {results_20['asymptotic']['mean'][1]:.3f}]")
print(f"n=100 Точный: [{results_100['normal']['mean'][0]:.3f}, {results_100['normal']['mean'][1]:.3f}]")
print(f"n=100 Асимпт: [{results_100['asymptotic']['mean'][0]:.3f}, {results_100['asymptotic']['mean'][1]:.3f}]")

# 2. Вывод информации о доверительных интервалах для стандартного отклонения
print("\nДоверительные интервалы для стандартного отклонения:")
print(f"n=20 Точный: [{results_20['normal']['sigma'][0]:.3f}, {results_20['normal']['sigma'][1]:.3f}]")
print(f"n=20 Асимпт: [{results_20['asymptotic']['sigma'][0]:.3f}, {results_20['asymptotic']['sigma'][1]:.3f}]")
print(f"n=100 Точный: [{results_100['normal']['sigma'][0]:.3f}, {results_100['normal']['sigma'][1]:.3f}]")
print(f"n=100 Асимпт: [{results_100['asymptotic']['sigma'][0]:.3f}, {results_100['asymptotic']['sigma'][1]:.3f}]")

# 3. График доверительных интервалов для математического ожидания
plt.figure(figsize=(12, 6))

# Для n=20
plt.errorbar(x=['Точный', 'Асимптотический'], 
             y=[results_20['mean'], results_20['mean']],
             yerr=[[results_20['mean'] - results_20['normal']['mean'][0], 
                    results_20['mean'] - results_20['asymptotic']['mean'][0]],
                  [results_20['normal']['mean'][1] - results_20['mean'], 
                   results_20['asymptotic']['mean'][1] - results_20['mean']]],
             fmt='o', capsize=5, label='n=20')

# Для n=100
plt.errorbar(x=['Точный', 'Асимптотический'], 
             y=[results_100['mean'], results_100['mean']],
             yerr=[[results_100['mean'] - results_100['normal']['mean'][0], 
                    results_100['mean'] - results_100['asymptotic']['mean'][0]],
                  [results_100['normal']['mean'][1] - results_100['mean'], 
                   results_100['asymptotic']['mean'][1] - results_100['mean']]],
             fmt='o', capsize=5, label='n=100')

plt.title("Доверительные интервалы для математического ожидания")
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.grid(True)
plt.legend()
plt.savefig('confidence_intervals_mean.png')
plt.close()

# 4. График доверительных интервалов для стандартного отклонения
plt.figure(figsize=(12, 6))

# Для n=20
plt.errorbar(x=['Точный', 'Асимптотический'], 
             y=[results_20['std'], results_20['std']],
             yerr=[[results_20['std'] - results_20['normal']['sigma'][0], 
                    results_20['std'] - results_20['asymptotic']['sigma'][0]],
                  [results_20['normal']['sigma'][1] - results_20['std'], 
                   results_20['asymptotic']['sigma'][1] - results_20['std']]],
             fmt='o', capsize=5, label='n=20')

# Для n=100
plt.errorbar(x=['Точный', 'Асимптотический'], 
             y=[results_100['std'], results_100['std']],
             yerr=[[results_100['std'] - results_100['normal']['sigma'][0], 
                    results_100['std'] - results_100['asymptotic']['sigma'][0]],
                  [results_100['normal']['sigma'][1] - results_100['std'], 
                   results_100['asymptotic']['sigma'][1] - results_100['std']]],
             fmt='o', capsize=5, label='n=100')

plt.title("Доверительные интервалы для стандартного отклонения")
plt.axhline(y=1, color='red', linestyle='--', alpha=0.5)
plt.grid(True)
plt.legend()
plt.savefig('confidence_intervals_std.png')
plt.close()

# 5. Гистограммы распределений
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(data_20, bins=10, density=True, alpha=0.7, color='blue')
plt.title(f'Выборка n=20\nСреднее: {results_20["mean"]:.2f}, σ: {results_20["std"]:.2f}')
plt.xlabel('Значение')
plt.ylabel('Плотность')
x = np.linspace(-4, 4, 100)
plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', alpha=0.5)

plt.subplot(1, 2, 2)
plt.hist(data_100, bins=15, density=True, alpha=0.7, color='green')
plt.title(f'Выборка n=100\nСреднее: {results_100["mean"]:.2f}, σ: {results_100["std"]:.2f}')
plt.xlabel('Значение')
plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', alpha=0.5)

plt.tight_layout()
plt.savefig('histograms.png')
plt.close()

print("\nГрафики сохранены в файлы:")
print("- confidence_intervals_mean.png")
print("- confidence_intervals_std.png")
print("- histograms.png")