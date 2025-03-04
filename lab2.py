import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats

def generate_samples(n):
    """Генерирует выборки различных распределений заданного размера."""
    return {
        'Normal': np.random.normal(0, 1, n),
        'Cauchy': np.random.standard_cauchy(n),
        'Poisson': np.random.poisson(10, n),
        'Uniform': np.random.uniform(-math.sqrt(3), math.sqrt(3), n)
    }

def plot_boxplots():
    """Строит бокс-плоты Тьюки для различных распределений, каждый на отдельном графике."""
    ns = [20, 100, 1000]
    for n in ns:
        samples = generate_samples(n)
        for name, data in samples.items():
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=data)
            plt.title(f'Бокс-плот для {name} распределения (n={n})')
            plt.xlabel("Значения")
            plt.ylabel("Частота")
            plt.show()

def count_outliers():
    """Определяет количество выбросов в выборках."""
    ns = [20, 100, 1000]
    outliers_data = []
    for n in ns:
        samples = generate_samples(n)
        outliers = {}
        for name, data in samples.items():
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers[name] = np.sum((data < lower_bound) | (data > upper_bound))
        outliers_data.append({'n': n, **outliers})
    df = pd.DataFrame(outliers_data)
    print(df)

def main():
    """Основная функция выполнения анализа."""
    plot_boxplots()
    count_outliers()

if __name__ == "__main__":
    main()
