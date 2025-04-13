import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

class NormalDistributionTest:
    def __init__(self):
        self.alpha = 0.05
        
    def generate_sample(self, size, dist_type='normal'):
        """Генерация выборки заданного типа"""
        if dist_type == 'normal':
            return np.random.standard_normal(size=size)
        elif dist_type == 'uniform':
            return np.random.uniform(-np.sqrt(3), np.sqrt(3), size=size)
    
    def estimate_parameters(self, sample):
        """Оценка параметров методом максимального правдоподобия"""
        mu = np.mean(sample)
        sigma = np.std(sample, ddof=0)  # Для ММП используем несмещенную оценку
        return mu, sigma
    
    def chi2_test(self, sample, mu, sigma):
        """Критерий согласия хи-квадрат"""
        n = len(sample)
        k = int(1 + 3.3 * np.log10(n))  # Правило Старджесса
        
        # Границы интервалов для N(mu, sigma)
        percentiles = np.linspace(0, 100, k+1)[1:-1]
        boundaries = np.percentile(sample, percentiles)
        boundaries = np.concatenate([[-np.inf], boundaries, [np.inf]])
        
        # Наблюдаемые частоты
        observed, _ = np.histogram(sample, boundaries)
        
        # Теоретические вероятности
        cdf = stats.norm(loc=mu, scale=sigma).cdf
        prob = np.diff(cdf(boundaries))
        expected = prob * n
        
        # Статистика хи-квадрат
        chi2 = np.sum((observed - expected)**2 / expected)
        critical = stats.chi2.ppf(1 - self.alpha, k-3)  # k-3 для N(mu,sigma)
        
        return chi2, critical, observed, expected, boundaries
    
    def run_tests(self):
        """Основная функция выполнения тестов"""
        # 1. Нормальное распределение n=100
        print("=== Нормальное распределение n=100 ===")
        sample_normal = self.generate_sample(100)
        mu, sigma = self.estimate_parameters(sample_normal)
        print(f"Оценки параметров: μ = {mu:.3f}, σ = {sigma:.3f}")
        
        chi2, critical, observed, expected, _ = self.chi2_test(sample_normal, mu, sigma)
        print(f"χ² наблюдаемое = {chi2:.3f}, критическое = {critical:.3f}")
        print(f"Гипотеза {'принимается' if chi2 < critical else 'отвергается'}")
        
        # 2. Равномерное распределение n=20
        print("\n=== Равномерное распределение n=20 ===")
        sample_uniform = self.generate_sample(20, 'uniform')
        mu_unif, sigma_unif = self.estimate_parameters(sample_uniform)
        
        chi2_unif, critical_unif, *_ = self.chi2_test(sample_uniform, mu_unif, sigma_unif)
        print(f"χ² наблюдаемое = {chi2_unif:.3f}, критическое = {critical_unif:.3f}")
        print(f"Гипотеза {'принимается' if chi2_unif < critical_unif else 'отвергается'}")
        
        # Визуализация
        self.plot_results(sample_normal, sample_uniform)
    
    def plot_results(self, sample_normal, sample_uniform):
        """Визуализация результатов"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(121)
        plt.hist(sample_normal, bins=20, density=True, alpha=0.7)
        x = np.linspace(-4, 4, 100)
        plt.plot(x, stats.norm.pdf(x), 'r-', lw=2)
        plt.title("Нормальное распределение (n=100)")
        
        plt.subplot(122)
        plt.hist(sample_uniform, bins=10, density=True, alpha=0.7)
        x = np.linspace(-2, 2, 100)
        plt.plot(x, stats.uniform.pdf(x, -np.sqrt(3), 2*np.sqrt(3)), 'r-', lw=2)
        plt.title("Равномерное распределение (n=20)")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    tester = NormalDistributionTest()
    tester.run_tests()