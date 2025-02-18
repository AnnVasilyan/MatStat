import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

def generate_samples(distribution, size, params):
    if distribution == "normal":
        return np.random.normal(*params, size)
    elif distribution == "cauchy":
        return np.random.standard_cauchy(size)
    elif distribution == "poisson":
        return np.random.poisson(*params, size)
    elif distribution == "uniform":
        return np.random.uniform(*params, size)
    else:
        raise ValueError("Unknown distribution")

def plot_hist_density(distribution, sizes, params, x_range=None):
    plt.figure(figsize=(10, 6))
    for size in sizes:
        sample = generate_samples(distribution, size, params)
        plt.hist(sample, bins=30, density=True, alpha=0.5, label=f"Sample size {size}")
        if x_range is not None:
            x = np.linspace(x_range[0], x_range[1], 1000)
        else:
            x = np.linspace(min(sample), max(sample), 1000)
        
        if distribution == "normal":
            plt.plot(x, stats.norm.pdf(x, *params), label=f"Density {distribution}")
        elif distribution == "cauchy":
            plt.plot(x, stats.cauchy.pdf(x, *params), label=f"Density {distribution}")
        elif distribution == "poisson":
            x_int = np.arange(min(sample), max(sample) + 1)
            plt.stem(x_int, stats.poisson.pmf(x_int, *params), linefmt='r-', markerfmt='ro', basefmt=" ")
        elif distribution == "uniform":
            plt.plot(x, stats.uniform.pdf(x, *params), label=f"Density {distribution}")
    
    plt.legend()
    plt.title(f"Histogram and Density for {distribution.capitalize()} Distribution")
    plt.show()

def compute_statistics(distribution, sizes, params, repeats=1000):
    results = []
    for size in sizes:
        mean_vals, median_vals, zq_vals = [], [], []
        for _ in range(repeats):
            sample = generate_samples(distribution, size, params)
            mean_vals.append(np.mean(sample))
            median_vals.append(np.median(sample))
            zq_vals.append((np.percentile(sample, 25) + np.percentile(sample, 75)) / 2)
        
        mean_mean = np.mean(mean_vals)
        mean_median = np.mean(median_vals)
        mean_zq = np.mean(zq_vals)
        
        var_mean = np.var(mean_vals)
        var_median = np.var(median_vals)
        var_zq = np.var(zq_vals)
        
        results.append([size, mean_mean, mean_median, mean_zq, var_mean, var_median, var_zq])
    
    df = pd.DataFrame(results, columns=["Sample Size", "E(Mean)", "E(Median)", "E(Zq)", "D(Mean)", "D(Median)", "D(Zq)"])
    return df

# Define distributions and parameters
distributions = {
    "normal": (0, 1),
    "cauchy": (0, 1),
    "poisson": (10,),
    "uniform": (-np.sqrt(3), np.sqrt(3))
}

# Part 1: Plot histograms and density functions
for dist, params in distributions.items():
    plot_hist_density(dist, [10, 50, 1000], params)

# Part 2: Compute statistics
sizes = [10, 100, 1000]
for dist, params in distributions.items():
    df = compute_statistics(dist, sizes, params)
    print(f"Statistics for {dist.capitalize()} Distribution:")
    print(df)
    print()
