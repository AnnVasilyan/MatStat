import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.stats as stats
from scipy.stats import chi2, t, norm, moment
from scipy.optimize import minimize
import os
import math

def get_least_squares_coefs(x, y):
    xy_m = np.mean(np.multiply(x, y))
    x_m = np.mean(x)
    x_2_m = np.mean(np.multiply(x, x))
    y_m = np.mean(y)
    b1_mnk = (xy_m - x_m * y_m) / (x_2_m - x_m * x_m)
    b0_mnk = y_m - x_m * b1_mnk

    return b0_mnk, b1_mnk


def abs_dev_val(b_arr, x, y):
    return np.sum(np.abs(y - b_arr[0] - b_arr[1] * x))


def get_linear_approx_coefs(x, y):
    init_b = np.array([0, 1])
    res = minimize(abs_dev_val, init_b, args=(x, y), method='COBYLA')

    return res.x


def draw(lsm_0, lsm_1, lam_0, lam_1, x, y, title, fname):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='pink', s=6, label='Выборка')
    y_lsm = np.add(np.full(20, lsm_0), x * lsm_1)
    y_lam = np.add(np.full(20, lam_0), x * lam_1)
    y_real = np.add(np.full(20, 2), x * 2)
    ax.plot(x, y_lsm, color='pink', label='МНК')
    ax.plot(x, y_lam, color='red', label='МНМ')
    ax.plot(x, y_real, color='orange', label='Модель')
    ax.set(xlabel='X', ylabel='Y',
       title=title)
    ax.legend()
    ax.grid()
    fig.savefig(fname + '.png', dpi=200)


def task():
    x = np.arange(-1.8, 2.1, 0.2)
    eps = np.random.normal(0, 1, size=20)
    # y = 2 + 2x + eps
    y = np.add(np.add(np.full(20, 2), x * 2), eps)
    # y2 with noise
    y2 = np.add(np.add(np.full(20, 2), x * 2), eps)
    y2[0] += 10
    y2[19] += -10

    if not os.path.exists("task_data"):
        os.mkdir("task_data")

    lsm_0, lsm_1 = get_least_squares_coefs(x, y)
    lam_0, lam_1 = get_linear_approx_coefs(x, y)

    lsm_02, lsm_12 = get_least_squares_coefs(x, y2)
    lam_02, lam_12 = get_linear_approx_coefs(x, y2)
# Истинные значения параметров
    a_true, b_true = 2, 2

    # Формируем таблицу для невозмущенной выборки
    with open("task_data/table1.tex", "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Результаты для невозмущенной выборки}\n")
        f.write("\\label{tab:unperturbed}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write(" & Метод & $\\hat{a}$ & $\\frac{\\hat{a}}{a}$ & $\\hat{b}$ & $\\frac{\\hat{b}}{b}$ \\\\\n")
        f.write("\\hline\n")
        f.write(f"1 & МНК & {lsm_0:.4f} & {lsm_0/a_true:.4f} & {lsm_1:.4f} & {lsm_1/b_true:.4f} \\\\\n")
        f.write("\\hline\n")
        f.write(f"2 & МНМ & {lam_0:.4f} & {lam_0/a_true:.4f} & {lam_1:.4f} & {lam_1/b_true:.4f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    # Формируем таблицу для возмущенной выборки
    with open("task_data/table2.tex", "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Результаты для возмущенной выборки}\n")
        f.write("\\label{tab:perturbed}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write(" & Метод & $\\hat{a}$ & $\\frac{\\hat{a}}{a}$ & $\\hat{b}$ & $\\frac{\\hat{b}}{b}$ \\\\\n")
        f.write("\\hline\n")
        f.write(f"1 & МНК & {lsm_02:.4f} & {lsm_02/a_true:.4f} & {lsm_12:.4f} & {lsm_12/b_true:.4f} \\\\\n")
        f.write("\\hline\n")
        f.write(f"2 & МНМ & {lam_02:.4f} & {lam_02/a_true:.4f} & {lam_12:.4f} & {lam_12/b_true:.4f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    with open("task_data/no_noise.tex", "w") as f:
        f.write("\\begin{enumerate}\n")
        f.write("\\item МНК, без возмущений:\n")
        f.write("$\\hat{a}\\approx " + str(lsm_0) + "$, $\\hat{b}\\approx " + str(lsm_1) + "$\n")
        f.write("\\item МНМ, без возмущений:\n")
        f.write("$\\hat{a}\\approx " + str(lam_0) + "$, $\\hat{b}\\approx " + str(lam_1) + "$\n")
        f.write("\\end{enumerate}\n")

    with open("task_data/noise.tex", "w") as f:
        f.write("\\begin{enumerate}\n")
        f.write("\\item МНК, с возмущенями:\n")
        f.write("$\\hat{a}\\approx " + str(lsm_02) + "$, $\\hat{b}\\approx " + str(lsm_12) + "$\n")
        f.write("\\item МНМ, с возмущенями:\n")
        f.write("$\\hat{a}\\approx " + str(lam_02) + "$, $\\hat{b}\\approx " + str(lam_12) + "$\n")
        f.write("\\end{enumerate}\n")

    draw(lsm_0, lsm_1, lam_0, lam_1, x, y, 'Выборка без возмущений', 'task2_data/no_noise')
    draw(lsm_02, lsm_12, lam_02, lam_12, x, y2, 'Выборка с возмущенями', 'task2_data/noise')



if __name__ == "__main__":
    task()