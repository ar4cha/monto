import numpy as np
import math
import random
from matplotlib import pyplot as plt
from IPython.display import clear_output

PI = 3.1415926
e = 2.71828

def get_rand_number(min_value, max_value):
    #Эта функция получает случайное число из равномерного распределения между
    #два входных значения [min_value, max_value] включительно
    #Аргументы:
    #минимальное_значение
    #максимальное_значение
    #Возвращение:
    #Случайное число из этого диапазона

    range = max_value - min_value
    choice = random.uniform(0,1)
    return min_value + range*choice
def f_of_x(x):
    #Это основная функция, которую мы хотим интегрировать.
    #Аргументы:
    #x: вход в функцию; должно быть в радианах
    #Возвращение:
    #вывод функции f(x)
    return (e**(-1*x))/(1+(x-1)**2)
def crude_monte_carlo(num_samples=5000):
    #Эта функция выполняет грубый Монте-Карло для нашего
    #конкретная функция f(x) в диапазоне от x=0 до x=5.
    #Обратите внимание, что этой границы достаточно, потому что f(x)
    #приближается к 0 в районе PI.
    #Аргументы:
    #num_samples : количество сэмплов
    #Возвращение:
    #Оценка сырой нефти по методу Монте-Карло
    lower_bound = 0
    upper_bound = 5

    sum_of_samples = 0
    for i in range(num_samples):
        x = get_rand_number(lower_bound, upper_bound)
        sum_of_samples += f_of_x(x)

    return (upper_bound - lower_bound) * float(sum_of_samples / num_samples)


def get_crude_MC_variance(num_samples):
    #Эта функция возвращает дисперсию сырой нефти Монте-Карло.
    #Обратите внимание, что введенное количество выборок не обязательно
    #должны соответствовать количеству образцов, используемых в Монте Симулятор Карло.
    #Аргументы:
    #num_samples (целое число)
    #Возвращение:
    #Дисперсия грубой аппроксимации Монте-Карло для f(x) (с плавающей запятой)
    int_max = 5  #это максимум нашего диапазона интегрирования

    # получить среднее значение квадратов
    running_total = 0
    for i in range(num_samples):
        x = get_rand_number(0, int_max)
        running_total += f_of_x(x) ** 2
    sum_of_sqs = running_total * int_max / num_samples

    # получить квадрат среднего
    running_total = 0
    for i in range(num_samples):
        x = get_rand_number(0, int_max)
        running_total = f_of_x(x)
    sq_ave = (int_max * running_total / num_samples) ** 2

    return sum_of_sqs - sq_ave
# Теперь мы запустим симуляцию Crude Monte Carlo с 10000 образцов.
# Мы также рассчитаем дисперсию с 10000 выборками и ошибкой

MC_samples = 10000
var_samples = 10000 # количество выборок, которые мы будем использовать для расчета дисперсии
crude_estimation = crude_monte_carlo(MC_samples)
variance = get_crude_MC_variance(var_samples)
error = math.sqrt(variance/MC_samples)

# отображать результаты
print(f"Аппроксимация Монте-Карло f (x): {crude_estimation}")
print(f"Дисперсия приближения: {variance}")
print(f"Ошибка в приближении: {error}")


#------------------------------------------------------------------------------------------------------------------

# это шаблон нашей весовой функции g(x)
def g_of_x(x, A, lamda):
    e = 2.71828
    return A*math.pow(e, -1*lamda*x)
# постройка функции
xs = [float(i/50) for i in range(int(50*PI))]
fs = [f_of_x(x) for x in xs]
gs = [g_of_x(x, A=1.4, lamda=1.4) for x in xs]
plt.plot(xs, fs)
plt.plot(xs, gs)
plt.title("f(x) and g(x)");
def inverse_G_of_r(r, lamda):
    return (-1 * math.log(float(r)))/lamda


def get_IS_variance(lamda, num_samples):
    #Эта функция вычисляет дисперсию, если метод Монте-Карло
    #с использованием выборки по важности.
    #Аргументы:
    #лямда : тестируемое лямбда-значение g(x)
    #Возвращение:
    #Дисперсия
    A = lamda
    int_max = 5

    # получить сумму квадратов
    running_total = 0
    for i in range(num_samples):
        x = get_rand_number(0, int_max)
        running_total += (f_of_x(x) / g_of_x(x, A, lamda)) ** 2

    sum_of_sqs = running_total / num_samples

    # получить квадрат среднего
    running_total = 0
    for i in range(num_samples):
        x = get_rand_number(0, int_max)
        running_total += f_of_x(x) / g_of_x(x, A, lamda)
    sq_ave = (running_total / num_samples) ** 2

    return sum_of_sqs - sq_ave


# получить дисперсию как функцию лямбда, протестировав множество
# разные лямбды

test_lamdas = [i * 0.05 for i in range(1, 61)]
variances = []

for i, lamda in enumerate(test_lamdas):
    print(f"lambda {i + 1}/{len(test_lamdas)}: {lamda}")
    A = lamda
    variances.append(get_IS_variance(lamda, 10000))
    clear_output(wait=True)

optimal_lamda = test_lamdas[np.argmin(np.asarray(variances))]
IS_variance = variances[np.argmin(np.asarray(variances))]

print(f"Оптимальная лямбда: {optimal_lamda}")
print(f"Оптимальный вариант: {IS_variance}")
print((IS_variance / 10000) ** 0.5)

plt.plot(test_lamdas[5:40], variances[5:40])
plt.title("Дисперсия MК при различных значениях лямбда");
