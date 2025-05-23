import numpy as np
from scipy.stats import norm

class BreakpointFinder():
    """Обнаружение разладок с помощью cumsum и процедуры Ширяева-Робертса"""
    
    def __init__(self, alpha = 0.05, beta = 0.05, mean_diff = 0.01, sigma_diff = 1, ceil = 1e6, mean_trsh=0.03, sigma_trsh=2, breaks_max=5, slice_length=7):
        
        self.mean_hat = 0
        self.std_hat = 1
        
        self.alpha = alpha
        self.beta = beta
        
        self.SRMetric = 0
        self.CumSumMetric = 0

        # ГИПЕРПАРАМЕТР: максимальная допустимая разность средних
        self.mean_diff = mean_diff 
        # ГИПЕРПАРАМЕТР: максимальная допустимая разность стандратного отклонения
        self.sigma_diff = sigma_diff 
        
        # ГИПЕРПАРАМЕТР: дельта для альтернативный гипотезы
        self.mean_diff = mean_diff 
        # ГИПЕРПАРАМЕТР: порог для критерия
        self.mean_trsh = mean_trsh
        self.sigma_trsh = sigma_trsh
        # ГИПЕРПАРАМЕТР: верхняя граница для значения критерия
        self.ceil = ceil
        # ГИПЕРПАРАМЕТРЫ: макс. число разладок для принятия мер
        self.breaks_max = breaks_max
        
        self.states = []
        self.breakpoints = []
        # ГИПЕРПАРАМЕТР: глубина среза значений метрики
        self.slice_length = slice_length
        self.colors=['green', 'red']
        
    def get_values(self):
        # оцениваем mean и std^2
        try:
            self.mean_hat = self.mean_values_sum / self.mean_weights_sum
            self.var_hat = self.var_values_sum / self.var_weights_sum
        except AttributeError:
            self.mean_hat = 0
            self.var_hat = 1
    
    def update(self, new_value):

        self.get_values()
        
        # нормализация и стандартизация
        self.new_value_normalized = (new_value - self.mean_hat) / np.sqrt(self.std_hat)

        # Считаем обновлённое значение std и среднее
        self.predicted_diff_value = (new_value - self.mean_hat) ** 2
        self.predicted_diff_mean = self.var_hat
        
        # обновляем значения mean и std^2
        try:
            self.mean_values_sum = (1 - self.alpha) * self.mean_values_sum + new_value
            self.mean_weights_sum = (1 - self.alpha) * self.mean_weights_sum + 1.0
        except AttributeError:
            self.mean_values_sum = new_value
            self.mean_weights_sum = 1.0 
        
        # новое значение std^2
        new_value_var = (self.new_value_normalized - self.mean_hat)**2
        
        try:
            self.var_values_sum = (1 - self.beta) * self.var_values_sum + new_value_var
            self.var_weights_sum = (1 - self.beta) * self.var_weights_sum + 1.0
        except:
            self.var_values_sum = new_value_var
            self.var_weights_sum = 1.0      

    def count_metric(self):
        # проверка гипотезы о том, что среднее действительно = 0, с учётом того, что std = 1
        zeta_k = np.log(norm.pdf(self.new_value_normalized, self.mean_diff, 1) / 
                  norm.pdf(self.new_value_normalized, 0, 1))
        self.CumSumMetric = max(0, self.CumSumMetric + zeta_k)

        # проверка гипотезы о том, что среднее у разницы между std действительно = 0
        adjusted_value = self.predicted_diff_value - self.predicted_diff_mean
        likelihood = np.exp(self.sigma_diff * (adjusted_value - self.sigma_diff / 2.))
        self.SRMetric = min(self.ceil, (1. + self.SRMetric) * likelihood)

        if self.CumSumMetric > self.mean_trsh or self.SRMetric > self.sigma_trsh:            
            self.states.append(1)
        else:
            self.states.append(0)
            
        if (np.array(self.states[-self.slice_length:]) == 1).sum() > self.breaks_max:
            self.breakpoints.append('red')
        else:
            self.breakpoints.append('green')