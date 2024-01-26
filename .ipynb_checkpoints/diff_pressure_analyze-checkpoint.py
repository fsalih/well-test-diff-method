# Библиотеки
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, TheilSenRegressor  # базовые линейные регрессоры
from sklearn.linear_model import BayesianRidge, OrthogonalMatchingPursuit, TweedieRegressor # дополнительные
from sklearn.linear_model import RANSACRegressor, ARDRegression, Lasso, Ridge, HuberRegressor

from sklearn.metrics import r2_score

import warnings

from dataclasses import dataclass

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import seaborn as sns
import json


@dataclass
class WellData:
    wct: float  # обводненность %
    q: float  # дебит жидкости преред остановкой, м3/сут    
    pw: float  # Забойное давление перед остановкой, ат.
    c_liq: float  # сжимаемость жидкости 1/ат.
    d_nkt_in: float  # Внутренний диаметр НКТ, м
    d_nkt_out: float  # Внешний диаметр НКТ, м
    d_tube_in: float  # Внутренний диаметр эксплуатационной колонны, м
    md_vdp: float  # Измеренная глубина ВДП, м
    md_device: float  # Измеренная глубина установки прибора, м
    md_bottom: float  # Измеренная глубина забоя, м
    wbs_type: str  # Место уровня жидкости. Варианты: ['Затрубное пространство', 'НКТ', 'НКТ+затруб', 'Сжимаемость']
    research_type: str  # Вид исследования. Варианты: ['КПД', 'КВД', 'КВУ']
    use_trajectory: bool  # Учитывать траекторию ствола скважины
    p_is_vdp: bool  # давление задано на глубину ВДП      
    tvd_vdp: float  # md_vdp  # глубина ВДП по вертикали
    tvd_device: float  # md_device  # глубина прибора по вертикали
    p_gs: float # гидростат, при необходимости можно заменить функцией. Пока так как есть.
    
    # Необязательные параметры
    q_calculation: str = 'basic'  # Коррекция расчетного дебита на измеренный дебит перед остановкой. Варианты: ['basic', 'by_first', 'by_max', 'by_volume'] 
    
    # все плотности необязательны, но обязательно надо задать один из плотностей, в зависимости от типа исследования
    rho_oil: float = 800  # Плотность нефти, кг/м3
    rho_water: float = 1000  # Плотность воды, кг/м3
    rho_inj: float = 1050  # Плотность жидкости закачки (для КПД), кг/м3        
    rho_liq: float = -1  # Плотность жидкости 
    
    tube_area: float = 0    
    changing_wbs_time: float = -1  # Время изменения типа ВСС
    use_changing_wbs: bool = False  # Учет изменения типа ВСС
    title: str = ''  # Титулка
    derivative_from: str = 'table'  # вариант расчета производной. Варианты: ['table', 'calculation']
    
    # траектория
    _df_inkl: pd.DataFrame = pd.DataFrame(data={'md': [-10_000, 0, 10_000], 
                                                'elong': [0, 0, 0], 
                                                'tvd': [-10_000, 0, 10_000], 
                                                'curvature': [1, 1, 1]})
    # свойства
    @property
    def df_inkl(self):
        # """Вызовется при запросе значения из атрибута df_inkl"""
        return self._df_inkl
        
    @df_inkl.setter
    def df_inkl(self, value):
        # """Вызовется при присвоении значения атрибуту df_inkl"""
        self._df_inkl = value
        # Выполняем нужные действия после присвоения
        self._df_inkl['tvd'] = self._df_inkl.md - self._df_inkl.elong        
        self._df_inkl['curvature'] = self._df_inkl['tvd'].diff()
        min_c = self._df_inkl[self._df_inkl['curvature']>0]['curvature'].min()
        self._df_inkl['curvature'].apply(lambda x: x if x>0 else min_c)
        self._df_inkl['curvature'] = self._df_inkl['md'].diff() / self._df_inkl['curvature']
        self._df_inkl.loc[0, 'curvature'] = 1
        

class DiffAnalyse():
    """
    Класс предназначен для анализа гидродинамических исследований КВУ, КВД, КПД дифференциальным методом.
    При инициализации класса необходимо задать начальные параметры исследуемого объекта в виде класса данных выше WellData.
    
    Методы класса (в порядке вызова):
        read_pressure_data - чтение исходных данных замеров в формате Excel таблицы со столбцами: 
            Дата, Время от начала исследования (ч), Давление (кгс/см2, ат.), Производная давления (ат./ч)
        calculate_changing_wbs_time - расчет времени изменения типа ВСС по уровню в стволе скважины
            На текущий момент величина затрубного давления не учитывается. 
        set_changing_wbs_time - задать время изменения типа ВСС вручную (для корректировки).
        fit_pressure_model - рассчитать модель: поиск левой границы интервала данных при закрепленной правой границе.
        
    """
    def __init__(self, diff_data: WellData):
        # constants
        self.G = 9.80665  # ускорение свободного падения
        
        # variables
        self.diff_data = diff_data
        self.model_fitted = False
        self.model_fitted_2d = False
        self.model = None
        self.left_bound = 0
        self.right_bound = -1
        self.ipr_data = None
        
        # плотность жидкости
        if diff_data.research_type == 'КПД':
            self.diff_data.rho_liq = diff_data.rho_inj
        else:
            self.diff_data.rho_liq = (diff_data.wct*diff_data.rho_water + (100 - diff_data.wct)*diff_data.rho_oil)/100  
        
        # траектория
        # self.diff_data._df_inkl['tvd'] = diff_data.df_inkl.md - diff_data.df_inkl.elong
        # self.diff_data._df_inkl['curvature'] = 1

        # расчет площади поперечного сечения места уровня жидкости
        if diff_data.wbs_type == 'Затрубное пространство':
            self.diff_data.tube_area = np.pi * (diff_data.d_tube_in**2 - diff_data.d_nkt_out**2)/4
        elif diff_data.wbs_type == 'НКТ':
            self.diff_data.tube_area = np.pi * (diff_data.d_nkt_in**2)/4
        elif diff_data.wbs_type == 'НКТ+затруб':
            self.diff_data.tube_area = np.pi * (diff_data.d_tube_in**2 - diff_data.d_nkt_out**2)/4 \
            + np.pi * (diff_data.d_nkt_in**2)/4
        elif diff_data.wbs_type == 'Сжимаемость':  # по идее это ошибка
            self.diff_data.tube_area = 1
            # ToDo продумать логику выбора типа ВСС. Пока рассчитывается напрямую по уровню.
        else:
            self.diff_data.tube_area = 0
            # raise error
        
        self.diff_data.tvd_vdp = self.tvd_by_md(diff_data.md_vdp)  # глубина ВДП по вертикали
        self.diff_data.tvd_device = self.tvd_by_md(diff_data.md_device)  # глубина прибора по вертикали
        
        # расчет гидростата, при необходимости можно заменить функцией. Пока так как есть.
        if self.diff_data.research_type == 'КПД':
            diff_data.p_gs = diff_data.tvd_vdp*diff_data.rho_inj/10_000
        else:
            self.diff_data.p_gs = diff_data.tvd_vdp*diff_data.rho_liq/10_000

        self.data = pd.DataFrame()
        
    # hidden variables
    # _research_sign
    @property
    def _research_sign(self):
        # """Вызовется при запросе значения из атрибута _research_sign"""
        if self.diff_data.research_type == 'КПД':
            return -1
        else:
            return 1

    
    # ToDo предполагается, что данные отсортированы по md, позже сделать проверку
    # функция для расчета глубины по стволу при заданной глубине по вертикали 
    def md_by_tvd(self, tvd_value):
        tvd_series = self.diff_data.df_inkl.tvd
        md_series = self.diff_data.df_inkl.md
        
        idx = (tvd_series < tvd_value).argmin()
        res = (tvd_value - tvd_series[idx-1])/(tvd_series[idx] - tvd_series[idx-1]) * (md_series[idx] - md_series[idx-1]) + md_series[idx-1] 
        return res    

    
    # функция для расчета глубины по стволу при заданной глубине по вертикали 
    def tvd_by_md(self, md_value):
        tvd_series = self.diff_data.df_inkl.tvd
        md_series = self.diff_data.df_inkl.md
        
        idx = (md_series < md_value).argmin()
        res = (md_value - md_series[idx-1])*(tvd_series[idx] - tvd_series[idx-1]) / (md_series[idx] - md_series[idx-1]) + tvd_series[idx-1] 
        return res 
    
    # функция кривизны на глубине уровня
    def get_curvature_by_md(self, md_value):
        curv_series = self.diff_data.df_inkl.curvature
        md_series = self.diff_data.df_inkl.md
        
        idx = (md_series < md_value).argmin()
        res = (md_value - md_series[idx-1])*(curv_series[idx] - curv_series[idx-1]) / (md_series[idx] - md_series[idx-1]) + curv_series[idx-1] 
        return res 
    
    
    # чтение замеров давления и производной из файла excel. 
    def read_pressure_data(self, filename):
        df = pd.read_excel(filename)
        df.columns = ['date_time', 'th', 'p', 'dp']
        # n = df.shape[0]
        df['delta_p'] = self._research_sign * (df.loc[:, 'p'] - self.diff_data.pw)  # расчет изменения давления относительно забойного давления
        df['H_tvd'] = self.diff_data.tvd_vdp - df.p*self.G*10_000/(self.G*self.diff_data.rho_liq)  # расчет уровня жидкости в стволе скважины 
        # ToDo нужно учесть замеры уровнемером типа if self.research_type == 'КВУ'...
        df['H_md'] = df['H_tvd'].apply(lambda x: self.md_by_tvd(x))
        df['curvature'] = df['H_md'].map(self.get_curvature_by_md)
        
        # Использовать траекторию для корректировки величины дебита
        flag = int(self.diff_data.use_trajectory)
        
        h_dyn_tvd = self.diff_data.tvd_vdp - self.diff_data.pw*self.G*10_000/(self.G*self.diff_data.rho_liq)
        h_dyn = self.md_by_tvd(h_dyn_tvd)
        if self.diff_data.derivative_from == 'table':
            df['dH_dt'] = df['dp'] * 98_066.5 / (9.80665 * self.diff_data.rho_liq) * (df['curvature']*flag + (1-flag))  # проверить знак для КПД !!!!!!!!!
        elif self.diff_data.derivative_from == 'calculation':
            df['dH_dt'] = - self._research_sign * (df.H_md.diff()/df.th.diff())  # скорость изменения уровня                    
            df.loc[0, 'dH_dt'] = - self._research_sign * (df.H_md[0] - h_dyn)/df.th[0]  # расчет скорости в первой точке, т.к. выше не рассчитывает
        else:
            print('To derivative_from use one of the values: ["table", "calculation"]. The value is set to "table"')            
        

        if self.diff_data.use_changing_wbs:
            key_h = (df['H_md']>0).astype('int')  # уровень выше  или ниже устья
        else:
            if self.diff_data.wbs_type == 'Сжимаемость':
                key_h = 0
            else:
                key_h = 1
            # get_wbs_type
            # # Место уровня жидкости. Варианты: ['Затрубное пространство', 'НКТ', 'НКТ+затруб', 'Сжимаемость']

        # df['q'] = -(df.dp*c_liq/(G*10_000)*md_bottom*tube_area)*(1-key_h) + (df.diff_h*tube_area/(G*rho_liq))*key_h
        
        df['q'] = (df.dp*self.diff_data.c_liq*self.diff_data.md_bottom*self.diff_data.tube_area)*(1-key_h)\
                                                                                                   + (df['dH_dt']*self.diff_data.tube_area)*key_h    
        # Варианты: ['basic', 'by_first', 'by_max', 'by_volume'] 
        if self.diff_data.q_calculation == 'by_first':  # нормировка первого расчетного дебита к дебиту перед остановкой
            coef = self.diff_data.q / df['q'][0]
        elif self.diff_data.q_calculation == 'by_max':  # нормировка максимального расчетного дебита к дебиту перед остановкой
            coef = self.diff_data.q / df['q'].max()  
        elif self.diff_data.q_calculation == 'by_volume':  # нормировка накопленного расчетного объема к объему ствола скважины при росте уровня
            volume_by_tube = (h_dyn - df['H_md'].iloc[-1]) * self.diff_data.tube_area
            dt = df['th'].diff()
            dt[0] = df['th'][0]
            volume_by_q_time = (df['q'] * dt).sum()
            coef = volume_by_tube / volume_by_q_time
        else:  # без нормировки
            coef = 1
            if self.diff_data.q_calculation != 'basic':
                print(f"q_calculation can not be equal {self.diff_data.q_calculation}. Value set to 'basic'")
                self.diff_data.q_calculation = 'basic'
        
        df['q'] = coef * df['q']
            
        self.data = df
    
    
    # Проверка критериев применимости метода
    def use_criteria(self, t_wbs=-1, t_line=-1, end_point=-1, verbose=False):
        """
            Функция для расчета критериев применимости дифференциального метода, 
            оценки восстановленности давления по эмпирически выведенным критериям.
            Параметры:
                t_wbs - время окончания ВСС, задается вручную или оценивается по диагностическому графику
                t_line - время окончания линейного и билинейного режимов, задается вручную или оценивается по диагностическому графику
                end_point - искуственное ограничение имеющихся данных справа, т.е. считаем длительность меньше, чем у нас имеется, для исследовательских целей
                verbose - вывод сообщений
        """
        research_score = 'средняя'
        
        if end_point > self.data['th'].shape[0]:
            end_point = self.data['th'].shape[0]
            print('Время окончания исследования скорректировано')
        
        if end_point > 2:
            time_series = self.data['th'][:end_point]
            p_series = self.data['p'][:end_point]
        else:
            time_series = self.data['th']
            p_series = self.data['p']
        
        research_time = time_series.max()
        max_time_index = time_series.argmax()
        
        # рассчитать время окончания ВСС или задано 
        if t_wbs < 0:  # == -1
            if verbose: print('Время окончания ВСС: не задано')
            # Актуальность расчета режимов течения пока отсутствует. 
            # При повышении актуальности есть идеи по проработке вопроса (ниже в блоке тестирования). 
        # 1.1 Длительность исследования менее длительности влияния ствола скважины - низкая
        # 1.2 Длительность исследования более длительности влияния ствола скважины - средняя
        criteria_1 = True
        if research_time < t_wbs:
            criteria_1 = False
        
        if t_line < 0:  # == -1
            if verbose: print('Время окончания линейного режима: не задано')
            # Актуальность расчета режимов течения пока отсутствует. 
            # При повышении актуальности есть идеи по проработке вопроса (ниже в блоке тестирования). 
        # Найти линейный режим, точнее линию на диагностическом графике и определить наклон
        # 2.1 Длительность исследования менее длительности окончания линейного/билинейного режима течения - низкая
        # 2.2 Длительность исследования более длительности окончания линейного/билинейного режима течения - средняя
        criteria_2 = True
        if research_time < t_line:
            criteria_2 = False
        
        # Определить стабилизацию по критериям
        # 3.1 Давление стабилизировалось, если изменение за последнии 30 часов составляет менее 1% от изменения за первые 30 часов
        last_30h = research_time - 30
        last_30h_index = np.abs(time_series - last_30h).argmin()
        first_30h_index = np.abs(time_series - 30).argmin()
        first_30h_dp = p_series[:first_30h_index].max() - p_series[:first_30h_index].min() 
        last_30h_dp = p_series[last_30h_index:].max() - p_series[last_30h_index:].min()
        
        if first_30h_dp != 0:
            c_30 = last_30h_dp / first_30h_dp
        else:
            c_30 = 1
            
        if c_30 < 0.01:
            criteria_3_1 = True
        else:
            criteria_3_1 = False                   
        
        # 3.2 Давление стабилизировалось, если изменение за последние сутки менее 0.1 атм
        last_1h = research_time - 1
        last_24h = research_time - 24
        if research_time > 1:
            last_1h_index = np.abs(time_series - last_1h).argmin()
            last_1h_dp = p_series[last_1h_index:].max() - p_series[last_1h_index:].min()            
        else:
            criteria_3_2 = False
            last_1h_dp = -1
            
        if research_time < 24:
            if last_1h_dp > 0:
                last_24h_dp = last_1h_dp * 24
            else:
                criteria_3_2 = False
                last_24h_dp = -1
        else:
            last_24h_index = np.abs(time_series - last_24h).argmin()
            last_24h_dp = p_series[last_24h_index:].max() - p_series[last_24h_index:].min()       
            
        criteria_3_2 = ((last_24h_dp < 0.1) and (last_24h_dp > 0))
                
        # 3.3 Давление стабилизировалось, если отношение изменения давления за последнии 5% длительности исследования 
        #         к депрессии за  90% длительности исследования(с точки остановки) не более 2.5 % (погрешность АГЗУ) 
        first_90t = research_time * 0.9
        first_90t_index = np.abs(time_series - first_90t).argmin()
        last_5t = research_time * 0.95
        last_5t_index = np.abs(time_series - last_5t).argmin()
        
        first_90t_dp = p_series[:first_90t_index].max() - p_series[:first_90t_index].min()  
        last_5t_dp = p_series[last_5t_index:].max() - p_series[last_5t_index:].min()  
        
        if first_90t_dp != 0:
            c_90 = last_5t_dp / first_90t_dp
        else:
            c_90 = 1
        
        criteria_3_3 = (c_90 < 0.025 )
        criteria_3 = criteria_3_1 or criteria_3_2 or criteria_3_3
        full_criteria = criteria_1 and criteria_2 and criteria_3
        if not full_criteria:
            research_score = 'низкая' 
        
        return {
            'score': research_score,
            'criteria_1': criteria_1,
            'criteria_2': criteria_2,
            'criteria_3_1': criteria_3_1,
            'criteria_3_2': criteria_3_2,
            'criteria_3_3': criteria_3_3,
            'first_30h_dp': first_30h_dp,
            'last_30h_dp': last_30h_dp,
            'last_1h_dp': last_1h_dp,
            'last_24h_dp': last_24h_dp,
            'first_90t_dp': first_90t_dp,
            'last_5t_dp': last_5t_dp,
            'conditions': '''1. Исследование длится больше времени ВСС. \n2. Исследование длится больше линейного режима.\n3.1 30 часов с начала и конца при 1% погрешности. \n3.2 Скорость меньше 0.1 ат./сут. \n3.3 Сравнение первых 90% с последними 5% при 2.5% погрешности.'''
        }
    
    
    # расчет времени, когда изменился тип ВСС
    def calculate_changing_wbs_time(self):
        # Для КПД 
        changing_idx = (self.data['p'] < self.diff_data.p_gs).argmax()
        self.diff_data.changing_wbs_time = self.data.th[changing_idx]
        
    
    # вручную задать врмя изменения ВСС и пересчитать плотность
    def set_changing_wbs_time(self, new_time):        
        self.diff_data.changing_wbs_time = new_time        
        df = self.data
        d_min = (-df.dp*df.th).min()
        d_max = (-df.dp*df.th).max()
        # changing_idx = df[df.th > changing_wbs_time].iloc[0].name
        changing_idx = (df.th > new_time).argmax()        
        # Изменить плотность по данным
        self.diff_data.rho_liq = df.p[changing_idx]*10_000/self.diff_data.tvd_vdp
        self.diff_data.p_gs = self.diff_data.tvd_vdp*self.diff_data.rho_liq/10_000
    
    
    # алгоритм автоматического определения Рпл дифф. методом
    def fit_pressure_model(self, model=LinearRegression(), left_bound=1, right_bound=-1, min_point_count = 5, use_log_point_score=False, invert_axes=False):
        """
        Алгоритм подбирает интервал данных для построения линии регрессии для определения пластового давления (Рпл). 
        В качестве метрики оценки качества модели используется коэффициент корреляции Пирсона (R2).
        Параметры:
            model - линейная модель из библиотеки sklearn;
            left_bound - int < n или float < 1: левая граница, из которой начинается поиск (вправо) наилучшего результата;
            right_bound - int < n или float < 1: правая граница, зафиксированная (индекс);
            min_point_count - int > 3: минимальное количество точек в выборке;
            # invert_axes - поменять местами оси при поиске модели, пока не реализовано;
            use_log_point_score - bool: при оценке качества модели дополнительно используется логарифм количества точек, 
                                  чтобы штрафовать модель за малое количество точек (малое количество точек приводит к получению более высокого качества).
        """
        # если поменять местами оси, то линия регрессии будет другой
        if invert_axes:      
            print('Значение invert_axes на данный момент не учитывается. Перезапустите расчет, изменив значение параметра.')
            return
            # пока только для линейной регрессии, в разработке 
        else:
            x_series = self.data[['q']].values
            y_series = self.data['p'].values
            
        max_score = 0
        n = x_series.shape[0]  
        # проверка данных на корректность
        if n < 5:
            print('Количество точек должно быть больше 5. Расчет не выполнен.')
            return
        if min_point_count < 3:
            min_point_count = 3
            print('Минимальное количество точек min_point_count установлено равным 3.')
            
        # установка границ
        if (0 < left_bound) and (left_bound < 1):
            left_bound = int(left_bound * n)
        if (0 < right_bound) and (right_bound < 1):
            right_bound = int(right_bound * n)
        
        # проверка данных на корректность
        if right_bound == -1:
            right_bound = n - 1 # установка правой границы
        elif right_bound > n - 1:
            right_bound = n - 1
            print(f'Правая граница скорректирована. Новое значение: {right_bound}')
        if left_bound > right_bound - min_point_count:
            print('Неверно заданы границы поиска. Расчет не выполнен.')
            return
                    
        # if (left_bound < 1) or (left_bound > n - min_point_count):
        #     left_bound = n - min_point_count
        # bound = left_bound
        
        # ToDo нужна проверка данных на предмет вылетов на последних точках
        # ToDo возможно нужно корректировать правую границу

        # поиск 
        scores = []
        lb_indexes = []
        pressures = []
        best_bound = left_bound
        
        for i_bound in range(left_bound, right_bound + 1 - min_point_count):
            x = x_series[i_bound:right_bound + 1]
            y = y_series[i_bound:right_bound + 1]
            
            if use_log_point_score:
                log_coef = ((np.log(right_bound - i_bound))/np.log(right_bound - min_point_count) + 1 )/2
            else:    
                log_coef = 1

            model.fit(x, y)
            line = model.predict(x)
            score = r2_score(y, line)*log_coef
            if score < 0:
                score = 0

            scores.append(score)
            lb_indexes.append(i_bound)
            pressures.append(model.predict([[0]])[0])

            # ToDo проверка знака для КПД !!!!!!!!!!!!!!!!!!
            if (score > max_score) and (model.coef_[0]<0) and (model.intercept_>self.data['p'].max()):  # ограничение на Рпл относительно всех точек данных
                max_score = score
                best_bound = i_bound

        if max_score == 0:  # В случае, если дифф.метод не смог получить результат
            x_res = np.array([[0], [self.diff_data.q]])
            y_res = np.array([self.data['p'].max(), self.diff_data.pw])
        else:
            x_res = x_series[best_bound:right_bound+1]
            y_res =y_series[best_bound:right_bound+1]
            
        model.fit(x_res, y_res)
        df = pd.DataFrame(data={'lb_indexes': lb_indexes, 
                                'pressures': pressures, 
                                'scores': scores})
        self.model_fitted = True
        self.model = model
        self.left_bound = best_bound
        self.right_bound = right_bound
        self.ipr_data = df
        # return model, df, best_bound 
        
    
    # алгоритм автоматического определения Рпл дифф. методом
    def fit_pressure_model_2d(self, model=LinearRegression(), left_bound=1, min_point_count = 5, invert_axes=False, use_log_point_score=True, verbose=False):
        """
        Алгоритм подбирает интервал данных для построения линии регрессии для определения пластового давления (Рпл). 
        В качестве метрики оценки качества модели используется коэффициент корреляции Пирсона (R2).
        Параметры:
            model - линейная модель из библиотеки sklearn
            не используется  # left_bound - левая граница, из которой начинается поиск (вправо) наилучшего результата
            min_point_count - минимальное количество точек в выборке
            invert_axes - поменять местами оси при поиске модели, пока не реализовано
            use_log_point_score - при оценке качества модели дополнительно используется логарифм количества точек, 
                                  чтобы штрафовать модель за малое количество точек (малое количество точек приводит к получению более высокого качества)
            verbose - вывод процесса расчета.
        """
        # если поменять местами оси, то линия регрессии будет другой
        if invert_axes:      
            print('Значение invert_axes на данный момент не учитывается')
            # пока только для линейной регрессии, в разработке 
        else:
            x_series = self.data[['q']].values
            y_series = self.data['p'].values
            
        max_score = 0
        # bound = left_bound
        n = x_series.shape[0]  
        right_bound = n  # пока расчет поиска по правой границе до конца таблицы замеров
        
        # проверка данных на корректность
        if n < 5:
            print('Количество точек должно быть больше 5. Расчет не выполнен.')
            return
        if min_point_count < 3:
            min_point_count = 3
            print('Минимальное количество точек min_point_count установлено равным 3.')
            
        # установка границ
        if (0 < left_bound) and (left_bound < 1):
            left_bound = int(left_bound * n)
        if (0 < right_bound) and (right_bound < 1):
            right_bound = int(right_bound * n)
        
        # проверка данных на корректность
        if right_bound == -1:
            right_bound = n - 1 # установка правой границы
        elif right_bound > n - 1:
            right_bound = n - 1
            print(f'Правая граница скорректирована. Новое значение: {right_bound}')
        if left_bound > right_bound - min_point_count:
            print('Неверно заданы границы поиска. Расчет не выполнен.')
            return

        # ToDo нужна проверка данных на предмет вылетов на последних точках

        # поиск 
        # проходимся по границам от 0 до n - min_point_count, матрица размерностью (левая граница х правая граница)
        k = n - min_point_count
        dim_index = range(0, n - min_point_count), range(min_point_count, n)
        
        shape = (n, n)
        scores = np.zeros(shape)
        lb_indexes = np.zeros(shape)
        rb_indexes = np.zeros(shape)
        pressures = np.zeros(shape)
        productivity_index = np.zeros(shape)
        best_left = 0
        best_right = n -1
        for r_bound in range(n-1, min_point_count, -1):
            if verbose: print(f'calculating r_bound={r_bound} in range({n-1}, {min_point_count}, -1)   max_score = {max_score}')
            for l_bound in range(r_bound - min_point_count - 1, -1, -1):
                x = x_series[l_bound: r_bound]
                y = y_series[l_bound: r_bound]
                if use_log_point_score:
                    log_coef = ((np.log(r_bound - l_bound))/np.log(n) + 1 )/2
                else:    
                    log_coef = 1

                model.fit(x, y)
                line = model.predict(x)
                score = r2_score(y, line) * log_coef
                if score < 0: 
                    score = 0

                scores[l_bound, r_bound] = score
                lb_indexes[l_bound, r_bound] =  l_bound  
                rb_indexes[l_bound, r_bound] =  r_bound
                pressures[l_bound, r_bound] =  model.predict([[0]])[0]
                if model.coef_[0] == 0:
                    max_prev = productivity_index[l_bound, r_bound].max()
                    productivity_index[l_bound, r_bound] = max_prev
                else:
                    productivity_index[l_bound, r_bound] = 1/model.coef_[0]

                # ToDo проверка знака для КПД !!!!!!!!!!!!!!!!!!
                if (score > max_score) and (model.coef_[0]<0) and (model.intercept_>self.data['p'].max()):
                    max_score = score
                    best_left = l_bound
                    best_right = r_bound

        if max_score == 0:  # В случае, если дифф.метод не смог получить результат
            # ИД при Рпл=Рпоследняя
            x_res = np.array([[0], [self.diff_data.q]])
            y_res = np.array([self.data['p'].max(), self.diff_data.pw])
        else:
            x_res = x_series[best_left: best_right]
            y_res =y_series[best_left: best_right]
            
        model.fit(x_res, y_res)
        
        df = {'lb_indexes': lb_indexes, 
              'rb_indexes': rb_indexes, 
              'pressures': pressures, 
              'scores': scores,
              'productivity_index': productivity_index}
        
        self.fit_pressure_model(left_bound=best_left-1, use_log_point_score=use_log_point_score, min_point_count=min_point_count, right_bound=best_right)
        
        self.model_fitted_2d = True
        self.model_2d = model
        self.left_bound = best_left
        self.right_bound = best_right
        self.ipr_data_add = df
        # return model, df, best_bound 
        
    
    def get_PI(self):
        if self.model_fitted:
            return self.diff_data.q / (self.model.intercept_ - self.diff_data.pw) * self._research_sign
        else:
            print('model is not fitted. use .fit before get PI')

    
    def plot_loglog(self, selected=None):
        # changing_wbs_time = 1
        if self.diff_data.changing_wbs_time is not None:
            x0 = self.diff_data.changing_wbs_time
        else:
            x0 = -1
        
        df = self.data
        
        d_min = (df.dp*df.th).min()
        d_max = (df.dp*df.th).max()
        # changing_idx = (df.th > changing_wbs_time)[0]
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df.th, y=df.delta_p, mode='markers', showlegend=True, name='p'))
        fig.add_trace(go.Scatter(x=df.th, y=df.dp*df.th, mode='markers', showlegend=True, name='dp'))
        fig.add_trace(go.Scatter(x=df.th, y=df.dp.diff()*df.th, mode='markers', showlegend=True, name='d2p'))
        
        fig.add_trace(go.Scatter(
            x=[x0, x0], 
            y=[d_min, d_max], 
            mode='lines', showlegend=True, name='Изменение ВСС'))
        # fig.add_shape(type="line", x0=x0, y0=d_min, x1=x0, y1=d_max, line=dict(color="red", width=2, ), showlegend=True, name='Изменение ВСС')
        
        if selected is not None:
            if len(selected) == 2:
                x_series = df.th[selected[0]: selected[1]]
                y_series = x_series * df.dp[selected[0]: selected[1]]
                fig.add_trace(go.Scatter(x=x_series, 
                                         y=y_series, 
                                         mode='markers', showlegend=True, name='dp selected'))

        fig.update_layout(width=800, height=500, title="Диагностический график", plot_bgcolor="white")
        
        fig.update_xaxes(type="log", dtick=1, gridcolor='black', linecolor='black', mirror=True)
        fig.update_yaxes(type="log", dtick=1, gridcolor='black', linecolor='black', mirror=True, scaleanchor = "x", scaleratio = 1)

        fig.show()

        
    def plot_complex(self, selected=None):
        """
            Функция для построения 2D графиков по результатам анализа:
                1. График замера давления с выделенным интервалом, по которому строится тренд для определения Рпл
                2. Информация по критериям применимости диффренциального метода
                3. Графики замера, трендов ИД в координатах дебит-давление
                4. График критерия выбора границ для тренда для определения пластового давления
                5. Диагностический график (log-log)
                6. График изменения уровня (расчетного) в стволе скважины от времени с прогнозом при допущении, что скорость изменения уровня не меняется со временем
            Параметры:
                selected - дополнительно выделяет интервал времени на диагностическом графике
        """
        if not self.model_fitted:
            print('Model not fitted. Use .fit_pressure_model to view results of calculations')
            return
        
        d_min = (self.data.dp*self.data.th).min()
        d_max = (self.data.dp*self.data.th).max()

        fig = make_subplots(rows=3, cols=2,
                            subplot_titles=('History plot [1]', 'Info [2]',
                                            'IPR [3]', 'R2 score [4]', 
                                            'Log-log [5]', 'Level [6]'),  
                            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                   [{"secondary_y": False}, {"secondary_y": True}],
                                   [{"secondary_y": False}, {"secondary_y": False}]]
                           )
        # Замер давления 1-1
        fig.add_trace(go.Scatter(x=self.data['th'], 
                                 y=self.data['p'], 
                                 mode='markers', 
                                 showlegend=True, 
                                 name='[1]: p', 
                                 yaxis='y', 
                                 marker=dict(color='green'))
                      , 1, 1)
        # fig.add_trace(go.Scatter(x=self.data['th'][self.left_bound: self.right_bound], 
        #                          y=self.data['p'][self.left_bound: self.right_bound], 
        #                          mode='markers', 
        #                          showlegend=True, 
        #                          name='[1]: Выбранные данные', 
        #                          yaxis='y')
        #               , 1, 1)        
        fig.add_vrect(
            x0=self.data['th'][self.left_bound], 
            x1=self.data['th'][self.right_bound],
            fillcolor="LightSalmon", 
            opacity=0.5,
            layer="below", 
            line_width=0, 
            col=1, row=1)
        
        fig.update_xaxes(title_text="time, h", row=1, col=1)
        fig.update_yaxes(title_text="pressure", row=1, col=1, scaleanchor='y')
        
        # Info 1-2
        criteria_map = {
            True: 'выполняется',
            False: 'не выполняется'
        }
        color_map = {
            True: 'green',
            False: 'red'
        }
        """
        'score': research_score,
            'criteria_3_1': criteria_3_1,
            'criteria_3_1': criteria_3_1,
            'criteria_3_1': criteria_3_1,
            'first_30h_dp': first_30h_dp,
            'last_30h_dp': last_30h_dp,
            'last_1h_dp': last_1h_dp,
            'last_24h_dp': last_24h_dp,
            'first_90t_dp': first_90t_dp,
            'last_5t_dp': last_5t_dp,
        """
        conditions = self.use_criteria()
        model = LinearRegression().fit(self.data[['th']][self.left_bound:self.right_bound], self.data['p'][self.left_bound:self.right_bound])
        fig.add_trace(go.Scatter(x=[30],
                                 y=[0],                                 
                                 showlegend=False,
                                 marker=dict(color='white')),
                      1, 2)                      
        # надписи в обратном порядке
        fig.add_trace(go.Scatter(x=[2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0],
                                 y=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                 mode="markers+text",
                                 name="[2]: Info Text",
                                 text=[f"  за выбранный период: {24*model.coef_[0]:.3f} ат./сут",
                                       "Скоровть восстановления давления",
                                       f"  Pt(95-100)/Pt(0-90): {conditions['last_5t_dp']:.3f} / {conditions['first_90t_dp']:.3f} = {conditions['last_5t_dp']/conditions['first_90t_dp']:.4f}",
                                       f" Критерий 3.3 {criteria_map[conditions['criteria_3_3']]} (2.5%)",
                                       f"  за крайние 24ч: {conditions['last_24h_dp']:.4f} ат./сут",
                                       f" Критерий 3.2 {criteria_map[conditions['criteria_3_2']]} (0.1 ат./сут)",
                                       f"   за 30ч: {conditions['last_30h_dp']:.3f} / {conditions['first_30h_dp']:.3f} = {conditions['last_30h_dp']/conditions['first_30h_dp']:.4f}",
                                       f" Критерий 3.1 {criteria_map[conditions['criteria_3_1']]} (1%)",
                                       f" Критерий 2 {criteria_map[conditions['criteria_2']]}",
                                       f" Критерий 1 {criteria_map[conditions['criteria_1']]}",
                                       f" Достоверность исследования: {conditions['score']}"],
                                 textposition="middle right",
                                 textfont=dict(
                                        family="sans serif",
                                        size=14,
                                        color=['black', 
                                               'black', 
                                               'black', 
                                               color_map[conditions['criteria_3_3']], 
                                               'black', 
                                               color_map[conditions['criteria_3_2']],
                                               'black', 
                                               color_map[conditions['criteria_3_1']],
                                               color_map[conditions['criteria_2']],
                                               color_map[conditions['criteria_1']],
                                               'black']),
                                marker=dict(color='black')),
                     1, 2)
                
        # ИД 2-1
        p_pl = self.model.intercept_
        q_left = self.data['q'].iloc[self.left_bound]
        p_left = self.model.predict([[q_left]])[0]
        
        fig.add_trace(go.Scatter(x=self.data['q'], 
                                 y=self.data['p'], 
                                 mode='markers', 
                                 showlegend=True, 
                                 name='[3]: Замеры', 
                                 yaxis='y3', 
                                 marker=dict(color='green'))
                      , 2, 1)
        # fig.add_trace(go.Scatter(x=self.data['q'][self.left_bound: self.right_bound], 
        #                          y=self.data['p'][self.left_bound: self.right_bound], 
        #                          mode='markers', 
        #                          showlegend=True, 
        #                          name='[3]: Выбранные данные')
        #               , 2, 1)
        fig.add_hrect(
            y0=self.data['p'][self.left_bound], 
            y1=self.data['p'][self.right_bound],
            fillcolor="LightSalmon", 
            opacity=0.5,
            layer="below", 
            line_width=0, 
            col=1, row=2)
        fig.add_trace(go.Scatter(x=[self.diff_data.q], 
                                 y=[self.diff_data.pw], 
                                 mode='markers', 
                                 showlegend=True, 
                                 name='[3]: Режим работы скважины')
                      , 2, 1)
        fig.add_trace(go.Scatter(x=[0, (self.data['p'][self.left_bound] - p_pl)/self.model.coef_[0]], 
                                 y=[p_pl, self.data['p'][self.left_bound]], 
                                 mode='lines', 
                                 showlegend=True, 
                                 name='[3]: Прогноз Рпл',
                                 line=dict(color='red'))
                      , 2, 1)
        fig.add_trace(go.Scatter(x=[0], 
                                 y=[p_pl], 
                                 mode='markers', 
                                 showlegend=True, 
                                 name=f'[3]: Рпл={p_pl:.2f}')
                      , 2, 1)
        fig.add_trace(go.Scatter(x=[0, self.diff_data.q], 
                                 y=[p_pl, self.diff_data.pw], 
                                 mode='lines', 
                                 showlegend=True, 
                                 name=f'[3]: ИД Кпрод = {self.diff_data.q / (p_pl - self.diff_data.pw):.3f}')
                      , 2, 1)
        # fig.add_trace(go.Scatter(x=[0, self.data['q'].max()], 
        #                          y=[self.data['p'].max(), self.data['p'].max()], 
        #                          mode='lines', 
        #                          showlegend=True, 
        #                          name='[3]: Р на последнюю точку')
        #               , 2, 1)
        fig.add_hrect(
            y0=self.data['p'].max()-0.2, 
            y1=self.data['p'].max()+0.2,
            fillcolor="red", 
            opacity=0.9,
            layer="below", 
            line_width=0, 
            col=1, row=2)
        fig.update_xaxes(title_text="Liquid rate (Q)", row=2, col=1)
        fig.update_yaxes(title_text="pressure", row=2, col=1, scaleanchor='y3')

        # Выбор Рпл
        fig.add_trace(go.Scatter(x=self.ipr_data['lb_indexes'], 
                                 y=self.ipr_data['pressures'], 
                                 mode='markers', 
                                 showlegend=True, 
                                 name='[4]: Расчетный Рпл', 
                                 yaxis='y3')
                      , 2, 2, secondary_y=False)
        fig.add_trace(go.Scatter(x=[self.left_bound], 
                                 y=[p_pl], 
                                 mode='markers', 
                                 showlegend=True, 
                                 name='[4]: Выбранное Рпл')
                      , 2, 2, secondary_y=False)
        fig.add_vrect(
            x0=self.left_bound, 
            x1=self.right_bound,
            fillcolor="LightSalmon", 
            opacity=0.5,
            layer="below", 
            line_width=0, 
            col=2, row=2)
        
        # fig.add_trace(go.Scatter(x=[self.ipr_data['lb_indexes'].min(), self.ipr_data['lb_indexes'].max()], 
        #                          y=[self.data['p'].max(), self.data['p'].max()], 
        #                          mode='lines', 
        #                          showlegend=True, 
        #                          name='Р на последнюю точку')
        #               , 2, 2, secondary_y=False)
        fig.add_hrect(
            y0=self.data['p'].max()-0.2, 
            y1=self.data['p'].max()+0.2,
            fillcolor="red", 
            opacity=0.9,
            layer="below", 
            line_width=0,
            editable=True ,
            col=2, row=2)
        fig.update_xaxes(title_text="left bound", row=2, col=2)
        fig.update_yaxes(title_text="pressure", row=2, col=2, secondary_y=False)
        fig.update_yaxes(matches='y3', row=2, secondary_y=False, scaleanchor='y3')
        
        fig.add_trace(go.Scatter(x=self.ipr_data['lb_indexes'], 
                                 y=self.ipr_data['scores'], 
                                 mode='markers', 
                                 showlegend=True, 
                                 yaxis='y2', # !!!!!!!!!!!!
                                 name='[4]: Score', 
                                 marker=dict(color='red'))
                      , 2, 2, secondary_y=True)
        
        fig.update_yaxes(title_text="R2 score", row=2, col=2, secondary_y=True, color='red')
        # fig.update_xaxes(title_text="left bound", row=2, col=2)
        # fig.update_yaxes(title_text="p", row=2, col=2)
        
        # Диагностический график
        fig.add_trace(go.Scatter(x=self.data['th'], 
                                 y=self.data['delta_p'], 
                                 mode='markers', 
                                 showlegend=True, 
                                 name='[5]: p', 
                                 xaxis='x5',
                                 marker=dict(color='green')),
                     3, 1)
        fig.add_trace(go.Scatter(x=self.data['th'], 
                                 y=self.data['dp'] * self.data['th'], 
                                 mode='markers', 
                                 showlegend=True, 
                                 name='[5]: dp',
                                 marker=dict(color='red')),
                     3, 1)
        # fig.add_trace(go.Scatter(x=df.th, y=df.dp.diff()*df.th, mode='markers', showlegend=True, name='d2p'))
        if self.diff_data.changing_wbs_time is not None:
            x0 = self.diff_data.changing_wbs_time
        else:
            x0 = -1
        d_min = (self.data.dp*self.data.th).min()
        d_max = (self.data.dp*self.data.th).max()
        
        fig.add_trace(go.Scatter(x=[x0, x0], 
                                 y=[d_min, d_max], 
                                 mode='lines', 
                                 showlegend=True, 
                                 name='[5]: Изменение ВСС'),
                     3, 1)
        # fig.add_shape(type="line", x0=x0, y0=d_min, x1=x0, y1=d_max, line=dict(color="red", width=2, ), showlegend=True, name='Изменение ВСС')
        fig.add_vrect(
            x0=self.data['th'][self.left_bound], 
            x1=self.data['th'][self.right_bound],
            fillcolor="LightSalmon", 
            opacity=0.5,
            layer="below", 
            line_width=0, 
            col=1, row=3)
        
        if selected is not None:
            if len(selected) == 2:
                # x_series = self.data.th[selected[0]: selected[1]]
                # y_series = x_series * self.data.dp[selected[0]: selected[1]]
                # fig.add_trace(go.Scatter(x=x_series, 
                #                          y=y_series, 
                #                          mode='markers', 
                #                          showlegend=True, 
                #                          name='dp selected'),
                #              3, 1)
                fig.add_vrect(
                    x0=self.data.th[selected[0]], 
                    x1=self.data.th[selected[1]],
                    fillcolor="lightblue", 
                    opacity=0.5,
                    layer="below", 
                    line_width=0, 
                    col=1, row=3)

        fig.update_layout(width=500, height=600, title="Диагностический график", plot_bgcolor="white")     
        fig.update_xaxes(title_text="time, h", type="log", dtick=1, gridcolor='black', linecolor='black', mirror=True, row=3, col=1)
        fig.update_yaxes(title_text="pressure", type="log", dtick=1, gridcolor='black', linecolor='black', mirror=True, row=3, col=1, scaleanchor = "x5", scaleratio = 1)
#         fig.update_yaxes(matches='x', row=3, col=1)
        
        
        # Рост уровня в стволе скважины**
        fig.add_trace(go.Scatter(x=self.data['th'], 
                                 y=self.data['H_md'], 
                                 mode='markers', 
                                 showlegend=True, 
                                 name='[6]: H_md'),
                     3, 2)
        model = LinearRegression().fit(self.data[['th']][self.left_bound:self.right_bound], self.data['H_md'][self.left_bound:self.right_bound])
        zero_level_time = -model.intercept_/model.coef_[0]
        fig.add_trace(go.Scatter(x=[self.data['th'][self.left_bound], self.data['th'][self.right_bound], zero_level_time], 
                                 y=[self.data['H_md'][self.left_bound], self.data['H_md'][self.right_bound], 0], 
                                 mode='lines', 
                                 showlegend=True, 
                                 name='[6]: Прогноз'),
                     3, 2)
        # fig.add_trace(go.Scatter(x=[0.001, 100], 
        #                          y=[0, 0], 
        #                          mode='lines', 
        #                          showlegend=False, 
        #                          name='0 line'),
        #              3, 2)
        fig.add_hrect(
            y0=-2, 
            y1=2,
            fillcolor="lightblue", 
            opacity=0.5,
            layer="below", 
            line_width=0, 
            col=2, row=3)
        
        fig.update_xaxes(type="log", dtick=1, gridcolor='black', linecolor='black', mirror=True, row=3, col=2)
        fig.update_xaxes(title_text="time, h", row=3, col=2)
        fig.update_yaxes(title_text="Level, m", row=3, col=2, autorange='reversed')
        
        fig.update_xaxes(gridcolor='black', linecolor='black', mirror=True, zerolinecolor='black')
        fig.update_yaxes(gridcolor='black', linecolor='black', mirror=True, zerolinecolor='black')
        
        fig.update_xaxes(visible=False, gridcolor=None, linecolor=None, mirror=True, zerolinecolor=None, col=2, row=1)
        fig.update_yaxes(visible=False, gridcolor=None, linecolor=None, mirror=True, zerolinecolor=None, col=2, row=1)

        fig.update_layout(title='Оценка пластового давления', 
                          # autosize=False,
                          width=1200, height=1200,)
            
        fig.show()
        
    def plot_criteria(self):
        n = self.data.shape[0]
        x = list(self.data['th'])
        criteries = [self.use_criteria(end_point=i) for i in range(n)]

        y1 = [int(criteries[i]['criteria_1']) for i in range(n)]
        y2 = [int(criteries[i]['criteria_2']) for i in range(n)]
        y31 = [int(criteries[i]['criteria_3_1']) for i in range(n)]
        y32 = [int(criteries[i]['criteria_3_2']) for i in range(n)]
        y33 = [int(criteries[i]['criteria_3_3']) for i in range(n)]

        trace1 = go.Scatter(x=x, y=y1, mode='markers+lines', name='Критерий 1', yaxis="y33")
        trace2 = go.Scatter(x=x, y=y2, mode='markers+lines', name='Критерий 2', yaxis="y32")
        trace31 = go.Scatter(x=x, y=y31, mode='markers+lines', name='Критерий 3.1', yaxis="y31")
        trace32 = go.Scatter(x=x, y=y32, mode='markers+lines', name='Критерий 3.2', yaxis="y2")
        trace33 = go.Scatter(x=x, y=y33, mode='markers+lines', name='Критерий 3.3', yaxis="y1")

        data = [trace1, trace2, trace31, trace32, trace33]
        layout = go.Layout(
            xaxis=dict(domain=[0, 1]),
            yaxis=dict(domain=[0, 0.19]),
            yaxis2=dict(domain=[0.21, 0.39]),
            yaxis31=dict(domain=[0.41, 0.59]),
            yaxis32=dict(domain=[0.61, 0.79]),
            yaxis33=dict(domain=[0.81, 1])    
        )
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(title='Критерии дифф.метода', 
                                  plot_bgcolor="white",
                                  width=1200, height=600,)
        fig.update_xaxes(title='time, h', gridcolor='black', linecolor='black', mirror=True, zerolinecolor='black', type='log', dtick=1)
        fig.update_yaxes(gridcolor='white', linecolor='black', mirror=True, zerolinecolor='black', range=[0,1])

        fig.show()
        
        