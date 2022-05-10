#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Описание-задачи" data-toc-modified-id="Описание-задачи-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Описание задачи</a></span><ul class="toc-item"><li><span><a href="#Цель-проекта" data-toc-modified-id="Цель-проекта-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Цель проекта</a></span></li><li><span><a href="#Задачи-проекта" data-toc-modified-id="Задачи-проекта-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Задачи проекта</a></span></li></ul></li><li><span><a href="#Подготовка" data-toc-modified-id="Подготовка-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Подготовка</a></span><ul class="toc-item"><li><span><a href="#Вывод" data-toc-modified-id="Вывод-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Вывод</a></span></li></ul></li><li><span><a href="#Анализ" data-toc-modified-id="Анализ-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Анализ</a></span><ul class="toc-item"><li><span><a href="#Выводы" data-toc-modified-id="Выводы-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Выводы</a></span></li></ul></li><li><span><a href="#Обучение" data-toc-modified-id="Обучение-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Обучение</a></span><ul class="toc-item"><li><span><a href="#Линейная-регрессия" data-toc-modified-id="Линейная-регрессия-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Линейная регрессия</a></span></li><li><span><a href="#Случайный-лес" data-toc-modified-id="Случайный-лес-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Случайный лес</a></span></li><li><span><a href="#LightGBM" data-toc-modified-id="LightGBM-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>LightGBM</a></span></li><li><span><a href="#Выводы" data-toc-modified-id="Выводы-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Выводы</a></span></li></ul></li><li><span><a href="#Тестирование" data-toc-modified-id="Тестирование-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Тестирование</a></span><ul class="toc-item"><li><span><a href="#Выводы" data-toc-modified-id="Выводы-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Выводы</a></span></li></ul></li><li><span><a href="#Общие-выводы" data-toc-modified-id="Общие-выводы-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Общие выводы</a></span></li></ul></div>

# #  Прогнозирование заказов такси
# ## Описание задачи

# Компания «Чётенькое такси» собрала исторические данные о заказах такси в аэропортах. Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час. Постройте модель для такого предсказания.
# 
# Значение метрики *RMSE* на тестовой выборке должно быть не больше 48.
# 
# Вам нужно:
# 
# 1. Загрузить данные и выполнить их ресемплирование по одному часу.
# 2. Проанализировать данные.
# 3. Обучить разные модели с различными гиперпараметрами. Сделать тестовую выборку размером 10% от исходных данных.
# 4. Проверить данные на тестовой выборке и сделать выводы.
# 
# 
# Данные лежат в файле `taxi.csv`. Количество заказов находится в столбце `num_orders` (от англ. *number of orders*, «число заказов»).

# ### Цель проекта
# Построить модель для предсказания количества заказов такси 

# ### Задачи проекта
# Для достижения целей проекта необходимо решить следующие задачи:
#  - Загрузить данные и выполнить их ресемплирование по одному часу.
#  - Проанализировать данные - определить трендовую составляющую и периодическую
#  - Обучить разные модели с различными гиперпараметрами. 
#  - Провести тестирование моделей на тестовой выборке размером 10% от исходной.
#  - Сделать выводы и рекомендовать модель для прогнозирования

# ## Подготовка

# Загрузим необходимые библиотеки

# In[1]:


import scipy.signal.signaltools
def _centered(arr, newsize):
    # Return the center newsize portion of the array. 
    newsize = np.asarray(newsize) 
    currsize = np.array(arr.shape) 
    startind = (currsize - newsize) // 2 
    endind = startind + newsize 
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))] 
    return arr[tuple(myslice)] 

scipy.signal.signaltools._centered =_centered


# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from scipy.signal.signaltools import _centered
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split,TimeSeriesSplit, GridSearchCV, ParameterGrid
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt


# Закгрузим файл исходных данных, при этом сразу распарсим дату и установим её как индекс

# In[3]:


df = pd.read_csv('/datasets/taxi.csv', parse_dates=[0], index_col=[0])


# Вывведем информацию о таблице

# In[4]:


df.info()


# И выведем первые 5 строк

# In[5]:


df.head()


# Пропусков в таблице нет, столбцы соответствуют заявленным. В талице 26496 записи. Проверим индекс (дату) на монотонность

# In[6]:


df.index.is_monotonic


# Построим график исходных данных по заказам

# In[7]:


df.plot(figsize=(15,5));


# Почасовой график заказов имеет некоторый тренд на повышение и некоторую переодичность, где пики сменяются минимумами.

# ### Вывод
# Файл открылся без проблем, колонки соответствуют описанию. Пропусков нет. К-во записей - 26496. Индекс - дата + время с периодичностью 10 минут. Индекс монотонен.

# ## Анализ

# Проведём ресемплирование по часам

# In[8]:


df = df.resample('1H').sum()


# In[9]:


df.plot(figsize=(15,5));


# Виден явный тренд на повышение. Что касается переодичной составляющей - попробуем вывести данные за месяц, чтобы лучше рассмотреть 

# In[10]:


df.plot(figsize=(15,5), xlim=('2018-03-01','2018-04-01'));


# Из графика видно, что есть периоды высоких значений и низких. Периодичность явно присутствует. Выведем данные за три дня.

# In[11]:


df.plot(figsize=(15,5), xlim=('2018-03-01','2018-03-03'));


# Видно, что максимальные значения наблюдаются часов в 11.00, а минимальное в районе 6.00. Периодичность примерно сутки. Выделим составляющи.

# In[12]:


decomposed = seasonal_decompose(df, period = 240) 


# Посмотрим на тренд

# In[13]:


decomposed.trend.plot(figsize=(15,5));


# Действительно есть явный тренд на повышение, хоть и колблющийся. Посмотрим на периодическую составлющую. Ограничимся 5 днями

# In[14]:


decomposed.seasonal.plot(figsize=(15,5), xlim=('2018-03-01','2018-03-6'));


# Посмотрим на остаток

# In[15]:


decomposed.resid.plot(figsize=(15,5));


# ### Выводы
# 
# Ряд имеет тренд на повышение с течением времени и наблюдается некоторая периодичная составляющая с периодом сутки.

# ## Обучение

# Копируем исходный датафрэйм

# In[16]:


X = df.copy()


# Объявим функцию ,которая будет добавлять доп. характеристики в фрейм

# In[17]:


def make_features(data, max_lag, rolling_mean_size):
    '''
    Функция добавляет в фрейм номер часа, дня, неделю, смещения и скользящее среднее 
    '''
    data['hour'] = data.index.hour
    data['day'] = data.index.dayofweek
    data['week'] = data.index.week
    
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['num_orders'].shift(lag)

    data['rolling_mean'] = data['num_orders'].shift().rolling(rolling_mean_size).mean()


# Создадим доп. характеристики

# In[18]:


make_features(X,2,24)


# Выведем итоговый фрейм

# In[19]:


X.head()


# Удалим появившиеся пропуски

# In[20]:


X = X.dropna()


# Выделим целевой признак - `num_orders`

# In[21]:


y = X.pop('num_orders')


# Разобъём выборку на тренировочную и тестовую часть. Для теста оставим 10%

# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)


# In[23]:


len(X_train), len(X_test), len(y_train), len(y_test)


# Для подбора параметров на кросс- валидации используем `TimeSeriesSplit`. Зададим кол-во батчей - 8

# In[24]:


tss = TimeSeriesSplit(n_splits=8)


# Для проекта используем модели линейной регрессии, случайного леса и LightGBM, которые будем записывать в *result_models*

# In[25]:


result_models = []


# Значение метрики *RMSE* на тренировочной выборке будем записывать в *train_score*, а на тестовой - в *test_score*

# In[26]:


train_score = []


# In[27]:


test_score = []


# ### Линейная регрессия

# Используем модель линейной регресии

# In[28]:


model = LinearRegression()


# Зададим параметры для перебора

# In[29]:


regression_grid={'fit_intercept':[True,False],'normalize':[True,False]}


# In[30]:


regression_grid_search = GridSearchCV(model, regression_grid, cv=tss, scoring='neg_mean_squared_error', verbose=1,n_jobs=-1)


# Запустим `GridSearch`

# In[31]:


regression_grid_search.fit(X_train, y_train)


# Выведем результаты

# In[32]:


pd.DataFrame(regression_grid_search.cv_results_)


# Лучший результат имеет модель с параметром `fit_intercept` равным *False*. Получим лучшую модель.

# In[33]:


regression_model = regression_grid_search.best_estimator_


# Оценим *RMSE* на тренировочной выборке

# In[34]:


mean_squared_error(y_train, regression_model.predict(X_train), squared=False)


# Метрика нас вполне устраивает. Визуализируем тренировочную выборку и прогноз

# In[35]:


y_train.plot(figsize=(15,5))
pd.Series(regression_model.predict(X_train), index = y_train.index).plot()


# Из графика видно, что регрессия хоть и похожа на тренировочные данные, но различия визуально довольно существенные. Модель регрессии запишем в итоговый список моделей.

# In[36]:


result_models.append(regression_model)


# In[37]:


train_score.append(mean_squared_error(y_train, regression_model.predict(X_train), squared=False))


# ### Случайный лес

# Следующей моделью попробуем `RandomForestRegressor`

# In[38]:


model=RandomForestRegressor(random_state = 42)


# Зададим параметры поиска

# In[39]:


forest_grid={'max_depth':[10, 20, 50, 100],'n_estimators':[10, 20, 30, 40, 50]}


# In[40]:


forest_grid_search = GridSearchCV(model, forest_grid, cv=tss, scoring='neg_mean_squared_error')


# Запустим поиск

# In[41]:


forest_grid_search.fit(X_train, y_train)


# Вывдем результат фреймом и отсортируем по рангу.

# In[42]:


pd.DataFrame(forest_grid_search.cv_results_).sort_values(by='rank_test_score')


# Лучший результат дали модели с глубиной 50 и 100 и к-вом деревьев 50. 

# In[43]:


forest_model = forest_grid_search.best_estimator_


# Оценим *RMSE* на тренировочной выборке

# In[44]:


mean_squared_error(y_train, forest_model.predict(X_train), squared=False)


# Результат лучше чем для регрессии. Визуализируем результат сравнением 2-х графиков

# In[45]:


y_train.plot(figsize=(15,5))
pd.Series(forest_model.predict(X_train), index = y_train.index).plot()


# Результат уже лучше линейной регрессии. Об этом свидетельствует меньшее значение метрики и графическое представление

# In[46]:


result_models.append(forest_model)


# In[47]:


train_score.append(mean_squared_error(y_train, forest_model.predict(X_train), squared=False))


# ### LightGBM

# Создадим модель на основе `LGBMRegressor`

# In[49]:


model = LGBMRegressor(random_state=42, metric='rmse')


# Зададим параметры для поиска

# In[50]:


lgbm_grid={'max_depth':[10, 20],'n_estimators':[30, 40, 50]}


# Объявим поиск на кросс-валидации и запустим

# In[51]:


lgbm_grid_search = GridSearchCV(model, lgbm_grid, cv=tss, scoring='neg_mean_squared_error')


# Запустим

# In[52]:


lgbm_grid_search.fit(X_train, y_train)


# Выведем результаты и отсортируем по лучшей метрике

# In[53]:


pd.DataFrame(lgbm_grid_search.cv_results_).sort_values(by='rank_test_score')


# Из таблицы видно, что лучший результат дала модель с глубиной 10 и к-вом деревьев 50. 

# In[54]:


lgbm_model = lgbm_grid_search.best_estimator_


# Оценим *RMSE* на тренировочной выборке

# In[55]:


mean_squared_error(y_train, lgbm_model.predict(X_train), squared=False)


# Результат несколько хуже леса, но и деревьев не 50, а 40.Визуализируем результат

# In[56]:


y_train.plot(figsize=(15,5))
pd.Series(lgbm_model.predict(X_train), index = y_train.index).plot()


# Результат лучше линейной регрессии на тренировочной выборке, но несколько хуже случайного леса. Запишем результаты

# In[57]:


result_models.append(lgbm_model)


# In[58]:


train_score.append(mean_squared_error(y_train, lgbm_model.predict(X_train), squared=False))


# ### Выводы
# Провели обучение 3-х моделей: линейной регрессии, случайного леса и LightGBM с перебором некоторых параметров. На тренировочной выборке модели расположились в следю порядке по возрастани метрики - Случайный лес, LightGBM, линейная регрессия

# ## Тестирование

# Проведём тестирование моделей. 

# In[59]:


for model in result_models:
    test_score_ = mean_squared_error(y_test, model.predict(X_test), squared=False)#определяем метрку
    print('Модель - {}\nRMSE на тестовой выборке {}'.format(model, test_score_))
    test_score.append(test_score_ )
    y_test.plot(figsize=(15,5))#строим график для визуализации
    pd.Series(model.predict(X_test), index = y_test.index).plot()
    plt.show()


# In[ ]:





# На тестовой выборке лучший результат показала модель LightGBM

# In[60]:


pd.DataFrame(
    {'model':['LinearRegresson', 'RandomForestRegressor', 'LGBMRegressor'], 
     'train_score':train_score, 
     'test_score':test_score}).sort_values(by='test_score')


# ### Выводы
# Модели случайного леса и LightGBM уложились в заданную метрику (менне 48). Модель регрессии показала метрику выше заданной. Для улучшения регрессии вероятно следует увеличить к-во лагов или добавить разности значений. Или вообще проверить на стационарность кого-нибудь порядка. Следует отметить, что (сходя из графиков) первые две модели ухудшают свой прогноз к концу горизонта прогнозирования (разница между фактическим и предсказываемым значением увеличивается). Для линейной регресии - расхождение более стабильно. Вероятно наилучшим решением будет использовать гибридные модели регрессия + LightGBM

# ## Общие выводы
# - При выполнении проекта выполнили их ресемплирование по одному часу, проанализировали данные - определили составляющие и установили, что наблюдается периодичность в заказах с периодом сутки и имеется тренд на повышение к-ва заказов с течением времени. 
# - Добполнили датасет признаками часа, дня и недели, сдвигов и скользящего среднего.
# - Обучили и определили лучшие параметры из списка для моделей линейной регрессии, случайного леса и LightGBM. 
# - Провели тестирование моделей и определили, модель линейной регресии не удовлетворяет требованиям по метрике *RMSE*
# 
# По результатам работы можем рекомендовать модель на основе для LightGBM для использования в задаче прогнозирования
