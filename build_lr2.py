# -*- coding: utf-8 -*-
"""Сборка и выполнение ноутбука ЛР2: качество данных, чистка, статистический анализ."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t))
code = lambda t: cells.append(nbf.v4.new_code_cell(t))

# ============================== ТИТУЛ ==============================
md("""# Лабораторная работа 2. Оценка качества данных, очистка и статистический анализ

**Датасет:** `9. Sales-Data-Analysis.csv` — заказы сети кафе (Лондон, Лиссабон, Мадрид, Париж, Берлин), ноябрь–декабрь 2022 г.

**Колонки:** `Order ID` — номер заказа, `Date` — дата, `Product` — товар, `Price` — цена за единицу, `Quantity` — проданное количество, `Purchase Type` — тип покупки, `Payment Method` — способ оплаты, `Manager` — менеджер, `City` — город.

**План работы:**
1. Загрузка данных
2. Анализ качества (пропуски, границы, дубликаты, корректность категорий)
3. Чистка (с контролем изменений после каждого этапа)
4. Преобразование и добавление новых полей
5. Статистический анализ (типы данных, меры центральной тенденции, меры разброса, выбросы, частотные ряды)""")

# ============================== 1. ЗАГРУЗКА ==============================
md("""## 1. Загрузка данных""")
code("""%matplotlib inline
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)
pd.set_option('display.float_format', lambda v: f'{v:,.2f}')""")

code("""# Универсальная загрузка: работает и локально, и в Google Colab.
CANDIDATES = [
    '9. Sales-Data-Analysis.csv',
    '/content/9. Sales-Data-Analysis.csv',   # путь в Colab после загрузки в панель «Файлы»
]
path = next((p for p in CANDIDATES if os.path.exists(p)), None)

if path is None:
    # В Colab: откроется диалог выбора файла
    from google.colab import files
    print('Файл не найден, выберите CSV в диалоге загрузки:')
    uploaded = files.upload()
    path = next(iter(uploaded))

df_raw = pd.read_csv(path)
df = df_raw.copy()
print('Загружен файл:', path)
print('Размер таблицы:', df.shape)""")

code("""df.head(10)""")
code("""df.info()""")
code("""df.describe(include='all')""")

# ============================== 2. КАЧЕСТВО ==============================
md("""## 2. Анализ качества данных

Проверяем: **пропущенные значения**, **дубликаты**, **границы значений** числовых полей и дат, **корректность категорий** в номинальных признаках.

### 2.1. Пропущенные значения""")
code("""na = df.isna().sum()
print('Пропущенных значений по колонкам:')
print(na)
print()
print('Итого пропусков в таблице:', int(na.sum()))""")

md("""**Вывод:** пропусков в данных нет — все 254 строки заполнены по всем 9 колонкам.""")

md("""### 2.2. Дубликаты""")
code("""print('Полных дубликатов строк:', int(df.duplicated().sum()))
print('Повторяющихся Order ID:', int(df['Order ID'].duplicated().sum()))

dup_id = df[df['Order ID'].duplicated(keep=False)]
if not dup_id.empty:
    display(dup_id.sort_values('Order ID'))
else:
    print('Каждый Order ID встречается ровно один раз — дубликатов заказов нет.')""")

md("""**Вывод:** полных дубликатов строк и повторяющихся номеров заказов нет. (Проверка повторяется после чистки, т.к. «дубликаты из-за лишних пробелов» могли бы появиться/исчезнуть при нормализации текста.)""")

md("""### 2.3. Границы значений числовых полей""")
code("""num_cols = ['Price', 'Quantity']
print('Диапазоны числовых полей:')
display(df[num_cols].agg(['min', 'max', 'mean']).T)

print('Уникальные значения Price:', sorted(df['Price'].unique()))
print()
print('Признак: у каждого товара должна быть одна справочная цена.')
print('Фактические цены по товарам (число строк с каждой ценой):')
display(df.groupby('Product')['Price'].value_counts().to_frame('n'))""")

md("""**Проблема найдена:** у каждого товара ровно одна «каталожная» цена, кроме двух строк-выбросов:
- `Fries` с ценой **25.5** (каталожная цена картофеля — 3.49),
- `Chicken Sandwiches` с ценой **29.05** (каталожная цена — 9.95).

Это опечатки/ошибки ввода: цены не соответствуют ассортименту (остальные цены лежат в диапазоне 2.95–12.99).

`Quantity` принимает **дробные** значения (например, 573.07 шт.) — количество единиц товара должно быть целым, это аномалия формата, исправим на этапе преобразований.""")

md("""### 2.4. Корректность дат""")
code("""print('Тип колонки Date до преобразования:', df['Date'].dtype)
print('Примеры значений:', df['Date'].head(3).tolist())

parsed = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
print('Невалидных дат (не парсятся):', int(parsed.isna().sum()))
print('Диапазон дат:', parsed.min().date(), '—', parsed.max().date())""")

md("""**Вывод:** все даты записаны в едином формате `ДД-ММ-ГГГГ` и валидны, диапазон — ноябрь–декабрь 2022 г. Проблем нет, но колонка хранится как **строка** (`object`), а не как дата — нужен каст в `datetime64`.""")

md("""### 2.5. Корректность категорий""")
code("""cat_cols = ['Product', 'Purchase Type', 'Payment Method', 'Manager', 'City']
for c in cat_cols:
    print(f'--- {c}: {df[c].nunique()} уникальных значений ---')
    print(df[c].value_counts(dropna=False).to_string())
    print()""")

md("""**Проблема найдена.** Реальных категорий в данных меньше, чем уникальных значений:
- `Manager`: **14 вариантов вместо 5** — лишние пробелы внутри имён и по краям (`\"Joao    Silva\"`, `\"       Pablo Perez\"` и т.п.);
- `Purchase Type`: 3 реальные категории (`Online`, `In-store`, `Drive-thru`), но у всех хвостовой пробел;
- `Payment Method`: 3 реальные категории (`Cash`, `Credit Card`, `Gift Card`), но у всех ведущий пробел;
- `Product` (5 значений) и `City` (5 значений) — корректны.""")

md("""### 2.6. Сводка проблем качества

| # | Проверка | Результат | Что делаем |
|---|----------|-----------|------------|
| 1 | Пропущенные значения | **Не найдены** | Контрольная проверка после чистки |
| 2 | Дубликаты строк / Order ID | **Не найдены** | Повторная проверка после нормализации |
| 3 | Границы `Price` | **2 аномалии**: 25.5 и 29.05 вне каталожных цен | Восстановить каталожную цену товара |
| 4 | Формат `Quantity` | Дробные значения вместо целых | Округлить и привести к `int` |
| 5 | Тип `Date` | Строка вместо даты | Каст в `datetime64` |
| 6 | Категории `Manager` | 14 «категорий» из-за пробелов | Нормализовать пробелы |
| 7 | Категории `Purchase Type`, `Payment Method` | Ведущие/хвостовые пробелы | Нормализовать пробелы |""")

# ============================== 3. ЧИСТКА ==============================
md("""## 3. Чистка данных

Каждый этап: фиксируем состояние «до» → выполняем преобразование → показываем «после».

### 3.1. Нормализация пробелов в текстовых колонках""")
code("""snapshot = df.copy()
text_cols = df.select_dtypes(include='object').columns

for c in text_cols:
    df[c] = (df[c].astype(str)
                   .str.strip()                              # пробелы по краям
                   .str.replace(r'\\s+', ' ', regex=True))    # двойные пробелы внутри

print('Менеджеры ДО :', snapshot['Manager'].nunique(), 'вариантов')
print('Менеджеры ПОСЛЕ:', df['Manager'].nunique(), 'вариантов')
print()
print('Категории после нормализации:')
for c in ['Manager', 'Purchase Type', 'Payment Method']:
    print(f'--- {c} ---')
    print(df[c].value_counts().to_string())
print()
n_changed = int((snapshot[text_cols] != df[text_cols]).any(axis=1).sum())
print(f'Строк, в которых изменился текст: {n_changed} из {len(df)}')""")

md("""**Изменение:** вместо 14 «разных менеджеров» осталось 5 реальных сотрудников; категории `Purchase Type` и `Payment Method` очищены от пробелов. Сами значения (люди, товары, города) не менялись — только представление текста.""")

md("""### 3.2. Исправление аномальных цен по каталожной цене товара""")
code("""catalog = df.groupby('Product')['Price'].agg(lambda s: s.mode().iloc[0])
print('Каталожные цены (мода по каждому товару):')
print(catalog.to_string())
print()

mask = df['Price'] != df['Product'].map(catalog)
fixed = df.loc[mask, ['Order ID', 'Product', 'Price']].copy()
fixed['Price до исправления'] = fixed['Price']
df.loc[mask, 'Price'] = df.loc[mask, 'Product'].map(catalog)
fixed['Price после исправления'] = df.loc[mask, 'Price'].values

print(f'Найдено и исправлено строк с некаталожной ценой: {int(mask.sum())}')
display(fixed.reset_index(drop=True))
print()
print('Уникальные цены после исправления:', sorted(df['Price'].unique()))""")

md("""**Изменение:** цена — справочная величина, зависящая только от товара, поэтому аномальные значения (25.5 → 3.49 для картофеля, 29.05 → 9.95 для сэндвичей) восстановлены по каталожной цене. Теперь у каждого товара ровно одна цена.""")

md("""### 3.3. Приведение дат к типу datetime""")
code("""print('ДО :', df['Date'].dtype, '| пример:', df['Date'].iloc[0])
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
print('ПОСЛЕ:', df['Date'].dtype, '| пример:', df['Date'].iloc[0].date())
print('Пропусков после парсинга (NaT):', int(df['Date'].isna().sum()))
print('Диапазон дат:', df['Date'].min().date(), '—', df['Date'].max().date())""")

md("""**Измерение:** `Date` теперь настоящий `datetime64` — с датами можно считать (сортировка, разности, извлечение месяца и дня недели).""")

md("""### 3.4. Контрольная проверка дубликатов и пропусков""")
code("""print('Дубликатов строк:', int(df.duplicated().sum()))
print('Повторяющихся Order ID:', int(df['Order ID'].duplicated().sum()))
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f'Строк было: {before}, стало: {len(df)} (удалено {before - len(df)})')
print()
print('Пропусков после чистки:', int(df.isna().sum().sum()))""")

md("""**Изменение:** дубликатов и пропусков нет, размер таблицы не изменился. Чистка завершилась без потери строк — исправлялись только значения внутри ячеек.""")

# ============================== 4. ПРЕОБРАЗОВАНИЯ ==============================
md("""## 4. Преобразование и добавление новых полей

### 4.1. `Quantity` → целое число""")
code("""print('ДО :', df['Quantity'].dtype, '| примеры:', df['Quantity'].head(3).tolist())
df['Quantity'] = df['Quantity'].round().astype(int)
print('ПОСЛЕ:', df['Quantity'].dtype, '| примеры:', df['Quantity'].head(3).tolist())
print('Диапазон Quantity:', df['Quantity'].min(), '—', df['Quantity'].max())""")

md("""Количество единиц товара — целое по смыслу; дробная часть (например, 573.07 шт.) — артефакт источника, устраняем округлением.""")

md("""### 4.2. Новые поля: выручка и календарные признаки""")
code("""df['Revenue'] = df['Price'] * df['Quantity']          # выручка заказа
df['Year'] = df['Date'].dt.year                        # год
df['Month'] = df['Date'].dt.month                      # номер месяца
df['MonthName'] = df['Date'].dt.strftime('%B')         # название месяца
df['Day'] = df['Date'].dt.day                          # день месяца
df['Weekday'] = df['Date'].dt.day_name()               # день недели

print('Новые колонки добавлены. Итоговая структура:')
df.info()""")

code("""df.head(8)""")

# ============================== 5. СТАТИСТИКА ==============================
md("""## 5. Статистический анализ

### 5.1. Типы данных после чистки и преобразований""")
code("""display(df.dtypes.to_frame('Тип данных'))""")

md("""Количественные (`Price`, `Quantity`, `Revenue` и календарные) — числовые типы; дата — `datetime64`; номинальные признаки — `object` (категориальные строки).""")

md("""### 5.2. Меры центральной тенденции""")
code("""num_cols = ['Price', 'Quantity', 'Revenue']
central = pd.DataFrame({
    'Среднее (mean)': df[num_cols].mean(),
    'Медиана (median)': df[num_cols].median(),
    'Мода (mode)': [df[c].mode().iloc[0] for c in num_cols],
})
display(central)

print('Мода номинальных признаков (наиболее частая категория):')
for c in ['Product', 'City', 'Manager', 'Payment Method', 'Purchase Type', 'Weekday']:
    print(f'  {c:15s} -> {df[c].mode().iloc[0]}')""")

md("""Для `Price` все три меры близки (цены почти равномерно распределены по 5 товарам). Для `Quantity` и `Revenue` медиана немного меньше среднего — распределение имеет лёгкую правостороннюю асимметрию из-за крупных заказов.""")

md("""### 5.3. Меры разброса""")
code("""disp = pd.DataFrame({
    'Минимум (min)': df[num_cols].min(),
    'Максимум (max)': df[num_cols].max(),
    'Размах (range)': df[num_cols].max() - df[num_cols].min(),
    'Дисперсия (var)': df[num_cols].var(),
    'Ст. отклонение (std)': df[num_cols].std(),
    'Q1 (25%)': df[num_cols].quantile(.25),
    'Q3 (75%)': df[num_cols].quantile(.75),
    'IQR': df[num_cols].quantile(.75) - df[num_cols].quantile(.25),
})
disp['Коэф. вариации, %'] = df[num_cols].std() / df[num_cols].mean() * 100
display(disp)""")

md("""Цены слабо рассеяны (CV ≈ 60% при всего 5 фиксированных значениях, разброс задаётся ассортиментом). `Quantity` и `Revenue` варьируют сильно (CV > 30%) — заказы заметно различаются по объёму продаж.""")

md("""### 5.4. Выбросы (метод IQR и z-оценка)""")
code("""def iqr_bounds(s):
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

print(f"{'Колонка':10s} {'Границы IQR':>22s} {'Выбросов (IQR)':>15s} {'Выбросов (|z|>3)':>17s}")
for c in num_cols:
    lo, hi = iqr_bounds(df[c])
    out_iqr = df[(df[c] < lo) | (df[c] > hi)]
    z = (df[c] - df[c].mean()) / df[c].std()
    out_z = df[z.abs() > 3]
    print(f'{c:10s} {f"[{lo:8.2f}; {hi:8.2f}]":>22s} {len(out_iqr):15d} {len(out_z):17d}')

print()
z = (df['Revenue'] - df['Revenue'].mean()) / df['Revenue'].std()
print('Единственная строка с |z| > 3 по Revenue:')
display(df.loc[z.abs() > 3, ['Order ID', 'Date', 'Product', 'Quantity', 'Revenue']])

print()
print('Проверка внутри групп (границы IQR отдельно по каждому товару):')
rows = []
for p, g in df.groupby('Product'):
    q1, q3 = g['Revenue'].quantile([.25, .75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = int(((g['Revenue'] < lo) | (g['Revenue'] > hi)).sum())
    rows.append({'Product': p, 'n': len(g), 'Q1': q1, 'Q3': q3, 'IQR': iqr,
                 'Границы': f'[{lo:,.0f}; {hi:,.0f}]', 'Выбросов': n})
display(pd.DataFrame(rows))""")

md("""Формальные методы дают разные результаты по разным признакам, и это важный методологический момент:

- `Price` и `Quantity`: **выбросов нет** — оба метода согласованы;
- `Revenue`: глобальный IQR помечает 102 из 254 строк (~40%). Это **не аномалии**: распределение выручки мультимодально — это смесь пяти ценовых уровней и кластеров количества, а метод IQR корректен для унимодальных распределений и на смесях даёт массовые ложные срабатывания. Групповая проверка внутри каждого товара показывает то же следствие дискретности данных: у `Fries` и `Sides & Other` IQR = 0 (более половины значений совпадают), границы вырождаются в точку;
- правило |z| > 3 находит единственный заказ № 10710 (Burgers, 754 шт., выручка 9 794.46) — это **правдоподобный крупный заказ**, а не ошибка: цена каталожная, количество в допустимых пределах.

**Общий вывод:** после чистки физически некорректных значений в данных не осталось. Показательно, что истинные ошибки ввода (цены 25.5 и 29.05) формальные fence-методы нашли бы не полностью (25.5 не выходит за границу 1.5·IQR) — обе аномалии были выявлены бизнес-правилом «у товара одна каталожная цена» на этапе 2.""")

code("""fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, c in zip(axes, num_cols):
    ax.boxplot(df[c], vert=False)
    ax.set_title(f'Boxplot: {c}')
    ax.set_xlabel(c)
fig.suptitle('Ящики с усами количественных признаков (после чистки)', y=1.04)
plt.tight_layout()
plt.show()""")

md("""### 5.5. Частотные ряды

**Категориальные признаки — дискретные частотные ряды:**""")
code("""fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, c in zip(axes, ['Product', 'City']):
    vc = df[c].value_counts()
    ax.bar(vc.index, vc.values, color='steelblue')
    ax.set_title(f'Частоты: {c}')
    ax.set_ylabel('Число заказов')
    ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.show()

for c in ['Product', 'City', 'Manager', 'Payment Method', 'Purchase Type']:
    freq = pd.DataFrame({
        'Частота': df[c].value_counts(),
        'Отн. частота': df[c].value_counts(normalize=True).round(3),
    })
    print(f'--- {c} ---')
    print(freq.to_string())
    print()""")

md("""**Количественные признаки — интервальные частотные ряды:**""")
code("""for c in ['Quantity', 'Revenue']:
    bins = pd.cut(df[c], bins=6)
    freq = pd.DataFrame({
        'Частота': bins.value_counts().sort_index(),
        'Отн. частота': bins.value_counts(normalize=True).sort_index().round(3),
    })
    freq.index = freq.index.astype(str)
    print(f'--- Интервальный ряд: {c} ---')
    print(freq.to_string())
    print()

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, c in zip(axes, ['Quantity', 'Revenue']):
    ax.hist(df[c], bins=12, color='steelblue', edgecolor='white')
    ax.set_title(f'Гистограмма: {c}')
    ax.set_xlabel(c)
    ax.set_ylabel('Число заказов')
plt.tight_layout()
plt.show()""")

md("""## Выводы

1. **Качество данных.** Пропусков и дубликатов в датасете нет; найденные проблемы — ошибки представления и ввода: лишние пробелы в категориях (14 фиктивных «менеджеров» вместо 5), две аномальные цены (25.5 и 29.05), дробный `Quantity`, дата-строка вместо типа `datetime`.
2. **Чистка.** Все проблемы устранены: текст нормализован, цены восстановлены по каталожной цене товара, `Date` приведена к `datetime64`, `Quantity` — к `int`. Потери строк — 0 (исправлялись значения, а не записи).
3. **Новые поля.** Добавлены выручка заказа `Revenue = Price × Quantity` и календарные признаки (год, месяц, день, день недели).
4. **Статистика.** Данные описаны мерами центральной тенденции (среднее/медиана/мода), разброса (размах, дисперсия, стандартное отклонение, IQR, коэффициент вариации), формально проверены на выбросы (IQR, z-оценка) и представлены частотными рядами — дискретными для категориальных и интервальными для количественных признаков.""")

nb['cells'] = cells
nb['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb['metadata']['language_info'] = {'name': 'python'}

OUT = 'LR2_Sales_Data_Analysis.ipynb'

# 1) чистый ноутбук (без выводов)
nbf.write(nb, OUT)
print('Ноутбук собран:', OUT, '| ячеек:', len(cells))

# 2) выполнение для проверки и встраивания результатов
from nbclient import NotebookClient
client = NotebookClient(nb, timeout=300, kernel_name='python3',
                        resources={'metadata': {'path': '/Users/dllnnx/Documents/vizdan'}})
client.execute()
nbf.write(nb, OUT)
print('Ноутбук выполнен без ошибок, выводы встроены:', OUT)
