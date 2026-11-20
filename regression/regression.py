import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

sns.set_style('whitegrid')

# ============================================================
# 1. Загрузка и объединение данных
# ============================================================
red   = pd.read_csv('winequality-red.csv',   sep=';')
white = pd.read_csv('winequality-white.csv', sep=';')

red['wine_type']   = 0   # 0 = красное
white['wine_type'] = 1   # 1 = белое

df = pd.concat([red, white], ignore_index=True)

print('Размер:', df.shape)
print('\nquality:')
print(df['quality'].value_counts().sort_index())
print('\nПропуски:', df.isnull().sum().sum())

# ============================================================
# 2. Распределение целевой переменной (EDA)
# ============================================================
plt.figure(figsize=(9, 5))
sns.countplot(x='quality', hue='wine_type', data=df,
              palette={0: '#8B0000', 1: '#F5DEB3'})
plt.title('Распределение оценок качества (0=красное, 1=белое)')
plt.xlabel('quality'); plt.ylabel('Количество')
plt.show()

# ============================================================
# 3. Корреляционная матрица (EDA)
# ============================================================
plt.figure(figsize=(13, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', center=0)
plt.title('Матрица корреляций (объединённый датасет)')
plt.tight_layout()
plt.show()

# ============================================================
# 4. Гистограммы признаков (EDA)
# ============================================================
df.hist(figsize=(16, 14), bins=30, color='steelblue', edgecolor='black')
plt.tight_layout()
plt.show()

# ============================================================
# 5. Feature Engineering
# ============================================================
df['total_acidity'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']
df['sulfur_ratio']  = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-6)

print('\nКолонки после FE:')
print(df.columns.tolist())
print('Размер:', df.shape)

# ============================================================
# 6. Разделение выборок
# ============================================================
X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'\nTrain: {X_train.shape}')
print(f'Test:  {X_test.shape}')

# ============================================================
# 7. Baseline
# ============================================================
baseline_value = y_train.mean()
y_pred_baseline = np.full_like(y_test, baseline_value, dtype=float)

rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
mae_baseline  = mean_absolute_error(y_test, y_pred_baseline)

print(f'\nBaseline (константа = {baseline_value:.3f})')
print(f'RMSE = {rmse_baseline:.4f}')
print(f'MAE  = {mae_baseline:.4f}')

# ============================================================
# 8. Linear Regression (по типам вина + log1p + клиппинг + полиномы)
# ============================================================
def lr_per_type_poly(X_tr, y_tr, X_te):
    mask_tr_red = X_tr['wine_type'] == 0
    mask_te_red = X_te['wine_type'] == 0
    pred = np.zeros(len(X_te))
    skewed = ['residual sugar', 'chlorides', 'free sulfur dioxide',
              'total sulfur dioxide', 'sulphates']

    for is_red in [True, False]:
        tr = mask_tr_red if is_red else ~mask_tr_red
        te = mask_te_red if is_red else ~mask_te_red
        Xtr = X_tr.loc[tr].drop(columns='wine_type').copy()
        Xte = X_te.loc[te].drop(columns='wine_type').copy()

        # log1p и клиппинг скошенных признаков
        for col in skewed:
            Xtr[col] = np.log1p(Xtr[col])
            Xte[col] = np.log1p(Xte[col])
        for col in Xtr.columns:
            lo, hi = Xtr[col].quantile([0.01, 0.99])
            Xtr[col] = Xtr[col].clip(lo, hi)
            Xte[col] = Xte[col].clip(lo, hi)

        # Масштабирование и полиномиальные взаимодействия
        sc = StandardScaler().fit(Xtr)
        Xtr_s = sc.transform(Xtr)
        Xte_s = sc.transform(Xte)
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        Xtr_p = poly.fit_transform(Xtr_s)
        Xte_p = poly.transform(Xte_s)

        m = LinearRegression().fit(Xtr_p, y_tr.loc[tr])
        pred[te] = m.predict(Xte_p)

    return pred

y_pred_lr = lr_per_type_poly(X_train, y_train, X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr  = mean_absolute_error(y_test, y_pred_lr)

print('\nLinear Regression (Poly взаимодействия по типам)')
print(f'RMSE = {rmse_lr:.4f}')
print(f'MAE  = {mae_lr:.4f}')

# ============================================================
# 9. Random Forest
# ============================================================
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf  = mean_absolute_error(y_test, y_pred_rf)

print('\nRandom Forest')
print(f'RMSE = {rmse_rf:.4f}')
print(f'MAE  = {mae_rf:.4f}')

importances_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(9, 7))
importances_rf.plot(kind='barh', color='forestgreen')
plt.title('Feature Importance — Random Forest')
plt.xlabel('Важность')
plt.tight_layout()
plt.show()

# ============================================================
# 10. Gradient Boosting
# ============================================================
gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
mae_gb  = mean_absolute_error(y_test, y_pred_gb)

print('\nGradient Boosting')
print(f'RMSE = {rmse_gb:.4f}')
print(f'MAE  = {mae_gb:.4f}')

importances_gb = pd.Series(gb.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(9, 7))
importances_gb.plot(kind='barh', color='darkorange')
plt.title('Feature Importance — Gradient Boosting')
plt.xlabel('Важность')
plt.tight_layout()
plt.show()

# ============================================================
# 11. Итоговая таблица
# ============================================================
results = pd.DataFrame({
    'Модель': ['Baseline', 'Linear Regression', 'Random Forest', 'Gradient Boosting'],
    'RMSE': [rmse_baseline, rmse_lr, rmse_rf, rmse_gb],
    'MAE':  [mae_baseline, mae_lr, mae_rf, mae_gb]
})
print('\n=== Итоговая таблица ===')
print(results.to_string(index=False))

print('\nПревзошли baseline:')
for _, row in results.iterrows():
    if row['Модель'] != 'Baseline' and row['RMSE'] < rmse_baseline:
        print(f"  ✓ {row['Модель']}  RMSE={row['RMSE']:.4f}")

print('\nМодели ниже порога 0.70:')
for _, row in results.iterrows():
    if row['Модель'] != 'Baseline' and row['RMSE'] < 0.70:
        print(f"  ✓ {row['Модель']}  RMSE={row['RMSE']:.4f}")

# ============================================================
# 12. Анализ ошибок лучшей модели
# ============================================================
best_idx  = results.loc[results['Модель'] != 'Baseline', 'RMSE'].idxmin()
best_name = results.loc[best_idx, 'Модель']
pred_map  = {'Linear Regression': y_pred_lr,
             'Random Forest':     y_pred_rf,
             'Gradient Boosting': y_pred_gb}
y_pred_best = pred_map[best_name]
print(f'\nЛучшая модель: {best_name}')

errors = y_test.values - y_pred_best

plt.figure(figsize=(8, 5))
sns.histplot(errors, bins=30, kde=True, color='steelblue')
plt.axvline(0, color='red', linestyle='--')
plt.title(f'Распределение ошибок — {best_name}')
plt.xlabel('Ошибка (true − pred)')
plt.tight_layout()
plt.show()

print(f'\nОшибки {best_name}:')
print(f'  Средняя: {errors.mean():.4f}')
print(f'  Std:     {errors.std():.4f}')
print(f'  MAE:     {np.abs(errors).mean():.4f}')
print(f'  RMSE:    {np.sqrt((errors**2).mean()):.4f}')

large_errors = pd.DataFrame({
    'true':  y_test.values,
    'pred':  y_pred_best,
    'error': np.abs(y_test.values - y_pred_best)
}).sort_values('error', ascending=False).head(10)

print('\nТоп-10 ошибок:')
print(large_errors.to_string(index=False))

print('\nRMSE по каждому quality:')
for q in sorted(y_test.unique()):
    mask = y_test.values == q
    if mask.sum() >= 3:
        r = np.sqrt(mean_squared_error(y_test.values[mask], y_pred_best[mask]))
        print(f'  quality={q}: n={mask.sum():4d}  RMSE={r:.4f}')

# ============================================================
# 13. Визуализация предсказаний (финальный график)
# ============================================================
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_best, alpha=0.3,
            c=X_test['wine_type'], cmap='coolwarm',
            edgecolor='k', linewidth=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'k--', label='Идеальное предсказание')
plt.xlabel('Истинное quality')
plt.ylabel('Предсказанное quality')
plt.title(f'{best_name}: истина vs предсказание')
plt.legend()
plt.tight_layout()
plt.show()

print('\nГотово. Все вычисления и графики выполнены.')