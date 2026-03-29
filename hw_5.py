import polars as pl
import bottleneck as bn


df_train = pl.read_csv("D:/train.csv")


print("=== Типы данных ===")
print(df_train.dtypes)
print("\n=== Описание (средние, std и т.д.) ===")
print(df_train.describe())

print("\n=== Количество пропусков в каждом столбце ===")
print(df_train.null_count())


print("\n=== Количество пассажиров по классам ===")
pclass_counts = df_train.get_column("Pclass").value_counts()
print(pclass_counts)


print("\n=== Количество выживших по полу ===")
survival_by_sex = (
    df_train
    .group_by("Sex")
    .agg(pl.col("Survived").sum().alias("Survived_count"))
)
print(survival_by_sex)


print("\n=== Пассажиры старше 44 лет ===")
older_passengers = df_train.filter(pl.col("Age") > 44)
print(older_passengers)

import pandas as pd


df_pandas = pd.read_csv("d:/train.csv")


mean_age = bn.nanmean(df_pandas['Age'])
std_age = bn.nanstd(df_pandas['Age'])
print(f"\nСредний возраст (bottleneck): {mean_age:.2f}")
print(f"Стандартное отклонение возраста (bottleneck): {std_age:.2f}")


df_pandas['Fare_new'] = df_pandas['Fare'].apply(lambda x: x * 1.3)

print("\nПервые 5 строк с новым столбцом Fare_new:")
print(df_pandas[['Fare', 'Fare_new']].head())


df_housing = pd.read_csv("d:/Housing.csv")


print("\n=== Потребление памяти до оптимизации ===")
memory_before = df_housing.memory_usage(deep=True).sum()
print(f"Объём памяти: {memory_before / 1024**2:.2f} MB")


print("\n=== Типы данных до оптимизации ===")
print(df_housing.dtypes)






df_housing['price'] = pd.to_numeric(df_housing['price'], downcast='unsigned')  
df_housing['area'] = pd.to_numeric(df_housing['area'], downcast='unsigned')   
df_housing['bedrooms'] = df_housing['bedrooms'].astype('int8')
df_housing['bathrooms'] = df_housing['bathrooms'].astype('int8')
df_housing['stories'] = df_housing['stories'].astype('int8')
df_housing['parking'] = df_housing['parking'].astype('int8')


binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df_housing[col] = df_housing[col].map({'yes': 1, 'no': 0}).astype('int8')


df_housing['furnishingstatus'] = df_housing['furnishingstatus'].astype('category')


print("\n=== Потребление памяти после оптимизации ===")
memory_after = df_housing.memory_usage(deep=True).sum()
print(f"Объём памяти: {memory_after / 1024**2:.2f} MB")
print(f"Экономия: {((memory_before - memory_after) / memory_before * 100):.2f}%")
