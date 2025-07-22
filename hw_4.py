import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('D:/train.csv')


fig, axes = plt.subplots(3, 2, figsize=(14, 12))

sns.countplot(data=df, x='Survived', ax=axes[0, 0])
axes[0, 0].set_title('Распределение Survived')

sns.countplot(data=df, x='Pclass', ax=axes[0, 1])
axes[0, 1].set_title('Распределение Pclass')

sns.histplot(data=df, x='Age', bins=20, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Распределение Age')

sns.countplot(data=df, x='Sex', ax=axes[1, 1])
axes[1, 1].set_title('Распределение Sex')

sns.countplot(data=df, x='Parch', ax=axes[2, 0])
axes[2, 0].set_title('Распределение Parch')

plt.tight_layout()
plt.savefig('distributions.png')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(data=df, y='Age')
plt.title('Boxplot для возраста пассажиров')
plt.ylabel('Возраст')
plt.savefig('boxplot_age.png')
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(14, 7))


df['Survived'].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[0], title='Доли выживших/погибших')


df['Pclass'].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[1], title='Доли по классам')

plt.tight_layout()
plt.savefig('pie_charts.png')
plt.show()


numeric_cols = df.select_dtypes(include=['number']).columns
sns.pairplot(df[numeric_cols].dropna())
plt.suptitle('Pairplot для числовых признаков', y=1.02)
plt.savefig('pairplot.png')
plt.show()

df_grouped = df.groupby(['Pclass', 'Sex']).size().reset_index(name='Count')

fig = px.sunburst(df_grouped, path=['Pclass', 'Sex'], values='Count',
                  title='Иерархия: класс пассажира -> пол')
fig.write_html("sunburst_plot.html")
fig.show()
