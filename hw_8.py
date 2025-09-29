
import requests
import pandas as pd
import psycopg2
from psycopg2 import OperationalError
import time


class IceAndFireAPI:
    def __init__(self):
        self.base_url = "https://www.anapioficeandfire.com/api"

    def get_all_books(self):
        "Получить все книги"
        try:
            response = requests.get(f"{self.base_url}/books")
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при получении книг: {e}")
            return pd.DataFrame()

    def get_all_houses(self):
        "Получить все дома"
        try:
            response = requests.get(f"{self.base_url}/houses")
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при получении домов: {e}")
            return pd.DataFrame()

    def get_houses_with_motto(self):
        "Получить дома с девизами"
        try:
            response = requests.get(f"{self.base_url}/houses?hasWords=true")
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при получении домов с девизами: {e}")
            return pd.DataFrame()


class DatabaseManager:
    def __init__(self, host, database, user, password, port=5432):
        self.connection_params = {
            'host': 'hh-pgsql-public.ebi.ac.uk',
            'database': 'pfmegrnargs',
            'user': 'reader',
            'password': 'NWDMCE5xdipIjRrp',
            'port': 5432
        }

    def connect(self):
        "Подключиться к БД"
        try:
            conn = psycopg2.connect(**self.connection_params)
            print("Успешное подключение к БД")
            return conn
        except OperationalError as e:
            print(f"Ошибка подключения к БД: {e}")
            return None

    def get_rnc_data_all_columns(self, limit=10):
        "Получить все столбцы из rnc_database"
        conn = self.connect()
        if not conn:
            return pd.DataFrame()

        try:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT * FROM rnc_database LIMIT {limit}")
                data = cursor.fetchall()

                # Получаем названия столбцов
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'rnc_database'
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in cursor.fetchall()]

                return pd.DataFrame(data, columns=columns)
        except Exception as e:
            print(f"Ошибка при получении данных: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def get_rnc_data_specific_columns(self, limit=10):
        "Получить конкретные столбцы из rnc_database"
        conn = self.connect()
        if not conn:
            return pd.DataFrame()

        try:
            with conn.cursor() as cursor:
                query = """
                SELECT display_name, num_sequences, num_organisms, url 
                FROM rnc_database 
                LIMIT %s
                """
                cursor.execute(query, (limit,))
                data = cursor.fetchall()

                return pd.DataFrame(data, columns=['display_name', 'num_sequences', 'num_organisms', 'url'])
        except Exception as e:
            print(f"Ошибка при получении данных: {e}")
            return pd.DataFrame()
        finally:
            conn.close()


def main():
    print("Начало выполнения задания...")

    # Работа с API
    api = IceAndFireAPI()

    print("\n1. Получение данных о книгах...")
    books_df = api.get_all_books()
    print(f"Получено книг: {len(books_df)}")

    print("\n2. Получение данных о всех домах...")
    houses_df = api.get_all_houses()
    print(f"Получено домов: {len(houses_df)}")

    print("\n3. Получение данных о домах с девизами...")
    houses_with_motto_df = api.get_houses_with_motto()
    print(f"Получено домов с девизами: {len(houses_with_motto_df)}")

    # Сохраняем результаты API
    books_df.to_csv('books.csv', index=False, encoding='utf-8')
    houses_df.to_csv('all_houses.csv', index=False, encoding='utf-8')
    houses_with_motto_df.to_csv('houses_with_motto.csv', index=False, encoding='utf-8')

    # Работа с БД (замените параметры на реальные)
    print("\n4. Работа с базой данных...")
    db = DatabaseManager(
        host="hh-pgsql-public.ebi.ac.uk",
        database="pfmegrnargs",
        user="reader",
        password="ваш_пароль"
    )

    print("\n5. Получение 10 строк из rnc_database...")
    rnc_all_data = db.get_rnc_data_all_columns(10)
    print(f"Получено строк: {len(rnc_all_data)}")

    print("\n6. Получение конкретных столбцов...")
    rnc_specific_data = db.get_rnc_data_specific_columns(10)
    print(f"Получено строк: {len(rnc_specific_data)}")

    # Сохраняем результаты БД
    rnc_all_data.to_csv('rnc_database_all.csv', index=False, encoding='utf-8')
    rnc_specific_data.to_csv('rnc_database_specific.csv', index=False, encoding='utf-8')

    print("\nЗадание выполнено! Результаты сохранены в CSV файлы.")


if __name__ == "__main__":
    main()