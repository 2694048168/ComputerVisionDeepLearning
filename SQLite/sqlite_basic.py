#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@Python Version: 3.12.8
@Author: Wei Li (Ithaca)
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2025/07/27
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Paper:
@Description: 
'''

import os
import glob
import sqlite3

from logger import Logger


class DatabaseSqlite:
    def __init__(self, folder, filename, log_filename="log.log"):
        self.folder = folder
        self.filename = filename
        self.database_list = []

        self.logger = Logger("DatabaseSqlite", log_filename).GetLogger()
        self.database = self.Connect()

    @staticmethod
    def SearchAllDatabase(self):
        for extensions in [".db"]:
            self.database_list += glob.glob(os.path.join(self.folder, f"*{extensions}"))
        assert len(self.database_list), f"there is not any DataBase(*.db) in the {self.folder}"

    def SearchDatabase(self, filepath):
        self.filename = glob.glob(filepath)

    def Connect(self):
        database = sqlite3.connect(os.path.join(self.folder, self.filename))
        self.logger.info("open the database successfully")
        
        return database
    
    def Close(self):
        self.database.close()
        self.logger.info("close the database successfully")

    def Create(self, table_name):
        cursor_handler = self.database.cursor()
        cursor_handler.execute(f'''CREATE TABLE {table_name}
            (
                ID      INT PRIMARY KEY     NOT NULL,
                NAME    TEXT    NOT NULL,
                AGE     INT     NOT NULL,
                ADDRESS CHAR(50),
                SALARY  REAL  
            )
        ''')
        self.database.commit()
        self.logger.info(f"create the table {table_name} in the {self.filename} database successfully")

    def Insert(self, table_name):
        cursor_handler = self.database.cursor()
        cursor_handler.execute(f"INSERT INTO {table_name} (ID,NAME,AGE,ADDRESS,SALARY) \
            VALUES (1, 'Paul', 32, 'California', 20000.00 )")

        cursor_handler.execute(f"INSERT INTO {table_name} (ID,NAME,AGE,ADDRESS,SALARY) \
            VALUES (2, 'Allen', 25, 'Texas', 15000.00 )")

        cursor_handler.execute(f"INSERT INTO {table_name} (ID,NAME,AGE,ADDRESS,SALARY) \
            VALUES (3, 'Teddy', 23, 'Norway', 20000.00 )")

        cursor_handler.execute(f"INSERT INTO {table_name} (ID,NAME,AGE,ADDRESS,SALARY) \
            VALUES (4, 'Mark', 25, 'Rich', 65000.00 )")

        self.database.commit()
        self.logger.info(f"insert the table {table_name} in the {self.filename} database successfully")

    def Select(self, table_name):
        cursor_handler = self.database.cursor()
        cursor_handler.execute(f"SELECT id, name, age, address, salary  from {table_name}")

        for row in cursor_handler:
            print(f"ID = {row[0]}")
            print(f"NAME = {row[1]}")
            print(f"AGE = {row[2]}")
            print(f"ADDRESS {row[3]}")
            print(f"SALARY = {row[4]}\n")

        self.logger.info(f"insert the table {table_name} in the {self.filename} database successfully")

    def Update(self, table_name):
        cursor_handler = self.database.cursor()
        cursor_handler.execute(f"UPDATE {table_name} set SALARY = 75000.00 where ID=1")
        self.database.commit()
        self.logger.info(f"Total number of rows updated : {self.database.total_changes}")

        cursor_handler.execute(f"SELECT id, name, age, address, salary  from {table_name}")
        for row in cursor_handler:
            print(f"ID = {row[0]}")
            print(f"NAME = {row[1]}")
            print(f"AGE = {row[2]}")
            print(f"ADDRESS {row[3]}")
            print(f"SALARY = {row[4]}\n")

        self.logger.info(f"update the table {table_name} in the {self.filename} database successfully")

    def Delete(self, table_name):
        cursor_handler = self.database.cursor()
        cursor_handler.execute(f"DELETE from {table_name} where ID=2;")
        self.database.commit()
        self.logger.info(f"Total number of rows updated : {self.database.total_changes}")

        cursor_handler.execute(f"SELECT id, name, age, address, salary  from {table_name}")
        for row in cursor_handler:
            print(f"ID = {row[0]}")
            print(f"NAME = {row[1]}")
            print(f"AGE = {row[2]}")
            print(f"ADDRESS {row[3]}")
            print(f"SALARY = {row[4]}\n")

        self.logger.info(f"delete the table {table_name} in the {self.filename} database successfully")


# ---------------------------
if __name__ == "__main__":
    folder = "./database/"
    os.makedirs(folder, exist_ok=True)
    filename = "basic.db"
    database = DatabaseSqlite(folder, filename)
    database.Connect()

    table_name = "User"
    database.Create(table_name)
    database.Insert(table_name)
    database.Select(table_name)
    database.Update(table_name)
    database.Delete(table_name)

    database.Close()
