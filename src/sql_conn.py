#load dependencies
import pyodbc

# Using SQL Server Auth
print("For the Live SQL Server Authentication, please enter your")
uid = input("Username: ")
pwd = input("Password: ")

connection = pyodbc.connect('DRIVER={SQL Server};'
                            'SERVER=52.17.58.152;'
                            'DATABASE=CPMS_BI;'
                            'UID=' + uid + ';'
                            'PWD=' + pwd + ';'
                           )