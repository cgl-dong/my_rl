

"""
D:\pythonProject\identifier.sqlite
"""

import sqlite3   #导入sqlite3模块

import pandas as pd

da = pd.read_excel("./SC.xlsx")
print(da)
"""Cno Cname  Cpno  Ccredit
    Sname Sno Ssex  Sage Sdept  Snumber
    Sno  Cno  Grade
"""
# print(da['Cno'],da['Cname'],da['Cpno'],da['Ccredit'])
# for i in range(5):
#     print(da['Cno'][i], da['Cname'][i], da['Cpno'][i], da['Ccredit'][i])
conn = sqlite3.connect("D:\pythonProject\identifier.sqlite")     #建立一个基于硬盘的数据库实例
cur = conn.cursor()        #创建关联数据库的游标实例
# cur.execute("select * from SC")  #对T_fish表执行数据查找命令

for i in range(12):
#     # sql1 = "insert into Course(Cno,Cname,Cpno,Ccredit) values({},'{}',{},{})".format(da['Cno'][i], da['Cname'][i], da['Cpno'][i], da['Ccredit'][i])
#     sql2 = "insert into Student(Sname,Sno,Ssex,Sage,Sdept) values('{}',{},'{}',{},'{}')".format(da['Sname'][i], da['Sno'][i], da['Ssex'][i], da['Sage'][i],da['Sdept'][i])
    sql3 = "insert into SC(Sno,Cno,Grade) values({},{},{})".format(da['Sno'][i], da['Cno'][i], da['Grade'][i])
    # print(sql3)
#     # print(da['Cno'][i], da['Cname'][i], da['Cpno'][i], da['Ccredit'][i])
    cur.execute(sql3)
# # for row in cur.fetchall():      #以一条记录为元组单位返回结果给row
# #     print(row)
    conn.commit()
conn.close()