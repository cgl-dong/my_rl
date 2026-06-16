import datetime
import time
from decimal import Decimal

import pymssql

# 创建连接字符串  （sqlserver默认端口为1433）
conn = pymssql.connect(host='ubuntu-cgl',  # 这里的host='_'可以用本机ip或ip+端口号
                       # server="******",#本地服务器
                       port="1433",  # TCP端口
                       user="sa", password="MyPassWord123",
                       database="TestDB",
                       # charset="UTF-8"
                       # 这里设置全局的GBK，如果设置的是UTF—8需要将数据库默认的GBK转化成UTF-8
                       )
if conn:
    print('连接数据库成功!')  # 测试是否连接上

cursor = conn.cursor()  # 使用cursor()方法获取操作游标


def print_result(results):
    for result in results:
        result = list(result)  # 元组转化为列表
        for res in range(len(result)):
            if isinstance(result[res], str):
                result[res] = result[res].replace(' ', '')
            if isinstance(result[res], datetime.datetime):
                result[res] = datetime.datetime.strftime(result[res], '%Y-%m-%d %H:%M:%S')
            if isinstance(result[res], Decimal):
                result[res] = "{}元".format(result[res])

        result = tuple(result)  # 列表再转换为元组
        print(result)


def task1_1():
    print("下达采购任务---->")
    count = int(input("请输入采购单物资类目数量: "))
    e_id = int(input("请输入员工id: "))
    buy_task_id = time.strftime("%Y%m%d%H%M%S")
    for i in range(count):
        m_id = int(input("输入第{}类物资id: ".format(i + 1)))
        m_num = int(input("输入第{}类物资数量: ".format(i + 1)))
        cursor.execute("exec task1_1 {},{},{},{}".format(buy_task_id, m_id, m_num, e_id))

    print("操作完成")
    conn.commit()


def task1_234(type):
    state = 0
    if type == 2:
        print("撤销采购任务---->")
        state = 2
    elif type == 3:
        print("修改采购任务---->")
    else:
        print("完成采购任务---->")
        state = 1
    e_id = int(input("请输入操作人id: "))
    buy_task_id = input("请输入采购单id: ")

    if type == 3:
        cursor.execute("select id,m_id,m_num,e_id,state,create_time from Buy_Task_Item where e_id = {} and buy_task_id = {}".format(e_id,buy_task_id))  # 执行语句
        results = cursor.fetchall()  # 获取所有记录列表
        print_result(results)
        id = int(input("请输入操作项id: "))
        m_id = int(input("请输入物料id: "))
        m_num = int(input("请输入物料数量: "))

        cursor.execute("UPDATE Buy_Task_Item set m_id = {},m_num = {} where id = {}".format(m_id,m_num,id))
        print("操作完成！")
        conn.commit()
        return

    cursor.execute("UPDATE Buy_Task_Item set state = {}  where buy_task_id = {} and e_id = {}".format(state,buy_task_id, e_id))

    print("操作完成！")
    conn.commit()


# 采购员或采购主任查看采购单
def task1_56(type):
    sql = "select * from Buy_Task_Item where 1 = 1"
    if type == 5:
        print("采购员查看采购单---->")
        e_id = int(input("请输入操作人id: "))
        sql = sql + "and e_id = {}"
        cursor.execute(sql.format(e_id))  # 执行语句
        results = cursor.fetchall()  # 获取所有记录列表
        print_result(results)
    else:
        print("采购主任查看采购单---->")
        e_id = input("请输入操作人id（非必填项）: ")
        buy_task_id = input("请输入采购单id（非必填项）: ")

        if e_id:
            sql = sql + "and e_id = {}".format(int(e_id))
        if buy_task_id:
            sql = sql + "and buy_task_id = {}".format(int(buy_task_id))

        cursor.execute(sql)  # 执行语句
        results = cursor.fetchall()  # 获取所有记录列表
        print_result(results)

    print("操作完成！")
    conn.commit()


# 填写询价单
def task2_1():
    print("采购员填写询价单---->")
    m_id = int(input("请输入物资id: "))
    supply_id = int(input("请输入供应商id: "))
    price = int(input("请输入单价: "))
    num = int(input("请输入数量: "))
    sum_price = int(input("请输入总价: "))
    cursor.execute("exec task2_1 {},{},{},{},{}".format(m_id, supply_id, price, sum_price, num))
    print("操作完成！")
    conn.commit()


# 采购部主任亩核询价单
def task2_2():
    print("采购部主任亩核询价单------>")
    id = int(input("请输入审价单id:"))
    state = int(input("请输入审核状态（审核状态 0未审核 1已审核）："))
    cursor.execute("exec task2_2 {},{}".format(id, state))
    print("操作完成！")
    conn.commit()


#  查看询价单
def task2_3():
    print("查看询价单---->")
    state = int(input("请输入审核状态（审核状态 0未审核 1已审核）："))
    sql = "select * from Inquiry_Item where state = {}".format(state)
    cursor.execute(sql)
    result = cursor.fetchall()
    print_result(result)
    print("操作完成！")


# 填写合同
def task3_1():
    print("填写合同---->")
    supply_id = int(input("请输入供应商id:"))
    buy_task_id = input("请输入采购任务单id:")
    m_id = int(input("请输入物料id:"))
    m_num = int(input("输入采购数量:"))
    sum_price = int(input("采购总价:"))

    cursor.execute("exec task3_1 {},{},{},{},{}".format(supply_id, buy_task_id, m_id, m_num, sum_price))
    print("操作完成！")
    conn.commit()


# 审核合同
def task3_2():
    print("物资部经理审核合同单------>")
    id = int(input("请输入合同id:"))
    state = int(input("请输入合同状态（0未签字 1 已签字）："))
    cursor.execute("exec task3_2 {},{}".format(id, state))
    print("操作完成！")
    conn.commit()


# 查看合同
'''
此处使用视图 将相关数据都拿到
'''

def task3_3():
    print("查看合同单------>")
    state = input("请输入合同状态：")
    e_id = input("请输入查看员工id:")
    sql = "select * from task3_3 where 1 =1"
    if e_id:
        sql = sql + "and e_id = {}".format(int(e_id))
    if state:
        sql = sql + "and state = {}".format(int(state))
    cursor.execute(sql)
    result = cursor.fetchall()
    print_result(result)
    print("操作完成！")


# 填写出入库申请单
def task4_1():
    print("填写出入库申请单------>")
    apply_id = int(input("输入申请人id："))
    reviewer_id = int(input("输入审查人id："))
    m_id = int(input("输入物料id："))
    s_id = int(input("输入仓库id："))
    num = int(input("输入物料数量："))
    type = int(input("输入出入库类型（0出库 1入库）："))
    cursor.execute("exec task4_1 {},{},{},{},{},{}".format(apply_id, reviewer_id, m_id, s_id, num, type))
    print("操作完成！")
    conn.commit()


# 审核出入库申请单
def task4_2():
    print("审核出入库申请单----->")
    id = int(input("请输入出入库申请单id:"))
    state = int(input("请输入审核状态（0 未审查 1 已审查)："))
    reviewer_id = int(input("请输入审核人id："))
    cursor.execute("exec task4_2 {},{},{}".format(id, state, reviewer_id))
    print("操作完成！")
    conn.commit()


# 查看出入库申请单
def task4_3():
    print("查看出入库申请单------>")
    state = input("请输入审查状态（选填）：")
    apply_id = input("请输入申请人id（选填）:")
    type = input("请输入出入库类型(0 未出库 1已出库 选填)：")
    reviewer_id = input("请输入审查人id(选填):")
    sql = "select * from Store_Operate where 1 =1"
    if apply_id:
        sql = sql + "and apply_id = {}".format(int(apply_id))
    if state:
        sql = sql + "and review_state = {}".format(int(state))
    if type:
        sql = sql + "and type = {}".format(int(type))
    if reviewer_id:
        sql = sql + "and reviewer_id = {}".format(int(reviewer_id))
    cursor.execute(sql)
    result = cursor.fetchall()
    print_result(result)
    print("操作完成！")


# 仓库库存查询 视图拼接
def task5_1():
    print("仓库库存查询------>")
    s_id = input("请输入仓库编号：")
    m_id = input("请输入物料id（选填）:")

    sql = "select * from task5_1 where s_id = {}".format(int(s_id))
    if m_id:
        sql = sql + "and m_id = {}".format(int(m_id))
    cursor.execute(sql)
    result = cursor.fetchall()
    print_result(result)
    print("操作完成！")


# 仓库预警
def task5_2():
    print("仓库预警信息查询------>")
    sql = "select *,case when num <50 then N'库存不足' when num>800 then N'库存冗余'  else N'库存合理' end as 'num_desc' from task5_1"
    cursor.execute(sql)
    result = cursor.fetchall()
    print_result(result)
    print("操作完成！")


# 生成相关发票
def task6_1():
    print("生成发票------>")
    cursor.execute('exec task6_1')
    conn.commit()
    print("操作完成！")

def task6_2():
    print("查看发票信息---->")
    no = input("请输入发票批次（选填）:")
    id = input("请输入发票id(选填):")

    sql = "select * from task6_2 where 1 = 1"
    if no:
        sql = sql + "and no = {}".format(int(no))
    if id:
        sql = sql + "and c_id = {}".format(int(id))
    cursor.execute(sql)
    result = cursor.fetchall()
    print_result(result)
    print("操作完成！")


parent_item_dict = {
    1: "采购任务下达",
    2: "采购审价",
    3: "合同填写与审核",
    4: "出入库申请与审核",
    5: "仓库库存查询",
    6: "发票生成与查询",
    0: "退出系统"
}

sub_item_dict = {
    1: {1: "下达采购任务", 2: "撤销采购任务", 3: "修改采购任务", 4: "完成采购任务", 5: "采购员查看采购任务单",
        6: "采购部主任查看采购任务单(通过采购员id或采购单号)"},
    2: {1: "填写询价单", 2: "审核询价单", 3: "查看询价单"},
    3: {1: "填写合同", 2: "审核合同", 3: "查看合同"},
    4: {1: "填写出入库申请单", 2: "审核出入库申请单", 3: "查看出入库申请单"},
    5: {1: "仓库库存查询", 2: "库存预警"},
    6: {1: "生成相关发票", 2: "查看发票信息"}
}


def welcome():
    hello = """
        =========一=采购与库存管理系统========一一一一=
            欢迎进入父功能菜单
    """""
    print(hello)
    for k, v in parent_item_dict.items():
        print("{}{}.{}".format(" " * 10, k, v))
    x = int(input("请输入要操作的序号进入子菜单："))
    if x == 0:
        print("退出系统")
        exit(0)
    operate(x)
    welcome()


def operate(key):
    hello = """
        =========一=欢迎进入{}子菜单========一一一一=
        """"".format(parent_item_dict.get(key))

    print(hello)
    for k, v in sub_item_dict.get(key).items():
        print("{}{}.{}".format(" " * 10, k, v))
    print("{}0.退出子菜单".format(" " * 10))
    x = int(input("请输入要操作的序号："))
    if x == 0:
        print("退出子菜单")
        return

    if key == 1:
        if x == 1:
            task1_1()
        elif x == 2:
            task1_234(2)
        elif x == 3:
            task1_234(3)
        elif x == 4:
            task1_234(4)
        elif x == 5:
            task1_56(5)
        elif x == 6:
            task1_56(6)
    elif key == 2:
        if x == 1:
            task2_1()
        if x == 2:
            task2_2()
        if x == 3:
            task2_3()
    elif key == 3:
        if x == 1:
            task3_1()
        if x == 2:
            task3_2()
        if x == 3:
            task3_3()
    elif key == 4:
        if x == 1:
            task4_1()
        if x == 2:
            task4_2()
        if x == 3:
            task4_3()

    elif key == 5:
        if x == 1:
            task5_1()
        if x == 2:
            task5_2()

    elif key == 6:
        if x == 1:
            task6_1()
        if x == 2:
            task6_2()


# 主函数运行
welcome()
# 关闭数据库
conn.close()
