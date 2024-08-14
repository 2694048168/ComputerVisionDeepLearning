#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 03_datetime_pylib.py
@Python Version: 3.12.1
@Platform: PyTorch 2.0.0 + cu117
@Author: Wei Li (Ithaca)
@Date: 2024-08-14.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V2.1
@License: Apache License Version 2.0, January 2004
    Copyright 2024. All rights reserved.

@Description: 
"""

# Python 标准库 datetime
import time
import datetime


if __name__ == "__main__":
    # 1. datetime.date 日期
    print("{:=^50s}".format("datetime.date"))
    print(f"the current date: {datetime.date.today()}")
    print(f"the current date: {datetime.date(2049, 10, 1)}")
    d = datetime.date.fromtimestamp(time.time())
    print(f"the current date: {d}")

    # 类属性
    print(f"the current MAX date: {datetime.date.max}")
    print(f"the current MIN date: {datetime.date.min}")
    print(f"the current resolution date: {datetime.date.resolution}")
    # 实例属性
    print(f"the current year: {d.year}")
    print(f"the current month: {d.month}")
    print(f"the current day: {d.day}")

    # 常用实例方法 datetime.date 对象 ---> 结构化时间对象
    print(d.timetuple())
    # 其他方法
    print(d)
    print(d.replace(2022))
    print(d.replace(d.year, 9))
    print(d.replace(d.year, d.month, 25))
    print(d.replace(day=25))

    # 0 表示周一,6表示周天
    print(f"today is weekday: {d.weekday()}")
    # 1 表示周一,0表示周天
    print(f"today is weekday: {d.isoweekday()}")
    print(f"today is format: {d.isoformat()}")
    print(f"today is YMD format: {d.strftime("%Y年%m月%d日")}")

    # 2. datetime.date 时间
    print("{:=^50s}".format("datetime.time"))
    print(f"the current time: {datetime.time()}")
    print(f"the current time: {datetime.time(15, 25, 45, 888888)}")

    # 类属性
    print(f"the MIN time: {datetime.time.min}")
    print(f"the MAX time: {datetime.time.max}")
    print(f"the resolution time: {datetime.time.resolution}")
    # 实例属性
    t = datetime.time(15, 25, 45, 888888)
    print(f"the hours of time: {t.hour}")
    print(f"the minute of time: {t.minute}")
    print(f"the second of time: {t.second}")
    print(f"the microsecond of time: {t.microsecond}")
    # 其他方法
    print(f"the format of time: {t.isoformat()}")
    print(f"the H-M-S of time: {t.strftime("%H时%M分%S秒 %f微秒")}")

    # 3. datetime.datetime 日期时间
    print("{:=^50s}".format("datetime.datetime"))
    dt = datetime.datetime(2024, 8, 12, 20, 21, 12, 888888)
    print(f"the type: {type(dt)} and value: {dt}")
    print(f"the today: {datetime.datetime.today()}")
    print(f"the today: {datetime.datetime.now(tz=None)}")
    # print(f"the today: {datetime.datetime.utcnow()}")
    print(f"the today: {datetime.datetime.now(datetime.UTC)}")
    print(f"the today: {datetime.datetime.fromtimestamp(time.time())}")
    # print(f"the today: {datetime.datetime.utcfromtimestamp(time.time())}")
    print(f"the today: {datetime.datetime.fromtimestamp(time.time(), datetime.UTC)}")

    print(f"the datetime combine: {datetime.datetime.combine(d, t)}")
    # 实例属性
    dt = datetime.datetime.now(datetime.UTC)
    print(f"the current year: {dt.year}")
    print(f"the current month: {dt.month}")
    print(f"the current day: {dt.day}")
    print(f"the hours of time: {dt.hour}")
    print(f"the minute of time: {dt.minute}")
    print(f"the second of time: {dt.second}")
    print(f"the microsecond of time: {dt.microsecond}")
    # replace method

    # datetime ---> 结构化对象
    print(dt.timetuple())
    # datetime ---> 时间戳
    print(f"the current timestamp {dt.timestamp()}")
    # datetime ---> 格式化字符串
    print(f"the format string: {dt.strftime("%Y年%m月%d日 %H时%M分%S秒 %f微秒")}")

    # 4. datetime.timedelta 时间间隔
    # Syntax : datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0) 
    # Returns : Date
    print("{:=^50s}".format("datetime.datetime"))
    td = datetime.timedelta(days=10)
    print(f"the type: {type(td)} and value: {td}")
    td = datetime.timedelta(days=10,hours=-21)
    print(f"the type: {type(td)} and value: {td}")

    # 计算日期差
    dt = datetime.datetime.now(datetime.UTC)
    print(f"the current datetime: {dt.strftime("%Y年%m月%d日 %H时%M分%S秒 %f微秒")}")
    delta = datetime.timedelta(days=12)
    target = dt + delta
    print(f"the after 12days datetime: {target.strftime("%Y年%m月%d日 %H时%M分%S秒 %f微秒")}")

    # 计算时间差
    dt1 = datetime.datetime.today()
    dt2 = datetime.datetime.now(datetime.UTC)
    td = dt1 = dt2
    # print(f"我们与UTC时差: {td.second/3600:.0f} 小时")
    print(f"我们与UTC时差: {td.second/3600} 小时")
