#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: data_logger.py
@Python Version: 3.12.1
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2025-08-11.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V1.0
@License: Apache License Version 2.0, January 2004
    Copyright 2025. All rights reserved.

@Description: 
"""

import csv
import json
import sqlite3
import pandas as pd
from datetime import datetime


class DataLogger:
    def __init__(self):
        self.system_data = {
            "cpu": [], "memory": [], "disks": [], 
            "disk_io": [], "network": [], "gpu": []
        }
        self.process_data = {}
    
    def log_system_data(self, data_type, data):
        """记录系统数据"""
        if data_type in self.system_data:
            # 添加时间戳
            data["time"] = datetime.now().isoformat()
            self.system_data[data_type].append(data)
    
    def log_process_data(self, process_data):
        """记录进程数据"""
        for pid, data in process_data.items():
            if pid not in self.process_data:
                self.process_data[pid] = []
            
            # 添加时间戳和进程ID
            data["time"] = datetime.now().isoformat()
            data["pid"] = pid
            self.process_data[pid].append(data)
    
    def export_csv(self, filename, data_type=None, pid=None):
        """导出数据到CSV"""
        if data_type and data_type in self.system_data:
            data = self.system_data[data_type]
            keys = data[0].keys() if data else []
            
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(data)
        
        elif pid and pid in self.process_data:
            data = self.process_data[pid]
            keys = data[0].keys() if data else []
            
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(data)
    
    def export_json(self, filename, data_type=None, pid=None):
        """导出数据到JSON"""
        data_to_export = {}
        
        if data_type and data_type in self.system_data:
            data_to_export[data_type] = self.system_data[data_type]
        elif pid and pid in self.process_data:
            data_to_export[f"process_{pid}"] = self.process_data[pid]
        else:
            # 导出所有数据
            data_to_export = {
                "system": self.system_data,
                "processes": self.process_data
            }
        
        with open(filename, 'w') as f:
            json.dump(data_to_export, f, indent=2)
    
    def export_sqlite(self, filename):
        """导出数据到SQLite数据库"""
        conn = sqlite3.connect(filename)
        c = conn.cursor()
        
        # 创建系统数据表
        c.execute('''CREATE TABLE IF NOT EXISTS system_cpu (
                     time TEXT, total REAL, cores TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS system_memory (
                     time TEXT, total INTEGER, used INTEGER, 
                     free INTEGER, percent REAL)''')
        # 其他系统表类似...
        
        # 创建进程数据表
        c.execute('''CREATE TABLE IF NOT EXISTS processes (
                     pid INTEGER, time TEXT, name TEXT, cpu REAL,
                     memory INTEGER, read_bytes INTEGER, 
                     write_bytes INTEGER, threads INTEGER)''')
        
        # 插入CPU数据
        for entry in self.system_data["cpu"]:
            cores_json = json.dumps(entry["cores"])
            c.execute('''INSERT INTO system_cpu (time, total, cores)
                         VALUES (?, ?, ?)''',
                      (entry["time"], entry["total"], cores_json))
        
        # 插入内存数据
        for entry in self.system_data["memory"]:
            c.execute('''INSERT INTO system_memory 
                         (time, total, used, free, percent)
                         VALUES (?, ?, ?, ?, ?)''',
                      (entry["time"], entry["total"], entry["used"],
                       entry["free"], entry["percent"]))
        
        # 插入进程数据
        for pid, entries in self.process_data.items():
            for entry in entries:
                c.execute('''INSERT INTO processes 
                             (pid, time, name, cpu, memory, 
                              read_bytes, write_bytes, threads)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                          (pid, entry["time"], entry["name"], 
                           entry["cpu"], entry["memory"],
                           entry["read_bytes"], entry["write_bytes"],
                           entry["threads"]))
        
        conn.commit()
        conn.close()
    
    def export_excel(self, filename):
        """导出数据到Excel文件"""
        with pd.ExcelWriter(filename) as writer:
            # 导出系统数据
            for data_type, entries in self.system_data.items():
                if entries:
                    df = pd.DataFrame(entries)
                    df.to_excel(writer, sheet_name=f"system_{data_type}", index=False)
            
            # 导出进程数据
            for pid, entries in self.process_data.items():
                if entries:
                    df = pd.DataFrame(entries)
                    df.to_excel(writer, sheet_name=f"process_{pid}", index=False)
