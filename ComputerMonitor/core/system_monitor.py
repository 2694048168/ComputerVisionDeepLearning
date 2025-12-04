#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: system_monitor.py
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

import psutil
import time
import threading
from PySide6.QtCore import QObject, Signal


class SystemMonitor(QObject):
    # 定义信号
    cpu_update = Signal(dict)           # CPU使用率数据
    memory_update = Signal(dict)         # 内存使用数据
    disk_update = Signal(list)           # 磁盘使用数据
    disk_io_update = Signal(dict)        # 磁盘IO数据
    network_update = Signal(dict)        # 网络流量数据
    gpu_update = Signal(dict)            # GPU使用数据
    process_update = Signal(dict)        # 进程监控数据
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._monitor_thread = None
        self._monitored_processes = {}  # {pid: process_obj}
        self._prev_disk_io = psutil.disk_io_counters()
        self._prev_net_io = psutil.net_io_counters()
        self._prev_time = time.time()
        
    def start_monitoring(self, interval=1):
        """启动监控线程"""
        if self._running:
            return
            
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
    
    def add_process_monitor(self, pid):
        """添加进程监控"""
        try:
            process = psutil.Process(pid)
            self._monitored_processes[pid] = process
        except psutil.NoSuchProcess:
            pass
    
    def remove_process_monitor(self, pid):
        """移除进程监控"""
        if pid in self._monitored_processes:
            del self._monitored_processes[pid]
    
    def _monitor_loop(self, interval):
        """监控主循环"""
        while self._running:
            try:
                # 获取系统数据
                self._collect_cpu_data()
                self._collect_memory_data()
                self._collect_disk_data()
                self._collect_disk_io_data()
                self._collect_network_data()
                self._collect_gpu_data()
                
                # 获取进程数据
                if self._monitored_processes:
                    self._collect_process_data()
                
            except Exception as e:
                print(f"监控错误: {e}")
            
            time.sleep(interval)
    
    def _collect_cpu_data(self):
        """收集CPU数据"""
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_total = sum(cpu_percent) / len(cpu_percent)
        
        core_data = {}
        for i, percent in enumerate(cpu_percent):
            core_data[f"core_{i}"] = percent
        
        self.cpu_update.emit({
            "total": cpu_total,
            "cores": core_data,
            "timestamp": time.time()
        })
    
    def _collect_memory_data(self):
        """收集内存数据"""
        mem = psutil.virtual_memory()
        self.memory_update.emit({
            "total": mem.total,
            "used": mem.used,
            "free": mem.free,
            "percent": mem.percent,
            "timestamp": time.time()
        })
    
    def _collect_disk_data(self):
        """收集磁盘使用数据"""
        disks = []
        for partition in psutil.disk_partitions():
            if 'cdrom' in partition.opts or partition.fstype == '':
                continue
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disks.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent
                })
            except Exception:
                continue
        
        self.disk_update.emit(disks)
    
    def _collect_disk_io_data(self):
        """收集磁盘IO数据"""
        current_time = time.time()
        current_io = psutil.disk_io_counters()
        time_diff = current_time - self._prev_time
        
        read_speed = (current_io.read_bytes - self._prev_disk_io.read_bytes) / time_diff
        write_speed = (current_io.write_bytes - self._prev_disk_io.write_bytes) / time_diff
        
        self.disk_io_update.emit({
            "read_speed": read_speed,
            "write_speed": write_speed,
            "timestamp": current_time
        })
        
        # 更新前值
        self._prev_disk_io = current_io
        self._prev_time = current_time
    
    def _collect_network_data(self):
        """收集网络流量数据"""
        current_io = psutil.net_io_counters()
        time_diff = time.time() - self._prev_time
        
        upload_speed = (current_io.bytes_sent - self._prev_net_io.bytes_sent) / time_diff
        download_speed = (current_io.bytes_recv - self._prev_net_io.bytes_recv) / time_diff
        
        self.network_update.emit({
            "upload": upload_speed,
            "download": download_speed,
            "timestamp": time.time()
        })
        
        # 更新前值
        self._prev_net_io = current_io
    
    def _collect_gpu_data(self):
        """收集GPU数据 - 需要平台特定实现"""
        # 实际实现会根据平台不同而不同
        # 这里提供Windows平台使用pywin32的实现示例
        gpu_data = {"utilization": 0}
        
        try:
            # Windows平台实现
            import wmi
            w = wmi.WMI()
            gpu_info = w.Win32_VideoController()[0]
            if hasattr(gpu_info, "LoadPercentage"):
                gpu_data["utilization"] = gpu_info.LoadPercentage
        except:
            # Linux平台实现
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_data["utilization"] = gpus[0].load * 100
            except:
                pass
        
        gpu_data["timestamp"] = time.time()
        self.gpu_update.emit(gpu_data)
    
    def _collect_process_data(self):
        """收集被监控进程的数据"""
        process_data = {}
        current_time = time.time()
        
        for pid, process in list(self._monitored_processes.items()):
            try:
                # 检查进程是否仍然存在
                if not process.is_running():
                    del self._monitored_processes[pid]
                    continue
                
                # 获取进程数据
                with process.oneshot():
                    cpu_percent = process.cpu_percent()
                    mem_info = process.memory_info()
                    io_counters = process.io_counters()
                    threads = process.num_threads()
                    
                process_data[pid] = {
                    "name": process.name(),
                    "cpu": cpu_percent,
                    "memory": mem_info.rss,
                    "read_bytes": io_counters.read_bytes,
                    "write_bytes": io_counters.write_bytes,
                    "threads": threads,
                    "timestamp": current_time
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                del self._monitored_processes[pid]
        
        self.process_update.emit(process_data)
