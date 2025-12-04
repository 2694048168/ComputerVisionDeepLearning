#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: main_window.py
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

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QTableWidget, QTableWidgetItem, QPushButton,
    QComboBox, QListWidget, QAbstractItemView, QFileDialog
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from core.system_monitor import SystemMonitor
from core.data_logger import DataLogger
import psutil
import time


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("工控机资源监控系统")
        self.resize(1200, 800)
        
        # 创建监控和数据记录对象
        self.monitor = SystemMonitor()
        self.logger = DataLogger()
        
        # 设置UI
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # 左侧系统监控面板
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.main_layout.addWidget(self.left_panel, 3)
        
        # 右侧进程监控面板
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.main_layout.addWidget(self.right_panel, 2)
        
        # 初始化UI组件
        self.init_system_monitor_ui()
        self.init_process_monitor_ui()
        self.init_export_ui()
        
        # 连接监控信号
        self.connect_monitor_signals()
        
        # 启动监控
        self.monitor.start_monitoring(interval=1)
        
        # 初始化进程列表
        self.refresh_process_list()
        
        # 状态栏
        self.statusBar().showMessage("监控运行中...")
    
    def init_system_monitor_ui(self):
        """初始化系统监控UI"""
        # CPU监控组
        cpu_group = QGroupBox("CPU监控")
        cpu_layout = QVBoxLayout(cpu_group)
        
        self.cpu_total_label = QLabel("总使用率: 0.0%")
        cpu_layout.addWidget(self.cpu_total_label)
        
        self.cpu_cores_table = QTableWidget()
        self.cpu_cores_table.setColumnCount(2)
        self.cpu_cores_table.setHorizontalHeaderLabels(["核心", "使用率"])
        self.cpu_cores_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        cpu_layout.addWidget(self.cpu_cores_table)
        
        self.left_layout.addWidget(cpu_group)
        
        # 内存监控组
        mem_group = QGroupBox("内存监控")
        mem_layout = QVBoxLayout(mem_group)
        
        self.mem_usage_label = QLabel("使用率: 0.0%")
        self.mem_used_label = QLabel("已用: 0 GB")
        self.mem_free_label = QLabel("可用: 0 GB")
        self.mem_total_label = QLabel("总计: 0 GB")
        
        mem_layout.addWidget(self.mem_usage_label)
        mem_layout.addWidget(self.mem_used_label)
        mem_layout.addWidget(self.mem_free_label)
        mem_layout.addWidget(self.mem_total_label)
        
        self.left_layout.addWidget(mem_group)
        
        # 磁盘监控组
        disk_group = QGroupBox("磁盘监控")
        disk_layout = QVBoxLayout(disk_group)
        
        self.disk_table = QTableWidget()
        self.disk_table.setColumnCount(5)
        self.disk_table.setHorizontalHeaderLabels(["设备", "挂载点", "总大小", "已用", "使用率"])
        self.disk_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        disk_layout.addWidget(self.disk_table)
        
        self.left_layout.addWidget(disk_group)
        
        # 磁盘IO监控组
        disk_io_group = QGroupBox("磁盘IO")
        disk_io_layout = QVBoxLayout(disk_io_group)
        
        self.disk_read_label = QLabel("读取速度: 0 KB/s")
        self.disk_write_label = QLabel("写入速度: 0 KB/s")
        
        disk_io_layout.addWidget(self.disk_read_label)
        disk_io_layout.addWidget(self.disk_write_label)
        
        self.left_layout.addWidget(disk_io_group)
        
        # 网络监控组
        net_group = QGroupBox("网络流量")
        net_layout = QVBoxLayout(net_group)
        
        self.net_upload_label = QLabel("上传: 0 KB/s")
        self.net_download_label = QLabel("下载: 0 KB/s")
        
        net_layout.addWidget(self.net_upload_label)
        net_layout.addWidget(self.net_download_label)
        
        self.left_layout.addWidget(net_group)
        
        # GPU监控组
        gpu_group = QGroupBox("GPU监控")
        gpu_layout = QVBoxLayout(gpu_group)
        
        self.gpu_usage_label = QLabel("使用率: 0.0%")
        gpu_layout.addWidget(self.gpu_usage_label)
        
        self.left_layout.addWidget(gpu_group)
    
    def init_process_monitor_ui(self):
        """初始化进程监控UI"""
        # 进程选择组
        process_select_group = QGroupBox("进程选择")
        process_select_layout = QVBoxLayout(process_select_group)
        
        # 进程列表
        self.process_list = QListWidget()
        self.process_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        # 刷新按钮
        self.refresh_btn = QPushButton("刷新进程列表")
        self.refresh_btn.clicked.connect(self.refresh_process_list)
        
        # 添加/移除按钮
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("添加监控")
        self.add_btn.clicked.connect(self.add_selected_processes)
        self.remove_btn = QPushButton("移除监控")
        self.remove_btn.clicked.connect(self.remove_selected_processes)
        
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        
        process_select_layout.addWidget(self.process_list)
        process_select_layout.addWidget(self.refresh_btn)
        process_select_layout.addLayout(btn_layout)
        
        self.right_layout.addWidget(process_select_group)
        
        # 进程监控表
        process_monitor_group = QGroupBox("进程监控")
        process_monitor_layout = QVBoxLayout(process_monitor_group)
        
        self.process_table = QTableWidget()
        self.process_table.setColumnCount(6)
        self.process_table.setHorizontalHeaderLabels(["PID", "名称", "CPU%", "内存(MB)", "读(KB)", "写(KB)"])
        self.process_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        process_monitor_layout.addWidget(self.process_table)
        self.right_layout.addWidget(process_monitor_group)
    
    def init_export_ui(self):
        """初始化数据导出UI"""
        export_group = QGroupBox("数据导出")
        export_layout = QVBoxLayout(export_group)
        
        # 导出选项
        option_layout = QHBoxLayout()
        self.export_type_combo = QComboBox()
        self.export_type_combo.addItems(["所有数据", "系统数据", "进程数据"])
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["CSV", "JSON", "SQLite", "Excel"])
        
        option_layout.addWidget(QLabel("数据类型:"))
        option_layout.addWidget(self.export_type_combo)
        option_layout.addWidget(QLabel("格式:"))
        option_layout.addWidget(self.format_combo)
        
        # 导出按钮
        self.export_btn = QPushButton("导出数据")
        self.export_btn.clicked.connect(self.export_data)
        
        export_layout.addLayout(option_layout)
        export_layout.addWidget(self.export_btn)
        self.right_layout.addWidget(export_group)
    
    def connect_monitor_signals(self):
        """连接监控信号到UI更新槽"""
        self.monitor.cpu_update.connect(self.update_cpu_ui)
        self.monitor.memory_update.connect(self.update_memory_ui)
        self.monitor.disk_update.connect(self.update_disk_ui)
        self.monitor.disk_io_update.connect(self.update_disk_io_ui)
        self.monitor.network_update.connect(self.update_network_ui)
        self.monitor.gpu_update.connect(self.update_gpu_ui)
        self.monitor.process_update.connect(self.update_process_ui)
    
    def refresh_process_list(self):
        """刷新进程列表"""
        self.process_list.clear()
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                self.process_list.addItem(f"{proc.info['pid']}: {proc.info['name']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    
    def add_selected_processes(self):
        """添加选中的进程到监控"""
        selected_items = self.process_list.selectedItems()
        for item in selected_items:
            pid_str = item.text().split(":")[0]
            try:
                pid = int(pid_str)
                self.monitor.add_process_monitor(pid)
            except ValueError:
                pass
    
    def remove_selected_processes(self):
        """从监控中移除选中的进程"""
        selected_items = self.process_list.selectedItems()
        for item in selected_items:
            pid_str = item.text().split(":")[0]
            try:
                pid = int(pid_str)
                self.monitor.remove_process_monitor(pid)
            except ValueError:
                pass
    
    def format_bytes(self, size):
        """格式化字节大小为易读格式"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"
    
    def format_speed(self, speed_bytes):
        """格式化速度值"""
        return self.format_bytes(speed_bytes) + "/s"
    
    def update_cpu_ui(self, data):
        """更新CPU显示"""
        self.cpu_total_label.setText(f"总使用率: {data['total']:.1f}%")
        
        # 更新核心使用率表
        cores = data['cores']
        self.cpu_cores_table.setRowCount(len(cores))
        
        for i, (core, percent) in enumerate(cores.items()):
            self.cpu_cores_table.setItem(i, 0, QTableWidgetItem(core))
            self.cpu_cores_table.setItem(i, 1, QTableWidgetItem(f"{percent:.1f}%"))
            
            # 根据使用率设置背景色
            if percent > 80:
                color = QColor(255, 100, 100)  # 红色
            elif percent > 60:
                color = QColor(255, 200, 100)  # 橙色
            else:
                color = QColor(200, 255, 200)  # 绿色
                
            self.cpu_cores_table.item(i, 1).setBackground(color)
        
        # 记录数据
        self.logger.log_system_data("cpu", data)
    
    def update_memory_ui(self, data):
        """更新内存显示"""
        total_gb = data['total'] / (1024 ** 3)
        used_gb = data['used'] / (1024 ** 3)
        free_gb = data['free'] / (1024 ** 3)
        percent = data['percent']
        
        self.mem_usage_label.setText(f"使用率: {percent:.1f}%")
        self.mem_used_label.setText(f"已用: {used_gb:.2f} GB")
        self.mem_free_label.setText(f"可用: {free_gb:.2f} GB")
        self.mem_total_label.setText(f"总计: {total_gb:.2f} GB")
        
        # 记录数据
        self.logger.log_system_data("memory", data)
    
    def update_disk_ui(self, disks):
        """更新磁盘显示"""
        self.disk_table.setRowCount(len(disks))
        
        for i, disk in enumerate(disks):
            total_gb = disk['total'] / (1024 ** 3)
            used_gb = disk['used'] / (1024 ** 3)
            
            self.disk_table.setItem(i, 0, QTableWidgetItem(disk['device']))
            self.disk_table.setItem(i, 1, QTableWidgetItem(disk['mountpoint']))
            self.disk_table.setItem(i, 2, QTableWidgetItem(f"{total_gb:.2f} GB"))
            self.disk_table.setItem(i, 3, QTableWidgetItem(f"{used_gb:.2f} GB"))
            self.disk_table.setItem(i, 4, QTableWidgetItem(f"{disk['percent']:.1f}%"))
            
            # 根据使用率设置背景色
            if disk['percent'] > 90:
                color = QColor(255, 100, 100)  # 红色
            elif disk['percent'] > 80:
                color = QColor(255, 200, 100)  # 橙色
            else:
                color = QColor(200, 255, 200)  # 绿色
                
            self.disk_table.item(i, 4).setBackground(color)
        
        # 记录数据
        self.logger.log_system_data("disks", {"disks": disks})
    
    def update_disk_io_ui(self, data):
        """更新磁盘IO显示"""
        read_speed = data['read_speed']
        write_speed = data['write_speed']
        
        self.disk_read_label.setText(f"读取速度: {self.format_speed(read_speed)}")
        self.disk_write_label.setText(f"写入速度: {self.format_speed(write_speed)}")
        
        # 记录数据
        self.logger.log_system_data("disk_io", data)
    
    def update_network_ui(self, data):
        """更新网络显示"""
        upload_speed = data['upload']
        download_speed = data['download']
        
        self.net_upload_label.setText(f"上传: {self.format_speed(upload_speed)}")
        self.net_download_label.setText(f"下载: {self.format_speed(download_speed)}")
        
        # 记录数据
        self.logger.log_system_data("network", data)
    
    def update_gpu_ui(self, data):
        """更新GPU显示"""
        self.gpu_usage_label.setText(f"使用率: {data['utilization']:.1f}%")
        
        # 记录数据
        self.logger.log_system_data("gpu", data)
    
    def update_process_ui(self, process_data):
        """更新进程监控显示"""
        self.process_table.setRowCount(len(process_data))
        
        for i, (pid, data) in enumerate(process_data.items()):
            # 内存转换为MB
            mem_mb = data['memory'] / (1024 * 1024)
            
            # 计算IO变化率 (需要实现)
            read_kb = data['read_bytes'] / 1024
            write_kb = data['write_bytes'] / 1024
            
            self.process_table.setItem(i, 0, QTableWidgetItem(str(pid)))
            self.process_table.setItem(i, 1, QTableWidgetItem(data['name']))
            self.process_table.setItem(i, 2, QTableWidgetItem(f"{data['cpu']:.1f}%"))
            self.process_table.setItem(i, 3, QTableWidgetItem(f"{mem_mb:.1f}"))
            self.process_table.setItem(i, 4, QTableWidgetItem(f"{read_kb:.1f}"))
            self.process_table.setItem(i, 5, QTableWidgetItem(f"{write_kb:.1f}"))
        
        # 记录数据
        self.logger.log_process_data(process_data)
    
    def export_data(self):
        """导出数据"""
        file_format = self.format_combo.currentText()
        data_type = self.export_type_combo.currentText()
        
        # 获取保存文件名
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "导出监控数据",
            "",
            f"{file_format} Files (*.{file_format.lower()})"
        )
        
        if not file_name:
            return
        
        # 根据选择的格式导出
        if file_format == "CSV":
            if data_type == "所有数据":
                # 实际应用中需要分别导出各系统数据和进程数据
                pass
            elif data_type == "系统数据":
                # 导出系统数据
                pass
            elif data_type == "进程数据":
                # 导出进程数据
                pass
            self.statusBar().showMessage("CSV导出功能需完善")
            
        elif file_format == "JSON":
            self.logger.export_json(file_name)
            self.statusBar().showMessage(f"数据已导出到 {file_name}")
            
        elif file_format == "SQLite":
            self.logger.export_sqlite(file_name)
            self.statusBar().showMessage(f"数据已导出到 {file_name}")
            
        elif file_format == "Excel":
            self.logger.export_excel(file_name)
            self.statusBar().showMessage(f"数据已导出到 {file_name}")
    
    def closeEvent(self, event):
        """关闭事件处理"""
        self.monitor.stop_monitoring()
        super().closeEvent(event)
