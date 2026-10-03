#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 20_mongodb.py
@Python Version: 3.12.13
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2026-10-03
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V0.1
@License: Apache License Version 2.0, January 2004
    Copyright 2026. All rights reserved.

@Description: Python 操作 NOSQL 数据库 MongoDB

server download link:
https://fastdl.mongodb.org/windows/mongodb-windows-x86_64-9.0.2.zip
"""

import asyncio
from datetime import datetime

from pymongo import ASCENDING, AsyncMongoClient


class UserService:
    def __init__(self, uri: str, db_name: str):
        self.client = AsyncMongoClient(uri)
        self.db = self.client[db_name]
        self.users = self.db.users

    async def init_indexes(self):
        """初始化索引"""
        await self.users.create_index("email", unique=True)
        await self.users.create_index([("created_at", ASCENDING)])

    async def create_user(self, name: str, email: str, age: int):
        """创建用户"""
        user = {
            "name": name,
            "email": email,
            "age": age,
            "tags": [],
            "created_at": datetime.now(),
        }
        result = await self.users.insert_one(user)
        return result.inserted_id

    async def find_users_by_age_range(self, min_age: int, max_age: int):
        """按年龄范围查询"""
        cursor = self.users.find({"age": {"$gte": min_age, "$lte": max_age}}).sort(
            "age", 1
        )
        return await cursor.to_list(length=100)

    async def add_tag(self, email: str, tag: str):
        """为用户添加标签"""
        await self.users.update_one({"email": email}, {"$push": {"tags": tag}})

    async def get_age_distribution(self):
        """聚合：年龄分布统计"""
        pipeline = [
            {
                "$group": {
                    "_id": {
                        "$switch": {
                            "branches": [
                                {"case": {"$lt": ["$age", 25]}, "then": "18-24"},
                                {"case": {"$lt": ["$age", 35]}, "then": "25-34"},
                                {"case": {"$lt": ["$age", 45]}, "then": "35-44"},
                            ],
                            "default": "45+",
                        }
                    },
                    "count": {"$sum": 1},
                    "avg_age": {"$avg": "$age"},
                }
            },
            {"$sort": {"_id": 1}},
        ]
        cursor = self.users.aggregate(pipeline)
        return await cursor.to_list(length=None)

    async def close(self):
        await self.client.close()


async def main():
    server_url = "mongodb://localhost:27017"
    database_name = "userdb"
    service = UserService(server_url, database_name)
    await service.init_indexes()

    # 创建用户
    await service.create_user("Alice", "alice@example.com", 30)
    await service.create_user("Bob", "bob@example.com", 24)
    await service.create_user("Charlie", "charlie@example.com", 38)
    await service.create_user("Diana", "diana@example.com", 42)

    # 添加标签
    await service.add_tag("alice@example.com", "vip")
    await service.add_tag("alice@example.com", "premium")

    # 查询
    users = await service.find_users_by_age_range(25, 40)
    print("年龄25-40的用户:", users)

    # 聚合统计
    distribution = await service.get_age_distribution()
    print("年龄分布:", distribution)

    await service.close()


if __name__ == "__main__":
    asyncio.run(main())
