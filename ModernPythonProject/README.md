 ## Modern Python Project

 ### Quick Start

 ```shell
 # download UV
winget install --id=astral-sh.uv  -e

# uv init 会生成 pyproject.toml、README.md、uv.lock 等基础文件
uv init --name VisionSmartAlgorithm --python 3.12

# 编辑 pyproject.toml，补充项目元数据、依赖分类、构建系统等信息
# 使用 UV 自动安装开发环境和依赖 && 仅安装生产依赖，直接运行 uv sync
uv sync --dev
uv sync
# UV 会根据 pyproject.toml 和 uv.lock 精确锁定依赖版本，保证环境可复现。
# 此后添加新依赖时使用 uv add <package>，删除用 uv remove <package>

# 项目结构设计 采用 src 布局（推荐用于可发布的包）以及清晰的功能模块划分

 ```
 