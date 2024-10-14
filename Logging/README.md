## Python Logger for project practice

> Loguru 是一个功能强大且易于使用的开源日志记录库, 建立在 Python 标准库中的 logging 模块之上, 提供了更加简洁直观的接口, 可以轻松地记录不同级别的日志消息, 并根据需求输出到终端、文件或其他目标.

loguru提供了七层日志层级,生产环境中，常常在不同场景下使用不用的日志类型，用于处理各种问题

每种类型的日志有一个整数值,表示日志层级 log level no
- TRACE (5): 用于记录程序执行路径的细节信息，以进行诊断;
- DEBUG (10): 开发人员使用该工具记录调试信息;
- INFO (20): 用于记录描述程序正常操作的信息消息;
- SUCCESS (25): 类似于INFO，用于指示操作成功的情况;
- WARNING (30): 警告类型，用于指示可能需要进一步调查的不寻常事件;
- ERROR (40): 错误类型，用于记录影响特定操作的错误条件;
- CRITICAL (50): 严重类型，用于记录阻止核心功能正常工作的错误条件;

### install
```shell
# https://github.com/Delgan/loguru
pip install loguru

```

### quick start
```python

import os
from loguru import logger

if __name__ == "__main__":
    # 调用 logger.add() 方法来配置文件输出
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, "log_file.log")
    logger.add(log_filepath)

    # 设置日志级别, 默认级别为 INFO
    logger.level("DEBUG")

    # 自定义日志格式, 调用 logger.add()方法并设置format 参数, 可以指定日志的格式
    logger.add(log_dir + "app.log", format="[{time:HH:mm:ss}] {level} - {message}")

    # 添加日志切割, 添加日志切割选项来分割文件，以便更好地管理和维护
    # 将日志文件每天切割为一个新文件
    logger.add(log_dir + "app.log", rotation="00:00")

    # Loguru 会自动添加时间戳、日志级别和日志消息内容，并将其输出到终端
    logger.info("this is a INFO level message")
    logger.warning("this is a WARNING level message")
    logger.error("this is a ERROR level message")

```

### 实际应用场景-自动化测试
```python
# 使用 Loguru 可以轻松地记录测试步骤、断言结果和异常信息
from loguru import logger

@logger.catch
def run_test():
    logger.info("开始执行测试")
    # 执行测试代码
    assert some_condition, "条件不符合"
    logger.info("测试通过")

if __name__ == "__main__":
    run_test()

```
