# ASC25 Test

## 任务目标及要求

在docker多容器中, 模拟scatter和reduce,gather操作实现简单的数据并行和张量并行

- 不限制通信协议
- 至少两个及以上的容器(worker)
- 不限制第三方库的使用

## 模型

两个线性层的MLP，为了便于实现，不要求实现bias

- 示例

```python
import numpy as np
class Model:
    self.weight_a = np.ones((4,4),dtype=np.float64)
    self.weight_b = np.full((4,4), dtype=np.float64)

    def forward(x):
        pass
```

## 构建测试模块

构建测试模块，可采用随机生成的数据进行测试

## 提交要求以及形式

1. (必做)你的baseline代码和对应的测试模块和数据，至少要求能跑通且结果正确，并且能说明为什么这样设计，原理至少能说清楚
2. (可选)在你的baseline上进行的上面优化
3. (可选)完成日志记录，记录所有容器的工作状态，格式自定义
