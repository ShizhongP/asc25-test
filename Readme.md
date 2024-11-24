# ASC25 AI Test

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
    self.weight_b = np.full((4,4),fill_value=2,dtype=np.float64)

    def forward(x):
        pass
```

## 构建测试模块

构建测试模块，可采用随机生成的数据进行测试

## 提交要求以及形式

1. (必做)你的baseline代码和对应的测试模块和数据，至少要求能跑通且结果正确，并且能说明为什么这样设计，原理至少能说清楚
2. (可选)在你的baseline上进行的上面优化
3. (可选)完成日志记录，记录所有容器的工作状态，格式自定义

## 参考

- [张量并行](https://github.com/wdndev/llm_interview_note/blob/main/04.%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83/4.%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C/4.%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C.md)
- [三大并行](https://github.com/ShizhongP/dl-notes/blob/main/notes/%E4%B8%89%E5%A4%A7%E5%B9%B6%E8%A1%8C%E6%89%8B%E6%AE%B5.md)
