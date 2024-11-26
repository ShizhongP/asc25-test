# ASC25 AI Test

## Task 1

### 任务目标及要求

在docker多容器中, 模拟scatter和reduce,gather操作实现简单的数据并行和张量并行

- 不限制通信协议
- 至少两个及以上的容器(worker)
- 不限制第三方库的使用

### 模型

两个线性层的MLP，为了便于实现，不要求实现bias和激活函数

- 示例

```python
import numpy as np
class Model:
    self.weight_a = np.ones((4,4),dtype=np.float64)
    self.weight_b = np.full((4,4),fill_value=2,dtype=np.float64)

    def forward(x):
        pass
```

### 构建测试模块

构建测试模块，可采用随机生成的数据进行测试

### 提交要求以及形式

1. (必做)你的baseline代码和对应的测试模块和数据，至少要求能跑通且结果正确，并且能说明为什么这样设计，原理至少能说清楚
2. (可选)在你的baseline上进行的上面优化
3. (可选)完成日志记录，记录所有容器的工作状态，格式自定义

### 参考

- [张量并行](https://github.com/wdndev/llm_interview_note/blob/main/04.%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83/4.%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C/4.%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C.md)
- [三大并行](https://github.com/ShizhongP/dl-notes/blob/main/notes/%E4%B8%89%E5%A4%A7%E5%B9%B6%E8%A1%8C%E6%89%8B%E6%AE%B5.md)

## Task 2

完形填空预测任务,使用你已掌握的任何方法，尽可能提升的预测任务精度，提供了一个`demo.py`示例代码

- 使用任何手段以方便你的训练和推理
- 模型可自由选择不做限制，根据自己可使用的计算资源决定
- 训练框架可自由选择
- 主要考察已经掌握和了解的知识

### 数据

数据存放在`data`文件夹下面

1. 数据集分为train数据，dev数据，test数据
2. 最终的精度(acc)使用test数据测试，test数据不允许用于训练

- 数据示例

```json
{
    "article": "At age 86, Millie Garfield is one of the world's oldest elderly bloggers . _ reading a newspaper article in 2003 and then asking her son for _ in getting online, Millie has been blogging ever since. We usually associate blogging with the _ : our children, grandchildren, nieces or nephews. While the blogging landscape was once _ almost entirely by teens, it has opened to different age groups now. After 38 years of marriage, Millie _ her husband in 1994. She has no siblings and has only one son. She has to live alone. Like many elderly people, her social network was beginning to _ in size as many of her friends were in assisted living. Blogging has _ Millie's universe. \"I have to blog once a week,\" she says. \"If I don't, they start _ about me.\" When I ask who \"they\" are, Millie says they are the 70 or 80 _ who visit her blog each day. When she was three days _ in posting one week, she began getting _ from them to see if she was okay. She has also got to _ other bloggers from around the country. Not only has blogging helped Millie make new _ , but it has also helped her learn about herself. \"I write about everyday living in a _ fashion, so I try to find interesting things in a TV show, a movie, or a(n) _ to the dentist, she says. \"I never knew I was funny but now people _ me I am. It is a big discovery.\" Millie _ loves blogging. \"My life would be _ and empty without it. I'm able to learn from people all over the world,\" she says. Then she adds, \"When you're older, you don't have many _ . The wonderful thing about blogging is that you can have many people hear what you think and no one _ you when you are speaking.\"", 
    "options": [["While", "Until", "After", "As"], ["help", "apology", "excuse", "permission"], ["old", "young", "rich", "sick"], ["damaged", "occupied", "prepared", "designed"], ["missed", "followed", "recognized", "lost"], ["grow", "develop", "decrease", "remain"], ["expanded", "concluded", "found", "ruined"], ["complaining", "thinking", "arguing", "worrying"], ["workers", "readers", "passengers", "speakers"], ["late", "away", "fast", "ready"], ["warnings", "suggestions", "emails", "books"], ["know", "see", "change", "ask"], ["comments", "connections", "contributions", "combinations"], ["popular", "famous", "similar", "humorous"], ["gift", "visit", "wave", "award"], ["warn", "prove", "order", "tell"], ["probably", "fortunately", "hardly", "clearly"], ["poor", "slow", "dull", "simple"], ["listeners", "managers", "interpreters", "lecturers"], ["fears", "interrupts", "controls", "treats"]], 
    "answers": ["C", "A", "B", "B", "D", "C", "A", "D", "B", "D", "C", "A", "B", "D", "B", "D", "D", "C", "A", "B"]
}
```

### 提交要求以及形式

1. (必作)你的训练代码/预测代码，如有必要，简要描述使用的方案和技术，至少有个测试结果，无需提交模型权重
2. (可选)你的训练日志，格式自定义

### 其它

1. 如果本地计算资源不够，可以借助kaggle来训练
2. 参考[使用bert做完形填空](https://developer.aliyun.com/article/1209150)