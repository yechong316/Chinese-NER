# ChineseNER

原作者地址在<https://github.com/buppt/ChineseNER>

首先非常作者，让我入门命名实体识别，非常感谢~~~~。在原作的基础进行优化改进，python2.7 升级为python3.5.2，进行重新优化。

# 本项目使用

+ python 3.5.2
+ tensorflow 1.10.0
+ pytorch 0.4.0 （本人暂未对torch版本进行优化）

## 数据
data文件夹中有三个开源数据集可供使用，玻森数据 (https://bosonnlp.com) 、1998年人民日报标注数据、MSRA微软亚洲研究院开源数据。其中boson数据集有6种实体类型，人民日报语料和MSRA一般只提取人名、地名、组织名三种实体类型。

先运行数据中的python文件处理数据，供模型使用。

# 工作目标

1. 使用更大容量的语料库，比如本人即将使用人民日报专属语料库，提升训练质量
2. 增加神经元单元个数，迭代次数，模型局部调优
3. 未来加入idCNN + CFR模型进行计算。
4. 加入tf.flags属性，便于baseline ==》 benchmark！

# 代码解析

## 数据集

有3个数据，分别运行该文件进行数据预处理

![1559806774562](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1559806774562.png)

我们来看人民日报的数据集

本代码很简洁，4个函数，分别对应4个方法

```
if __name__ == '__main__':
    
    originHandle()
    originHandle2()
    sentence2split()
    data2pkl()
```

### originHandle()

首先看函数的功能

```
with open('./renmin.txt','r', encoding='utf-8') as inp,open('./renmin2.txt','w', encoding='utf-8') as outp:
    for line in inp.readlines():
        line = line.split('  ')
        i = 1
        ~~~~~~~~~~省略~~~~~~~~~~~~~~~~~~
```

![1559806883576](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1559806883576.png)

转变为：

![1559806898690](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1559806898690.png)

最终结果为：

![1559807060262](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\1559807060262.png)

# 核心模型

```
input_embedded = tf.nn.embedding_lookup(word_embeddings, self.input_data)
input_embedded = tf.nn.dropout(input_embedded,self.dropout_keep)

lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_dim, forget_bias=1.0, state_is_tuple=True)
lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_dim, forget_bias=1.0, state_is_tuple=True)
(output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, 
                                                                 lstm_bw_cell, 
                                                                 input_embedded,
                                                                 dtype=tf.float32,
                                                                 time_major=False,
                                                                 scope=None)
```