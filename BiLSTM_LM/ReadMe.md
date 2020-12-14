# Bi LSTM语言模型

模型：

![LSTM-Page-1](.\image\LSTM-Page-1.png)


1. 环境需求：python 3.6 pytorch 1.4 cuda 10.1
2. 需要将数据命名为dataset.txt 来展开训练（数据下载地址：https://pan.baidu.com/s/1wdXesmTVBtoh1XNvJOD-dg 提取码：pecg）
3. 预训练WordEmbedding(https://pan.baidu.com/s/1AmXYWVgkxrG4GokevPtNgA)
4. 自行生成w2v_map存入data目录下
5. 训练好的模型在model文件夹下
6. 运行BiLSTM.py 开始训练
7. 测试：运行get_prop2.py使用两次采样预测下个字的概率

