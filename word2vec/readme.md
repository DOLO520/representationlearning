w2v.py模型介绍

w2v.py模型是采用TensorFlow框架实现网络搭建，建立skip-gram模型，使用负采样方法进行训练。
模型的输入是 test_seg.txt ，它是一个分词之后的语料库，每个词语用空格隔开，词语总共有2262896个，去掉公共词之后有199247个词语。


首先，读取数据read_data后得到data,此时它是一个列表，把它命名为words


第二步，build_dataset,建立count列表，计算每个单词的频率传入到列表中。此时count里面的数据为 [（'十岁时',2），（'白描',2）,……]
接着为每个词创建id ,在建立data=list(),把每个词语放到data里面，此时data为词语所对应的id序号。即data把词语id化
我们的build_data返回四个值，data,count,dictionary,reverse_dictionary

第三步，为skip-gram模型生成可训练的批次，batch_size=8,num_skip=2,总窗口，skip_windows=1，上文和下文个1个字，embedding_size=128,这四个值均为超参数。每个词语对应labels,分别为上文和下文的词语。


第四步，建立和训练skip-gram。
创建计算图，为NCE（noise-contrastive estimation）随机初始化权重和偏置，定义损失nce_loss，负采样个数设置为60。使用SGD优化，学习率设为1.0，
计算每个minibatch和all embedding 的余弦相似度，这里要对embeddings归一化，进行embedding_lookup查询，得到valid—embeddings ,计算valid-embeddings和归一化后的embeddings之间的余弦相似度。静态图就被定义好了。


第五步，session.run()训练模型参数，并进行更新。


第六步，打印出结果，相似度评估，打印最接近的词语，这里取top-K=8,打印前8个最相似的词语。








