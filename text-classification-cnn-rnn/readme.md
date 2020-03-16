run_cnn代码调式

首先，从主文件中调式，加载配置文件TCNNConfig，里面包含 embedding_dim词向量维度,seq_length序列长度,num_classes类别数,num_filters卷积核数目,kernel_size卷积核尺寸,voacb_size,词汇表大小hidden_dim全连接层神经元,dropout_keep_prob保留比例,learning_rate学习率,batch_size每批训练大小,num_epochs总迭代轮次,print_per_batch每多少轮输出一次结果,save_per_batch每轮存入多少tensorboard

其次
读取read_category,总共有10个类，建立cat_to_id
读取read_vocab,词汇表vocab_dir是一行一个字，，按行读取到words，在建立word_to_id,总共有5000个词汇量。

初始化TextCNN模型，进入train模式，使用process_file函数加载文件，获得content,labels，这是一个列表。5万句话。每句话长度为600，查字典转换成data_id,label_id，
初始化x，x是50000*600的矩阵   预处理每句话，当每句话超过600个字的时候，我们采取截断操作，即是从-600开始，最后位置往前数600个位置得到的矩阵。我们只取后600个单词。

在对模型进行write.add_graph

最后，我们训练模型，分批次读取文件，每次读64句话，用batch_iter生成batch_train，喂模型,创建字典feed_dict。若执行到10，20，我们session.run摘要信息，若执行100，200次我们session.run模型损失和模型精度，使用x_valy_val评价loss_val,acc_val并保存最好的结果。

进行session.run模型优化。若验证正确率长期不提升，提前结束训练。

run_cnn.py详细说明

初始化模型model=TextCNN（config）
调用init方法 input_x,input_y他们是一个tensor,加载cnn模型，#词向量映射，embedding是一个矩阵表，5000*64
查边得到embedding_inputs大小为600*64，进行conv1d卷积，参数大小embedding_inputs，num_filters=256,kernel_size=25，卷积之后得到conv大小为596*256
按列取最大值，这相当于最大池化，得到gmp大小为1*256
进入全连接层fc大小为1*128 ，其中128为隐藏层的维数，进行dropout，relu激活。
再次进行全连接输入为128，输出为10类，得到logits大小为1*10
最后softmax输出，预测类别
损失函数为交叉熵cross_entropy这是一个均值，它是一个数值。
accuracy 根据inPut_y和y_pred_cls是否相等得到准确率correct_pred
输出acc数值

run_rnn代码说明

#词向量映射得到embedding_inputs(每次放多少句话*字数*字的维度)

#2层rnn网络得到run_cell,之后有两个输出_output，_.前者_output为纵向的输出（？*600*128）
取最后一个时序输出作为最终结果[:,-1,:]这是一个二维的（？*128)
接着进行全连接层，dropout,relu层
最后在全连接转化为10分类，最后预测类别输出。
更新参数，使用交叉熵函数，求平均损失。




