# 提交日志
## 10.27 提交
总体上完成了对googlenet的训练，下一步就是阅读相关代码，完成freeze（lora）微调。

### 新增
inference_flask.py  
作为flask后端，方便在前端查看。

meanAndStd.py  
作为数据预处理部分，计算train和test的mean和std

lr_finder.py  
修改数据集以达到使用现有数据集  
同时计算学习率，左眼是 0.003-0.005
存放相关的在result.jpg下  

tenplates--flask.html  
作为flask简陋的前端（后续可以改进）

附今天训练：  
Evaluating Network.....  
Test set: Epoch: 100  
Average loss: 0.0066  
Accuracy: 0.7981  
Precision: 0.7752  
Recall: 0.7981  
F1 Score: 0.7739  
AUC: 0.8063  
Time consumed: 17.84s  

saving weights file to   
checkpoint/googlenet/Monday_27_October_2025_11h_56m_21s/googlenet-100-regular.pth  
✅ Training started. You can visualize with TensorBoard:  
```tensorboard --logdir=runs --port=6006 --host=localhost```

## 10.30 修改
对googlenet代码调试
同时对相关参数修改