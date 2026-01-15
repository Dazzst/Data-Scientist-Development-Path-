# Data-Scientist-Development-Path
重新整理规划所有关于数据科学的知识点以方便复习。


马上要回国了，我需要复习所有已经学到的知识和技能，并且计划以后可能会接触并更新的技能。

内容将包含数据分析，商业分析，统计学，可视化数据处理，机器学习，神经网络。

本仓库包含成为数据科学家/数据分析师的学习路线与知识点整理。


## 数据分析基础 Basic Data Analysis
- 编程与开发工具 Programming and Development Tools
  - [Python](Python/Python_Intro.md)
  - [SQL](SQL/SQL_Intro.md)
  - [R](R/R_Intro.md) 
  - [Excel](Excel/Excel_Intro.md)
- 数据分析
  - 数据采集与导入
  - 探索性数据分析 
  - 数据预处理
  - 指标体系构建
- 商业分析
  - 分析框架
    - 5W2H分析方法
    - 多维度拆解分析方法
    - 对比分析方法
    - 假设检验分析方法
    - 相关性分析方法
    - 群组分析方法
    - RFM用户分群分析方法
    - AARRR模型分析方法
    - 漏斗分析方法
  - 案例分析
- 统计学
  - 描述性统计
  - 概率论
  - 推断性统计
  - A/B 测试
- 可视化与数据呈现
  - 可视化工具
    - Matplotlib(py)
    - Seaborn(py)
    - Tableau
    - PowerBI 
  - 图表选择
  - 数据叙事 
- 数据清洗与特征工程
  - 特征工程:
    - 特征编码（One-Hot, Label Encoding）
    - 特征缩放
    - 日期时间特征处理
## 数据实战
- [Kaggle 实操手册](Kaggle/Kaggle_handbook.md)
- https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce 
## 学习资料
- [100天学习Python](https://github.com/jackfrued/Python-100-Days/blob/master/Day66-80/71.NumPy%E7%9A%84%E5%BA%94%E7%94%A8-4.md)
- [Data Science cheatsheets](https://github.com/FavioVazquez/ds-cheatsheets?tab=readme-ov-file)
- [Data Science in Python Handbook](https://github.com/wangyingsm/Python-Data-Science-Handbook/blob/master/notebooks/01.05-IPython-And-Shell-Commands.ipynb)
- [数据分析师成长](https://zhuanlan.zhihu.com/p/478792950)
- [Python数据分析路径](https://zhuanlan.zhihu.com/p/29813260)
- [如何寻找数据分析项目](https://www.zhihu.com/question/68476755/answer/1084320060)
- [Python黑魔法](https://www.zhihu.com/question/431725755/answer/1596178843)


## 数据分析中级 Intermediate Data Analysis

- 经典机器学习
  - 模型理论: 理解线性回归 (Linear Regression)、逻辑回归 (Logistic Regression)、决策树 (Decision Tree)、支持向量机 (SVM) 等经典模型的数学原理和假设。
  - 模型评估与选择: 掌握准确率 (Accuracy)、精确率 (Precision)、召回率 (Recall)、F1-Score、AUC/ROC 等评估指标，了解交叉验证 (Cross-Validation) 和超参数调优 (Hyperparameter Tuning) 的方法。
  - 集成学习 (Ensemble Learning): 了解 Bagging（如 Random Forest）和 Boosting（如 GBDT、XGBoost、LightGBM）的原理和应用场景。
  - 无监督学习: 掌握聚类算法（如 K-Means、DBSCAN）和降维方法（如 PCA、t-SNE）。
- 神经网络与深度学习
  - 基础概念: 理解神经元、激活函数（如 ReLU、Sigmoid）、损失函数、梯度下降 (Gradient Descent) 等核心组件。
  - 网络架构: 熟悉前馈神经网络 (FNN)、卷积神经网络 (CNN) 和循环神经网络 (RNN/LSTM/GRU) 的基本结构和应用。
  - 深度学习框架: 能够使用 TensorFlow 或 PyTorch 搭建、训练和评估深度学习模型。
- 数据工程与存储
  - 大数据基础: 了解大数据生态系统（如 Hadoop、Spark）的基本概念和作用。
  - 分布式计算: 掌握 Spark 的核心概念（如 RDD、DataFrame）和基本操作，用于处理大规模数据。
  - 数据仓库: 了解数据仓库 (Data Warehouse) 的概念、架构（如 Inmon、Kimball）和 ETL/ELT 流程。
  - 云平台 (Cloud Platforms): 了解 AWS、Azure 或 Google Cloud 等主流云平台在数据存储（如 S3/Blob Storage）、计算和数据服务方面的基本知识。
- 模型部署与 MLOps
  - 模型持久化: 掌握使用 pickle 或 ONNX 等方式保存和加载训练好的模型。
  - API 部署: 能够使用 Flask 或 Streamlit 等框架将模型封装为 RESTful API 服务，实现模型推理。
  - 容器化 (Containerization): 熟悉 Docker 的基本用法，将模型及其运行环境打包成容器。
  - MLOps 概念: 了解模型监控、版本控制、自动化训练流水线 (Pipeline) 等 MLOps 流程和工具（如 MLflow、Kubeflow）的基础知识。
  ##  学习资料
  - [AI Learning](https://github.com/apachecn/ailearning/blob/master/docs/ml/1.md)

    
## 数据分析高级
- 专项方向（NLP / CV / 时序 / 推荐 等）
- 大数据与分布式计算
- 项目实践与软技能

## 数据分析大神
- 前沿研究与算法（论文复现、模型创新、因果推断、强化学习等）
- 大规模系统与架构（分布式训练、在线大规模推理、可扩展 ML 平台）
- 影响力与领导力（开源贡献、论文/技术博客、团队与产品层面的技术决策）
