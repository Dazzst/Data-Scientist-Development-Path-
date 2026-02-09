# Data Scientist Development Path  
数据科学家成长路径

本仓库用于系统整理成为 **Data Scientist** 所需的知识与技能，同时覆盖 Data Analyst / Business Analyst 相关能力，既是我的个人复习路线，也希望能为正在学习的数据同学提供参考。[page:1]

This repository is a structured **Data Scientist Development Path**, covering the full spectrum from data analysis to machine learning, deep learning and data engineering. It is designed for both **personal review** and **public learning reference**.[page:1]

---

## 目录 / Table of Contents

1. 路线说明 / Path Overview  
2. 数据分析基础（Basic Data Analysis）  
3. 数据分析中级（Intermediate Data Analysis）  
4. 数据分析高级（Advanced Data Analysis）  
5. 数据分析大神（Expert Level）  
6. 数据实战（Projects & Kaggle）  
7. 学习资料（Resources）  

---

## 1. 路线说明  
Path Overview

这是一条从「数据分析基础 → 机器学习与深度学习 → 数据工程与系统化实践 → 高阶数据科学」的完整路径，中间自然包含了数据分析师所需的工具、分析框架和业务能力。[page:1]

This path goes from **basic data analysis** to **advanced data science**, including programming, statistics, machine learning, deep learning, data engineering, and practical projects. Data analyst skills are treated as an essential foundation for data scientists.[page:1]

学习方式建议：  
- README 只提供结构和导航，**所有具体细节与示例放在子文件 / 子目录中**（例如 `Python_Intro.md`、`SQL_Intro.md`、`Kaggle_handbook.md` 等）。[page:1]  
- 可以按阶段顺序学习，也可以按主题（如 Python、SQL、ML、MLOps）跳转查阅。[page:1]

---

## 2. 数据分析基础  
Basic Data Analysis

这一阶段主要面向：**数据分析师 / 数据科学入门者**，目标是打牢数据分析和业务理解的基础。[page:1]  
This stage builds core skills for data analysis and business understanding.[page:1]

涵盖内容（细节见各子文件）：[page:1]

- 编程与开发工具 Programming & Tools  
  - [Python Intro](./Python/Python_Intro.md)  
  - [SQL Intro](./SQL/SQL_Intro.md)  
  - [R Intro](./R/R_Intro.md)  
  - [Excel Intro](./Excel/Excel_Intro.md)  

- 数据分析 Data Analysis  
  - 数据采集与导入  
  - 探索性数据分析（EDA）  
  - 数据预处理 / 清洗  
  - 指标体系构建  

- 商业分析 Business & Analytical Frameworks  
  - 详见目录：`Analytical Framework/` 与相关文档  
  - 包含：5W2H、多维度拆解、对比分析、假设检验分析、相关性分析、群组分析、RFM 分群、AARRR 模型、漏斗分析等。[page:1]

- 统计学 Statistics  
  - 描述性统计、概率论、推断性统计、A/B 测试基础。[page:1]

- 可视化与数据呈现 Visualization & Data Storytelling  
  - 工具：Matplotlib、Seaborn、Tableau、Power BI  
  - 图表选择与数据叙事原则。[page:1]

- 数据清洗与特征工程 Data Cleaning & Feature Engineering  
  - 缺失值与异常值处理  
  - 特征编码（One-Hot、Label Encoding）、特征缩放、日期时间特征处理等。[page:1]

---

## 3. 数据分析中级  
Intermediate Data Analysis

这一阶段开始从「分析」走向「建模」，逐步接近 Data Scientist 的核心技能。[page:1]  
This stage focuses on classical machine learning, deep learning basics, data engineering, and model deployment.[page:1]

- 经典机器学习 Classical Machine Learning  
  - 线性/逻辑回归、决策树、SVM 等模型理论与假设  
  - 模型评估与选择：Accuracy、Precision、Recall、F1、AUC、交叉验证、超参数调优  
  - 集成学习：Random Forest、GBDT、XGBoost、LightGBM  
  - 无监督学习：聚类（K-Means、DBSCAN）、降维（PCA、t-SNE）  

- 神经网络与深度学习 Neural Networks & Deep Learning  
  - 基础概念：神经元、激活函数、损失函数、梯度下降  
  - 常见结构：FNN、CNN、RNN/LSTM/GRU  
  - 深度学习框架：TensorFlow / PyTorch 基本使用  

- 数据工程与存储 Data Engineering & Storage  
  - 大数据生态：Hadoop、Spark 基本概念  
  - 分布式计算：RDD、DataFrame 核心思想与常用操作  
  - 数据仓库：架构、ETL/ELT 流程  
  - 云平台入门：AWS / Azure / GCP 中与数据相关的核心服务。[page:1]

- 模型部署与 MLOps Intro  
  - 模型持久化（pickle、ONNX 等）  
  - 使用 Flask / Streamlit 封装推理 API  
  - 容器化（Docker）基础  
  - 初步了解 MLOps：模型监控、版本管理、Pipeline 等。[page:1]

（部分内容链接：`Data Analysis/`、`Kaggle/` 等子目录及学习资料小节。）[page:1]

---

## 4. 数据分析高级  
Advanced Data Analysis

这一阶段强调「**方向选择 + 深度**」，从通才走向在某一领域具备专业度的数据科学家。[page:1]

可能的专项方向包括（各方向细分内容在后续小文件中具体展开）：[page:1]

- 自然语言处理（NLP）  
- 计算机视觉（CV）  
- 时间序列分析（Time Series）  
- 推荐系统（Recommender Systems）  
- 大数据与分布式计算  

同时关注：  
- 模型可解释性：特征重要性、SHAP/LIME 等方法（直觉层面）  
- 模型监控与数据漂移  
- 更复杂的实验设计与指标体系。[page:1]

---

## 5. 数据分析大神  
Expert Level

面向希望在 Data Science 方向做到「**专家 / 影响者**」的人群。[page:1]

方向要点（具体内容将以更细的 Markdown 文档补充）：[page:1]

- 前沿研究与算法  
  - 论文阅读与复现  
  - 因果推断、强化学习等高阶方法  
  - 模型创新与算法改进思路  

- 大规模系统与架构  
  - 分布式训练与大规模在线推理  
  - 可扩展的 ML 平台、特征平台、统一特征存储  

- 影响力与领导力  
  - 开源项目贡献  
  - 论文 / 技术博客 / 技术分享  
  - 在团队与产品层面的技术决策。[page:1]

---

## 6. 数据实战  
Projects & Kaggle

这一部分将前面知识落地到真实项目和数据集。[page:1]

- Kaggle 实战  
  - [Kaggle 实操手册](./Kaggle/Kaggle_handbook.md)  
  - 推荐从结构化数据任务入手（如 Tabular），再逐步尝试 NLP / CV 等方向。[page:1]

- 典型数据集  
  - [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)  
    - 可用于：电商业务分析、用户行为分析、留存与复购、预测任务等。[page:1]

- 项目建议  
  - 基础：端到端数据分析报告  
  - 中级：经典 ML 项目（回归 / 分类）  
  - 高级：专项方向项目（推荐、NLP、CV、时序等）。[page:1]

---

## 7. 学习资料  
Resources

辅助整个 Data Scientist Path 的参考资料。[page:1]

- 编程与工具  
  - [100天学习Python](https://github.com/jackfrued/Python-100-Days/blob/master/Day66-80/71.NumPy%E7%9A%84%E5%BA%94%E7%94%A8-4.md)  
  - [Python Data Science Handbook](https://github.com/wangyingsm/Python-Data-Science-Handbook/blob/master/notebooks/01.05-IPython-And-Shell-Commands.ipynb)  

- 综合 Cheatsheets  
  - [Data Science cheatsheets](https://github.com/FavioVazquez/ds-cheatsheets?tab=readme-ov-file)  

- 学习路线与经验分享  
  - [数据分析师成长](https://zhuanlan.zhihu.com/p/478792950)  
  - [Python数据分析路径](https://zhuanlan.zhihu.com/p/29813260)  
  - [如何寻找数据分析项目](https://www.zhihu.com/question/68476755/answer/1084320060)  
  - [Python黑魔法](https://www.zhihu.com/question/431725755/answer/1596178843)  

- 中级 / 高级机器学习与 AI  
  - [AI Learning](https://github.com/apachecn/ailearning/blob/master/docs/ml/1.md) 等。[page:1]

---

欢迎 Issue / PR，一起完善这条 Data Scientist 成长路径。[page:1]  
Issues and pull requests are welcome to improve and extend this path.[page:1]

