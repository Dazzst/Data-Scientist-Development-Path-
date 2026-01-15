# 首先是Data Exploration，也就是 EDA (Exploratory Data Analysis)，也就是对数据进行探索性的分析，从而为之后的处理和建模提供必要的结论。

一般这一步是用Pandas和numpy进行一些基本分析，或者做题进行观察。

做图Visulazation，通常来说 matplotlib 和 seaborn 提供的绘图功能就可以满足需求了。

比如散点图Scatter Plot，查看目标变量的分布,查看它们的分布趋势和是否有离群点的存在。
比如分布图 Distribution Plot / Histogram，查看 Feature 是否符合正态分布和是否需要将有偏度的图形进行调整。
比如箱线图Box Plot和提琴图Violin Plot，来直观地查看数据的分布。
对于分类问题，将数据绘制Pairplot根据 Label 的不同着不同的颜色绘制出来。
比如correlation matrix, 绘制变量之间两两的分布和相关度图表。
# Data Exploration (EDA) —— 数据探索与假设生成

EDA 不仅仅是画几个图看一看，它是**生成假设 (Hypothesis Generation)** 的过程。
你需要通过 EDA 回答：数据脏不脏？特征和 Target 有什么关系？是否需要做特殊的预处理？

**1. 自动化 EDA (Automated EDA) —— 效率为王**
* **痛点**：手动写代码画几十个 Feature 的直方图非常耗时。
* **神器**：使用 `ydata-profiling` (原 pandas-profiling) 或 `sweetviz`。
* **做法**：一行代码生成详细的 HTML 报告，自动计算缺失率、相关性、偏度、零值占比等。
    * `ProfileReport(df).to_file("report.html")`
* **注意**：如果数据量很大，建议先 `df.sample(100000)` 抽样后再跑，否则会很慢。

**2. 核心可视化检查清单 (The Checklist)**

* **目标变量分析 (Target Analysis)**
    * **分类问题**：必须看 `Target` 的分布比例。是否存在**类别不平衡 (Class Imbalance)**？如果是 1:99 的比例，后续必须考虑 Stratified K-Fold 或重采样。
    * **回归问题**：必须看 Target 是否符合正态分布。如果 Target 严重右偏（长尾），务必对其进行 `log1p` 变换，这通常能显著降低 RMSE。
* **单变量分析 (Univariate Analysis)**
    * **数值型**：看 Histogram。关注**偏度 (Skewness)** 和 **峰度 (Kurtosis)**。长尾分布通常需要 Log 变换或 Box-Cox 变换。
    * **类别型**：看 Countplot。关注是否存在 High Cardinality (类别极多) 的特征，或者只有 1-2 个样本的 Rare Labels。
* **双变量关系 (Bi-variate Analysis)**
    * **Correlation Matrix**：热力图 (Heatmap) 是标配。
        * *注意*：Pearson 相关系数只能衡量线性关系。建议同时查看 **Spearman** (秩相关)，它能发现非线性的单调关系。
    * **Scatter Plot**：查看 Feature 与 Target 的散点图。如果发现明显的非线性模式（如抛物线），这暗示你需要构造平方特征 ($x^2$)。
**3. 高维数据可视化 (High-Dimensional Visualization)**
* 当特征成百上千时，Pairplot 已经看不清了。
* **降维可视化**：使用 **t-SNE** 或 **UMAP** 将高维数据映射到 2D 平面。
    * 如果降维后的图明显聚成了几堆，且不同类别的点分得比较开，说明数据是可分的 (Separable)，模型很有希望。
    * 如果混成一团乱麻，说明可能需要强力的特征工程。
**4. 交互式绘图 (Interactive Plots)**
* 静态图有时候看不清细节。推荐使用 **Plotly**。
* 它可以让你在图表上缩放、悬停查看具体数值，特别适合排查离群点 (Outliers) 的具体 ID。
  
# 其次是Data Preprocessing，我们需要对比赛提供的数据集进行一些处理。

多个数据集，或者多个数据库时候，需要 Join 起来。
处理 Missing Data。
处理 Outlier。
必要时转换某些 Categorical Variable 的表示方式。
有些 Float 变量可能是从未知的 Int 变量转换得到的，这个过程中发生精度损失会在数据中产生不必要的 Noise，即两个数值原本是相同的却在小数点后某一位开始有不同。这对 Model 可能会产生很负面的影响，需要设法去除或者减弱 Noise。
查看之前的图表，对数据进行转换。
对于 Categorical Variable，常用的做法就是 One-hot encoding。
在 Kaggle 中，预处理不仅仅是“清洗”，更是为了“生存”（防止内存爆炸）。
**1. 内存优化 (Memory Reduction) —— 必做动作**
* **痛点**：Pandas 默认读取数值为 `float64` 和 `int64`，这非常浪费内存。
* **做法**：编写一个 `reduce_mem_usage` 函数，遍历所有列，检查数值范围。如果最大值不超过 255，就转为 `int8`；不超过 65535 转 `int16`；浮点数转 `float32`。
* **收益**：通常能将内存占用减少 50%-70%，防止 OOM (Out Of Memory)，且能加快模型训练速度。

**2. 缺失值处理 (Missing Values)**
* **树模型 (XGB/LGB/Cat)**：通常**不需要**填充缺失值。这些模型内部有机制（Default Direction）能自动学习“缺失”代表的含义。
* **神经网络/线性模型**：**必须**填充。
    * **统计填充**：均值 (Mean)、中位数 (Median)。
    * **模型填充**：使用 `KNNImputer` 或 `IterativeImputer` 利用其他列的信息预测缺失值。
* **重要技巧**：总是为重要的特征添加一个 `is_null` 的 0/1 新列。因为“数据缺失”这个行为本身往往就包含重要信息（比如用户不愿意填写年龄，可能暗示了某些特征）。

**3. 异常值处理 (Outliers)**
* **截断 (Clipping)**：不要直接删除异常值（这会减少训练样本）。建议使用 `clip` 函数，将超过 99% 分位数的值强制设为 99% 分位数值。
* **Log 变换**：对于长尾分布（如房价、销量），取 `log1p` (log(x+1)) 可以有效拉近异常值的距离，使其服从正态分布。

**4. 数据去噪 (Denoising) —— 针对你提到的 Float 问题**
* **精度修复**：很多脱敏数据原本是整数 (Int)，但因为除以了某个系数变成了浮点数 (Float)，导致出现了 `2.999999` 和 `3.000001` 这种本该相同却不同的噪声。
* **Trick**：尝试将数据 `round` 到特定小数位，或者尝试寻找那个“系数”把它还原回 Int。这能显著降低树模型寻找 Split Point 的难度。

**5. 类别清洗 (Categorical Cleaning)**
* **Rare Labels**：如果某个类别出现的次数极少（比如 < 1%），把它单独作为一类特征意义不大，反而会引入噪声。建议将这些长尾类别统一重命名为 `"Other"`。
# 接着是Feature Engineering，这是决定结果的很重要一个步骤。Feature Selection (特征筛选)

总的来说，现在的强力模型（如 XGBoost/LightGBM）虽然能自动处理一定的冗余，但主动进行 Feature Selection 仍然至关重要。我们通常的策略是：**先尽可能多地生成特征，再大刀阔斧地进行筛选**。

1.  **抗过拟合 (Curse of Dimensionality)**：Feature 越少，模型越不容易迷失在噪声中，泛化能力往往越强。
2.  **提升效率**：特征越少，训练和迭代越快，这对比赛后期争分夺秒非常关键。
3.  **处理共线性 (Multicollinearity)**：虽然树模型对共线性不敏感，但去除高度相关的冗余特征能让模型更稳定。
4.  **为高阶特征铺路**：通过筛选出最重要的 Top N 特征，我们可以将它们两两组合（加减乘除），往往能挖掘出意想不到的强力交互特征 (Interaction Features)。

**实用的筛选方法：**

* **基于模型的重要性 (Model-based Importance)**
    * 最直接的方法：训练一个 Random Forest 或 GBDT (XGBoost/LightGBM)，查看 `feature_importances_`。
    * **进阶技巧**：传统的 impurity-based importance 容易偏向高基数（High Cardinality）特征。更稳健的做法是使用 **Permutation Importance**（打乱某一列特征的值看模型精度下降多少）或者 **SHAP Values**，它们能给出更真实的特征贡献度。
    
* **递归特征消除 (RFE - Recursive Feature Elimination)**
    * 这是一种“贪心”算法。反复训练模型，每次剔除最不重要的末位特征，直到剩想保留的特征数量。这比单纯看一次重要性排名更靠谱。

* **对抗验证 (Adversarial Validation) —— Kaggle 核心技巧**
    * 如果训练集和测试集分布不一致（Distribution Shift），模型线下再好也没用。
    * **做法**：构建一个分类器，Target 为 `Is_Train` (0或1)，尝试区分训练集和测试集。如果 AUC 很高（比如 > 0.7），说明两者差异明显。
    * **筛选**：查看这个分类器的 Feature Importance，排在前面的就是导致分布差异的“罪魁祸首”特征。剔除它们通常能显著提升线上成绩。

* **脱敏数据处理**
    * 看 Feature Importance 对于某些数据经过脱敏处理（Feature 名字是 f1, f2...）的比赛尤其重要。这可以免得你浪费大把时间在琢磨一个完全不重要的变量的物理意义上。
 
# Feature Encoding (特征编码)，将非数值型的类别特征 (Categorical Features) 转化为数值，是模型能读懂数据的关键。不同的编码方式对模型效果影响巨大。

**1. Label Encoding / Ordinal Encoding**
* **做法**：将类别映射为整数（如 A->1, B->2, C->3）。
* **适用场景**：**树模型 (XGBoost/LightGBM/CatBoost)**。树模型可以很好地处理这种非线性关系，不需要 One-Hot 带来的稀疏矩阵。
* **注意**：对于线性模型（Linear Regression/SVM/NN），这种编码暗示了 3 > 1 的数学关系，通常是不合理的。

**2. One-Hot Encoding**
* **做法**：为每个类别创建一个新的 0/1 列。
* **适用场景**：**低基数 (Low Cardinality) 特征**，以及**线性模型**或**神经网络**。
* **缺点**：如果类别特别多（如 City 有 1000 个），会导致维度爆炸，且树模型在处理这种极度稀疏的特征时效率很低（树得切很多刀才能切分出一个类别）。

**3. Count Encoding / Frequency Encoding (Kaggle 常用技巧)**
* **做法**：用该类别在训练集中出现的次数（或频率）来替换类别本身。
* **逻辑**：它反映了该类别的“流行度”。比如“商品ID”本身没意义，但“购买次数”是有物理意义的强特征。
* **代码示例**：`df['col_count'] = df['col'].map(df['col'].value_counts())`

**4. Target Encoding / Mean Encoding (双刃剑)**
* **做法**：用该类别对应的 Target 均值来替换类别。例如，类别 A 的样本中，Label 为 1 的比例是 0.8，那么 A 就编码为 0.8。
* **威力**：对于高基数特征（如 UserID, IP地址），这是最强的编码方式之一。
* **致命风险**：**极易过拟合 (Data Leakage)**。
* **解决方案**：必须使用 **K-Fold Regularization**（用 K-Fold 交叉验证的方式，用其他折的数据来计算当前折的编码值）或者添加噪声 (Smoothing)。建议使用 `category_encoders` 库中的 `TargetEncoder` 或 `CatBoostEncoder`。

**5. Embedding (嵌入层)**
* **做法**：在神经网络中学习一个低维稠密向量来表示类别。
* **适用场景**：深度学习模型，或者将训练好的 Embedding 向量提取出来作为特征喂给树模型（Entity Embeddings）。

# 接着是很重要一部分内容 Model Selection (模型选择)

准备好 Feature 以后，就可以开始选用一些常见的模型进行训练了。在 Kaggle 的表格型数据比赛 (Tabular Data Competitions) 中，**Gradient Boosting Decision Tree (GBDT)** 类的模型占据了绝对的统治地位。

**1. GBDT 三巨头 (The Holy Trinity)**
这是你必须掌握的三个大杀器，几乎所有高分方案都是它们的 Ensemble：
* **XGBoost**: 开山鼻祖，极其稳定，生态完善。虽然训练速度稍慢，但精度依然是顶尖的。
* **LightGBM (LGBM)**: 微软开发。特点是**快**！内存占用低，训练速度比 XGBoost 快很多，是目前做 Baseline 和快速迭代的首选。
* **CatBoost**: Yandex 开发。特点是**对 Categorical Features 处理极佳**（无需繁琐的 One-Hot），且默认参数往往就能跑出极好的效果（过拟合风险低）。

**2. 辅助模型 (Base Models for Ensembling)**
单一模型很难拿金牌，我们需要差异化大的模型来做融合 (Ensemble)。以下模型虽然单打独斗不如 GBDT，但它们提供了宝贵的“不同视角”，是 Stacking 的最佳基模型：
* **Linear Models**: Logistic Regression, Ridge/Lasso (Sklearn)。
* **SVM / KNN**: 也就是 Support Vector Machines 和 K-Nearest Neighbors。
* **Extra Trees / Random Forest**: 虽同为树模型，但机制不同（Bagging vs Boosting），能提供很好的多样性。

**3. 神经网络 (Neural Networks)**
* **非表格数据**：在图像 (CV) 或文本 (NLP) 比赛中，Deep Learning (CNN, Transformer) 是绝对的主角。
* **表格数据**：千万不要忽视简单的 **MLP (多层感知机)** 或 **TabNet**。虽然它们在表格数据上单模分数通常略低于 GBDT，但因为原理完全不同（线性/非线性变换 vs 树状切割），它们能捕捉到树模型遗漏的信息，在最后的 Ensemble 阶段往往能带来关键的提分。

**实战建议**
* **起手式**：通常先用 **LightGBM** 快速跑通流程，验证 Feature Engineering 的有效性（因为它最快）。
* **冲刺阶段**：同时训练 XGBoost, LightGBM, CatBoost 以及 1-2 个神经网络模型，最后通过 **Stacking** 或 **Weighted Blending** 融合结果。

# Model Training & Hyperparameter Tuning (模型训练与调参)

在特征工程完成后，我们的目标是找到一组参数，让模型在**不过拟合 (Overfitting)** 的前提下尽可能学到规律。

**1. 调参工具：抛弃 Grid Search**
* **Grid Search (网格搜索)**：暴力遍历所有组合，效率极低，只适合参数极少的情况。
* **Optuna (Kaggle 首选)**：基于贝叶斯优化的自动化调参工具。它能“吸取教训”，根据之前的尝试结果智能猜测下一组更好的参数。
    * *优势*：速度快、代码简洁、支持剪枝（Pruning，发现某组参数效果不好就提前终止训练）。

**2. 核心调参策略 (The Tuning Strategy)**
不要一开始就陷入 0.0001 的微小提升中。
1.  **Baseline**：使用默认参数跑通模型，确立基线分数。
2.  **找树的数量 (n_estimators)**：固定一个较高的 `learning_rate` (如 0.05 或 0.1)，利用 **Early Stopping** 自动确定最佳的迭代次数（trees）。
3.  **主要参数微调**：使用 Optuna 搜索树结构参数（深度、叶子数等）和随机采样参数。
4.  **降维打击**：确定好结构参数后，将 `learning_rate` 调低（如 0.01 或 0.005），并按比例增加 `n_estimators`，通常能获得更好的泛化能力。

**3. GBDT 通用关键参数对照表**
不同的库参数名不同，但物理意义是一样的：

| 物理意义 | XGBoost | LightGBM | CatBoost | 作用 |
| :--- | :--- | :--- | :--- | :--- |
| **学习率** | `eta` / `learning_rate` | `learning_rate` | `learning_rate` | 步长。越小越稳但越慢。 |
| **树的数量** | `num_round` / `n_estimators` | `n_estimators` | `iterations` | 迭代次数。需配合 Early Stopping。 |
| **树的深度** | `max_depth` | `max_depth` | `depth` | 控制模型复杂度。LGBM 更常用 `num_leaves`。 |
| **叶子节点数**| (无直接对应) | `num_leaves` | (自动推导) | LGBM 的核心参数，控制复杂度。 |
| **行采样** | `subsample` | `subsample` / `bagging_fraction` | `subsample` | 防止过拟合，每次只练一部分数据。 |
| **列采样** | `colsample_bytree` | `feature_fraction` | `rsm` | 防止过拟合，每次只用一部分特征。 |
| **正则化** | `alpha`, `lambda` | `lambda_l1`, `lambda_l2` | `l2_leaf_reg` | 惩罚项，数值越大模型越保守。 |

**4. 重要的 Early Stopping**
永远不要手动指定 `n_estimators` 为一个固定的死数字（比如 1000）。
* **做法**：设置 `n_estimators=10000` (一个很大的数)，然后开启 `early_stopping_rounds=50` (或是 100)。
* **含义**：如果验证集 (Validation Set) 的分数连续 50 轮没有提升，就停止训练，并回滚到分数最高的那一轮。这是防止过拟合最简单有效的手段。

**5. 随机种子 (Random Seed)**
* 所有涉及随机性的参数（如 `seed`, `random_state`）都要固定下来，确保你的实验结果是**可复现 (Reproducible)** 的。
* **进阶技巧**：训练 5 个不同 Seed 的模型然后取平均（Seed Averaging），通常能稳稳提升分数。

# Cross Validation (交叉验证) 

CV 是你在比赛中唯一能相信的指标。Public LB (Leaderboard) 经常具有误导性（特别是当 Public/Private 切分不均匀时），**“相信你的 CV”** 是无数金牌选手的血泪经验。

**1. 常见的 CV 策略**
* **K-Fold (K折交叉验证)**：最通用。5-Fold 是标配，10-Fold 更准但更慢。
* **Stratified K-Fold (分层K折)**：**强烈推荐**。确保每一折中 Target 的分布（如正负样本比例）与总训练集一致。对于类别不平衡（Imbalanced）的数据集，这是必须的。
* **Group K-Fold (分组K折)**：**防止过拟合的神器**。如果数据集中包含同一用户的多条记录，必须用 UserID 进行 Group 分割，确保同一个用户的数据要么全在训练集，要么全在验证集，绝不能跨越。否则模型会“记住”这个用户的特征，导致严重的 Data Leakage。
* **Time Series Split (时间序列分割)**：如果比赛是预测未来（如股票、销量），千万不能用随机 K-Fold。必须按照时间顺序切分（Train: Jan-Mar, Val: Apr），否则会用到未来的信息预测过去（Lookahead Bias）。

**2. 这里的核心原则**
* **Consistency (一致性)**：如果你的 CV 分数涨了，但 LB 分数跌了，检查你的 CV 策略是否与测试集分布一致（Adversarial Validation 可派上用场）。如果确认 CV 设置无误，**请坚信 CV，忽略 LB 的波动**。

# Ensemble Generation (模型融合)

单模型决定下限，融合决定上限。在 Kaggle 想要拿牌，Ensemble 是必经之路。现在的融合早已不只是简单的投票，而是一套系统的工程。

**1. 融合的基石：Diversity (多样性)**
* 好的 Ensemble 需要模型之间“和而不同”。
* **高相关性模型**（如两个参数略有不同的 XGBoost）融合效果**差**。
* **低相关性模型**（如 XGBoost + 神经网络 + 线性回归）融合效果**极好**。这就是为什么即便 Neural Network 单模分数一般，也必须加入豪华午餐的原因。

**2. 进阶融合技术**
* **Weighted Blending (加权融合)**
    * 不要只做平均。根据 CV 分数给模型分配权重（如 `0.4*LGBM + 0.3*XGB + 0.3*CatBoost`）。
    * **技巧**：可以使用 `scipy.optimize` 自动寻找最佳权重组合以最大化 CV 分数。
    
* **Stacking (堆叠法)**
    * **原理**：把第一层模型（Base Models）的输出（OOF Predictions）作为特征，输入给第二层模型（Meta Model，通常是 Linear Regression 或简单的 MLP）。
    * **注意**：必须使用 Out-of-Fold (OOF) 预测值来训练第二层，严禁使用训练集上的预测值，否则会严重过拟合。
    
* **Power Averaging (幂平均)**
    * 适用于评估指标不仅关注顺序还关注数值的情况。公式：`(Pred1^p + Pred2^p + ...)/N)^(1/p)`。有时比算术平均更有效。

**3. 什么时候做 Ensemble？**
* **早期**：先跑通流程，用简单的加权平均验证 pipeline。
* **中期**：专注于提升单模型（Single Model）的分数和多样性。
* **最后一周**：停止单模型调优，全力做 Stacking 和 Blending，榨干最后 0.001 的提升。
  
# Pipeline (工程化与流水线)

理想的 Pipeline 应该像一条现代化的生产线：**配置驱动、自动记录、结果可复现**。与其每次手动改代码跑模型，不如搭建一个能让机器替你“打工”的系统。

**1. 实验追踪 (Experiment Tracking)：告别 Excel**
* **旧做法**：手动把 CV 分数、参数记在 Excel 或 txt 里。容易记错，且无法对比。
* **新标准**：使用 **Weights & Biases (WandB)**, **MLflow** 或 **Neptune.ai**。
    * **自动记录**：一行代码自动记录所有的超参数 (Hyperparams)、CV 分数、Loss 曲线甚至 Feature Importance 图表。
    * **对比神器**：可以在网页端直接对比 Experiment A 和 Experiment B 的差异，一眼看出哪个参数改动带来了提分。

**2. 配置管理 (Configuration Management)**
* **原则**：Code is Logic, Config is State. **代码和配置必须分离**。
* **工具**：使用 **Hydra** 或 **OmegaConf** (配合 YAML 文件)。
* **做法**：不要在 python 脚本里硬编码 `learning_rate = 0.01`。而是通过命令行参数 `python train.py model=xgboost lr=0.01` 来运行。这样你可以轻松写一个 Shell 脚本，一晚上跑 10 组不同的参数。

**3. 特征缓存 (Feature Caching) —— 效率倍增器**
* **痛点**：每次跑模型都要重新计算 `groupby` 或 `text embedding`，浪费大量时间。
* **解决方案**：检查特征文件是否存在。
    * 如果存在 `features/user_group_mean.parquet`，直接 Load。
    * 如果不存在，计算并 Save。
* 这能让你在后期调试模型时，将迭代周期从“小时级”缩短到“分钟级”。

**4. 统一的 OOF 管理 (Standardized OOF Saving)**
* 为了后续的 Stacking，Pipeline 必须自动保存每一折的 **Out-of-Fold (OOF)** 预测结果。
* **格式规范**：保存为 `oof_{model_name}_{cv_score}.csv`。确保索引 (Index) 与训练集完全一致，这样在做 Ensemble 时，只需要读取这些 CSV 文件，放入逻辑回归里即可自动训练。

**5. 现代 Pipeline 架构参考**
一个成熟的 Kaggle 代码库通常长这样：
```text
├── configs/          # 存放 yaml 配置文件
├── data/             # 原始数据
├── features/         # 缓存生成的特征 (parquet)
├── models/           # 保存训练好的模型文件
├── src/
│   ├── feature_eng/  # 特征工程逻辑 (只定义怎么算，不存储值)
│   ├── training/     # 训练循环 (Train Loop, CV Split)
│   └── utils/        # 工具函数 (Seeding, Metrics)
├── train.py          # 入口脚本
└── inference.py      # 提交用的预测脚本
参考https://www.dataapplab.com/how-to-be-top-10-at-kaggle-competition/
