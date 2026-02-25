# Python 编程语言

**Python** 是一种高级的、解释型的、通用的编程语言，由 Guido van Rossum 于 1991 年创建。它以其优雅、简洁的语法和强大的功能而闻名，是全球最受欢迎的编程语言之一。

**Python** is an interpreted, object-oriented, high-level programming language with dynamic semantics. Its high-level built in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together. Python's simple, easy to learn syntax emphasizes readability and therefore reduces the cost of program maintenance. Python supports modules and packages, which encourages program modularity and code reuse. The Python interpreter and the extensive standard library are available in source or binary form without charge for all major platforms, and can be freely distributed.

Often, programmers fall in love with Python because of the increased productivity it provides. Since there is no compilation step, the edit-test-debug cycle is incredibly fast. Debugging Python programs is easy: a bug or bad input will never cause a segmentation fault. Instead, when the interpreter discovers an error, it raises an exception. When the program doesn't catch the exception, the interpreter prints a stack trace. A source level debugger allows inspection of local and global variables, evaluation of arbitrary expressions, setting breakpoints, stepping through the code a line at a time, and so on. The debugger is written in Python itself, testifying to Python's introspective power. On the other hand, often the quickest way to debug a program is to add a few print
statements to the source: the fast edit-test-debug cycle makes this simple approach very effective.

<img width="2774" height="6137" alt="NotebookLM Mind Map" src="https://github.com/user-attachments/assets/8b01a510-25fc-4d3e-a820-b2c7992b006b" />
[Official introduction](https://en.wikipedia.org/wiki/Python_(programming_language))

## 核心特性

* **简洁易读（可读性强）：** Python 强调代码的可读性和简洁性，其语法设计允许程序员用更少的代码行表达概念，使其非常适合初学者。
* **动态类型和自动内存管理：** 变量不需要事先声明类型，且 Python 自动处理内存的分配与回收。
* **解释型语言：** 代码在运行时逐行解释执行，无需编译，加快了开发周期。
* **面向对象：** Python 完全支持面向对象编程（OOP），允许创建类和对象。
* **庞大的标准库：** 提供了大量的内置模块和功能，涵盖了网络、文件操作、数学运算等。

主要应用领域

Python 的多功能性使其在多个行业中得到广泛应用：

1.  **数据科学与机器学习 (Data Science & ML)：**
    * **NumPy** 用于科学计算和数组处理。
    * **Pandas** 用于数据分析和操作。
    * **Scikit-learn** 用于机器学习算法。
    * **TensorFlow/PyTorch** 用于深度学习。
2.  **Web 开发 (Web Development)：**
    * **Django：** 功能齐全、用于快速开发的重量级框架。
    * **Flask：** 轻量级微框架，适用于小型项目和API开发。
3.  **自动化、脚本与系统管理：**
    * 广泛用于编写自动化脚本（如爬虫），处理重复性任务，以及进行系统配置和管理。
4.  **教育与科研：**
    * 因其易学性，被许多大学选为编程入门教学语言。

## 1. 数据分析与 Python 概述
• 什么是数据分析：数据分析是将海量数据转化为有用信息的过程（例如，将沃尔玛的一整年交易记录转化为“Pop-Tarts在周二卖得更好”这样的商业洞察）。其流程包括：收集数据、清洗、转换、分析建模，最后输出报告以支持决策。
• 为什么选择 Python：相比于 Excel 或 Tableau 这样易学但受限的“封闭式工具”，Python 作为一门开源编程语言，提供了极大的灵活性（可以连接任何 API 或数据库）和处理海量数据的能力。
• 数据分析师 vs 数据科学家：数据科学家通常具备更强的编程和数学基础，侧重于机器学习；而数据分析师则侧重于沟通和“讲故事”的能力，通过报告支持商业决策。

## 2. 开发环境：Jupyter Notebook

Jupyter Notebook 是数据分析师 99% 的时间都在使用的实时交互式工具。
• 无视觉参考的工作方式：不同于 Excel 可以随时看到所有数据，在 Python 中处理几百万行数据时，我们不会一直盯着数据看。我们只需在脑海中对数据的形态（Shape）和统计属性保持认知，这种方式极大提升了处理速度。
• 单元格 (Cells)：笔记本由单元格组成，主要分为 代码 (Code) 和 Markdown (格式化文本) 两种类型。这让你可以将分析代码和精美的文本报告结合在一起导出为 PDF 或 HTML。
• 双模式与快捷键：
    ◦ 编辑模式 (Edit Mode)：用于在单元格内打字输入，按 Enter 进入，按 Esc 退出。
    ◦ 命令模式 (Command Mode)：用于操作单元格，按 A 在上方插入，按 B 在下方插入，双击 D 删除单元格，按 M 切换为 Markdown，按 Y 切换回代码。
    ◦ 运行代码：按 Shift + Enter 运行当前单元格并跳转至下一格。
    
## 3. NumPy：高性能科学计算底层

NumPy 是处理数据的核心基础，Pandas 和 Matplotlib 都建立在它之上。
• 为什么需要 NumPy：原生 Python 是一种高级语言，为了方便，它将数字封装成了复杂的对象。Python 中一个简单的整数 1 甚至会占用 28 个字节的内存。NumPy 允许你创建精确占用内存（如 int8 仅占 1 字节，或 int64、float64）的连续内存数组，极大节省空间并提升速度。
• 向量化与广播 (Vectorized Operations & Broadcasting)：这是 NumPy 的精髓。当你想让数组每个元素都加 10 时，不需要写 for 循环。直接写 array + 10，底层会自动将操作“广播”到每个元素，速度极快。
• 布尔数组 (Boolean Arrays)：通过条件判断（如 array >= 2），NumPy 会返回一个由 True 和 False 组成的同维度数组。你可以直接用这个布尔数组来极速过滤所需的数据。
• 不可变性：NumPy 操作默认是不可变的（Immutable），运算后会返回一个新的数组，而不会改变原数组。

## 4. Pandas：数据处理的核心利器

Pandas 提供了类似 Excel 的表格操作体验，是日常分析的主力。
• 两大核心数据结构：
    ◦ Series：类似于 Python 列表或字典，是一维的有序数据结构，但它拥有显式的标签索引 (Index)，底层是由 NumPy 数组驱动的。
    ◦ DataFrame：类似于 Excel 表格的二维数据结构，由多列 Series 共享同一个索引构成。
• 快速了解数据：加载数据后，通常第一时间使用 .info() 查看列的数据类型和空值情况，使用 .shape 查看行/列数，使用 .describe() 查看均值、最大/小值等快速统计信息。
• 数据的选择与提取：
    ◦ df['列名']：直接提取一整列。
    ◦ df.loc[]：基于名称/标签提取特定的行或列。
    ◦ df.iloc[]：基于**数字顺序（物理位置）**提取 特定的行或列。
• 创建与修改列：可以直接通过如 df['GDP per capita'] = df['GDP'] / df['Population'] 这样的广播操作，瞬间完成新列的计算。绝大多数 Pandas 的过滤和删除操作（如 drop）也都是不可变的，除非重新赋值。

## 5. 数据清洗 (Data Cleaning)

现实中的数据往往充满脏乱，清洗数据通常分为几个层次：
• 处理缺失值 (Missing Data)：
    ◦ 识别：使用 pd.isna() 或 isnull() 配合 .sum() 可以统计出缺失值的数量。
    ◦ 丢弃：使用 dropna() 删除含有空值的行，也可以设置 thresh 阈值保留包含一定数量有效值的行。
    ◦ 填充：使用 fillna() 填入特定值（如 0 或平均值），或使用前向填充 (ffill)、后向填充 (bfill) 顺延上方或下方的数据。
• 处理重复值与无效值：
    ◦ 识别与删除重复：使用 duplicated() 识别重复行，并用 drop_duplicates() 删除它们，可以设置是以第一次出现还是最后一次出现为准保留数据。
    ◦ 替换无效值：使用 .replace() 将错误的值（如性别列中的“D”或“？”）替换为正确的值。
• 高级字符串处理：对于字符串列，Pandas 提供了一个特殊的 .str 属性，可以直接调用类似于纯 Python 的 .str.split()（切割字符串）、.str.contains()（包含特定字符）等方法批量处理文本列。
6. 数据的导入与导出 (Import/Export)
Pandas 支持几乎所有主流数据格式，方法名均以 read_ 和 to_ 开头：
• CSV与文本文件：read_csv()。该方法非常强大，你可以自定义列名 (names)、分隔符 (sep，应对非逗号分隔的文件)、处理日期列 (parse_dates) 以及定义哪些字符应被视为空值 (na_values)。它甚至可以直接读取网络 URL 上的 CSV 文件。
• 数据库 (SQL)：通过 read_sql() 配合数据库连接对象，可以直接将 SQL 查询的结果转化为 DataFrame。如果只需一整张表，也可以使用 read_sql_table()。写回数据库则使用 to_sql()，可设置是替换还是追加数据。
• HTML 表格：read_html() 可以自动抓取网页（如维基百科）上的所有 <table> 标签并转化为 DataFrame 列表。不过，由于网页表格是给人看而非给机器看的（往往存在合并单元格），抓取后通常需要繁琐的清洗工作。
• Excel：read_excel() 允许你指定读取具体的 Sheet (sheet_name)，也可以跳过特定的表头行。

## 7. 数据可视化：Matplotlib 基础

虽然可以直接通过 df.plot() 快速生成图表（底层调用的就是 Matplotlib），但了解 Matplotlib 本身的机制非常重要。 Matplotlib 有两种 API，强烈推荐使用第二种：
• 全局 API (Global API)：受 MATLAB 启发，使用 plt.plot() 等直接绘图。这种方式代码可读性差，当有一张图包含多个子图时，你很难分清当前的代码到底在修改哪个子图。
• 面向对象 API (OOP API)：通过 fig, axes = plt.subplots() 先创建一个画布（Figure）和几个坐标系（Axes）。之后只需对特定的坐标系发出指令，如 axes.plot()，代码结构清晰明确。 支持常见的图表类型：折线图、散点图 (scatter，可同时展示 X, Y, 大小, 颜色四维信息)、直方图 (hist) 和箱线图（用于查看异常值）。

## 8. (补充模块) Python 速成指南

视频最后为零基础学员提供了一个十分钟的 Python 语法回顾：
• Python 是一门解释型、动态类型且面向对象的高级语言。
• 放弃了其他语言的大括号 {}，严格使用缩进 (Indentation) 来定义代码块，这让代码极其易读。
• 包含四种核心集合数据结构：
    ◦ 列表 (List)：可包含不同数据类型的有序组合。
    ◦ 元组 (Tuple)：类似列表，但是不可变的 (Immutable)。
    ◦ 字典 (Dictionary)：键值对映射，类似其他语言的哈希表，方便按名称查找属性。
    ◦ 集合 (Set)：包含唯一值的无序结构，其查找某个元素是否在集合中的性能极高 (O(1) 复杂度)。
• 推荐使用 for 循环遍历集合，而不是使用 while 循环以避免死循环。

