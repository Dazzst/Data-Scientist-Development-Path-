# Python 编程语言

**Python** 是一种高级的、解释型的、通用的编程语言，由 Guido van Rossum 于 1991 年创建。它以其优雅、简洁的语法和强大的功能而闻名，是全球最受欢迎的编程语言之一。

**Python** is an interpreted, object-oriented, high-level programming language with dynamic semantics. Its high-level built in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together. Python's simple, easy to learn syntax emphasizes readability and therefore reduces the cost of program maintenance. Python supports modules and packages, which encourages program modularity and code reuse. The Python interpreter and the extensive standard library are available in source or binary form without charge for all major platforms, and can be freely distributed.

Often, programmers fall in love with Python because of the increased productivity it provides. Since there is no compilation step, the edit-test-debug cycle is incredibly fast. Debugging Python programs is easy: a bug or bad input will never cause a segmentation fault. Instead, when the interpreter discovers an error, it raises an exception. When the program doesn't catch the exception, the interpreter prints a stack trace. A source level debugger allows inspection of local and global variables, evaluation of arbitrary expressions, setting breakpoints, stepping through the code a line at a time, and so on. The debugger is written in Python itself, testifying to Python's introspective power. On the other hand, often the quickest way to debug a program is to add a few print statements to the source: the fast edit-test-debug cycle makes this simple approach very effective.

<img width="2774" height="6137" alt="NotebookLM Mind Map" src="https://github.com/user-attachments/assets/8b01a510-25fc-4d3e-a820-b2c7992b006b" />

[Official introduction](https://en.wikipedia.org/wiki/Python_(programming_language))

## 核心特性

* **简洁易读（可读性强）：** Python 强调代码的可读性和简洁性，其语法设计允许程序员用更少的代码行表达概念，使其非常适合初学者。
* **动态类型和自动内存管理：** 变量不需要事先声明类型，且 Python 自动处理内存的分配与回收。
* **解释型语言：** 代码在运行时逐行解释执行，无需编译，加快了开发周期。
* **面向对象：** Python 完全支持面向对象编程（OOP），允许创建类和对象。
* **庞大的标准库：** 提供了大量的内置模块和功能，涵盖了网络、文件操作、数学运算等。

## 主要应用领域

Python 的多功能性使其在多个行业中得到广泛应用：

1. **数据科学与机器学习 (Data Science & ML)：**
   * **NumPy** 用于科学计算和数组处理。
   * **Pandas** 用于数据分析和操作。
   * **Scikit-learn** 用于机器学习算法。
   * **TensorFlow/PyTorch** 用于深度学习。

2. **Web 开发 (Web Development)：**
   * **Django：** 功能齐全、用于快速开发的重量级框架。
   * **Flask：** 轻量级微框架，适用于小型项目和API开发。

3. **自动化、脚本与系统管理：**
   * 广泛用于编写自动化脚本（如爬虫），处理重复性任务，以及进行系统配置和管理。

4. **教育与科研：**
   * 因其易学性，被许多大学选为编程入门教学语言。

## 1. 数据分析与 Python 概述

• 什么是数据分析：数据分析是将海量数据转化为有用信息的过程（例如，将沃尔玛的一整年交易记录转化为"Pop-Tarts在周二卖得更好"这样的商业洞察）。其流程包括：收集数据、清洗、转换、分析建模，最后输出报告以支持决策。

• 为什么选择 Python：相比于 Excel 或 Tableau 这样易学但受限的"封闭式工具"，Python 作为一门开源编程语言，提供了极大的灵活性（可以连接任何 API 或数据库）和处理海量数据的能力。

• 数据分析师 vs 数据科学家：数据科学家通常具备更强的编程和数学基础，侧重于机器学习；而数据分析师则侧重于沟通和"讲故事"的能力，通过报告支持商业决策。

## 2. 开发环境：Jupyter Notebook

Jupyter Notebook 是数据分析师 99% 的时间都在使用的实时交互式工具。

• 无视觉参考的工作方式：不同于 Excel 可以随时看到所有数据，在 Python 中处理几百万行数据时，我们不会一直盯着数据看。我们只需在脑海中对数据的形态（Shape）和统计属性保持认知，这种方式极大提升了处理速度。
