# SQL（结构化查询语言）

**SQL**（Structured Query Language，结构化查询语言）是用于管理关系数据库系统（RDBMS）的标准语言。SQL 不仅仅是一种编程语言，它更像是一种用于与数据库沟通的声明性语言。

## SQL 的重要性

* **数据管理核心：** SQL 是所有与关系型数据库（如 MySQL, PostgreSQL, SQL Server, Oracle）进行交互的基础。
* **数据科学家必备技能：** 数据科学家需要使用 SQL 从数据库中提取、筛选、聚合和转换数据，为后续的分析和建模做准备。

## SQL 的四大主要分类

SQL 命令根据其功能分为以下四大类（D-系列）：

1. DDL（Data Definition Language）：数据定义语言，用来定义数据库对象：库、表、列等； 
2. DML（Data Manipulation Language）：数据操作语言，用来定义数据库记录（数据）； 
3. DCL（Data Control Language）：数据控制语言，用来定义访问权限和安全级别； 
4. DQL（Data Query Language）：数据查询语言，用来查询记录（数据）

1.  **DQL (Data Query Language) - 数据查询语言：**
    * **用途：** 用于从数据库中检索数据。
    * **核心命令：** `SELECT` (最常用)。
    * **示例：** `SELECT * FROM table_name WHERE condition;`

2.  **DML (Data Manipulation Language) - 数据操作语言：**
    * **用途：** 用于处理数据库表中的记录（数据）。
    * **核心命令：** `INSERT` (插入数据)、`UPDATE` (修改数据)、`DELETE` (删除数据)。
    * **示例：** `INSERT INTO table_name (col1, col2) VALUES (val1, val2);`

3.  ****DDL (Data Definition Language) - 数据定义语言：****
    * **用途：** 用于定义数据库对象，包括库、表、视图、索引等结构。
    * **核心命令：** `CREATE` (创建)、`ALTER` (修改结构)、`DROP` (删除结构)。
    * **示例：** `CREATE TABLE table_name (column_name datatype);`

4.  **DCL (Data Control Language) - 数据控制语言：**
    * **用途：** 用于定义访问权限和安全级别。
    * **核心命令：** `GRANT` (授予权限)、`REVOKE` (撤销权限)。
    * **示例：** `GRANT SELECT ON table_name TO user;`

## 关键查询操作和概念

### 1. 过滤与排序
* **`WHERE`：** 用于基于指定条件过滤记录。
* **`ORDER BY`：** 对结果集进行排序（升序 `ASC` 或降序 `DESC`）。

### 2. 聚合与分组
* **聚合函数：** `COUNT()` (计数)、`SUM()` (求和)、`AVG()` (平均值)、`MAX()` (最大值)、`MIN()` (最小值)。
* **`GROUP BY`：** 将结果集按一个或多个列进行分组。
* **`HAVING`：** 用于在 `GROUP BY` 之后对分组结果进行过滤。

### 3. 表连接 (JOINs)
连接是 SQL 中最重要和常用的功能，用于将来自两个或多个表的行组合起来。
* **`INNER JOIN`：** 只返回两个表中匹配的行。
* **`LEFT JOIN` (或 `LEFT OUTER JOIN`)：** 返回左表中的所有行，以及右表中匹配的行。
* **`RIGHT JOIN` (或 `RIGHT OUTER JOIN`)：** 返回右表中的所有行，以及左表中匹配的行。
* **`FULL JOIN` (或 `FULL OUTER JOIN`)：** 返回当任一表中存在匹配时，所有行。

## 示例代码

**查询示例 (DQL):**
```sql
-- 1. 查询所有员工的姓名和薪水，并按薪水降序排列
SELECT employee_name, salary
FROM employees
ORDER BY salary DESC;

-- 2. 统计每个部门的员工数量
SELECT department, COUNT(employee_id) AS total_employees
FROM employees
GROUP BY department
HAVING COUNT(employee_id) > 5;
