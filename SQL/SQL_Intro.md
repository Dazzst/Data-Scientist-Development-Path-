# SQL（结构化查询语言）

**SQL**（Structured Query Language）是用于管理关系数据库系统（RDBMS）的标准语言。SQL 可以应用到所有关系型数据库中，例如 MySQL、Oracle、SQL Server 等。

<div align="center">
<img width="599" height="649" alt="v2-6186746a271bbf404eeaefba729b09a8_1440w" src="https://github.com/user-attachments/assets/a969fd72-4094-4190-a964-bb1731d3a970"/>
</div>

## 一、SQL 语言分类

SQL 语句根据其功能分为以下四大类（D-系列）：

| 分类 | 英文全称 | 中文描述 | 核心命令 | 用途 |
| :--- | :--- | :--- | :--- | :--- |
| **DDL** | Data Definition Language | 数据定义语言 | `CREATE`, `ALTER`, `DROP` | 定义数据库对象（库、表、列等结构）。 |
| **DML** | Data Manipulation Language | 数据操作语言 | `INSERT`, `UPDATE`, `DELETE` | 处理数据库表中的记录（数据）。 |
| **DCL** | Data Control Language | 数据控制语言 | `GRANT`, `REVOKE` | 定义访问权限和安全级别。 |
| **DQL** | Data Query Language | 数据查询语言 | `SELECT` | 查询记录（数据）。 |


## 二、DDL（数据定义语言）：定义数据库结构

### 1. 数据库操作
| 操作 | 描述 | 示例 |
| :--- | :--- | :--- |
| **创建** | 创建数据库 | `CREATE DATABASE [IF NOT EXISTS] mydb1;` |
| **删除** | 删除数据库 | `DROP DATABASE [IF EXISTS] mydb1;` |
| **修改** | 修改数据库编码 | `ALTER DATABASE mydb1 CHARACTER SET utf8;` |
| **切换** | 切换当前使用的数据库 | `USE mydb1;` |

### 2. 常用数据类型
| 类型 | 描述 | 示例 |
| :--- | :--- | :--- |
| `INT` | 整型 | `age INT` |
| `DECIMAL` | 浮点型/定点型 (金融常用) | `sal DECIMAL(7, 2)` |
| `CHAR` | 固定长度字符串 | `sid CHAR(6)` |
| `VARCHAR` | 可变长度字符串 | `sname VARCHAR(50)` |
| `DATE` | 日期类型 | `hiredate DATE` |
| `TIME` | 时间类型 | `time TIME` |
| `TIMESTAMP` | 时间戳类型 | |

### 3. 表结构操作
* **创建表：**
    ```sql
    CREATE TABLE stu (
        sid CHAR(6),
        sname VARCHAR(20),
        age INT,
        gender VARCHAR(10)
    );
    ```
* **修改表：**
    * **添加列：** `ALTER TABLE stu ADD (classname VARCHAR(100));`
    * **修改列类型：** `ALTER TABLE stu MODIFY gender CHAR(2);`
    * **修改列名：** `ALTER TABLE stu CHANGE gender sex CHAR(2);`
    * **删除列：** `ALTER TABLE stu DROP classname;`
    * **修改表名：** `ALTER TABLE stu RENAME TO student;`
* **查看表结构：** `DESC table_name;`
* **删除表：** `DROP TABLE table_name;`

## 三、DML（数据操作语言）：操作表中的数据

1.  **插入数据 (`INSERT`)：**
    ```sql
    -- 指定列名插入
    INSERT INTO stu (sid, sname) VALUES ('s_1001', 'zhangSan');
    -- 插入所有列（按创建时的顺序）
    INSERT INTO stu VALUES ('s_1002', 'liSi', 32, 'female');
    ```
2.  **修改数据 (`UPDATE`)：**
    ```sql
    UPDATE table_name SET col1 = val1, col2 = val2 [WHERE condition];
    -- 示例：
    UPDATE stu SET sname = 'zhangSanSan', age = 32 WHERE sid = 's_1001';
    ```
3.  **删除数据 (`DELETE` / `TRUNCATE`)：**
    * **`DELETE`：** 逐行删除，可带条件，属于 DML，支持事务回滚。
        `DELETE FROM stu WHERE sid = 's_1001';`
    * **`TRUNCATE`：** 先删除表再创建新表，效率更高，属于 DDL，**不可回滚**。
        `TRUNCATE TABLE stu;`

## 四、DQL（数据查询语言）：查询数据

DQL 是 SQL 的核心，用于从数据库中检索数据，命令为 `SELECT`。

### 1. 基础查询
* **查询所有列：** `SELECT * FROM table_name;`
* **查询指定列：** `SELECT col1, col2 FROM table_name;`
* **去重查询：** 使用 `DISTINCT` 关键字。
    `SELECT DISTINCT deptno FROM emp;`
* **使用别名：** 给列或表起别名 (`AS` 可省略)。
    `SELECT sal + IFNULL(comm, 0) AS total FROM emp;`

### 2. 条件过滤 (`WHERE` 子句)
使用运算符和关键字来筛选行记录：
* **比较运算：** `=`, `!=` 或 `<>`, `>`, `<`, `>=`, `<=`
* **范围查询：** `BETWEEN value1 AND value2`
* **集合查询：** `IN (val1, val2, ...)` 或 `NOT IN`
* **空值判断：** `IS NULL` 或 `IS NOT NULL`
* **逻辑运算：** `AND`, `OR`, `NOT`

### 3. 模糊查询 (`LIKE`)
用于匹配部分字符串。`%` 表示任意 0 个或多个字符；`_` 表示任意单个字符。
* **以 'z' 开头：** `SELECT * FROM stu WHERE sname LIKE 'z%';`
* **包含 'a' 字母：** `SELECT * FROM stu WHERE sname LIKE '%a%';`

### 4. 排序 (`ORDER BY`)
用于对结果集进行排序。
* **升序：** `ORDER BY col ASC;` (ASC 可省略)
* **降序：** `ORDER BY col DESC;`
* **多列排序：** `ORDER BY col1 DESC, col2 ASC;` (先按 col1 降序，再按 col2 升序)

### 5. 聚合函数
用于对一组值进行计算，返回单个值。
* `COUNT()`：统计行数（不统计 `NULL` 值的行）。
* `SUM()`：计算总和。
* `AVG()`：计算平均值。
* `MAX()`：计算最大值。
* `MIN()`：计算最小值。

### 6. 分组查询 (`GROUP BY` 与 `HAVING`)
* **`GROUP BY`：** 按指定列对结果进行分组。
    `SELECT deptno, SUM(sal) FROM emp GROUP BY deptno;`
* **`HAVING`：** 对 **分组后** 的结果进行过滤（`WHERE` 是对分组前的原始数据进行过滤）。
    `SELECT deptno, SUM(sal) FROM emp GROUP BY deptno HAVING SUM(sal) > 9000;`

### 7. 多表连接查询 (`JOIN`)
用于组合来自两个或多个表的行。

| 连接类型 | 描述 | 示例关键词 |
| :--- | :--- | :--- |
| **内连接** | 只返回两个表中匹配的行。 | `INNER JOIN` 或 `table1, table2 WHERE condition` |
| **左连接** | 返回左表所有行，以及右表中匹配的行。 | `LEFT JOIN` |
| **右连接** | 返回右表所有行，以及左表中匹配的行。 | `RIGHT JOIN` |

### 8. 结果限定 (`LIMIT`)
（主要为 MySQL 方言）用于限定查询结果的起始行和总行数，常用于分页。
* **语法：** `LIMIT 起始行, 总行数` (起始行从 0 开始)
* **示例：** `SELECT * FROM emp LIMIT 0, 10;` (查询前 10 行)

---
**提示：** 在使用此内容时，请注意像 `IFNULL` 和 `LIMIT` 这样的函数和子句在不同的数据库系统（如 MySQL, Oracle, SQL Server 等）中可能存在细微的语法差异或不同的替代方案。



