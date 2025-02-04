-- TRUNCATE TABLE [Employee Table]
-- CREATE TABLE [Employee Table] (
--     EmployeeNumber int NOT NULL,
--     EmployeeFirstName varchar(50) NOT NULL,
--     EmployeeLastName varchar(50) NOT NULL,
--     EmployeeMiddleName varchar(50) NULL,
--     EmployeeGovermentID char(10) NULL,
--     DateOfBirth Date NOT NULL,
--     Department varchar(50) NULL,
--     Manager int NULL
-- )
-- SELECT
--     *
-- FROM
--     [Employee Table]
-- INSERT INTO
--     [Employee Table](
--         [EmployeeNumber],
--         [EmployeeFirstName],
--         [EmployeeMiddleName],
--         [EmployeeLastName],
--         [EmployeeGovermentID],
--         [DateOfBirth],
--         [Department],
--         [Manager]
--     )
-- VALUES
--     (
--         2,
--         'Jane',
--         null,
--         'Zwilling',
--         'AB123456G',
--         '1994-12-30',
--         'Customer Relations',
--         NULL
--     )
-- CREATE TABLE [tbltransaction] (
--     Amount smallmoney NOT NULL,
--     DateOfTransaction Date NOT NULL,
--     EmployeeNumber smallint NOT NULL
-- )
-- INSERT INTO
--     [dbo].[tblTransaction]
-- VALUES
--     (858.16, '2024-08-07', 1),
--     (804.4, '2025-01-02', 1),
--     (-808.17, '2025-10-30', 1),
--     (957.03, '2024-05-20', 2),
--     (786.22, '2024-11-11', 2),
--     (-179.47, '2025-03-15', 2),
--     (-967.36, '2025-10-22', 2),
--     (-576.77, '2025-11-12', 3),
--     (-693.26, '2024-11-21', 4),
--     (390.52, '2024-11-29', 5),
--     (-500.73, '2025-09-15', 5),
--     (228.51, '2025-12-28', 5),
--     (-491.37, '2024-01-15', 6),
--     (-571, '2024-08-17', 6),
--     (817.11, '2025-07-16', 7),
--     (-369.69, '2025-04-06', 8),
--     (-573.18, '2025-05-06', 8),
--     (117.21, '2024-05-28', 10),
--     (981.18, '2025-04-18', 10),
--     (861.16, '2024-05-18', 11),
--     (-912.11, '2024-07-11', 11),
--     (-589.77, '2025-02-11', 11),
--     (-2.77, '2025-05-12', 11),
--     (-946.12, '2025-06-21', 11),
--     (-920.27, '2025-01-04', 12);
-- SELECT
--     *
-- FROM
--     [tblEmployee] AS E
--     LEFT JOIN [tbltransaction] AS T ON E.EmployeeNumber = T.EmployeeNumber
-- WHERE Amount is null
-- SELECT
--     *
-- FROM
--     tbltransaction AS T
--     JOIN tblEmployee AS E ON T.EmployeeNumber = E.EmployeeNumber
-- where T.EmployeeNumber IS NULL
-- SELECT
--     *
-- FROM
--     tblEmployee AS E1
--     LEFT JOIN tblEmployee as E2 ON E1.Manager = E2.EmployeeNumber
CREATE TABLE tblStampNames (
    [StampID] tinyint NOT NULL,
    [StampName] varchar(17) NOT NULL,
    [StampCountry] varchar(7) NOT NULL,
    [StampYear] smallint NOT NULL
);

INSERT INTO
    tblStampNames (
        [StampID],
        [StampName],
        [StampCountry],
        [StampYear]
    )
VALUES
    (1, 'Inverted Alison', 'Italy', 1917),
    (2, 'John Adams', 'USA', 1867),
    (3, 'Stars and Stripes', 'USA', 1882),
    (5, 'Queen Victoria', 'UK', 1937),
    (6, 'Rio de Janeiro', 'Brazil', 1898),
    (7, 'Kiev Standard', 'Ukraine', 1876),
    (8, 'William IV', 'UK', 1936),
    (9, 'Yellow Tree', 'Italy', 1865);

CREATE TABLE tblStampPurchases (
    [StampID] tinyint NOT NULL,
    [PurchaseDate] date NOT NULL,
    [PurchasePrice] int NOT NULL
);

INSERT INTO
    tblStampPurchases([StampID], [PurchaseDate], [PurchasePrice])
VALUES
    (2, '2001-02-09', 47500),
    (3, '1988-04-26', 46600),
    (4, '1994-01-04', 10700),
    (5, '1982-03-06', 15400),
    (6, '1980-11-02', 10800),
    (6, '1986-03-10', 10300);
