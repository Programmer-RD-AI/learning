CREATE TABLE tblEmployee
(EmployeeNumber int NOT NULL,
EmployeeFirstName varchar(50) NOT NULL,
EmployeeMiddleName varchar(50) NULL,
EmployeeLastName varchar(50) NOT NULL,
EmployeeGovernmentID char(10) NULL,
DateOfBirth date NOT NULL,
Department varchar(50) NULL,
Manager int NULL);

INSERT INTO [dbo].[tblEmployee]
([EmployeeNumber],[EmployeeFirstName],[EmployeeMiddleName],[EmployeeLastName],[EmployeeGovernmentID]
,[DateOfBirth],[Department],[Manager])
VALUES (2, 'Jane', NULL, 'Zwilling', 'AB123456G ', '1994-12-30', 'Customer Relations', NULL),
(3, 'Carolyn', 'Andrea', 'Zimmerman', 'AB234578H ', '1985-05-29', 'Commercial', '2'),
(4, 'Jane', NULL, 'Zabokritski', 'LU778728T ', '1987-12-07', 'Commercial', '2'),
(5, 'Ken', 'J', 'Yukish', 'PO201903O ', '1979-12-25', 'HR', '2'),
(6, 'Terri', 'Lee', 'Yu', 'ZH206496W ', '1996-11-11', 'Customer Relations', '2'),
(7, 'Roberto', NULL, 'Young', 'EH793082D ', '1977-04-02', 'Customer Relations', '3'),
(8, 'Rob', NULL, 'Yalovsky', 'WF039886Z ', '1991-08-29', 'Litigation', '4'),
(9, 'Gail', 'A', 'Wu', 'SR883921U ', '1982-02-16', 'HR', '6'),
(10, 'Jossef', 'H', 'Wright', 'FU781952O ', '1990-07-30', 'Commercial', '5'),
(11, 'Dylan', 'A', 'Word', 'SU416128W ', '1999-11-26', 'Customer Relations', '5');

CREATE TABLE tblTransaction
(Amount smallmoney NOT NULL,
DateOfTransaction date NOT NULL,
EmployeeNumber int NOT NULL);

INSERT INTO [dbo].[tblTransaction] VALUES 
(858.16, '2024-08-07', 1),
(804.4, '2025-01-02', 1),
(-808.17, '2025-10-30', 1),
(957.03, '2024-05-20', 2),
(786.22, '2024-11-11', 2),
(-179.47, '2025-03-15', 2),
(-967.36, '2025-10-22', 2),
(-576.77, '2025-11-12', 3),
(-693.26, '2024-11-21', 4),
(390.52, '2024-11-29', 5),
(-500.73, '2025-09-15', 5),
(228.51, '2025-12-28', 5),
(-491.37, '2024-01-15', 6),
(-571, '2024-08-17', 6),
(817.11, '2025-07-16', 7),
(-369.69, '2025-04-06', 8),
(-573.18, '2025-05-06', 8),
(117.21, '2024-05-28', 10),
(981.18, '2025-04-18', 10),
(861.16, '2024-05-18', 11),
(-912.11, '2024-07-11', 11),
(-589.77, '2025-02-11', 11),
(-2.77, '2025-05-12', 11),
(-946.12, '2025-06-21', 11),
(-920.27, '2025-01-04', 12);

