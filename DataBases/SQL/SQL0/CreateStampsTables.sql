CREATE TABLE tblStampNames (		
	[StampID] tinyint NOT NULL,	
	[StampName] varchar(17) NOT NULL,	
	[StampCountry] varchar(7) NOT NULL,	
	[StampYear] smallint NOT NULL);

INSERT INTO tblStampNames ( [StampID], [StampName], [StampCountry], [StampYear]) 
VALUES  (1, 'Inverted Alison', 'Italy', 1917),
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
	[PurchasePrice] int NOT NULL);

INSERT INTO tblStampPurchases( [StampID], [PurchaseDate], [PurchasePrice]) 
VALUES 	(2, '2001-02-09', 47500),
	(3, '1988-04-26', 46600),
	(4, '1994-01-04', 10700),
	(5, '1982-03-06', 15400),
	(6, '1980-11-02', 10800),
	(6, '1986-03-10', 10300);

SELECT * FROM tblStampPurchases;
