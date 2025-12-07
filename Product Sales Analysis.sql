CREATE TABLE Products(productID SERIAL PRIMARY KEY NOT NULL,productName VARCHAR(50),price NUMERIC(5,2));
INSERT INTO products(productid,productname,price) VALUES(1,'Apple',2.50),(2,'Banana',1.50),(3,'Orange',3.00),(4,'Mango',2.00)
SELECT * FROM products;
CREATE TABLE Orders(orderID SERIAL PRIMARY KEY NOT NULL,productid INT,quantity INT,sales NUMERIC(10,2));
INSERT INTO orders(orderid,productid,quantity,sales) VALUES(1,1,10,25.00),(2,1,5,12.50),(3,2,8,12.00),(4,3,12,36.00),(5,4,6,12.00);
SELECT * FROM orders;
DROP TABLE orders;
SELECT p.productid,p.productname, sum(o.sales) AS total_sales FROM products p JOIN orders o ON p.productid=o.productid GROUP BY p.productid,p.productname ORDER BY total_sales DESC;