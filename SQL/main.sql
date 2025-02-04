DROP TABLE students;
CREATE TABLE students
(
    student_id INT AUTO_INCREMENT,
    student_name VARCHAR(20) NOT NULL,
    student_major VARCHAR(20) UNIQUE DEFAULT 'No Major',
    PRIMARY KEY(student_id),
);


SELECT * FROM students;

INSERT INTO students VALUES ('Jack', NULL);
INSERT INTO students VALUES (2, 'Kate', 'Sociolody');
-- INSERT INTO students VALUES (3, 'Claire',NULL);
INSERT INTO students(student_id, student_name) VALUES (3, 'Hagerg');
INSERT INTO students VALUES (4, 'Back','Biology');
INSERT INTO students VALUES (5, 'Mike','Computer Science');

-- INSERT INTO students(student_id, student_name) VALUES (3, 'Ranuga');

-- DROP TABLE students;

-- ALTER TABLE students ADD gpa DECIMAL(3,2);

-- ALTER TABLE students DROP COLUMN gpa;

SELECT * FROM students;
