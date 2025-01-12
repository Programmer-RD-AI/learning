DROP TABLE students;
CREATE TABLE students
(
    student_id INT IDENTITY(1,1) ,
    student_name VARCHAR(20) NOT NULL,
    student_major VARCHAR(20) UNIQUE DEFAULT 'No Major',
    CONSTRAINT student_id PRIMARY KEY(student_id),
);


SELECT *
FROM students;


INSERT INTO students
    (student_name,student_major)
VALUES
    ('Jack', 'IDK');
INSERT INTO students
    (student_name,student_major)
VALUES
    ('Kate', 'Sociolody');
INSERT INTO students
    (student_name,student_major)
VALUES
    ('Claire', 'hrthet');
INSERT INTO students
    ( student_name)
VALUES
    ('Hagerg');
INSERT INTO students
    (student_name,student_major)
VALUES
    ('Back', 'Biology');
INSERT INTO students
    (student_name,student_major)
VALUES
    ('Mike', 'Computer Science');
INSERT INTO students
    (student_major, student_name)
VALUES
    ('Mgergerike', 'Computer Science');
INSERT INTO students
    (student_name)
VALUES
    ('Ranuga');


-- UPDATE students SET student_major = 'bio' WHERE student_major = 'biology'
-- UPDATE students SET student_major = 'CS' WHERE student_major = 'Computer Science'
-- UPDATE students SET student_major = 'CS' WHERE student_id = 1;
-- UPDATE students SET student_major = 'biochemistry' WHERE student_major = 'bio' or student_major = 'Chemistry'
-- UPDATE students SET student_name = 'Hagerg', student_major = 'No Major' WHERE student_major = 1
-- UPDATE students SET student_major = 'test';


-- SELECT *
-- FROM students;

-- DELETE FROM students WHERE student_id = 5 AND student_major = 'CS';

-- SELECT *
-- FROM students;

-- DROP TABLE students;

-- ALTER TABLE students ADD gpa DECIMAL(3,2);

-- ALTER TABLE students DROP COLUMN gpa;

-- SELECT * FROM students;

-- SELECT *
-- FROM students;
-- SELECT *
-- FROM students
-- WHERE student_id = 5;

-- SELECT student_major
-- FROM students
-- WHERE student_id = 5;

-- SELECT students.student_major
-- FROM students
-- WHERE student_id = 5 OR student_id = 4 OR student_id = 3
-- ORDER BY student_id ASC;
-- SELECT students.student_major
-- FROM students
-- WHERE student_id = 5 OR student_id = 4 OR student_id = 3
-- ORDER BY student_id DESC;


-- SELECT students.student_major
-- FROM students
-- WHERE student_id = 5 OR student_id = 4 OR student_id = 3
-- ORDER BY student_id DESC;

-- SELECT *
-- FROM students
-- WHERE student_id = 5 OR student_id = 4 OR student_id = 3
-- ORDER BY student_major, student_id;

-- SELECT *
-- FROM students;

-- SELECT student_name
-- FROM students
-- WHERE student_major != 'biology';

SELECT * FROM students WHERE student_name IN ('Back','Mike','Jack') AND student_id > 2;
