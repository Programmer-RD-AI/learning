#include <stdio.h>
struct Student
{
    char name[50];
    char major[50];
    int age;
    double gpa;
};
int main()
{
    struct Student student1;
    student1.age = 22;
    student1.gpa = 3.2;

    return 0;
}
