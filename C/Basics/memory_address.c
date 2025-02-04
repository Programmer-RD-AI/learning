#include <stdio.h>

int main()
{
    int age = 30;
    double gpa = 3.4;
    char grade = 'A';
    printf("%p\n", &age); // This is the place in memory that the variable age is stored in
    printf("%p\n", &gpa);
    printf("%p\n", &grade);
    return 0;
}
