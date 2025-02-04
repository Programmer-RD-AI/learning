#include <stdio.h>

int main()
{
    int age = 30;
    int *pAge = &age;
    printf("%p", pAge);
    return 0;
}
