#include <stdio.h>

int main()
{
    int age = 30;
    int *pAge = &age;
    printf("%d\n", *pAge);
    printf("%p", pAge);
    return 0;
}
