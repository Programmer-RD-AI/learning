#include <stdio.h>

int main()
{
    int age;
    printf("Enter your age : ");
    scanf("%d", &age); // & because its pointing to the variables address I think... :)
    printf("%d", age);
    return 0;
}
