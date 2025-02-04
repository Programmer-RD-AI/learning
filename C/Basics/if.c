#include <stdio.h>

int max(int num1, int num2)
{
    // if (num1 < num2 || num1 <= num2) // Or ||
    // if (num1 < num2 && num1 <= num2) // And &&
    if (num1 < num2)
    {
        return num2;
    }
    else if (num1 == num2)
    {
        return num1 - num2;
    }
    else
    {
        return num1;
    }
    return 0;
}
int main()
{
    int max_out_of_40_and_50 = max(40, 50);
    printf("%d", max_out_of_40_and_50);
}
