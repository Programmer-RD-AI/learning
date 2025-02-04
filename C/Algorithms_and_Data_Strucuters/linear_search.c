#include <stdio.h>
int LinearSearch(int Arr[], int size, int val);
int main()
{
    int Arr[4] = {2, 4, 6, 8};
    int index = LinearSearch(Arr, 4, 8);
    printf("%d", index);
    return 0;
}
int LinearSearch(int Arr[], int size, int val)
{
    int iter;
    for (iter = 0; iter < size; iter++)
    {
        if (Arr[iter] == val)
        {
            return iter;
        }
    }
    return 0;
}
