#include <stdio.h>
int binary_search(int arr[], int val, int l_idx, int r_idx);
int main()
{
    int unsorted[] = {9, 5, 13, 3, 8, 7, 2, 12, 6, 10, 4, 11, 1};
    int sorted[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    int index = binary_search(sorted, 12, 0, 12);
    printf("%d", index);
    return 0;
}
int binary_search(int arr[], int val, int l_idx, int r_idx)
{
    int mid = l_idx + (r_idx - l_idx) / 2;
    if (l_idx > r_idx)
    {
        return -1;
    }
    if (arr[mid] == val)
    {
        return mid;
    }
    else if (arr[mid] > val)
    {
        return binary_search(arr, val, l_idx, mid - 1);
    }
    else
    {
        return binary_search(arr, val, mid + 1, r_idx);
    }
    // The Number is 2
    // 1,2,3,4,5
    // 1,2 (0 + (1-0)/2 = 0.5 (1))
    // 2 ?
    // 2!
    // ----------------------------------------------------------------
