import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        int array[] = new int[100];
        int target = 42;
        for (int i = 0; i < array.length; i++){
            array[i] = i;
        }

//        int idx = Arrays.binarySearch(array, target);
        int idx = binarySearch(array, target);
        System.out.println(idx);


        return;
    }

    private static int binarySearch(int[] array, int target){
        int low = 0;
        int high = array.length - 1;
        while (low <= high){
            int mid = high + (high - low) / 2;
            int value = array[mid];

            if (value < target) low = mid +1;
            else if (value > target) high = mid - 1;
            else return mid;
        }
        return -1;
    }
}
