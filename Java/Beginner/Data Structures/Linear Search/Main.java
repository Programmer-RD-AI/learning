public class Main {
    public static void main(String args[]){
        int[] array = {9,1,8,2,7,3,6,4,5};
        int idx = linearSearch(array, 1);
        System.out.println(idx);
    }

    private static int linearSearch(int[] array, int value){
        for (int i=0; i<array.length; i++){
            if (array[i] == value){
                return array[i];
            }
        }
        return -1;
    }
}
