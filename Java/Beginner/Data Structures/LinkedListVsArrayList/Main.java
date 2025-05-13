import java.util.ArrayList;
import java.util.LinkedList;

public class Main {
    public static void main(String args[]){
        LinkedList<Integer> linkedList = new LinkedList<>();
        ArrayList<Integer> arrayList = new ArrayList<>();

        long startTime;
        long endTime;
        long elapsedTime;

        for (int i = 0; i < 1000001; i++){
            linkedList.add(i);
            arrayList.add(i);
        }

        // LinkedList
        startTime = System.nanoTime();
//        linkedList.remove(50000);
//        linkedList.get(1000000);
        linkedList.remove(1000000);

        endTime = System.nanoTime();

        System.out.println(endTime-startTime);

        // ArrayList
        startTime = System.nanoTime();

//        arrayList.get(1000000);
        arrayList.remove(1000000);
        endTime = System.nanoTime();

        System.out.println(endTime-startTime);
    }
}
