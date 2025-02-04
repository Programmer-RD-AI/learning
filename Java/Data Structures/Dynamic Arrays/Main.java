import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        // ArrayList<String> arrayList = new ArrayList<>();
        DynamicArray dynamicArray = new DynamicArray();
        System.out.println(dynamicArray);
        dynamicArray.add("A");
        dynamicArray.add("B");
        dynamicArray.add("C");
        System.out.println("empty: " + dynamicArray.isEmpty());
        return;
    }
}
