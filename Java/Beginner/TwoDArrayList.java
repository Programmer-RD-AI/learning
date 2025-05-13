import java.util.ArrayList;

public class TwoDArrayList{
    public static void main(String[] args){
        ArrayList<ArrayList<String>> groceryList = new ArrayList();

        ArrayList<String> bakeryList = new ArrayList();
        bakeryList.add("pasta");
        bakeryList.add("garlic bread");
        bakeryList.add("donuts");
        System.out.println(bakeryList);

        ArrayList<String> produceList = new ArrayList();
        produceList.add("tomatos");
        produceList.add("zuuchini");
        produceList.add("peppers");
        System.out.println(bakeryList);

        groceryList.add(bakeryList);
        groceryList.add(produceList);

        System.out.println(groceryList);
        return;
    }
}