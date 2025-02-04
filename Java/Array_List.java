public class Array_List{
    public static void main(String[] args){
        java.util.ArrayList<String> food = new java.util.ArrayList<String>();
        food.add("pizza");
        food.add("hamburger");
        food.add("hotdog");
        food.set(0, "sushi");
        food.remove(2);
        food.clear();
        for (int i = 0; i < food.size(); i++){
            System.out.println(food.get(i));
        }
//        when using a int or smthn you gotta use the Integer wrapper class
        return;
    }
}