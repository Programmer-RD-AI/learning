public class Main {
    public static void Main(String[] args){
        String name = "Bro";
        char symbol = '!';
        MyInterface myInterface = (x, y) -> System.out.println("test");
        myInterface.message(name, symbol);
        return;
    }
}
