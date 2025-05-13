public class Main {
    public static void main(String[] args){
        // method chaining = a common syntax for invoking multiple method calls in OOP condense code into less lines

        String name = "  bro";
//        name = name.concat(" code  ");
//        name = name.toUpperCase();
//        name = name.trim();
        name = name.concat(" code  ").toUpperCase().trim();
        System.out.println(name);
    }
}
