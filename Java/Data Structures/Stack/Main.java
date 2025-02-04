import java.util.Stack;

public class Main {
    public static void main(String[] args) {
        Stack<String> stack = new Stack();
        stack.push("Minecraft");
        stack.push("Skyrim");
        stack.push("DOOM");
        stack.push("Borderlands");
        stack.push("FFVII");
        String myFavGame = stack.pop();
        System.out.println(myFavGame);
        System.out.println(stack);
        System.out.println(stack.get(0));
        System.out.println(stack.peek());
        System.out.println(stack.search("Minecraft")); // if not found -1
        for (int i = 0; i < 1000000000; i++){
            stack.push("FFVII");
        }
        return;
    }
}
