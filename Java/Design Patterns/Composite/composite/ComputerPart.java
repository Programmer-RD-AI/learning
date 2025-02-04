package composite;

import java.util.ArrayList;
import java.util.List;

interface Component{
    void showPrice();
}

class Leaf implements Component{
    public int price;
    String name;

    Leaf(int price, String name){
        this.price = price;
        this.name = name;
    }

    @Override
    public void showPrice() {
        System.out.println(name + ": " +  price);
    }
}

class Composite implements Component{
    String name;
    List<Component> componentList = new ArrayList<>();

    Composite(String name){
        this.name = name;
    }

    public void addComponent(Component com){
       componentList.add(com);
    }

    @Override
    public void showPrice() {
        System.out.println(name);
        for (Component c: componentList){
            c.showPrice();
        }
        return;
    }
}