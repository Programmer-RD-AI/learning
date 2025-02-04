public class Main {
    public static void main(String[] args) throws InterruptedException {
        MyThread thread1 = new MyThread();
        MyRunnable runnable1 = new MyRunnable();
        Thread thread2 = new Thread(runnable1);
        thread1.start();
        thread1.join(); // wait till thread1 is finished then thread2 starts yk?
        thread2.start();
        // thread1.setDaemon(true); then if the main thread ides they will also stop yk?
        return;
    }
}
