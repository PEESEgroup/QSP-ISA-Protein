public class BigBlock implements Gate {
    final int bot, top;
    final String type;
    
    public BigBlock(int bot, int top, String type) {
        this.bot = bot;
        this.top = top;
        this.type = type;
    }
    public <T> T invoke(Operation<T> o) {
        return o.processBigBlock(this);
    }
    public int maxQubit() {
        return top;
    }
    public int minQubit() {
        return bot;
    }
}