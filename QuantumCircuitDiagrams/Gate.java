interface Gate {
    default boolean overlaps(Gate other) {
        if(this.minQubit() >= other.minQubit()) {
            return other.maxQubit() >= this.minQubit();
        }
        return this.maxQubit() >= other.minQubit();
    }
    int minQubit();
    int maxQubit();
    static int findMin(int[] c, int t) {
        int output = t;
        for(int i : c) {
            if(i < output) output = i;
        }
        return output;
    }
    static int findMax(int[] c, int t) {
        int output = t;
        for(int i : c) {
            if(i > output) output = i;
        }
        return output;
    }
    <T> T invoke(Operation<T> o);
}
