class RotationGate implements Gate {
    final String type;
    final int target;
    RotationGate(int target, String type) {
        this.type = type;
        this.target = target;
    }
    public int minQubit() {
        return target;
    }
    public int maxQubit() {
        return target;
    }
    
    public <T> T invoke(Operation<T> o) {
        return o.processRotationGate(this);
    }
}
