class UniformControlGate implements Gate {
    final int target;
    final int[] controls;
    final String type;
    UniformControlGate(int[] controls, int target, String type) {
        this.target = target;
        this.controls = new int[controls.length];
        System.arraycopy(controls, 0, this.controls, 0, controls.length);
        this.type = type;
    }
    public int minQubit() {
        return Gate.findMin(controls, target);
    }
    public int maxQubit() {
        return Gate.findMax(controls, target);
    }
    
    public <T> T invoke(Operation<T> o) {
        return o.processUniformControlGate(this);
    }
}
