class ControlGate implements Gate {
    final int[] controls;
    final int target;
    final String type;

    ControlGate(int[] controls, int target, String type) {
        this.controls = new int[controls.length];
        System.arraycopy(controls, 0, this.controls, 0, controls.length);
        this.target = target;
        this.type = type;
    }
    public int minQubit() {
        return Gate.findMin(controls, target);
    }
    public int maxQubit() {
        return Gate.findMax(controls, target);
    }
    
    public <T> T invoke(Operation<T> o) {
        return o.processControlGate(this);
    }
}
