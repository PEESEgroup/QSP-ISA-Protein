import java.util.*;

class QuantumCircuit {
    final int numQubits;
    final ArrayList<ArrayList<Gate>> gates;
    
    public QuantumCircuit(int n) {
        this.numQubits = n;
        this.gates = new ArrayList<ArrayList<Gate>>();
        this.newBlock();
    }
    public void cx(int control, int target) {
        this.cx(new int[] {control}, target);
    }
    public void cx(int[] control, int target) {
        this.appendGate(new ControlXGate(control, target));
    }
    public void rotation(int target, String type) {
        this.appendGate(new RotationGate(target, type));
    }
    public void uniformControl(int control, int target, String type) {
        this.uniformControl(new int[] {control}, target, type);
    }
    public void uniformControl(int[] control, int target, String type) {
        this.appendGate(new UniformControlGate(control, target, type));
    }
    public void controlRotation(int control, int target, String type) {
        this.controlRotation(new int[] {control}, target, type);
    }
    public void controlRotation(int[] control, int target, String type) {
        this.appendGate(new ControlGate(control, target, type));
    }
    public void bigBlock(int bot, int top, String type) {
        this.appendGate(new BigBlock(bot, top, type));
    }
    public GateIterator iterator() {
        return new GateIteratorImpl(gatesDeepCopy());
    }
    public void newBlock() {
        this.gates.add(new ArrayList<>());
    }
    private void appendGate(Gate g) {
        int index = this.gates.size() - 1;
        /*
        while(index > -1) {
            ArrayList<Gate> currentBlock = gates.get(index);
            boolean overlap = false;
            for(Gate gg : currentBlock) {
                if(g.overlaps(gg)) overlap = true;
            }
            if(overlap) break;
            index--;
        }
        */
        if(index == this.gates.size() - 1) newBlock();
        index++;
        this.gates.get(index).add(g);
    }
    public ArrayList<ArrayList<Gate>> gatesDeepCopy() {
        ArrayList<ArrayList<Gate>> output = new ArrayList<>();
        for(ArrayList<Gate> a : this.gates) {
            ArrayList<Gate> aa = new ArrayList<>();
            for(Gate g : a) aa.add(g);
            output.add(aa);
        }
        return output;
    }
    private class GateIteratorImpl implements GateIterator {
        private int index;
        private ArrayList<ArrayList<Gate>> gates;
        GateIteratorImpl(ArrayList<ArrayList<Gate>> gates) {
            this.gates = gates;
            this.index = 0;
        }
        public boolean hasNext() {
            return this.index < gates.size();
        }
        public Gate[] nextBlock() {
            if(!hasNext()) throw new RuntimeException();
            Gate[] output = gates.get(index).toArray(new Gate[]{});
            index++;
            return output;
        }
    }
}

