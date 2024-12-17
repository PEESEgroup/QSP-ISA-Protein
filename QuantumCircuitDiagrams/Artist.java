import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import java.util.*;

class Artist implements Operation<Object> {
    public static final double WIRE_WIDTH = 55;
    public static final double CTRL_WIDTH = 10;
    public static final double XOR_WIDTH = 25;
    public static final double BIG_BOX_HEIGHT = 50;
    public static final double SMALL_BOX_WIDTH = 12;
    public static final double SMALL_BOX_HEIGHT = 12;
    public static final double PADDING_OUTER = 10;
    public static final double PADDING_INNER = 10;
    public static final String FONT = "Arial";
    public static final int FONT_SIZE = 24;
    
    public static void draw(QuantumCircuit circuit, GraphicsContext gc, 
        double startX, double startY) {
        Artist a = new Artist(gc, startX, startY, circuit.numQubits);
        GateIterator iter = circuit.iterator();
        while(iter.hasNext()) {
            Gate[] next = iter.nextBlock();
            a.nextBlock(Width.computeBlockWidth(next));
            for(Gate g : next) {
                g.invoke(a);
            }
        }
    }
    
    private double currentBlockWidth;
    private GraphicsContext gc;
    private double x, y;
    private int numQubits;
    
    private Artist(GraphicsContext gc, double startX, double startY, int n) {
        this.currentBlockWidth = 0;
        this.gc = gc;
        this.x = startX;
        this.y = startY;
        this.numQubits = n;
    }
    
    public void nextBlock(double width) {
        x += currentBlockWidth / 2;
        this.currentBlockWidth = width;
        for(int i = 0; i < numQubits; i++) {
            CanvasFunctions.horizontalLine(gc, x, y + i * WIRE_WIDTH, width);
        }
        x += width / 2;
    }
    
    public Object processControlXGate(ControlXGate g) {
        int top = g.minQubit();
        int bot = g.maxQubit();
        CanvasFunctions.verticalLine(gc, x, y + top * WIRE_WIDTH, 
            (bot - top) * WIRE_WIDTH);
        for(int i : g.controls) {
            CanvasFunctions.fillCircle(gc, x, y + i * WIRE_WIDTH, 
                CTRL_WIDTH / 2);
        }
        CanvasFunctions.openCircle(gc, x, y + g.target * WIRE_WIDTH, 
            XOR_WIDTH / 2);
        CanvasFunctions.horizontalLine(gc, x - XOR_WIDTH / 2, y + g.target * WIRE_WIDTH, 
            XOR_WIDTH);
        CanvasFunctions.verticalLine(gc, x, 
            y + g.target * WIRE_WIDTH - XOR_WIDTH / 2, XOR_WIDTH);
        return null;
    }
    
    public Object processControlGate(ControlGate g) {
        int top = g.minQubit();
        int bot = g.maxQubit();
        CanvasFunctions.verticalLine(gc, x, y + top * WIRE_WIDTH,
            (bot - top) * WIRE_WIDTH);
        for(int i : g.controls) {
            CanvasFunctions.fillCircle(gc, x, y + i * WIRE_WIDTH,
                CTRL_WIDTH / 2);
        }
        CanvasFunctions.openRect(gc, x, y + g.target * WIRE_WIDTH,
            Width.computeWidth(g), Artist.BIG_BOX_HEIGHT);
        gc.fillText(g.type, x, y + g.target * WIRE_WIDTH);
        return null;
    }
    
    public Object processUniformControlGate(UniformControlGate g) {
        int top = g.minQubit();
        int bot = g.maxQubit();
        CanvasFunctions.verticalLine(gc, x, y + top * WIRE_WIDTH,
            (bot - top) * WIRE_WIDTH);
        for(int i : g.controls) {
            CanvasFunctions.openRect(gc, x, y + i * WIRE_WIDTH,
                SMALL_BOX_WIDTH, SMALL_BOX_HEIGHT);
        }
        CanvasFunctions.openRect(gc, x, y + g.target * WIRE_WIDTH,
            Width.computeWidth(g), Artist.BIG_BOX_HEIGHT);
        gc.fillText(g.type, x, y + g.target * WIRE_WIDTH);
        return null;
    }
    
    public Object processRotationGate(RotationGate g) {
        CanvasFunctions.openRect(gc, x, y + g.target * WIRE_WIDTH,
            Width.computeWidth(g), Artist.BIG_BOX_HEIGHT);
        gc.fillText(g.type, x, y + g.target * WIRE_WIDTH);
        return null;
    }
    
    public Object processBigBlock(BigBlock b) {
        double yy = y + (b.bot + b.top) * WIRE_WIDTH / 2;
        CanvasFunctions.openRect(gc, x, yy,
            Width.computeWidth(b), Artist.BIG_BOX_HEIGHT 
            + WIRE_WIDTH * (b.top - b.bot));
        gc.fillText(b.type, x, yy);
        return null;
    }
}
