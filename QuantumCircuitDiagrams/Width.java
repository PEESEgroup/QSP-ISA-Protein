import javafx.scene.text.Text;
import javafx.scene.text.Font;
import java.util.*;

/**
 * Class representing a function that takes a gate and returns how wide it is,
 * including all forms of padding.
 */
class Width implements Operation<Double> {
    private static final Font FONT = new Font(Artist.FONT, Artist.FONT_SIZE);
    private static final Width UNIT = new Width();
    
    /**
     * Computes the width of the gate [g], including inner padding but
     * excluding outer padding
     */
    public static double computeWidth(Gate g) {
        return g.invoke(UNIT);
    }
    
    /**
     * Computes the width of the block of gates [g]. Essentially, width of
     * widest gate + outer padding.
     */
    public static double computeBlockWidth(Gate[] g) {
        double output = 0;
        for(Gate gg : g) {
            double width = Width.computeWidth(gg);
            if(width > output) output = width;
        }
        return output + 2 * Artist.PADDING_OUTER;
    }
    
    private Width() {
        
    }
    
    public Double processControlXGate(ControlXGate g) {
        return Artist.XOR_WIDTH;
    }
    public Double processControlGate(ControlGate g) {
        return Width.computeGateWidth(g.type);
    }
    public Double processRotationGate(RotationGate g) {
        return Width.computeGateWidth(g.type);
    }
    public Double processUniformControlGate(UniformControlGate g) {
        return Width.computeGateWidth(g.type);
    }
    public Double processBigBlock(BigBlock b) {
        return Width.computeGateWidth(b.type);
    }
    private static double computeStringWidth(String s) {
        Text t = new Text(s);
        t.setFont(FONT);
        return t.getLayoutBounds().getWidth();
    }
    private static double computeGateWidth(String s) {
        return Math.max(50, computeStringWidth(s) + 2 * Artist.PADDING_INNER);
    }
}