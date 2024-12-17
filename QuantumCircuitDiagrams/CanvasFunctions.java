import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

class CanvasFunctions {
    public static final double LINE_WIDTH = 2;
    public static final Color BLACK = Color.BLACK;
    public static final Color WHITE = Color.WHITE;
    
    public static void fillRect(GraphicsContext gc, double center_x, double center_y,
        double width, double height) {
        gc.fillRect(center_x - width / 2, center_y - height / 2, width, height);
    }
    
    public static void openRect(GraphicsContext gc, double center_x, double center_y,
        double width, double height) {
        fillRect(gc, center_x, center_y, width, height);
        gc.setFill(WHITE);
        fillRect(gc, center_x, center_y, width - 2 * LINE_WIDTH,
            height - 2 * LINE_WIDTH);
        gc.setFill(BLACK);
    }
    
    public static void fillCircle(GraphicsContext gc, double center_x, 
        double center_y, double radius) {
        double width = 2 * radius;
        gc.fillOval(center_x - radius, center_y - radius, width, width);
    }
    
    public static void openCircle(GraphicsContext gc, double center_x,
        double center_y, double radius) {
        fillCircle(gc, center_x, center_y, radius);
        gc.setFill(WHITE);
        fillCircle(gc, center_x, center_y, radius - LINE_WIDTH);
        gc.setFill(BLACK);
    }
    
    public static void horizontalLine(GraphicsContext gc, double x, double y, 
        double length) {
        gc.fillRect(x, y - LINE_WIDTH / 2, length, LINE_WIDTH);
    }
    
    public static void verticalLine(GraphicsContext gc, double x, double y,
        double length) {
        gc.fillRect(x - LINE_WIDTH / 2, y, LINE_WIDTH, length);
    }
}