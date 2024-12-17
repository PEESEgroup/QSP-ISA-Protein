import javafx.application.Application;
import javafx.stage.Stage;
import javafx.scene.*;
import javafx.scene.paint.*;
import javafx.scene.canvas.*;
import javafx.scene.text.*;
import javafx.geometry.VPos;
import javafx.geometry.Orientation;

/**
 * Main class
 */
public class Main extends Application {
    public static void main(String[] args) {
        System.out.println("Hello World");
    }
    
    public void start(Stage stage) {
        final Canvas canvas = new Canvas(1200, 600);
        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.setFont(new Font("Arial", 24));
        gc.setTextAlign(TextAlignment.CENTER);
        gc.setTextBaseline(VPos.CENTER);
        gc.setFill(CanvasFunctions.BLACK);
        
        QuantumCircuit circuit = new QuantumCircuit(5);
        circuit.bigBlock(0, 2, "Controlled\nSubstate Merging");
        circuit.bigBlock(2, 4, "Disentangle\nthird qubit");
        circuit.bigBlock(0, 2, "Controlled\nSubstate Merging");
        circuit.bigBlock(2, 4, "SP3 base case");

        Artist.draw(circuit, gc, 100, 100);
        
        Group root = new Group(canvas);
        Scene s = new Scene(root);
        stage.setScene(s);
        stage.show();
    }
    
}
