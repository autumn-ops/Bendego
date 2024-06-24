package application;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;

public class Main extends Application {
	@Override
	public void start(Stage primaryStage) {
		try {
			BorderPane root = (BorderPane)FXMLLoader.load(getClass().getResource("Viewer.fxml"));
			Scene scene = new Scene(root,720,480);
			scene.getStylesheets().add(getClass().getResource("application.css").toExternalForm());
			primaryStage.setScene(scene);
			primaryStage.show();
			if (Controller.mainThread != null && Controller.mainThread.isAlive()) {
                System.out.println("終了操作の検知");
                Training.stop.set(true);
                Controller.mainThread.interrupt();
                try {
                    Controller.mainThread.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		launch(args);
	}
}
