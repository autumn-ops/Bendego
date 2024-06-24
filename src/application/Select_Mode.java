package application;

import java.io.File;

class Select_Mode {

	public Runnable runnable;
	
	void run(String mode, File in_path, File out_path) {

		switch(mode){
		
		  case "Analysis":  //
			  runnable = new Analysis(mode, in_path, out_path);
			  break;
			  
		  case "Count":  //
			  runnable = new Count(mode, in_path, out_path);
			  break;
			  
		  case "Cut":  //
			  runnable = new Cut(mode, in_path, out_path);
			  break;
			    
		  case "PDF":  //
			  runnable = new PDF(mode, in_path, out_path);
			  break;
			    
		  case "Thumb":  //
			  runnable = new Thumb(mode, in_path, out_path);
			  break;
			    
		  case "Sort":  //
			  runnable = new Sort(mode, in_path, out_path);
			  break;
			    
		  case "Train":  //
			  runnable = new Training(mode, in_path, out_path);
			  break;
						  
			  
			  
		}
		if (Controller.mainThread == null || !Controller.mainThread.isAlive()) {
			Controller.mainThread = new Thread(runnable);
			Controller.mainThread.start();
        }
	}
}