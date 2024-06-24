package application;

import java.io.File;
import java.util.ArrayList;

class DumpDir {
	
	ArrayList<File> list = new ArrayList<File>();
	
	ArrayList<File> run(File in_path) {
		dumpDir(in_path, list);
		return list;
	}
	
	private ArrayList<File> dumpDir(File file ,ArrayList<File> array){
		
    	File[] files = file.listFiles();
    	
    	for (File tmpFile : files) {
    		if(tmpFile.isHidden() == false ) {
    			
    			if(tmpFile.isDirectory()){
    				
    				if(!tmpFile.getName().equals("CaptureOne")) {
        				array.add(tmpFile);
    					dumpDir(tmpFile, array);
    				}
    			}
    		}
        }
    	return array;
    }
}