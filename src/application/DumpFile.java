package application;

import java.io.File;
import java.util.ArrayList;

class DumpFile {
	
	ArrayList<File> list = new ArrayList<File>();
	
	ArrayList<File> run(File in_path) {
		dumpFile(in_path, list);
		return list;
	}
	
	private void dumpFile(File file ,ArrayList<File> array){
		
    	File[] files = file.listFiles();
    	
    	for (File tmpFile : files) {
    		if(tmpFile.isHidden() == false ) {
    			
    			if(tmpFile.isDirectory()){
    				
    				if(!tmpFile.getName().equals("CaptureOne")) {
    					dumpFile(tmpFile, array);
    				}
    				
    			}else{
    				array.add(tmpFile);
                }
    		}
        }
    }
}